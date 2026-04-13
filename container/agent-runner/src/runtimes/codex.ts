/**
 * OpenAI Codex app-server runtime for the container agent-runner.
 *
 * Uses JSON-RPC over stdio to communicate with `codex app-server`.
 * Replaces the previous SDK-based approach (thread.runStreamed) with the
 * richer app-server protocol that supports native compaction, skill
 * discovery, thread health checks, and eliminates config.toml generation.
 *
 * Self-registers with the container runtime registry.
 */
import fs from 'fs';
import path from 'path';

import {
  ContainerInput,
  drainIpcInput,
  getContainerBaseUrl,
  getContainerModel,
  log,
  shouldClose,
  writeOutput,
} from '../shared.js';
import { registerContainerRuntime, type QueryResult } from '../runtime-registry.js';
import {
  type AppServer,
  type JsonRpcNotification,
  CODEX_TOOL_GUIDANCE,
  attachCodexAutoApproval,
  buildCodexMcpConfig,
  createCodexConfigOverrides,
  initializeCodexAppServer,
  killCodexAppServer,
  sendCodexRequest,
  spawnCodexAppServer,
  startCodexTurn,
  startOrResumeCodexThread,
  writeCodexMcpConfigToml,
} from './codex-app-server.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Input tokens before triggering native compaction */
const COMPACT_THRESHOLD = 40_000;

/** Timeout for the entire turn (5 minutes) */
const TURN_TIMEOUT_MS = 5 * 60 * 1000;

// ---------------------------------------------------------------------------
// AGENTS.md assembly — persona only, no skills (app-server handles those)
// ---------------------------------------------------------------------------

function assembleAgentsMd(): void {
  const agentsParts: string[] = [];

  for (const dir of ['/workspace/global', '/workspace/group']) {
    for (const filename of ['AGENT.md', 'CLAUDE.md']) {
      const filePath = path.join(dir, filename);
      if (fs.existsSync(filePath)) {
        agentsParts.push(fs.readFileSync(filePath, 'utf-8'));
        break;
      }
    }
  }

  agentsParts.push(CODEX_TOOL_GUIDANCE);

  if (agentsParts.length > 0) {
    fs.writeFileSync(
      '/workspace/group/AGENTS.md',
      agentsParts.join('\n\n---\n\n'),
    );
    log(`Assembled AGENTS.md from ${agentsParts.length} source(s)`);
  }
}

// ---------------------------------------------------------------------------
// Main query handler
// ---------------------------------------------------------------------------

async function runCodexQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
): Promise<QueryResult> {
  const modelRef = getContainerModel(containerInput, 'gpt-5.4-mini');
  const baseUrl = getContainerBaseUrl(containerInput) || process.env.OPENAI_BASE_URL;

  // Assemble AGENTS.md — persona + tool guidance only.
  // Skills are discovered natively by the app-server via skills/list.
  assembleAgentsMd();

  // Build config overrides (replaces config.toml)
  const configOverrides = createCodexConfigOverrides(baseUrl);

  // Build MCP server configs and write config.toml
  const mcpServers = buildCodexMcpConfig(mcpServerPath, containerInput, modelRef);
  writeCodexMcpConfigToml(mcpServers);

  // Spawn the app-server
  const server = spawnCodexAppServer(configOverrides);
  attachCodexAutoApproval(server);

  let newThreadId: string | undefined;
  let closedDuringQuery = false;

  try {
    await initializeCodexAppServer(server);

    // Pass AGENTS.md content explicitly via baseInstructions.
    // The JSON-RPC protocol does not auto-read AGENTS.md from cwd.
    const agentsMdContent = fs.existsSync('/workspace/group/AGENTS.md')
      ? fs.readFileSync('/workspace/group/AGENTS.md', 'utf-8')
      : undefined;

    const threadParams = {
      model: modelRef,
      cwd: '/workspace/group',
      sandbox: 'danger-full-access' as const,
      approvalPolicy: 'never' as const,
      personality: 'friendly' as const,
      baseInstructions: agentsMdContent,
    };

    const threadId = await startOrResumeCodexThread(server, sessionId, threadParams);
    newThreadId = threadId;

    // Collect streaming output
    const turnState = {
      resultText: '',
      toolCalls: [] as string[],
      turnComplete: false,
      totalInputTokens: 0,
    };

    const turnPromise = new Promise<void>((resolve, reject) => {
      const turnTimeout = setTimeout(() => {
        reject(new Error(`Turn timed out after ${TURN_TIMEOUT_MS}ms`));
      }, TURN_TIMEOUT_MS);

      server.notificationHandlers.push((notification: JsonRpcNotification) => {
        const method = notification.method;
        const params = notification.params;

        switch (method) {
          case 'thread/started': {
            const thread = params.thread as { id?: string } | undefined;
            if (thread?.id) {
              newThreadId = thread.id;
              log(`Thread started: ${newThreadId}`);
            }
            break;
          }

          case 'item/agentMessage/delta': {
            const delta = params.delta as string;
            if (delta) turnState.resultText += delta;
            break;
          }

          case 'item/started': {
            const item = params.item as { type?: string; command?: string; server?: string; tool?: string; query?: string; changes?: { path: string }[] } | undefined;
            if (!item) break;
            if (item.type === 'commandExecution' && item.command) {
              log(`[tool] Running: ${item.command}`);
              turnState.toolCalls.push(`$ ${item.command}`);
            } else if (item.type === 'mcpToolCall' && item.server && item.tool) {
              log(`[tool] MCP: ${item.server}/${item.tool}`);
              turnState.toolCalls.push(`MCP: ${item.server}/${item.tool}`);
            } else if (item.type === 'webSearch' && item.query) {
              log(`[tool] Web search: ${item.query}`);
              turnState.toolCalls.push(`Search: ${item.query}`);
            } else if (item.type === 'fileChange' && item.changes) {
              const paths = item.changes.map((c) => c.path).join(', ');
              log(`[tool] File changes: ${paths}`);
              turnState.toolCalls.push(`Files: ${paths}`);
            }
            break;
          }

          case 'item/completed': {
            const item = params.item as { type?: string; aggregated_output?: string; text?: string } | undefined;
            if (item?.type === 'commandExecution' && item.aggregated_output) {
              turnState.toolCalls.push(item.aggregated_output.slice(0, 500));
            }
            if (item?.type === 'agentMessage' && item.text) {
              turnState.resultText = item.text;
            }
            break;
          }

          case 'item/commandExecution/outputDelta':
            break;

          case 'thread/tokenUsage/updated': {
            const usage = params.tokenUsage as { total?: { inputTokens?: number; outputTokens?: number } } | undefined;
            if (usage?.total) {
              turnState.totalInputTokens = usage.total.inputTokens || 0;
              log(`Token usage: ${usage.total.inputTokens} in, ${usage.total.outputTokens} out (total)`);
            }
            break;
          }

          case 'thread/compacted':
            log('Thread compacted by server');
            break;

          case 'turn/started':
            log('Turn started');
            break;

          case 'turn/completed':
            turnState.turnComplete = true;
            clearTimeout(turnTimeout);
            resolve();
            break;

          case 'thread/status/changed':
            log(`Thread status: ${params.status}`);
            break;

          case 'item/reasoning/summaryTextDelta':
          case 'turn/diff/updated':
          case 'turn/plan/updated':
            break;

          default:
            if (!method.startsWith('item/')) {
              log(`[notification] ${method}`);
            }
            break;
        }
      });
    });

    await startCodexTurn(server, {
      threadId,
      inputText: prompt,
      model: modelRef,
      cwd: '/workspace/group',
    });

    // Poll for IPC messages during the turn
    let ipcPolling = true;
    const pollIpc = () => {
      if (!ipcPolling) return;
      if (shouldClose()) {
        closedDuringQuery = true;
        ipcPolling = false;
        return;
      }
      setTimeout(pollIpc, 500);
    };
    setTimeout(pollIpc, 500);

    await turnPromise;
    ipcPolling = false;

    log(`Turn complete. Result: ${turnState.resultText.length} chars, ${turnState.toolCalls.length} tool calls`);

    // Trigger native compaction if cumulative tokens exceed threshold
    if (turnState.totalInputTokens >= COMPACT_THRESHOLD) {
      log(`Compaction threshold reached (${turnState.totalInputTokens}/${COMPACT_THRESHOLD} tokens). Compacting...`);
      const compactResp = await sendCodexRequest(server, 'thread/compact/start', {
        threadId: newThreadId,
      });
      if (compactResp.error) {
        log(`Compaction failed: ${compactResp.error.message}. Continuing uncompacted.`);
      } else {
        log('Native compaction completed');
      }
    }

    writeOutput({ status: 'success', result: turnState.resultText || null, newSessionId: newThreadId });

    // Process IPC messages that arrived during the turn
    const pendingMessages = drainIpcInput();
    if (pendingMessages.length > 0 && !closedDuringQuery) {
      log(`Processing ${pendingMessages.length} IPC message(s) that arrived during turn`);
      const followUp = pendingMessages.join('\n');

      turnState.resultText = '';
      turnState.turnComplete = false;

      const followUpTurnPromise = new Promise<void>((resolve) => {
        const checkComplete = () => {
          if (turnState.turnComplete) { resolve(); return; }
          setTimeout(checkComplete, 100);
        };
        setTimeout(checkComplete, 100);
      });

      try {
        await startCodexTurn(server, {
          threadId: newThreadId,
          inputText: followUp,
          model: modelRef,
        });
        await followUpTurnPromise;
        if (turnState.resultText) {
          writeOutput({ status: 'success', result: turnState.resultText, newSessionId: newThreadId });
        }
      } catch (err) {
        log(`Follow-up turn failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    log(`Codex error: ${error}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: newThreadId,
      error,
    });
  } finally {
    killCodexAppServer(server);
  }

  return { newSessionId: newThreadId, closedDuringQuery };
}

// --- Self-register ---

registerContainerRuntime('codex', runCodexQuery);
