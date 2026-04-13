/**
 * OpenAI Codex SDK runtime for the container agent-runner.
 * Self-registers with the container runtime registry.
 */
import fs from 'fs';
import path from 'path';
import { Codex } from '@openai/codex-sdk';

import {
  ContainerInput,
  drainIpcInput,
  formatTranscriptMarkdown,
  getContainerBaseUrl,
  getContainerModel,
  log,
  ParsedMessage,
  sanitizeFilename,
  shouldClose,
  writeOutput,
} from '../shared.js';
import { getProviderCodexToml } from '../provider-registry.js';
import { registerContainerRuntime, type QueryResult } from '../runtime-registry.js';

// --- Auto-compaction ---
// Mimics Claude's auto_compact: when cumulative input tokens exceed
// the threshold, the current thread generates a summary, and the next
// query starts a fresh thread with the summary as context.

const COMPACT_THRESHOLD = 40_000; // input tokens before triggering compaction
const COMPACT_STATE_FILE = '/workspace/group/.codex-compact-state.json';

interface CompactState {
  cumulativeInputTokens: number;
  sessionId?: string;
}

function loadCompactState(): CompactState {
  try {
    if (fs.existsSync(COMPACT_STATE_FILE)) {
      return JSON.parse(fs.readFileSync(COMPACT_STATE_FILE, 'utf-8'));
    }
  } catch { /* ignore corrupt state */ }
  return { cumulativeInputTokens: 0 };
}

function saveCompactState(state: CompactState): void {
  fs.writeFileSync(COMPACT_STATE_FILE, JSON.stringify(state));
}

/** Codex-specific tool guidance — injected into AGENTS.md during assembly */
const CODEX_TOOL_GUIDANCE = `
## File and Shell Best Practices

**IMPORTANT: Use MCP file tools instead of bash for file operations.** They are faster (no shell overhead) and return cleaner output.

- **Reading files:** Use \`mcp__nanoclaw__file_read\` (not cat/head/tail/sed)
- **Writing files:** Use \`mcp__nanoclaw__file_write\` (not echo/cat heredoc)
- **Editing files:** Use \`mcp__nanoclaw__file_edit\` (not sed/awk)
- **Finding files:** Use \`mcp__nanoclaw__file_glob\` (not find/ls)
- **Searching content:** Use \`mcp__nanoclaw__file_grep\` (not grep/rg)
- **Running commands:** Use bash only for system commands, git, and tools that aren't file operations

These MCP tools execute locally with no shell overhead. Each bash tool call requires a full API round-trip — MCP tools are significantly faster.
`;

// --- Codex query ---

async function runCodexQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
): Promise<QueryResult> {
  const runtimeOptions = containerInput.runtimeOptions || {};
  const modelRef = getContainerModel(containerInput, 'gpt-5.4-mini');
  const baseUrl = getContainerBaseUrl(containerInput) || process.env.OPENAI_BASE_URL;
  const sandboxProfile =
    runtimeOptions.sandboxProfile === 'operator' ? 'operator' : 'safe';

  // Assemble AGENTS.md from global + group AGENT.md files
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

  // Provider docs (MS365, GWS usage instructions) are NOT injected globally.
  // They live in the skills that use them (/email-archive, /add-email-account)
  // and are only loaded when the user's prompt matches a skill trigger.

  // Pre-inject skills into AGENTS.md to avoid Codex reading them via tool calls.
  // Match the user's prompt against skill names/descriptions to inject the full
  // SKILL.md for relevant skills. Include a summary index of all skills so the
  // agent knows what's available without reading each file.
  const skillsDir = path.join(process.env.HOME || '/home/node', '.codex', 'skills');
  if (fs.existsSync(skillsDir)) {
    const skillIndex: string[] = ['## Available Skills'];
    const promptLower = containerInput.prompt.toLowerCase();

    for (const entry of fs.readdirSync(skillsDir)) {
      const skillMd = path.join(skillsDir, entry, 'SKILL.md');
      if (!fs.existsSync(skillMd)) continue;

      const content = fs.readFileSync(skillMd, 'utf-8');
      // Parse frontmatter name and description
      const nameMatch = content.match(/^name:\s*(.+)$/m);
      const descMatch = content.match(/^description:\s*(.+)$/m);
      const name = nameMatch?.[1]?.trim() || entry;
      const desc = descMatch?.[1]?.trim() || '';

      skillIndex.push(`- **/${name}** — ${desc}`);

      // Inject full content if the prompt references this skill
      const triggers = [
        `/${name}`,
        `/${entry}`,
        name.replace(/-/g, ' '),
        entry.replace(/-/g, ' '),
      ];
      const isRelevant = triggers.some((t) => promptLower.includes(t.toLowerCase()));
      if (isRelevant) {
        agentsParts.push(`## Skill: /${name}\n\n${content}`);
        log(`Pre-injected skill: ${name}`);
      }
    }

    agentsParts.push(skillIndex.join('\n'));
  }

  if (agentsParts.length > 0) {
    fs.writeFileSync(
      '/workspace/group/AGENTS.md',
      agentsParts.join('\n\n---\n\n'),
    );
    log(`Assembled AGENTS.md from ${agentsParts.length} source(s)`);
  }

  // Write MCP server config for Codex.
  // Strip all [mcp_servers.*] from any existing config and rewrite fresh.
  // This prevents stale/duplicate MCP blocks from crashing Codex.
  const codexConfigDir = path.join(
    process.env.HOME || '/home/node',
    '.codex',
  );
  fs.mkdirSync(codexConfigDir, { recursive: true });
  const configTomlPath = path.join(codexConfigDir, 'config.toml');

  // Preserve non-MCP settings (plugins, features, projects, sandbox)
  let baseConfig = '';
  if (fs.existsSync(configTomlPath)) {
    const existing = fs.readFileSync(configTomlPath, 'utf-8');
    // Keep everything before the first [mcp_servers section
    const mcpIdx = existing.indexOf('[mcp_servers');
    baseConfig = mcpIdx !== -1 ? existing.slice(0, mcpIdx).trimEnd() : existing.trimEnd();
  }

  // Ensure sandbox settings exist
  if (!baseConfig.includes('[features]')) {
    baseConfig += `\n\n# Disable bwrap sandbox — container is already sandboxed by Docker\n[features]\nuse_linux_sandbox_bwrap = false\n\n[sandbox_workspace_write]\nnetwork_access = true`;
  }

  // Build fresh MCP server config
  const mcpConfig = `
[mcp_servers.nanoclaw]
type = "stdio"
command = "node"
args = ["${mcpServerPath}"]
[mcp_servers.nanoclaw.env]
NANOCLAW_CHAT_JID = "${containerInput.chatJid}"
NANOCLAW_GROUP_FOLDER = "${containerInput.groupFolder}"
NANOCLAW_IS_MAIN = "${containerInput.isMain ? '1' : '0'}"
NANOCLAW_RUNTIME = "codex"
NANOCLAW_MODEL = "${modelRef}"
`;

  // Provider-based MCP servers (MS365, etc.)
  const providerToml = getProviderCodexToml();

  fs.writeFileSync(
    configTomlPath,
    baseConfig + '\n\n' + mcpConfig + (providerToml ? '\n' + providerToml + '\n' : ''),
  );
  log('Wrote MCP config to Codex config.toml (clean rebuild)');

  // Discover additional directories (Fix #3)
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) extraDirs.push(fullPath);
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  const codex = new Codex({
    apiKey: process.env.OPENAI_API_KEY,
    baseUrl,
    config: {
      // Disable bwrap sandbox — container is already sandboxed by Docker
      features: { use_linux_sandbox_bwrap: false },
      sandbox_workspace_write: { network_access: true },
    },
  });

  // ARCHITECTURE DECISION: sandboxMode is 'danger-full-access' because the
  // Docker container IS the sandbox. The Codex inner sandbox (bubblewrap) is
  // disabled since the container already restricts mounts, network, and UID.
  // The host launcher (container-runner.ts) controls what's mounted and only
  // applies seccomp=unconfined for Codex (needed for user namespaces).
  // If NanoClaw becomes multi-tenant, enforce sandboxProfile here per group.
  const threadOptions = {
    model: modelRef,
    workingDirectory: '/workspace/group',
    sandboxMode: 'danger-full-access' as const,
    networkAccessEnabled: true,
    approvalPolicy: 'never' as const,
    skipGitRepoCheck: true,
    additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
  };
  log(`Codex sandbox profile: ${sandboxProfile}`);
  // TODO: Enforce safe/operator at the container launch boundary instead of
  // only carrying the profile through runtimeOptions into the runner.

  // Auto-compaction: check if we should start fresh due to context size
  const compactState = loadCompactState();
  let compactedSessionId = sessionId;
  let compactionSummary: string | null = null;

  if (
    sessionId &&
    compactState.cumulativeInputTokens >= COMPACT_THRESHOLD &&
    compactState.sessionId === sessionId
  ) {
    log(`Auto-compaction triggered (${compactState.cumulativeInputTokens} input tokens). Generating summary...`);
    try {
      // Ask the current thread for a summary before discarding it
      const summaryThread = codex.resumeThread(sessionId, threadOptions);
      const summaryTurn = await summaryThread.runStreamed(
        'Summarize our conversation so far in 300 words. Focus on: what tasks were completed, what decisions were made, what the user is currently working on, and any pending items. Be concise and factual.',
      );
      for await (const event of summaryTurn.events) {
        if (event.type === 'item.completed' && event.item.type === 'agent_message') {
          compactionSummary = event.item.text;
        }
      }
      log(`Compaction summary: ${compactionSummary?.length || 0} chars`);
    } catch (err) {
      log(`Failed to generate compaction summary: ${err instanceof Error ? err.message : String(err)}`);
    }
    // Reset state — fresh session
    compactedSessionId = undefined;
    saveCompactState({ cumulativeInputTokens: 0 });
  }

  // Try resuming a previous thread if we have a session ID.
  // Fall back to a fresh thread if resume fails (e.g. "no rollout found").
  let thread;
  if (compactedSessionId) {
    try {
      thread = codex.resumeThread(compactedSessionId, threadOptions);
      log(`Resuming Codex thread: ${compactedSessionId}`);
    } catch (err) {
      log(`Resume failed (${err instanceof Error ? err.message : String(err)}), starting fresh thread`);
      thread = codex.startThread(threadOptions);
    }
  } else {
    thread = codex.startThread(threadOptions);
  }

  // If we just compacted, prepend the summary to the user's prompt
  if (compactionSummary) {
    prompt = `[CONTEXT FROM PREVIOUS SESSION]\n${compactionSummary}\n\n[NEW MESSAGE]\n${prompt}`;
    log('Injected compaction summary into prompt');
  }

  let closedDuringQuery = false;
  let newSessionId: string | undefined;

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

  try {
    // Use streaming for progress visibility
    const streamedTurn = await thread.runStreamed(prompt);
    let resultText: string | null = null;
    let usage: { input_tokens: number; output_tokens: number } | null = null;

    // Build rich archive from stream events (Fix #2)
    const archiveMessages: ParsedMessage[] = [
      { role: 'user', content: prompt },
    ];
    const toolCalls: string[] = [];

    for await (const event of streamedTurn.events) {
      if (event.type === 'thread.started') {
        newSessionId = event.thread_id;
        log(`Codex thread: ${newSessionId} (new)`);
      }

      // Log and capture tool activity
      if (event.type === 'item.started') {
        const item = event.item;
        if (item.type === 'command_execution') {
          log(`[tool] Running: ${item.command}`);
          toolCalls.push(`$ ${item.command}`);
        } else if (item.type === 'mcp_tool_call') {
          log(`[tool] MCP: ${item.server}/${item.tool}`);
          toolCalls.push(`MCP: ${item.server}/${item.tool}`);
        } else if (item.type === 'web_search') {
          log(`[tool] Web search: ${item.query}`);
          toolCalls.push(`Search: ${item.query}`);
        } else if (item.type === 'file_change') {
          const paths = item.changes.map((c: { path: string }) => c.path).join(', ');
          log(`[tool] File changes: ${paths}`);
          toolCalls.push(`Files: ${paths}`);
        }
      }

      // Capture command output for archive
      if (event.type === 'item.completed' && event.item.type === 'command_execution') {
        const cmd = event.item;
        if (cmd.aggregated_output) {
          toolCalls.push(cmd.aggregated_output.slice(0, 500));
        }
      }

      if (event.type === 'item.completed' && event.item.type === 'agent_message') {
        resultText = event.item.text;
      }

      if (event.type === 'turn.completed') {
        usage = event.usage;
      }
    }

    if (!newSessionId) {
      newSessionId = thread.id || undefined;
    }

    if (usage) {
      log(`Codex usage: ${usage.input_tokens} in, ${usage.output_tokens} out`);
      // Track cumulative input tokens for auto-compaction
      const updatedState = loadCompactState();
      updatedState.cumulativeInputTokens += usage.input_tokens;
      updatedState.sessionId = newSessionId;
      saveCompactState(updatedState);
      log(`Cumulative input tokens: ${updatedState.cumulativeInputTokens}/${COMPACT_THRESHOLD}`);
    }

    // Rich conversation archive (Fix #2)
    if (resultText || toolCalls.length > 0) {
      try {
        if (toolCalls.length > 0) {
          archiveMessages.push({
            role: 'assistant',
            content: `[Tool calls]\n${toolCalls.join('\n')}`,
          });
        }
        if (resultText) {
          archiveMessages.push({ role: 'assistant', content: resultText });
        }

        const conversationsDir = '/workspace/group/conversations';
        fs.mkdirSync(conversationsDir, { recursive: true });
        const date = new Date().toISOString().split('T')[0];
        const name = sanitizeFilename(prompt.slice(0, 50).replace(/\n/g, ' '));
        const filePath = path.join(conversationsDir, `${date}-${name || 'conversation'}.md`);
        fs.writeFileSync(
          filePath,
          formatTranscriptMarkdown(
            archiveMessages,
            prompt.slice(0, 50),
            containerInput.assistantName,
          ),
        );
        log(`Archived Codex conversation to ${filePath}`);
      } catch (err) {
        log(`Failed to archive: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    writeOutput({ status: 'success', result: resultText, newSessionId });

    // Check for IPC messages that arrived during this turn (Fix #1)
    // Feed them as a follow-up turn so the user's mid-turn messages
    // get processed immediately instead of waiting for next container cycle.
    ipcPolling = false;
    const pendingMessages = drainIpcInput();
    if (pendingMessages.length > 0 && !closedDuringQuery) {
      log(`Processing ${pendingMessages.length} IPC message(s) that arrived during turn`);
      const followUp = pendingMessages.join('\n');
      const followUpTurn = await thread.runStreamed(followUp);
      let followUpResult: string | null = null;

      for await (const event of followUpTurn.events) {
        if (event.type === 'item.completed' && event.item.type === 'agent_message') {
          followUpResult = event.item.text;
        }
      }

      if (followUpResult) {
        writeOutput({ status: 'success', result: followUpResult, newSessionId });
      }
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    log(`Codex error: ${error}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: newSessionId || thread.id || undefined,
      error,
    });
  }

  ipcPolling = false;
  return { newSessionId, closedDuringQuery };
}

// --- Self-register ---

registerContainerRuntime('codex', runCodexQuery);
