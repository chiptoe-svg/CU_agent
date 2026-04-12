/**
 * Google Gemini runtime for the container agent-runner.
 * Self-registers with the container runtime registry.
 *
 * Uses Google ADK (Agent Development Kit) — a Python FastAPI sidecar
 * that handles reasoning, tool calling, sub-agents, and session state.
 */
import fs from 'fs';
import path from 'path';
import { spawn, ChildProcess } from 'child_process';

import {
  ContainerInput,
  drainIpcInput,
  formatTranscriptMarkdown,
  log,
  ParsedMessage,
  sanitizeFilename,
  shouldClose,
  writeOutput,
} from '../shared.js';
import { getProviderAgentDocs } from '../provider-registry.js';
import { registerContainerRuntime, type QueryResult } from '../runtime-registry.js';

// --- ADK server management ---

const ADK_PORT = 8765;
const ADK_HOST = '127.0.0.1';
const ADK_BASE = `http://${ADK_HOST}:${ADK_PORT}`;
const ADK_APP = 'nanoclaw_agent';
const ADK_USER = 'default';

let adkProcess: ChildProcess | null = null;

async function startAdkServer(containerInput: ContainerInput, mcpServerPath: string): Promise<void> {
  if (adkProcess) return;

  const adkAgentDir = '/app/adk';
  if (!fs.existsSync(path.join(adkAgentDir, ADK_APP, '__init__.py'))) {
    throw new Error('ADK agent not found at /app/adk/nanoclaw_agent/. Rebuild the container image.');
  }

  const model = containerInput.model || 'gemini-2.5-flash';
  log(`Starting ADK server on port ${ADK_PORT} (model: ${model})`);

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    NANOCLAW_MODEL: model,
    NANOCLAW_MCP_SERVER: mcpServerPath,
    NANOCLAW_CHAT_JID: containerInput.chatJid,
    NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
    NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
    NANOCLAW_WORKSPACE: '/workspace/group',
  };

  if (process.env.GEMINI_API_KEY) env.GOOGLE_API_KEY = process.env.GEMINI_API_KEY;
  if (process.env.GOOGLE_API_KEY) env.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
  if (containerInput.baseUrl) env.GOOGLE_GEMINI_BASE_URL = containerInput.baseUrl;

  adkProcess = spawn('adk', [
    'api_server',
    '--host', ADK_HOST,
    '--port', String(ADK_PORT),
    '--session_service_uri', 'sqlite:///workspace/group/.adk-sessions.db',
    '--auto_create_session',
    adkAgentDir,
  ], {
    cwd: '/workspace/group',
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  adkProcess.stdout?.on('data', (chunk: Buffer) => {
    for (const line of chunk.toString().trim().split('\n')) {
      if (line) log(`[adk] ${line}`);
    }
  });

  adkProcess.stderr?.on('data', (chunk: Buffer) => {
    for (const line of chunk.toString().trim().split('\n')) {
      if (line) log(`[adk] ${line}`);
    }
  });

  adkProcess.on('close', (code) => {
    log(`ADK server exited with code ${code}`);
    adkProcess = null;
  });

  const ready = await waitForHealth(15_000);
  if (!ready) {
    adkProcess?.kill();
    adkProcess = null;
    throw new Error('ADK server failed to start. Check that google-adk is installed in the container.');
  }

  log('ADK server ready');
}

async function waitForHealth(timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${ADK_BASE}/health`);
      if (res.ok) return true;
    } catch {
      // Not ready yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

// --- ADK query ---

async function runAdkQuery(
  prompt: string,
  sessionId: string | undefined,
): Promise<{ resultText: string | null; toolCalls: string[]; newSessionId?: string }> {
  let sid = sessionId;
  if (!sid) {
    const res = await fetch(
      `${ADK_BASE}/apps/${ADK_APP}/users/${ADK_USER}/sessions`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
    );
    if (!res.ok) throw new Error(`Failed to create ADK session: ${res.status}`);
    const data = await res.json() as { id?: string; session_id?: string };
    sid = data.id || data.session_id;
    log(`ADK session created: ${sid}`);
  }

  if (!sid) throw new Error('Failed to create ADK session');

  const res = await fetch(`${ADK_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      app_name: ADK_APP,
      user_id: ADK_USER,
      session_id: sid,
      new_message: {
        role: 'user',
        parts: [{ text: prompt }],
      },
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ADK /run failed (${res.status}): ${text.slice(0, 200)}`);
  }

  const events = await res.json() as Array<{
    author?: string;
    content?: { role?: string; parts?: Array<{ text?: string; function_call?: { name: string } }> };
  }>;

  let resultText: string | null = null;
  const toolCalls: string[] = [];

  for (const event of events) {
    if (!event.content?.parts) continue;
    for (const part of event.content.parts) {
      if (part.function_call) {
        toolCalls.push(`Tool: ${part.function_call.name}`);
        log(`[adk-tool] ${part.function_call.name}`);
      }
      if (part.text && event.content.role === 'model') {
        resultText = (resultText || '') + part.text;
      }
    }
  }

  return { resultText, toolCalls, newSessionId: sid };
}

// --- Main entry point ---

async function runGeminiQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
): Promise<QueryResult> {
  // Symlink additional directories into the working directory
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      const linkPath = path.join('/workspace/group', `_extra_${entry}`);
      if (fs.statSync(fullPath).isDirectory() && !fs.existsSync(linkPath)) {
        try {
          fs.symlinkSync(fullPath, linkPath);
          log(`Symlinked extra dir: ${entry} → ${linkPath}`);
        } catch (err) {
          log(`Failed to symlink ${entry}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }
  }

  // Inject provider docs into AGENT.md so ADK agent sees them
  const providerDocs = getProviderAgentDocs();
  if (providerDocs) {
    const agentMdPath = '/workspace/group/AGENT.md';
    if (fs.existsSync(agentMdPath)) {
      const existing = fs.readFileSync(agentMdPath, 'utf-8');
      if (!existing.includes('<!-- provider-docs -->')) {
        fs.appendFileSync(agentMdPath, `\n\n<!-- provider-docs -->\n${providerDocs}\n`);
        log('Injected provider docs into AGENT.md for Gemini');
      }
    }
  }

  const model = containerInput.model || 'gemini-2.5-flash';
  let closedDuringQuery = false;

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
    await startAdkServer(containerInput, mcpServerPath);

    const { resultText, toolCalls, newSessionId } = await runAdkQuery(prompt, sessionId);

    // Archive conversation
    const archiveMessages: ParsedMessage[] = [
      { role: 'user', content: prompt },
    ];
    if (toolCalls.length > 0) {
      archiveMessages.push({
        role: 'assistant',
        content: `[Tool calls]\n${toolCalls.join('\n')}`,
      });
    }
    if (resultText) {
      archiveMessages.push({ role: 'assistant', content: resultText });
    }

    if (archiveMessages.length > 1) {
      try {
        const conversationsDir = '/workspace/group/conversations';
        fs.mkdirSync(conversationsDir, { recursive: true });
        const date = new Date().toISOString().split('T')[0];
        const name = sanitizeFilename(prompt.slice(0, 50).replace(/\n/g, ' '));
        const filePath = path.join(conversationsDir, `${date}-${name || 'conversation'}.md`);
        fs.writeFileSync(
          filePath,
          formatTranscriptMarkdown(archiveMessages, prompt.slice(0, 50), containerInput.assistantName),
        );
        log(`Archived Gemini conversation to ${filePath}`);
      } catch (err) {
        log(`Failed to archive: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    writeOutput({ status: 'success', result: resultText, newSessionId });

    // Post-turn follow-ups
    ipcPolling = false;
    const pendingMessages = drainIpcInput();
    if (pendingMessages.length > 0 && !closedDuringQuery) {
      log(`Processing ${pendingMessages.length} IPC message(s) that arrived during turn`);
      const followUp = pendingMessages.join('\n');
      try {
        const followUpResult = await runAdkQuery(followUp, newSessionId);
        if (followUpResult.resultText) {
          writeOutput({ status: 'success', result: followUpResult.resultText, newSessionId });
        }
      } catch (err) {
        log(`Follow-up error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    log(`Gemini error: ${error}`);
    writeOutput({ status: 'error', result: null, error });
  }

  ipcPolling = false;
  if (adkProcess) {
    adkProcess.kill('SIGTERM');
    adkProcess = null;
  }

  return { newSessionId: undefined, closedDuringQuery };
}

// --- Self-register ---

registerContainerRuntime('gemini', runGeminiQuery);
