/**
 * Codex app-server helpers — reusable JSON-RPC primitives.
 *
 * Used by the main Codex runtime (codex.ts). Keeps all Codex-specific
 * protocol logic in the container/runtime layer.
 */
import fs from 'fs';
import path from 'path';
import { spawn, type ChildProcess } from 'child_process';
import { createInterface, type Interface as ReadlineInterface } from 'readline';

import { log } from '../shared.js';
import { getProviderMcpConfigs } from '../provider-registry.js';
import type { ContainerInput } from '../shared.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Timeout for initialize handshake */
const INIT_TIMEOUT_MS = 30_000;

/** Codex-specific tool guidance — injected into AGENTS.md during assembly */
export const CODEX_TOOL_GUIDANCE = `
## File and Shell Best Practices

**IMPORTANT: Use MCP file tools instead of bash for file operations.** They are faster (no shell overhead) and return cleaner output.

- **Reading files:** Use \`mcp__nanoclaw__file_read\` (not cat/head/tail/sed)
- **Writing files:** Use \`mcp__nanoclaw__file_write\` (not echo/cat heredoc)
- **Editing files:** Use \`mcp__nanoclaw__file_edit\` (not sed/awk)
- **Finding files:** Use \`mcp__nanoclaw__file_glob\` (not find/ls)
- **Searching content:** Use \`mcp__nanoclaw__file_grep\` (not grep/rg)
- **Running commands:** Use bash only for system commands, git, and tools that aren't file operations

## Subagent Delegation

You have native multi-agent tools. For complex tasks that benefit from parallel work or focused expertise:

- \`spawnAgent\` — create a subagent with a specific task and persona
- \`sendInput\` — send additional context or follow-up to a running subagent
- \`wait\` — wait for a subagent to complete and get its results
- \`closeAgent\` — terminate a subagent when done

Use subagents for research, analysis, writing, or any task that can run independently. Each subagent gets its own thread and works in parallel. Always include full context in the spawn — subagents cannot see your conversation.
`;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

let nextRequestId = 1;

interface JsonRpcRequest {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

export interface JsonRpcResponse {
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

export interface JsonRpcNotification {
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcServerRequest {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

type JsonRpcMessage = JsonRpcResponse | JsonRpcNotification | JsonRpcServerRequest;

function makeRequest(method: string, params: Record<string, unknown>): JsonRpcRequest {
  return { id: nextRequestId++, method, params };
}

function isResponse(msg: JsonRpcMessage): msg is JsonRpcResponse {
  return 'id' in msg && ('result' in msg || 'error' in msg) && !('method' in msg);
}

function isServerRequest(msg: JsonRpcMessage): msg is JsonRpcServerRequest {
  return 'id' in msg && 'method' in msg;
}

// ---------------------------------------------------------------------------
// App-server process
// ---------------------------------------------------------------------------

export interface AppServer {
  process: ChildProcess;
  readline: ReadlineInterface;
  pending: Map<number, { resolve: (r: JsonRpcResponse) => void; reject: (e: Error) => void }>;
  notificationHandlers: ((n: JsonRpcNotification) => void)[];
  serverRequestHandlers: ((r: JsonRpcServerRequest) => void)[];
}

export function spawnCodexAppServer(configOverrides: string[]): AppServer {
  const args = ['app-server', '--listen', 'stdio://'];
  for (const override of configOverrides) {
    args.push('-c', override);
  }

  log(`Spawning: codex ${args.join(' ')}`);
  const proc = spawn('codex', args, {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env },
  });

  const rl = createInterface({ input: proc.stdout! });

  const server: AppServer = {
    process: proc,
    readline: rl,
    pending: new Map(),
    notificationHandlers: [],
    serverRequestHandlers: [],
  };

  proc.stderr?.on('data', (chunk: Buffer) => {
    const text = chunk.toString().trim();
    if (text) log(`[codex-stderr] ${text}`);
  });

  rl.on('line', (line: string) => {
    if (!line.trim()) return;
    let msg: JsonRpcMessage;
    try {
      msg = JSON.parse(line);
    } catch {
      log(`[codex-parse-error] ${line.slice(0, 200)}`);
      return;
    }

    if (isResponse(msg)) {
      const handler = server.pending.get(msg.id);
      if (handler) {
        server.pending.delete(msg.id);
        handler.resolve(msg);
      }
    } else if (isServerRequest(msg)) {
      for (const h of server.serverRequestHandlers) h(msg);
    } else if ('method' in msg) {
      for (const h of server.notificationHandlers) h(msg as JsonRpcNotification);
    }
  });

  proc.on('error', (err) => {
    log(`[codex-process-error] ${err.message}`);
    for (const [, handler] of server.pending) handler.reject(err);
    server.pending.clear();
  });

  proc.on('exit', (code, signal) => {
    log(`[codex-exit] code=${code} signal=${signal}`);
    const err = new Error(`Codex app-server exited: code=${code} signal=${signal}`);
    for (const [, handler] of server.pending) handler.reject(err);
    server.pending.clear();
  });

  return server;
}

export function sendCodexRequest(server: AppServer, method: string, params: Record<string, unknown>, timeoutMs = 60_000): Promise<JsonRpcResponse> {
  const req = makeRequest(method, params);
  const line = JSON.stringify(req) + '\n';

  return new Promise<JsonRpcResponse>((resolve, reject) => {
    const timer = setTimeout(() => {
      server.pending.delete(req.id);
      reject(new Error(`Timeout waiting for ${method} response (${timeoutMs}ms)`));
    }, timeoutMs);

    server.pending.set(req.id, {
      resolve: (r) => { clearTimeout(timer); resolve(r); },
      reject: (e) => { clearTimeout(timer); reject(e); },
    });

    try {
      server.process.stdin!.write(line);
    } catch (err) {
      clearTimeout(timer);
      server.pending.delete(req.id);
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

function sendResponse(server: AppServer, id: number, result: unknown): void {
  const line = JSON.stringify({ id, result }) + '\n';
  try {
    server.process.stdin!.write(line);
  } catch (err) {
    log(`[codex-send-error] Failed to send response for id=${id}: ${err instanceof Error ? err.message : String(err)}`);
  }
}

export function killCodexAppServer(server: AppServer): void {
  try {
    server.readline.close();
    server.process.kill('SIGTERM');
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Auto-approval
// ---------------------------------------------------------------------------

export function attachCodexAutoApproval(server: AppServer): void {
  server.serverRequestHandlers.push((req) => {
    const method = req.method;
    log(`[approval] Auto-approving: ${method}`);

    switch (method) {
      case 'item/commandExecution/requestApproval':
        sendResponse(server, req.id, { decision: 'accept' });
        break;
      case 'item/fileChange/requestApproval':
        sendResponse(server, req.id, { decision: 'accept' });
        break;
      case 'item/permissions/requestApproval':
        sendResponse(server, req.id, {
          permissions: { fileSystem: { read: ['/'], write: ['/'] }, network: { enabled: true } },
          scope: 'session',
        });
        break;
      case 'applyPatchApproval':
      case 'execCommandApproval':
        sendResponse(server, req.id, { decision: 'approved' });
        break;
      case 'item/tool/call': {
        const toolName = (req.params as { tool?: string }).tool || 'unknown';
        log(`[approval] Unexpected dynamic tool call: ${toolName}`);
        sendResponse(server, req.id, {
          success: false,
          contentItems: [{ type: 'inputText', text: `Tool "${toolName}" is not available. Use MCP tools instead.` }],
        });
        break;
      }
      case 'item/tool/requestUserInput':
      case 'mcpServer/elicitation/request':
        sendResponse(server, req.id, { input: null });
        break;
      default:
        log(`[approval] Unknown server request method: ${method}, attempting generic approval`);
        sendResponse(server, req.id, { decision: 'accept' });
        break;
    }
  });
}

// ---------------------------------------------------------------------------
// High-level helpers
// ---------------------------------------------------------------------------

export async function initializeCodexAppServer(server: AppServer): Promise<void> {
  log('Sending initialize...');
  const resp = await sendCodexRequest(server, 'initialize', {
    clientInfo: { name: 'nanoclaw', version: '1.0.0' },
    capabilities: { experimentalApi: false },
  }, INIT_TIMEOUT_MS);

  if (resp.error) {
    throw new Error(`Initialize failed: ${resp.error.message}`);
  }
  log('Initialize successful');
}

export interface ThreadParams {
  model: string;
  cwd: string;
  sandbox?: string;
  approvalPolicy?: string;
  personality?: string;
  baseInstructions?: string;
}

/**
 * Start or resume a Codex thread. Returns the thread ID.
 * If sessionId is provided, attempts resume first, falling back to fresh start.
 */
export async function startOrResumeCodexThread(
  server: AppServer,
  sessionId: string | undefined,
  params: ThreadParams,
): Promise<string> {
  if (sessionId) {
    log(`Resuming thread: ${sessionId}`);
    const resp = await sendCodexRequest(server, 'thread/resume', {
      threadId: sessionId,
      ...params as unknown as Record<string, unknown>,
    });
    if (!resp.error) {
      log(`Thread resumed: ${sessionId}`);
      return sessionId;
    }
    log(`Resume failed: ${resp.error.message}. Starting fresh thread.`);
  }

  log('Starting new thread...');
  const resp = await sendCodexRequest(server, 'thread/start', { ...params as unknown as Record<string, unknown> });
  if (resp.error) {
    throw new Error(`thread/start failed: ${resp.error.message}`);
  }

  const result = resp.result as { thread?: { id?: string } } | undefined;
  const threadId = result?.thread?.id;
  if (!threadId) {
    throw new Error('thread/start response missing thread ID');
  }
  log(`New thread started: ${threadId}`);
  return threadId;
}

export interface TurnParams {
  threadId: string | undefined;
  inputText: string;
  model?: string;
  cwd?: string;
}

/**
 * Start a turn. Throws on error.
 */
export async function startCodexTurn(server: AppServer, params: TurnParams): Promise<void> {
  const resp = await sendCodexRequest(server, 'turn/start', {
    threadId: params.threadId,
    input: [{ type: 'text', text: params.inputText }],
    model: params.model,
    cwd: params.cwd,
  });
  if (resp.error) {
    throw new Error(`turn/start failed: ${resp.error.message}`);
  }
}

// ---------------------------------------------------------------------------
// MCP config
// ---------------------------------------------------------------------------

export interface McpServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

export function buildCodexMcpConfig(mcpServerPath: string, containerInput: ContainerInput, modelRef: string): Record<string, McpServerConfig> {
  const servers: Record<string, McpServerConfig> = {};

  servers['nanoclaw'] = {
    command: 'node',
    args: [mcpServerPath],
    env: {
      NANOCLAW_CHAT_JID: containerInput.chatJid,
      NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
      NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
      NANOCLAW_RUNTIME: 'codex',
      NANOCLAW_MODEL: modelRef,
    },
  };

  const providerConfigs = getProviderMcpConfigs();
  for (const [name, config] of Object.entries(providerConfigs)) {
    servers[name] = {
      command: config.command,
      args: config.args || [],
      env: config.env || {},
    };
  }

  return servers;
}

export function writeCodexMcpConfigToml(servers: Record<string, McpServerConfig>): void {
  const codexConfigDir = path.join(process.env.HOME || '/home/node', '.codex');
  fs.mkdirSync(codexConfigDir, { recursive: true });
  const configTomlPath = path.join(codexConfigDir, 'config.toml');

  const lines: string[] = [];
  for (const [name, config] of Object.entries(servers)) {
    lines.push(`[mcp_servers.${name}]`);
    lines.push('type = "stdio"');
    lines.push(`command = "${config.command}"`);
    if (config.args && config.args.length > 0) {
      const argsStr = config.args.map((a) => `"${a}"`).join(', ');
      lines.push(`args = [${argsStr}]`);
    }
    if (config.env && Object.keys(config.env).length > 0) {
      lines.push(`[mcp_servers.${name}.env]`);
      for (const [key, value] of Object.entries(config.env)) {
        lines.push(`${key} = "${value}"`);
      }
    }
    lines.push('');
  }

  fs.writeFileSync(configTomlPath, lines.join('\n'));
  log(`Wrote MCP config.toml (${Object.keys(servers).length} servers, clean rebuild)`);
}

export function createCodexConfigOverrides(baseUrl?: string | null): string[] {
  const overrides = ['features.use_linux_sandbox_bwrap=false'];
  if (baseUrl) {
    overrides.push(`model_provider_base_url="${baseUrl}"`);
  }
  return overrides;
}

