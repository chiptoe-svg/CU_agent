import fs from 'fs';
import path from 'path';
import { spawn, type ChildProcess } from 'child_process';
import { createInterface, type Interface as ReadlineInterface } from 'readline';

import type { ContainerInput } from '../shared.js';
import { log } from '../shared.js';
import { getProviderMcpConfigs } from '../provider-registry.js';

export const INIT_TIMEOUT_MS = 30_000;

export const CODEX_TOOL_GUIDANCE = `
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

type JsonRpcMessage =
  | JsonRpcResponse
  | JsonRpcNotification
  | JsonRpcServerRequest;

export interface AppServer {
  process: ChildProcess;
  readline: ReadlineInterface;
  pending: Map<
    number,
    { resolve: (r: JsonRpcResponse) => void; reject: (e: Error) => void }
  >;
  notificationHandlers: Array<(n: JsonRpcNotification) => void>;
  serverRequestHandlers: Array<(r: JsonRpcServerRequest) => void>;
}

export interface CodexThreadParams {
  model: string;
  cwd: string;
  sandbox?: string;
  approvalPolicy?: string;
  personality?: string;
  baseInstructions?: string;
}

export interface CodexMcpServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

function makeRequest(
  method: string,
  params: Record<string, unknown>,
): JsonRpcRequest {
  return { id: nextRequestId++, method, params };
}

function isResponse(msg: JsonRpcMessage): msg is JsonRpcResponse {
  return 'id' in msg && ('result' in msg || 'error' in msg) && !('method' in msg);
}

function isServerRequest(msg: JsonRpcMessage): msg is JsonRpcServerRequest {
  return 'id' in msg && 'method' in msg;
}

export function createCodexConfigOverrides(baseUrl?: string): string[] {
  const overrides = ['features.use_linux_sandbox_bwrap=false'];
  if (baseUrl) {
    overrides.push(`model_provider_base_url="${baseUrl}"`);
  }
  return overrides;
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
      for (const handler of server.serverRequestHandlers) handler(msg);
    } else if ('method' in msg) {
      for (const handler of server.notificationHandlers) handler(msg);
    }
  });

  proc.on('error', (err) => {
    log(`[codex-process-error] ${err.message}`);
    for (const [, handler] of server.pending) {
      handler.reject(err);
    }
    server.pending.clear();
  });

  proc.on('exit', (code, signal) => {
    log(`[codex-exit] code=${code} signal=${signal}`);
    const err = new Error(
      `Codex app-server exited: code=${code} signal=${signal}`,
    );
    for (const [, handler] of server.pending) {
      handler.reject(err);
    }
    server.pending.clear();
  });

  return server;
}

export function sendCodexResponse(
  server: AppServer,
  id: number,
  result: unknown,
): void {
  const line = JSON.stringify({ id, result }) + '\n';
  try {
    server.process.stdin!.write(line);
  } catch (err) {
    log(
      `[codex-send-error] Failed to send response for id=${id}: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
  }
}

export function sendCodexRequest(
  server: AppServer,
  method: string,
  params: Record<string, unknown>,
  timeoutMs = 60_000,
): Promise<JsonRpcResponse> {
  const req = makeRequest(method, params);
  const line = JSON.stringify(req) + '\n';

  return new Promise<JsonRpcResponse>((resolve, reject) => {
    const timer = setTimeout(() => {
      server.pending.delete(req.id);
      reject(new Error(`Timeout waiting for ${method} response (${timeoutMs}ms)`));
    }, timeoutMs);

    server.pending.set(req.id, {
      resolve: (response) => {
        clearTimeout(timer);
        resolve(response);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      },
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

export function killCodexAppServer(server: AppServer): void {
  try {
    server.readline.close();
    server.process.kill('SIGTERM');
  } catch {
    /* ignore */
  }
}

function handleServerRequest(server: AppServer, req: JsonRpcServerRequest): void {
  log(`[approval] Auto-approving: ${req.method}`);

  switch (req.method) {
    case 'item/commandExecution/requestApproval':
    case 'item/fileChange/requestApproval':
      sendCodexResponse(server, req.id, { decision: 'accept' });
      break;

    case 'item/permissions/requestApproval':
      sendCodexResponse(server, req.id, {
        permissions: {
          fileSystem: { read: ['/'], write: ['/'] },
          network: { enabled: true },
        },
        scope: 'session',
      });
      break;

    case 'applyPatchApproval':
    case 'execCommandApproval':
      sendCodexResponse(server, req.id, { decision: 'approved' });
      break;

    case 'item/tool/call': {
      const toolName = (req.params as { tool?: string }).tool || 'unknown';
      log(`[approval] Unexpected dynamic tool call: ${toolName}`);
      sendCodexResponse(server, req.id, {
        success: false,
        contentItems: [
          {
            type: 'inputText',
            text: `Tool "${toolName}" is not available. Use MCP tools instead.`,
          },
        ],
      });
      break;
    }

    case 'item/tool/requestUserInput':
    case 'mcpServer/elicitation/request':
      sendCodexResponse(server, req.id, { input: null });
      break;

    default:
      log(
        `[approval] Unknown server request method: ${req.method}, attempting generic approval`,
      );
      sendCodexResponse(server, req.id, { decision: 'accept' });
      break;
  }
}

export function attachCodexAutoApproval(server: AppServer): void {
  server.serverRequestHandlers.push((req) => handleServerRequest(server, req));
}

export async function initializeCodexAppServer(server: AppServer): Promise<void> {
  log('Sending initialize...');
  const initResp = await sendCodexRequest(
    server,
    'initialize',
    {
      clientInfo: { name: 'nanoclaw', version: '1.0.0' },
      capabilities: { experimentalApi: false },
    },
    INIT_TIMEOUT_MS,
  );

  if (initResp.error) {
    throw new Error(`Initialize failed: ${initResp.error.message}`);
  }
  log('Initialize successful');
}

export async function startOrResumeCodexThread(
  server: AppServer,
  sessionId: string | undefined,
  params: CodexThreadParams,
): Promise<string> {
  if (sessionId) {
    log(`Resuming thread: ${sessionId}`);
    const resumeResp = await sendCodexRequest(server, 'thread/resume', {
      threadId: sessionId,
      ...params,
    });

    if (!resumeResp.error) {
      log(`Thread resumed: ${sessionId}`);
      return sessionId;
    }

    log(`Resume failed: ${resumeResp.error.message}. Starting fresh thread.`);
  }

  log('Starting new thread...');
  const startResp = await sendCodexRequest(server, 'thread/start', params);
  if (startResp.error) {
    throw new Error(`thread/start failed: ${startResp.error.message}`);
  }

  const result = startResp.result as { thread?: { id?: string } } | undefined;
  const threadId = result?.thread?.id;
  if (!threadId) {
    throw new Error('thread/start response missing thread ID');
  }
  log(`New thread started: ${threadId}`);
  return threadId;
}

export async function startCodexTurn(
  server: AppServer,
  opts: { threadId: string; inputText: string; model: string; cwd?: string },
): Promise<void> {
  const turnResp = await sendCodexRequest(server, 'turn/start', {
    threadId: opts.threadId,
    input: [{ type: 'text', text: opts.inputText }],
    model: opts.model,
    ...(opts.cwd ? { cwd: opts.cwd } : {}),
  });

  if (turnResp.error) {
    throw new Error(`turn/start failed: ${turnResp.error.message}`);
  }
}

export function buildCodexMcpConfig(
  mcpServerPath: string,
  containerInput: ContainerInput,
  modelRef: string,
): Record<string, CodexMcpServerConfig> {
  const servers: Record<string, CodexMcpServerConfig> = {
    nanoclaw: {
      command: 'node',
      args: [mcpServerPath],
      env: {
        NANOCLAW_CHAT_JID: containerInput.chatJid,
        NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
        NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        NANOCLAW_RUNTIME: 'codex',
        NANOCLAW_MODEL: modelRef,
      },
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

export function writeCodexMcpConfigToml(
  servers: Record<string, CodexMcpServerConfig>,
): void {
  const codexConfigDir = path.join(process.env.HOME || '/home/node', '.codex');
  fs.mkdirSync(codexConfigDir, { recursive: true });
  const configTomlPath = path.join(codexConfigDir, 'config.toml');

  const lines: string[] = [];
  for (const [name, config] of Object.entries(servers)) {
    lines.push(`[mcp_servers.${name}]`);
    lines.push('type = "stdio"');
    lines.push(`command = "${config.command}"`);
    if (config.args && config.args.length > 0) {
      lines.push(`args = [${config.args.map((arg) => `"${arg}"`).join(', ')}]`);
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
  log(
    `Wrote MCP config.toml (${Object.keys(servers).length} servers, clean rebuild)`,
  );
}
