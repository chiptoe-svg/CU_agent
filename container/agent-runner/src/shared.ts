/**
 * Shared container plumbing — runtime-agnostic.
 *
 * IO, IPC polling, message streaming, script execution, and protocol types
 * used by any agent runner regardless of which AI SDK it calls.
 *
 * When a second runner is added (e.g. openai-runner), move this file to
 * container/shared/ and update import paths.
 */

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';

// --- Protocol types ---

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
  runtime?: 'claude' | 'codex' | string;
  model?: string;
  baseUrl?: string;
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

export interface ScriptResult {
  wakeAgent: boolean;
  data?: unknown;
}

// --- IO ---

export const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
export const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

export function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

export function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

export async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

// --- IPC ---

export const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;

/** Check for _close sentinel. */
export function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try {
      fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
    } catch {
      /* ignore */
    }
    return true;
  }
  return false;
}

/** Drain all pending IPC input messages. Returns messages found, or empty array. */
export function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(
          `Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`,
        );
        try {
          fs.unlinkSync(filePath);
        } catch {
          /* ignore */
        }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
export function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

// --- MessageStream ---

/**
 * Push-based async iterable for streaming user messages into an SDK.
 * Yields plain strings — the runner converts to SDK-specific message format.
 * Keeps the iterable alive until end() is called.
 */
export class MessageStream {
  private queue: string[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push(text);
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

// --- Script runner ---

const SCRIPT_TIMEOUT_MS = 30_000;

export async function runScript(script: string): Promise<ScriptResult | null> {
  const scriptPath = '/tmp/task-script.sh';
  fs.writeFileSync(scriptPath, script, { mode: 0o755 });

  return new Promise((resolve) => {
    execFile(
      'bash',
      [scriptPath],
      {
        timeout: SCRIPT_TIMEOUT_MS,
        maxBuffer: 1024 * 1024,
        env: process.env,
      },
      (error, stdout, stderr) => {
        if (stderr) {
          log(`Script stderr: ${stderr.slice(0, 500)}`);
        }

        if (error) {
          log(`Script error: ${error.message}`);
          return resolve(null);
        }

        // Parse last non-empty line of stdout as JSON
        const lines = stdout.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        if (!lastLine) {
          log('Script produced no output');
          return resolve(null);
        }

        try {
          const result = JSON.parse(lastLine);
          if (typeof result.wakeAgent !== 'boolean') {
            log(
              `Script output missing wakeAgent boolean: ${lastLine.slice(0, 200)}`,
            );
            return resolve(null);
          }
          resolve(result as ScriptResult);
        } catch {
          log(`Script output is not valid JSON: ${lastLine.slice(0, 200)}`);
          resolve(null);
        }
      },
    );
  });
}

// --- Shared utilities used by runtime modules ---

export interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

export function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

export function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

export function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const formatDateTime = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

export function getMcpServerConfig(
  mcpServerPath: string,
  containerInput: ContainerInput,
) {
  const config: Record<string, { command: string; args: string[]; env: Record<string, string> }> = {
    nanoclaw: {
      command: 'node',
      args: [mcpServerPath],
      env: {
        NANOCLAW_CHAT_JID: containerInput.chatJid,
        NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
        NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        NANOCLAW_RUNTIME: containerInput.runtime || 'claude',
        NANOCLAW_MODEL: containerInput.model || '',
      },
    },
  };

  // Microsoft 365 MCP server (email, calendar, tasks, Teams, OneDrive, etc.)
  // Only registered when token cache is mounted from host
  const ms365TokenDir = '/workspace/.ms365-tokens';
  const ms365TokenCache = path.join(ms365TokenDir, '.token-cache.json');
  if (fs.existsSync(ms365TokenCache)) {
    config.ms365 = {
      command: 'npx',
      args: [
        '@softeria/ms-365-mcp-server', '--toon',
        '--enabled-tools', '^(list-mail-(?!rule)|get-mail-(?!box)|create-mail-(?!rule)|delete-mail-(?!rule)|move-mail|update-mail-(?!rule)|add-mail|create-draft|list-calendar|get-calendar|create-calendar|update-calendar|delete-calendar|accept-calendar|decline-calendar|tentatively-accept|list-specific|get-specific|create-specific|update-specific|delete-specific|get-calendar-view|list-todo|get-todo|create-todo|update-todo|delete-todo|list-planner|get-planner|create-planner|update-planner|list-plan-tasks)',
      ],
      env: {
        MS365_MCP_CLIENT_ID: '7556c30a-2955-4186-86cc-4ebc34809e4b',
        MS365_MCP_TENANT_ID: '0c9bf8f6-ccad-4b87-818d-49026938aa97',
        MS365_MCP_TOKEN_CACHE_PATH: ms365TokenCache,
        MS365_MCP_SELECTED_ACCOUNT_PATH: path.join(ms365TokenDir, '.selected-account.json'),
        SILENT: '1',
      },
    };
  }

  return config;
}

/**
 * Set up Google Workspace CLI (`gws`) credentials if mounted from host.
 * Copies credentials.json to gws config dir and sets env vars so
 * gws uses plain-text storage (no keyring in containers).
 */
export function setupGwsCredentials(): void {
  const gwsCredsSource = '/workspace/.gws-tokens/credentials.json';
  if (!fs.existsSync(gwsCredsSource)) return;

  const gwsConfigDir = path.join(process.env.HOME || '/home/node', '.config', 'gws');
  fs.mkdirSync(gwsConfigDir, { recursive: true });

  // Copy credentials as plain-text (container has no keyring)
  const destPath = path.join(gwsConfigDir, 'credentials.json');
  fs.copyFileSync(gwsCredsSource, destPath);

  // Also copy client_secret.json if present
  const clientSecretSource = '/workspace/.gws-tokens/client_secret.json';
  if (fs.existsSync(clientSecretSource)) {
    fs.copyFileSync(clientSecretSource, path.join(gwsConfigDir, 'client_secret.json'));
  }

  // Tell gws to use plain-text storage (no keyring)
  process.env.GWS_CREDENTIAL_STORE = 'plaintext';

  log('Configured Google Workspace CLI credentials');
}
