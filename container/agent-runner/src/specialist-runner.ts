/**
 * Specialist subagent runner.
 *
 * Runs a subtask using a specialist persona via the current container runtime.
 * Called by the run_specialist MCP tool. Works with all runtimes:
 *
 * - Claude: spawns a fresh query() with the specialist persona as system prompt
 * - Codex: opens a dedicated app-server worker thread with the specialist persona
 * - Gemini: runs gemini CLI with the specialist persona prepended to the prompt
 *
 * Specialist personas are defined in AGENT.md (as guidance for the main agent)
 * and optionally as standalone files in /workspace/group/specialists/<name>.md
 * for detailed instructions.
 */
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

import { log } from './shared.js';
import {
  CODEX_TOOL_GUIDANCE,
  attachCodexAutoApproval,
  buildCodexMcpConfig,
  createCodexConfigOverrides,
  initializeCodexAppServer,
  killCodexAppServer,
  spawnCodexAppServer,
  startCodexTurn,
  startOrResumeCodexThread,
  writeCodexMcpConfigToml,
} from './runtimes/codex-app-server.js';

const SPECIALISTS_DIR = '/workspace/group/specialists';
const AGENT_MD_PATH = '/workspace/group/AGENT.md';
const DEFAULT_CODEX_MODEL = 'gpt-5.4-mini';
const DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash';
const SPECIALIST_TIMEOUT_MS = 90_000;

export interface SpecialistResult {
  summary: string;
  resultText: string;
  toolCalls: string[];
  threadId?: string;
}

/**
 * Load specialist persona by name.
 * First checks specialists/<name>.md, then falls back to parsing
 * the ## Specialists section of AGENT.md for an inline definition.
 */
export function loadSpecialistPersona(name: string): string | null {
  const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) return null;

  const personaFile = path.join(SPECIALISTS_DIR, `${safeName}.md`);
  if (fs.existsSync(personaFile)) {
    return fs.readFileSync(personaFile, 'utf-8');
  }

  if (fs.existsSync(AGENT_MD_PATH)) {
    const agentMd = fs.readFileSync(AGENT_MD_PATH, 'utf-8');
    const persona = extractInlineSpecialist(agentMd, safeName);
    if (persona) return persona;
  }

  return null;
}

function extractInlineSpecialist(markdown: string, name: string): string | null {
  const pattern = new RegExp(
    `###\\s+${name}\\s*\\n([\\s\\S]*?)(?=\\n###\\s|\\n##\\s|$)`,
    'i',
  );
  const match = markdown.match(pattern);
  return match ? match[1].trim() : null;
}

export function listSpecialists(): string[] {
  const names = new Set<string>();

  if (fs.existsSync(SPECIALISTS_DIR)) {
    for (const file of fs.readdirSync(SPECIALISTS_DIR)) {
      if (file.endsWith('.md')) {
        names.add(file.replace(/\.md$/, ''));
      }
    }
  }

  if (fs.existsSync(AGENT_MD_PATH)) {
    const content = fs.readFileSync(AGENT_MD_PATH, 'utf-8');
    const specialistsSection = content.match(
      /## Specialists\s*\n([\s\S]*?)(?=\n## |$)/i,
    );
    if (specialistsSection) {
      const headings = specialistsSection[1].matchAll(/###\s+(\w+)/g);
      for (const match of headings) {
        names.add(match[1].toLowerCase());
      }
    }
  }

  return [...names].sort();
}

function summarizeResult(resultText: string): string {
  const compact = resultText.replace(/\s+/g, ' ').trim();
  return compact.slice(0, 280) || '(no response from specialist)';
}

async function runCodexSpecialist(
  persona: string,
  task: string,
  model: string,
): Promise<SpecialistResult> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');
  const modelRef = model || DEFAULT_CODEX_MODEL;
  const workerInstructions = [
    'You are a focused specialist worker thread. Complete the assigned task and return a concise answer for the parent agent. You do not have access to the parent conversation beyond the explicit task text you were given.',
    persona,
    CODEX_TOOL_GUIDANCE,
  ].join('\n\n---\n\n');

  const containerInput = {
    prompt: task,
    groupFolder: process.env.NANOCLAW_GROUP_FOLDER || '',
    chatJid: process.env.NANOCLAW_CHAT_JID || '',
    isMain: process.env.NANOCLAW_IS_MAIN === '1',
    runtime: 'codex',
  };

  const mcpServers = buildCodexMcpConfig(mcpServerPath, containerInput, modelRef);
  writeCodexMcpConfigToml(mcpServers);

  const server = spawnCodexAppServer(
    createCodexConfigOverrides(process.env.OPENAI_BASE_URL),
  );
  attachCodexAutoApproval(server);

  const turnState = {
    resultText: '',
    toolCalls: [] as string[],
    threadId: undefined as string | undefined,
  };

  try {
    await initializeCodexAppServer(server);
    turnState.threadId = await startOrResumeCodexThread(server, undefined, {
      model: modelRef,
      cwd: '/workspace/group',
      sandbox: 'danger-full-access',
      approvalPolicy: 'never',
      personality: 'friendly',
      baseInstructions: workerInstructions,
    });

    const turnPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Codex specialist timed out after 90s'));
      }, SPECIALIST_TIMEOUT_MS);

      server.notificationHandlers.push((notification) => {
        switch (notification.method) {
          case 'item/agentMessage/delta': {
            const delta = notification.params.delta as string | undefined;
            if (delta) turnState.resultText += delta;
            break;
          }

          case 'item/started': {
            const item = notification.params.item as
              | {
                  type?: string;
                  command?: string;
                  server?: string;
                  tool?: string;
                  query?: string;
                }
              | undefined;
            if (!item) break;
            if (item.type === 'commandExecution' && item.command) {
              turnState.toolCalls.push(`$ ${item.command}`);
            } else if (item.type === 'mcpToolCall' && item.server && item.tool) {
              turnState.toolCalls.push(`MCP: ${item.server}/${item.tool}`);
            } else if (item.type === 'webSearch' && item.query) {
              turnState.toolCalls.push(`Search: ${item.query}`);
            }
            break;
          }

          case 'item/completed': {
            const item = notification.params.item as
              | { type?: string; text?: string; aggregated_output?: string }
              | undefined;
            if (item?.type === 'agentMessage' && item.text) {
              turnState.resultText = item.text;
            } else if (
              item?.type === 'commandExecution' &&
              item.aggregated_output
            ) {
              turnState.toolCalls.push(item.aggregated_output.slice(0, 300));
            }
            break;
          }

          case 'thread/started': {
            const thread = notification.params.thread as
              | { id?: string }
              | undefined;
            if (thread?.id) turnState.threadId = thread.id;
            break;
          }

          case 'turn/completed':
            clearTimeout(timeout);
            resolve();
            break;
        }
      });
    });

    await startCodexTurn(server, {
      threadId: turnState.threadId,
      inputText: task,
      model: modelRef,
      cwd: '/workspace/group',
    });
    await turnPromise;

    const resultText = turnState.resultText.trim() || '(no response from specialist)';
    return {
      summary: summarizeResult(resultText),
      resultText,
      toolCalls: turnState.toolCalls,
      threadId: turnState.threadId,
    };
  } finally {
    killCodexAppServer(server);
  }
}

async function runGeminiSpecialist(
  persona: string,
  task: string,
  model: string,
): Promise<SpecialistResult> {
  const fullPrompt = `${persona}\n\n---\n\nTask: ${task}`;

  return new Promise((resolve, reject) => {
    const args = ['-p', fullPrompt, '--yolo', '--model', model];
    const proc = spawn('gemini', args, {
      cwd: '/workspace/group',
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    proc.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    const timeout = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error('Specialist timed out after 90s'));
    }, SPECIALIST_TIMEOUT_MS);

    proc.on('close', (code) => {
      clearTimeout(timeout);
      if (code !== 0 && !stdout.trim()) {
        reject(new Error(`Gemini specialist exited ${code}: ${stderr.slice(-200)}`));
        return;
      }
      const resultText = stdout.trim() || '(no response from specialist)';
      resolve({
        summary: summarizeResult(resultText),
        resultText,
        toolCalls: [],
      });
    });

    proc.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

export async function runSpecialist(
  specialistName: string,
  task: string,
  runtime: string,
  model: string,
): Promise<SpecialistResult> {
  const persona = loadSpecialistPersona(specialistName);
  if (!persona) {
    const available = listSpecialists();
    const resultText = `Unknown specialist "${specialistName}". Available: ${available.join(', ') || 'none defined. Add specialist definitions to AGENT.md or create files in specialists/'}`;
    return {
      summary: summarizeResult(resultText),
      resultText,
      toolCalls: [],
    };
  }

  const resolvedModel =
    model ||
    (runtime === 'codex'
      ? DEFAULT_CODEX_MODEL
      : runtime === 'gemini'
        ? DEFAULT_GEMINI_MODEL
        : '');

  log(
    `[specialist] Running ${specialistName} via ${runtime} (model: ${
      resolvedModel || 'default'
    })`,
  );

  try {
    if (runtime === 'codex') {
      return await runCodexSpecialist(persona, task, resolvedModel);
    }

    if (runtime === 'gemini') {
      return await runGeminiSpecialist(persona, task, resolvedModel);
    }

    const resultText = await runClaudeSpecialistFallback(persona, task);
    return {
      summary: summarizeResult(resultText),
      resultText,
      toolCalls: [],
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`[specialist] ${specialistName} failed: ${msg}`);
    const resultText = `Specialist "${specialistName}" failed: ${msg}`;
    return {
      summary: summarizeResult(resultText),
      resultText,
      toolCalls: [],
    };
  }
}

async function runClaudeSpecialistFallback(
  persona: string,
  task: string,
): Promise<string> {
  const fullPrompt = `${persona}\n\n---\n\nTask: ${task}`;

  return new Promise((resolve, reject) => {
    const proc = spawn(
      'claude',
      [
        '-p',
        fullPrompt,
        '--allowedTools',
        'Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch',
      ],
      {
        cwd: '/workspace/group',
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );

    let stdout = '';
    proc.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    const timeout = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error('Specialist timed out after 90s'));
    }, SPECIALIST_TIMEOUT_MS);

    proc.on('close', (code) => {
      clearTimeout(timeout);
      if (code !== 0 && !stdout.trim()) {
        reject(new Error(`Claude specialist exited ${code}`));
        return;
      }
      resolve(stdout.trim() || '(no response from specialist)');
    });

    proc.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}
