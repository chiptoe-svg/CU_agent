/**
 * Claude Agent SDK runtime for the container agent-runner.
 * Self-registers with the container runtime registry.
 */
import fs from 'fs';
import path from 'path';
import {
  query,
  HookCallback,
  PreCompactHookInput,
} from '@anthropic-ai/claude-agent-sdk';

import {
  ContainerInput,
  MessageStream,
  drainIpcInput,
  formatTranscriptMarkdown,
  generateFallbackName,
  getMcpServerConfig,
  log,
  ParsedMessage,
  sanitizeFilename,
  shouldClose,
  writeOutput,
} from '../shared.js';
import { registerContainerRuntime, type QueryResult } from '../runtime-registry.js';

// --- Claude SDK types ---

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

// --- Transcript parsing (Claude JSONL format) ---

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    const entry = index.entries.find((e) => e.sessionId === sessionId);
    if (entry?.summary) return entry.summary;
  } catch {
    // ignore
  }
  return null;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {}
  }
  return messages;
}

// --- Pre-compact hook ---

function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);
      if (messages.length === 0) return {};

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filePath = path.join(conversationsDir, `${date}-${name}.md`);
      fs.writeFileSync(
        filePath,
        formatTranscriptMarkdown(messages, summary, assistantName),
      );
      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(
        `Failed to archive: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    return {};
  };
}

// --- SDK message converter ---

async function* toSdkMessages(
  stream: MessageStream,
): AsyncGenerator<SDKUserMessage> {
  for await (const text of stream) {
    yield {
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    };
  }
}

// --- Claude query ---

async function runClaudeQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<QueryResult> {
  const stream = new MessageStream();
  stream.push(prompt);

  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpc = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      stream.push(text);
    }
    setTimeout(pollIpc, 500);
  };
  setTimeout(pollIpc, 500);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Ensure CLAUDE.md exists from AGENT.md
  const groupAgentMd = '/workspace/group/AGENT.md';
  const groupClaudeMd = '/workspace/group/CLAUDE.md';
  if (fs.existsSync(groupAgentMd) && !fs.existsSync(groupClaudeMd)) {
    fs.copyFileSync(groupAgentMd, groupClaudeMd);
  }

  // Load global instructions
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain) {
    for (const p of ['/workspace/global/AGENT.md', '/workspace/global/CLAUDE.md']) {
      if (fs.existsSync(p)) {
        globalClaudeMd = fs.readFileSync(p, 'utf-8');
        break;
      }
    }
  }

  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) extraDirs.push(fullPath);
    }
  }

  for await (const message of query({
    prompt: toSdkMessages(stream),
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: globalClaudeMd
        ? { type: 'preset' as const, preset: 'claude_code' as const, append: globalClaudeMd }
        : undefined,
      allowedTools: [
        'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
        'WebSearch', 'WebFetch', 'Task', 'TaskOutput', 'TaskStop',
        'TeamCreate', 'TeamDelete', 'SendMessage', 'TodoWrite',
        'ToolSearch', 'Skill', 'NotebookEdit', 'mcp__nanoclaw__*',
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: getMcpServerConfig(mcpServerPath, containerInput),
      hooks: {
        PreCompact: [
          { hooks: [createPreCompactHook(containerInput.assistantName)] },
        ],
      },
    },
  })) {
    messageCount++;
    const msgType = message.type === 'system'
      ? `system/${(message as { subtype?: string }).subtype}`
      : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }
    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }
    if (message.type === 'result') {
      resultCount++;
      const textResult = 'result' in message ? (message as { result?: string }).result : null;
      log(`Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`);
      writeOutput({ status: 'success', result: textResult || null, newSessionId });
    }
  }

  ipcPolling = false;
  log(`Query done. Messages: ${messageCount}, results: ${resultCount}, closedDuringQuery: ${closedDuringQuery}`);
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

// --- Self-register ---

registerContainerRuntime('claude', runClaudeQuery);
