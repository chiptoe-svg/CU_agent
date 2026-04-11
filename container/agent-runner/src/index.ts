/**
 * NanoClaw Agent Runner
 *
 * Runs inside a container. Dispatches to registered SDK runtimes.
 * Each runtime self-registers via runtime-registry.ts when imported.
 *
 * Shared plumbing (IO, IPC, MessageStream) lives in shared.ts.
 * SDK-specific code lives in runtimes/claude.ts and runtimes/codex.ts.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  ContainerInput,
  IPC_INPUT_DIR,
  drainIpcInput,
  log,
  readStdin,
  runScript,
  setupGwsCredentials,
  waitForIpcMessage,
  writeOutput,
} from './shared.js';
import {
  getContainerRuntime,
  getRegisteredContainerRuntimeNames,
} from './runtime-registry.js';

// --- Register available runtimes ---
// Each import triggers self-registration. Missing SDKs are silently skipped.

// claude
try { await import('./runtimes/claude.js'); } catch { /* SDK not installed */ }

// codex (codex)
try { await import('./runtimes/codex.js'); } catch { /* SDK not installed */ }

// gemini (google)
try { await import('./runtimes/gemini.js'); } catch { /* CLI not installed */ }

// --- Main ---

async function main(): Promise<void> {
  const registeredRuntimes = getRegisteredContainerRuntimeNames();
  log(`Available runtimes: ${registeredRuntimes.join(', ') || 'none'}`);

  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder} (runtime: ${containerInput.runtime || 'default'})`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  // Set up Google Workspace CLI if credentials are mounted
  setupGwsCredentials();

  const runtime = containerInput.runtime || 'claude';
  const handler = getContainerRuntime(runtime);

  if (!handler) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Runtime '${runtime}' not available. Installed: ${registeredRuntimes.join(', ') || 'none'}. Run /add-agentSDK-${runtime}`,
    });
    process.exit(1);
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel
  try { fs.unlinkSync(path.join(IPC_INPUT_DIR, '_close')); } catch { /* ignore */ }

  // Build initial prompt
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Script phase
  if (containerInput.script && containerInput.isScheduledTask) {
    log('Running task script...');
    const scriptResult = await runScript(containerInput.script);
    if (!scriptResult || !scriptResult.wakeAgent) {
      log(`Script decided not to wake agent: ${scriptResult ? 'wakeAgent=false' : 'script error'}`);
      writeOutput({ status: 'success', result: null });
      return;
    }
    log('Script wakeAgent=true, enriching prompt with data');
    prompt = `[SCHEDULED TASK]\n\nScript output:\n${JSON.stringify(scriptResult.data, null, 2)}\n\nInstructions:\n${containerInput.prompt}`;
  }

  // Query loop
  let resumeAt: string | undefined;
  const sdkEnv: Record<string, string | undefined> = {
    ...process.env,
    CLAUDE_CODE_AUTO_COMPACT_WINDOW: '165000',
  };

  try {
    while (true) {
      log(`Starting ${runtime} query (session: ${sessionId || 'new'})...`);

      const queryResult = await handler(
        prompt,
        sessionId,
        mcpServerPath,
        containerInput,
        sdkEnv,
        resumeAt,
      );

      if (queryResult.newSessionId) sessionId = queryResult.newSessionId;
      if (queryResult.lastAssistantUuid) resumeAt = queryResult.lastAssistantUuid;

      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      writeOutput({ status: 'success', result: null, newSessionId: sessionId });
      log('Query ended, waiting for next IPC message...');

      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({ status: 'error', result: null, newSessionId: sessionId, error: errorMessage });
    process.exit(1);
  }
}

main();
