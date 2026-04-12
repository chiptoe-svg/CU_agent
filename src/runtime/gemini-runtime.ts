/**
 * GeminiRuntime — AgentRuntime implementation for Google Gemini (ADK).
 *
 * Like ClaudeRuntime and CodexRuntime, delegates to the container via
 * ContainerManager.runAgentSession(). The container agent-runner detects
 * runtime='gemini' and starts the Google ADK sidecar.
 */
import type {
  AgentEvent,
  AgentRuntime,
  AgentRuntimeConfig,
  ContainerManager,
  ContainerOutput,
  RuntimeId,
} from './types.js';
import { registerAgentSdk } from './registry.js';

export class GeminiRuntime implements AgentRuntime {
  readonly id: RuntimeId = 'gemini';

  private containerManager: ContainerManager | null = null;
  private groupFolder: string | null = null;

  async *run(
    prompt: string,
    config: AgentRuntimeConfig,
  ): AsyncGenerator<AgentEvent> {
    this.containerManager = config.containerManager;
    this.groupFolder = config.group.folder;

    const model = config.group.containerConfig?.model || 'gemini-2.5-flash';

    const output = await config.containerManager.runAgentSession({
      group: config.group,
      input: {
        prompt,
        sessionId: config.sessionId,
        groupFolder: config.group.folder,
        chatJid: config.chatJid,
        isMain: config.isMain,
        isScheduledTask: config.isScheduledTask,
        assistantName: config.assistantName,
        script: config.script,
        runtime: 'gemini',
        runtimeOptions: {
          model,
          ...(config.group.containerConfig?.baseUrl
            ? { baseUrl: config.group.containerConfig.baseUrl }
            : {}),
        },
      },
      onProcess: (proc, containerName) =>
        config.onProcess(proc, containerName, config.group.folder),
      onOutput: async (streamedOutput: ContainerOutput) => {
        if (config._onStreamedOutput) {
          await config._onStreamedOutput(streamedOutput);
        }
      },
    });

    if (output.newSessionId) {
      yield {
        type: 'session_update',
        runtime: this.id,
        sessionId: output.newSessionId,
      };
    }

    if (output.status === 'error') {
      yield {
        type: 'error',
        runtime: this.id,
        error: output.error,
        sessionId: output.newSessionId,
      };
    } else {
      yield {
        type: 'result',
        runtime: this.id,
        result: output.result,
        sessionId: output.newSessionId,
      };
    }
  }

  sendFollowUp(text: string): boolean {
    if (!this.containerManager || !this.groupFolder) return false;
    return this.containerManager.sendToContainer(this.groupFolder, text);
  }

  close(): void {
    if (this.containerManager && this.groupFolder) {
      this.containerManager.closeSession(this.groupFolder);
    }
  }

  shouldClearSession(error: string): boolean {
    // Clear on Gemini ADK errors that indicate stale state
    return /session.*not found|cache.*expired|invalid.*session/i.test(error);
  }
}

// Self-register
registerAgentSdk('gemini', () => new GeminiRuntime());
