/**
 * CodexRuntime — AgentRuntime implementation using OpenAI Codex SDK.
 *
 * Like ClaudeRuntime, the agent loop runs inside a container via the
 * agent-runner. The agent-runner detects runtime='openai' and uses
 * @openai/codex-sdk instead of @anthropic-ai/claude-agent-sdk.
 *
 * Both runtimes share the same container image, MCP server, IPC protocol,
 * and output format. The only difference is which SDK drives the agent loop.
 */
import path from 'path';

import { DATA_DIR } from '../config.js';
import { validateProviderAuth } from '../auth/backends.js';
import { logger } from '../logger.js';
import {
  resolveCodexRuntimeOptions,
  type CodexResolvedOptions,
} from './codex-policy.js';
import type {
  AgentEvent,
  AgentRuntime,
  AgentRuntimeConfig,
  ContainerManager,
  ContainerOutput,
  RuntimeCapabilities,
  RuntimePreflightResult,
  RuntimeId,
} from './types.js';
import { registerAgentSdk } from './registry.js';

export class CodexRuntime implements AgentRuntime {
  readonly id: RuntimeId = 'codex';

  private containerManager: ContainerManager | null = null;
  private groupFolder: string | null = null;

  private toRuntimeOptions(
    options: CodexResolvedOptions,
  ): Record<string, unknown> {
    return {
      modelRef: options.modelRef,
      sandboxProfile: options.sandboxProfile,
      ...(options.baseUrl ? { baseUrl: options.baseUrl } : {}),
    };
  }

  capabilities(): RuntimeCapabilities {
    return {
      supportsResume: true,
      supportsToolStreaming: true,
      supportsSkills: true,
      supportsProjectInstructions: true,
      supportsScheduledTasks: true,
      supportsDelegation: 'manual',
    };
  }

  async preflight(
    config: AgentRuntimeConfig,
  ): Promise<RuntimePreflightResult> {
    const resolvedOptions = resolveCodexRuntimeOptions(
      config.group,
      config.runtimeOptions,
    );
    const runtimeOptions = this.toRuntimeOptions(resolvedOptions);
    const warnings: string[] = [];

    if (config.runtimeOptions?.model && config.runtimeOptions?.modelRef) {
      warnings.push('Codex runtimeOptions received both model and modelRef; modelRef wins.');
    }

    if (
      config.runtimeOptions?.sandboxProfile &&
      config.runtimeOptions.sandboxProfile !== 'safe' &&
      config.runtimeOptions.sandboxProfile !== 'operator'
    ) {
      warnings.push('Unknown codex sandboxProfile; defaulting to safe.');
    }

    const authValidation = await validateProviderAuth({
      group: config.group,
      runtime: this.id,
      groupSessionsBase: path.join(DATA_DIR, 'sessions', config.group.folder),
      projectRoot: process.cwd(),
      runtimeOptions,
    });

    if (authValidation.warnings) warnings.push(...authValidation.warnings);

    return {
      ok: authValidation.ok,
      resolved: runtimeOptions,
      warnings: warnings.length > 0 ? warnings : undefined,
      errors: authValidation.errors,
    };
  }

  async *run(
    prompt: string,
    config: AgentRuntimeConfig,
  ): AsyncGenerator<AgentEvent> {
    this.containerManager = config.containerManager;
    this.groupFolder = config.group.folder;
    const preflight = await this.preflight(config);
    const resolvedOptions = resolveCodexRuntimeOptions(
      config.group,
      preflight.resolved || config.runtimeOptions,
    );
    const runtimeOptions = this.toRuntimeOptions(resolvedOptions);

    if (preflight.warnings) {
      for (const warning of preflight.warnings) {
        logger.warn({ group: config.group.name, runtime: this.id }, warning);
      }
    }

    if (!preflight.ok) {
      yield {
        type: 'error',
        runtime: this.id,
        error: preflight.errors?.join(' ') || 'Codex runtime preflight failed.',
        sessionId: config.sessionId,
      };
      return;
    }

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
        runtime: 'codex',
        runtimeOptions,
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
    // Clear session on thread resume failures
    return /no rollout found|thread.*not found|resume.*failed/i.test(error);
  }
}

// Self-register
registerAgentSdk('codex', () => new CodexRuntime());
