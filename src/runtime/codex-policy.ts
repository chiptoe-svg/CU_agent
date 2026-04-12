import { DEFAULT_MODEL } from '../config.js';
import type { RegisteredGroup } from '../types.js';

export type SandboxProfile = 'safe' | 'operator';

export interface CodexResolvedOptions {
  modelRef: string;
  baseUrl?: string;
  sandboxProfile: SandboxProfile;
}

function readStringOption(
  runtimeOptions: Record<string, unknown> | undefined,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = runtimeOptions?.[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

export function resolveCodexRuntimeOptions(
  group: RegisteredGroup,
  runtimeOptions?: Record<string, unknown>,
): CodexResolvedOptions {
  const modelRef =
    readStringOption(runtimeOptions, 'modelRef', 'model') ||
    group.containerConfig?.model ||
    DEFAULT_MODEL;

  const baseUrl =
    readStringOption(runtimeOptions, 'baseUrl') ||
    group.containerConfig?.baseUrl;

  const sandboxProfile =
    runtimeOptions?.sandboxProfile === 'operator' ? 'operator' : 'safe';

  return {
    modelRef,
    baseUrl,
    sandboxProfile,
  };
}
