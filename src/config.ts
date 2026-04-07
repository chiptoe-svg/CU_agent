import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { isValidTimezone } from './timezone.js';

// Read config values from .env (falls back to process.env).
// Secrets (API keys, tokens) are NOT read here — they are loaded only
// by the credential proxy (credential-proxy.ts), never exposed to containers.
const envConfig = readEnvFile([
  'ASSISTANT_NAME',
  'ASSISTANT_HAS_OWN_NUMBER',
  'DEFAULT_RUNTIME',
  'LITELLM_URL',
  'TZ',
]);

export const ASSISTANT_NAME =
  process.env.ASSISTANT_NAME || envConfig.ASSISTANT_NAME || 'Andy';
export const ASSISTANT_HAS_OWN_NUMBER =
  (process.env.ASSISTANT_HAS_OWN_NUMBER ||
    envConfig.ASSISTANT_HAS_OWN_NUMBER) === 'true';
export const POLL_INTERVAL = 2000;
export const SCHEDULER_POLL_INTERVAL = 60000;

// Absolute paths needed for container mounts
const PROJECT_ROOT = process.cwd();
const HOME_DIR = process.env.HOME || os.homedir();

// Mount security: allowlist stored OUTSIDE project root, never mounted into containers
export const MOUNT_ALLOWLIST_PATH = path.join(
  HOME_DIR,
  '.config',
  'nanoclaw',
  'mount-allowlist.json',
);
export const SENDER_ALLOWLIST_PATH = path.join(
  HOME_DIR,
  '.config',
  'nanoclaw',
  'sender-allowlist.json',
);
export const STORE_DIR = path.resolve(PROJECT_ROOT, 'store');
export const GROUPS_DIR = path.resolve(PROJECT_ROOT, 'groups');
export const DATA_DIR = path.resolve(PROJECT_ROOT, 'data');

export const CONTAINER_IMAGE =
  process.env.CONTAINER_IMAGE || 'nanoclaw-agent:latest';

// Per-runtime container images. Both runtimes share the same image by default
// (agent-runner detects runtime from ContainerInput and uses the right SDK).
const RUNTIME_IMAGES: Record<string, string | undefined> = {
  claude: process.env.CONTAINER_IMAGE_CLAUDE || undefined, // falls back to CONTAINER_IMAGE
  codex: process.env.CONTAINER_IMAGE_CODEX || undefined, // falls back to CONTAINER_IMAGE
};

export const DEFAULT_RUNTIME = envConfig.DEFAULT_RUNTIME || 'claude';
export const DEFAULT_MODEL = process.env.OPENAI_MODEL || 'gpt-5.4-mini';

/** Available models grouped by provider. Used by /model command. */
export const AVAILABLE_MODELS: Record<
  string,
  Array<{ id: string; name: string }>
> = {
  openai: [
    { id: 'gpt-5.4-mini', name: 'GPT-5.4 Mini — fast, light quota' },
    { id: 'gpt-5.2', name: 'GPT-5.2 — general purpose' },
    { id: 'gpt-5.3-codex', name: 'GPT-5.3 Codex — tool/code focused' },
    { id: 'gpt-5.4', name: 'GPT-5.4 — max capability' },
  ],
  claude: [
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
    { id: 'claude-sonnet-4-5-20250514', name: 'Claude Sonnet 4.5' },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5' },
  ],
  // Local models via OMLX (direct, no LiteLLM needed).
  // Requires OMLX running: brew services start omlx
  local: [
    { id: 'mlx-community/Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B (local)' },
    { id: 'mlx-community/Qwen2.5-7B-Instruct', name: 'Qwen 2.5 7B (local)' },
    {
      id: 'mlx-community/DeepSeek-R1-0528-Qwen3-8B',
      name: 'DeepSeek R1 8B (local)',
    },
  ],
  // Custom models via LiteLLM. Requires LITELLM_URL in .env.
  custom: [
    {
      id: 'huggingface/meta-llama/Llama-3.1-70B-Instruct',
      name: 'Llama 3.1 70B (HuggingFace)',
    },
  ],
};

export const OMLX_URL = process.env.OMLX_URL || 'http://localhost:8000/v1';
export const LITELLM_URL = envConfig.LITELLM_URL || '';

/** Resolve the container image for a given runtime. */
export function getContainerImage(runtime?: string): string {
  const r = runtime || DEFAULT_RUNTIME;
  return RUNTIME_IMAGES[r] || CONTAINER_IMAGE;
}
export const CONTAINER_TIMEOUT = parseInt(
  process.env.CONTAINER_TIMEOUT || '1800000',
  10,
);
export const CONTAINER_MAX_OUTPUT_SIZE = parseInt(
  process.env.CONTAINER_MAX_OUTPUT_SIZE || '10485760',
  10,
); // 10MB default
export const CREDENTIAL_PROXY_PORT = parseInt(
  process.env.CREDENTIAL_PROXY_PORT || '3001',
  10,
);
export const MAX_MESSAGES_PER_PROMPT = Math.max(
  1,
  parseInt(process.env.MAX_MESSAGES_PER_PROMPT || '10', 10) || 10,
);
export const IPC_POLL_INTERVAL = 1000;
export const IDLE_TIMEOUT = parseInt(process.env.IDLE_TIMEOUT || '1800000', 10); // 30min default — how long to keep container alive after last result
export const MAX_CONCURRENT_CONTAINERS = Math.max(
  1,
  parseInt(process.env.MAX_CONCURRENT_CONTAINERS || '5', 10) || 5,
);

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function buildTriggerPattern(trigger: string): RegExp {
  return new RegExp(`^${escapeRegex(trigger.trim())}\\b`, 'i');
}

export const DEFAULT_TRIGGER = `@${ASSISTANT_NAME}`;

export function getTriggerPattern(trigger?: string): RegExp {
  const normalizedTrigger = trigger?.trim();
  return buildTriggerPattern(normalizedTrigger || DEFAULT_TRIGGER);
}

export const TRIGGER_PATTERN = buildTriggerPattern(DEFAULT_TRIGGER);

// Timezone for scheduled tasks, message formatting, etc.
// Validates each candidate is a real IANA identifier before accepting.
function resolveConfigTimezone(): string {
  const candidates = [
    process.env.TZ,
    envConfig.TZ,
    Intl.DateTimeFormat().resolvedOptions().timeZone,
  ];
  for (const tz of candidates) {
    if (tz && isValidTimezone(tz)) return tz;
  }
  return 'UTC';
}
export const TIMEZONE = resolveConfigTimezone();
