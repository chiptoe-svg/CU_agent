/**
 * Container-side provider registry.
 *
 * Loads provider JSON configs from /workspace/.providers/ (mounted read-only
 * from ~/.nanoclaw/providers/) and exposes helpers for:
 * - MCP server config (shared.ts)
 * - Allowed tools (claude.ts, codex.ts)
 * - Init hooks (index.ts)
 * - Agent docs injection (AGENT.md assembly)
 *
 * Adding a new provider requires only a JSON file — no code changes.
 */
import fs from 'fs';
import path from 'path';

import { log } from './shared.js';

// --- Types ---

interface ProviderMcpConfig {
  command: string;
  args: string[];
  env: Record<string, string>;
}

interface ProviderTokenPaths {
  host: string;
  container: string;
  readonly: boolean;
  requiredFile: string;
}

interface ProviderConfig {
  id: string;
  name: string;
  tokenPaths: ProviderTokenPaths;
  mcp: ProviderMcpConfig | null;
  allowedTools: string[];
  init: string | null;
  agentDocs: string;
}

// --- Init hook registry ---

type InitHook = () => void;
const initHooks: Record<string, InitHook> = {};

/** Register an init hook by name (called from provider init modules). */
export function registerProviderInit(name: string, hook: InitHook): void {
  initHooks[name] = hook;
}

// --- Paths ---

const PROVIDERS_DIR = '/workspace/.providers';

// --- Registry ---

let cachedProviders: ProviderConfig[] | null = null;

/** Load all provider configs from /workspace/.providers/. */
export function loadProviders(): ProviderConfig[] {
  if (cachedProviders) return cachedProviders;

  const providers: ProviderConfig[] = [];

  if (!fs.existsSync(PROVIDERS_DIR)) {
    log('No providers directory found at /workspace/.providers/');
    cachedProviders = providers;
    return providers;
  }

  for (const file of fs.readdirSync(PROVIDERS_DIR)) {
    if (!file.endsWith('.json')) continue;
    try {
      const config: ProviderConfig = JSON.parse(
        fs.readFileSync(path.join(PROVIDERS_DIR, file), 'utf-8'),
      );
      if (!config.id) continue;
      providers.push(config);
    } catch (err) {
      log(`Failed to load provider ${file}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  log(`Loaded ${providers.length} provider(s): ${providers.map((p) => p.id).join(', ')}`);
  cachedProviders = providers;
  return providers;
}

/**
 * Check if a provider's tokens are available in the container.
 * requiredFile may be an exact filename or a shell-style glob (containing
 * `*` or `?`). Globs are matched against entries in the token dir.
 */
function isProviderAvailable(provider: ProviderConfig): boolean {
  const tokenDir = provider.tokenPaths.container;
  const pattern = provider.tokenPaths.requiredFile;

  if (!pattern.includes('*') && !pattern.includes('?')) {
    return fs.existsSync(path.join(tokenDir, pattern));
  }

  if (!fs.existsSync(tokenDir)) return false;
  const regex = new RegExp(
    '^' +
      pattern
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*/g, '.*')
        .replace(/\?/g, '.') +
      '$',
  );
  return fs.readdirSync(tokenDir).some((entry) => regex.test(entry));
}

/**
 * Resolve template variables in MCP env values.
 * Supports:
 *   ${tokenDir}  → provider's container token directory
 *   ${env.VAR}   → process.env.VAR (forwarded from host via containerEnv)
 */
function resolveEnvValue(value: string, provider: ProviderConfig): string {
  return value
    .replace(/\$\{tokenDir\}/g, provider.tokenPaths.container)
    .replace(/\$\{env\.([A-Z_][A-Z0-9_]*)\}/g, (_, name) => process.env[name] ?? '');
}

/**
 * Get MCP server configs for all available providers.
 * Returns a record that can be merged into the mcpServers config.
 */
export function getProviderMcpConfigs(): Record<
  string,
  { command: string; args: string[]; env: Record<string, string> }
> {
  const configs: Record<
    string,
    { command: string; args: string[]; env: Record<string, string> }
  > = {};

  for (const provider of loadProviders()) {
    if (!provider.mcp) continue;
    if (!isProviderAvailable(provider)) continue;

    const resolvedEnv: Record<string, string> = {};
    for (const [key, value] of Object.entries(provider.mcp.env)) {
      resolvedEnv[key] = resolveEnvValue(value, provider);
    }

    configs[provider.id] = {
      command: provider.mcp.command,
      args: provider.mcp.args,
      env: resolvedEnv,
    };
  }

  return configs;
}

/**
 * Get all allowed tool patterns from available providers.
 * Returns an array of tool patterns (e.g., ['mcp__ms365__*']).
 */
export function getProviderAllowedTools(): string[] {
  const tools: string[] = [];

  for (const provider of loadProviders()) {
    if (!isProviderAvailable(provider)) continue;
    tools.push(...provider.allowedTools);
  }

  return tools;
}

/**
 * Run all provider init hooks for available providers.
 * Each provider's init field names a registered hook function.
 */
export function runProviderInits(): void {
  for (const provider of loadProviders()) {
    if (!provider.init) continue;
    if (!isProviderAvailable(provider)) continue;

    const hook = initHooks[provider.init];
    if (hook) {
      log(`Running init hook for provider: ${provider.id}`);
      try {
        hook();
      } catch (err) {
        log(`Init hook failed for ${provider.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      log(`No init hook registered for: ${provider.init} (provider: ${provider.id})`);
    }
  }
}

/**
 * Get agent docs for all available providers.
 * Returns concatenated markdown sections for AGENT.md injection.
 */
export function getProviderAgentDocs(): string {
  const docs: string[] = [];

  for (const provider of loadProviders()) {
    if (!isProviderAvailable(provider)) continue;
    if (provider.agentDocs) {
      docs.push(provider.agentDocs);
    }
  }

  return docs.join('\n\n');
}

/**
 * Get Codex TOML config blocks for MCP servers from available providers.
 * Returns TOML string to append to config.toml.
 */
export function getProviderCodexToml(): string {
  const blocks: string[] = [];

  for (const provider of loadProviders()) {
    if (!provider.mcp) continue;
    if (!isProviderAvailable(provider)) continue;

    const resolvedEnv: Record<string, string> = {};
    for (const [key, value] of Object.entries(provider.mcp.env)) {
      resolvedEnv[key] = resolveEnvValue(value, provider);
    }

    const lines = [
      `[mcp_servers.${provider.id}]`,
      `type = "stdio"`,
      `command = "${provider.mcp.command}"`,
      `args = [${provider.mcp.args.map((a) => `"${a.replace(/"/g, '\\"')}"`).join(', ')}]`,
      `[mcp_servers.${provider.id}.env]`,
    ];

    for (const [key, value] of Object.entries(resolvedEnv)) {
      lines.push(`${key} = "${value.replace(/"/g, '\\"')}"`);
    }

    blocks.push(lines.join('\n'));
  }

  return blocks.join('\n\n');
}
