/**
 * Host-side provider registry.
 *
 * Loads provider JSON configs from ~/.nanoclaw/providers/ and exposes
 * helpers for container-runner.ts (token mounts) and index.ts (startup copy).
 *
 * Provider configs are JSON files declaring token paths, MCP server config,
 * allowed tools, init hooks, and agent docs. Adding a new provider is a
 * single JSON file — no code changes needed.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';

import { logger } from './logger.js';

// --- Types ---

export interface ProviderTokenPaths {
  host: string;
  container: string;
  readonly: boolean;
  requiredFile: string;
}

export interface ProviderConfig {
  id: string;
  name: string;
  description: string;
  tokenPaths: ProviderTokenPaths;
  mcp: unknown;
  allowedTools: string[];
  init: string | null;
  agentDocs: string;
  /** Host env var names to forward into the container (read from .env). */
  containerEnv?: string[];
  auth: {
    loginCommand: string;
    postLogin?: string;
    description: string;
  };
}

// --- Paths ---

const PROVIDERS_DIR = path.join(os.homedir(), '.nanoclaw', 'providers');

// --- Registry ---

let cachedProviders: ProviderConfig[] | null = null;

/** Load all provider configs from ~/.nanoclaw/providers/. Caches result. */
export function loadProviders(): ProviderConfig[] {
  if (cachedProviders) return cachedProviders;

  const providers: ProviderConfig[] = [];

  if (!fs.existsSync(PROVIDERS_DIR)) {
    cachedProviders = providers;
    return providers;
  }

  for (const file of fs.readdirSync(PROVIDERS_DIR)) {
    if (!file.endsWith('.json')) continue;
    const filePath = path.join(PROVIDERS_DIR, file);
    try {
      const config: ProviderConfig = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      );
      if (!config.id || !config.tokenPaths) {
        logger.warn(
          { file },
          'Skipping invalid provider config (missing id or tokenPaths)',
        );
        continue;
      }
      providers.push(config);
    } catch (err) {
      logger.warn(
        { file, error: err instanceof Error ? err.message : String(err) },
        'Failed to load provider config',
      );
    }
  }

  logger.info(
    { count: providers.length, ids: providers.map((p) => p.id) },
    'Loaded provider configs',
  );

  cachedProviders = providers;
  return providers;
}

/** Clear cached providers (for testing or reload). */
export function clearProviderCache(): void {
  cachedProviders = null;
}

/**
 * Resolve a host path with ~ expansion.
 */
function resolveHostPath(hostPath: string): string {
  if (hostPath.startsWith('~/')) {
    return path.join(os.homedir(), hostPath.slice(2));
  }
  return hostPath;
}

/**
 * Get volume mounts for all providers whose token directories exist on the host.
 * Called by container-runner.ts to add provider-specific mounts.
 */
export function getProviderMounts(opts?: { includeTokens?: boolean }): Array<{
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}> {
  const mounts: Array<{
    hostPath: string;
    containerPath: string;
    readonly: boolean;
  }> = [];

  if (opts?.includeTokens) {
    for (const provider of loadProviders()) {
      const hostDir = resolveHostPath(provider.tokenPaths.host);
      if (fs.existsSync(hostDir)) {
        mounts.push({
          hostPath: hostDir,
          containerPath: provider.tokenPaths.container,
          readonly: provider.tokenPaths.readonly,
        });
      }
    }
  }

  // Always mount the providers directory itself (read-only)
  if (fs.existsSync(PROVIDERS_DIR)) {
    mounts.push({
      hostPath: PROVIDERS_DIR,
      containerPath: '/workspace/.providers',
      readonly: true,
    });
  }

  return mounts;
}

/**
 * Copy default provider configs from container/providers/ to ~/.nanoclaw/providers/
 * if the user directory doesn't have them yet. Called on startup.
 */
export function ensureDefaultProviders(projectRoot: string): void {
  const srcDir = path.join(projectRoot, 'container', 'providers');
  if (!fs.existsSync(srcDir)) return;

  fs.mkdirSync(PROVIDERS_DIR, { recursive: true });

  for (const file of fs.readdirSync(srcDir)) {
    if (!file.endsWith('.json')) continue;
    const dst = path.join(PROVIDERS_DIR, file);
    if (!fs.existsSync(dst)) {
      fs.copyFileSync(path.join(srcDir, file), dst);
      logger.info({ file }, 'Copied default provider config');
    }
  }

  // Clear cache so next load picks up new files
  clearProviderCache();
}
