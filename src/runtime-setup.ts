/**
 * Runtime-specific container setup strategies.
 *
 * Each runtime declares how it wants its home directory laid out,
 * what credentials/env vars it needs injected, and where skills go.
 * container-runner.ts consults this map instead of branching on
 * runtime names.
 *
 * To add a new runtime (e.g. google-adk): add an entry here.
 * No changes needed in container-runner.ts.
 */
import fs from 'fs';
import path from 'path';

import {
  materializeAuthMaterial,
  prepareProviderAuthSync,
  validateProviderAuthSync,
} from './auth/backends.js';
import type { AuthContext } from './auth/types.js';
import { CREDENTIAL_PROXY_PORT } from './config.js';
import { CONTAINER_HOST_GATEWAY } from './container-runtime.js';
import { detectAuthMode } from './credential-proxy.js';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';

// --- Types ---

export interface RuntimeSetupContext extends AuthContext {}

export interface HomeMount {
  hostPath: string;
  containerPath: string;
}

export interface RuntimeSetup {
  /** Prepare the home directory and return the mount point. */
  prepareHome(ctx: RuntimeSetupContext): HomeMount;
  /** Return env vars to inject into the container for auth/credentials. */
  getCredentialEnv(ctx: RuntimeSetupContext): Record<string, string>;
}

// --- Shared helpers ---

/** Sync skill directories from host container/skills/ into a target dir. */
function syncSkills(dstDir: string, projectRoot: string): void {
  const srcDir = path.join(projectRoot, 'container', 'skills');
  if (!fs.existsSync(srcDir)) return;
  for (const entry of fs.readdirSync(srcDir)) {
    const s = path.join(srcDir, entry);
    if (fs.statSync(s).isDirectory()) {
      fs.cpSync(s, path.join(dstDir, entry), { recursive: true });
    }
  }
}

/** Create a per-group home subdir, sync skills into it, return the mount. */
function prepareHomeDir(
  ctx: RuntimeSetupContext,
  subdir: string,
  containerPath: string,
): { homeDir: string; mount: HomeMount } {
  const homeDir = path.join(ctx.groupSessionsBase, subdir);
  fs.mkdirSync(homeDir, { recursive: true });
  syncSkills(path.join(homeDir, 'skills'), ctx.projectRoot);
  return { homeDir, mount: { hostPath: homeDir, containerPath } };
}

/** Copy a file from src to dst if src exists. */
function copyIfExists(src: string, dst: string): void {
  if (fs.existsSync(src)) fs.copyFileSync(src, dst);
}

function createMinimalHome(ctx: RuntimeSetupContext): HomeMount {
  const homeDir = path.join(ctx.groupSessionsBase, 'home');
  fs.mkdirSync(homeDir, { recursive: true });
  return { hostPath: homeDir, containerPath: '/home/node' };
}

// --- Claude ---

const claudeSetup: RuntimeSetup = {
  prepareHome(ctx) {
    const { homeDir, mount } = prepareHomeDir(
      ctx,
      '.claude',
      '/home/node/.claude',
    );

    // SDK settings — only written once (not overwritten on restart)
    const settingsFile = path.join(homeDir, 'settings.json');
    if (!fs.existsSync(settingsFile)) {
      fs.writeFileSync(
        settingsFile,
        JSON.stringify(
          {
            env: {
              CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
              CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
              CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
            },
          },
          null,
          2,
        ) + '\n',
      );
    }

    return mount;
  },

  getCredentialEnv() {
    const env: Record<string, string> = {
      ANTHROPIC_BASE_URL: `http://${CONTAINER_HOST_GATEWAY}:${CREDENTIAL_PROXY_PORT}`,
    };
    if (detectAuthMode() === 'api-key') {
      env.ANTHROPIC_API_KEY = 'placeholder';
    } else {
      env.CLAUDE_CODE_OAUTH_TOKEN = 'placeholder';
    }
    return env;
  },
};

// --- Codex ---

const codexSetup: RuntimeSetup = {
  prepareHome(ctx) {
    const material = prepareProviderAuthSync(ctx);
    const authValidation = validateProviderAuthSync(ctx);
    if (authValidation.warnings) {
      for (const warning of authValidation.warnings) {
        logger.warn({ runtime: ctx.runtime, group: ctx.group.name }, warning);
      }
    }
    if (authValidation.errors) {
      for (const error of authValidation.errors) {
        logger.warn({ runtime: ctx.runtime, group: ctx.group.name }, error);
      }
    }

    const { homeDir, mount } = prepareHomeDir(
      ctx,
      '.codex',
      '/home/node/.codex',
    );
    if (material.files && material.files.length > 0) {
      materializeAuthMaterial(homeDir, material);
    }

    // TODO: Enforce sandboxProfile-aware launch policy in container-runner once
    // the runtime-specific security profiles are wired through container args.
    return mount;
  },

  getCredentialEnv(ctx) {
    return prepareProviderAuthSync(ctx).env || {};
  },
};

// --- Gemini ---

const geminiSetup: RuntimeSetup = {
  prepareHome(ctx) {
    return prepareHomeDir(ctx, '.gemini', '/home/node/.gemini').mount;
  },

  getCredentialEnv() {
    const env: Record<string, string> = {};
    const secrets = readEnvFile(['GEMINI_API_KEY', 'GOOGLE_API_KEY']);
    const apiKey = secrets.GEMINI_API_KEY || secrets.GOOGLE_API_KEY;
    if (apiKey) {
      env.GEMINI_API_KEY = apiKey;
      env.GOOGLE_API_KEY = apiKey;
    }
    return env;
  },
};

// --- Fallback (unknown runtimes) ---

const fallbackSetup: RuntimeSetup = {
  prepareHome(ctx) {
    logger.warn(
      { runtime: ctx.runtime, group: ctx.group.name },
      'Unknown runtime — using minimal home directory with no credentials',
    );
    return createMinimalHome(ctx);
  },

  getCredentialEnv(ctx) {
    logger.warn(
      { runtime: ctx.runtime },
      'No credential injection for unknown runtime',
    );
    return {};
  },
};

// --- Registry ---

const RUNTIME_SETUP: Record<string, RuntimeSetup> = {
  claude: claudeSetup,
  codex: codexSetup,
  gemini: geminiSetup,
};

export function getRuntimeSetup(runtime: string): RuntimeSetup {
  return RUNTIME_SETUP[runtime] || fallbackSetup;
}

/** Visible for testing. */
export const _testing = {
  RUNTIME_SETUP,
  fallbackSetup,
  syncSkills,
  prepareHomeDir,
  copyIfExists,
  createMinimalHome,
};
