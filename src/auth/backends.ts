import fs from 'fs';
import path from 'path';

import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';

import type {
  AuthContext,
  AuthMaterial,
  AuthValidationResult,
  ProviderAuthBackend,
} from './types.js';

const LEGACY_CODEX_HOME = path.join(process.env.HOME || '/home/node', '.codex');

function runtimeOptionString(
  runtimeOptions: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const value = runtimeOptions?.[key];
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function getLegacyCodexAuthPath(): string {
  return path.join(LEGACY_CODEX_HOME, 'auth.json');
}

function getLegacyCodexConfigPath(): string {
  return path.join(LEGACY_CODEX_HOME, 'config.toml');
}

function getCodexBaseUrl(ctx: AuthContext): string | undefined {
  return (
    runtimeOptionString(ctx.runtimeOptions, 'baseUrl') ||
    ctx.group.containerConfig?.baseUrl ||
    readEnvFile(['OPENAI_BASE_URL']).OPENAI_BASE_URL
  );
}

const compatibilityEnvBackend: ProviderAuthBackend = {
  id: 'compatibility-env',
  supports(runtime) {
    return runtime === 'codex';
  },

  prepare(ctx) {
    return Promise.resolve(prepareCompatibilityEnv(ctx));
  },

  validate(ctx) {
    return Promise.resolve(validateCompatibilityEnv(ctx));
  },
};

function prepareCompatibilityEnv(ctx: AuthContext): AuthMaterial {
  const env: Record<string, string> = {};
  const hasLegacyAuth = fs.existsSync(getLegacyCodexAuthPath());
  if (!hasLegacyAuth) {
    const secrets = readEnvFile(['OPENAI_API_KEY']);
    if (secrets.OPENAI_API_KEY) {
      env.OPENAI_API_KEY = secrets.OPENAI_API_KEY;
    }
  }

  const baseUrl = getCodexBaseUrl(ctx);
  if (baseUrl) {
    env.OPENAI_BASE_URL = baseUrl;
  }

  return {
    env,
    metadata: {
      source: 'compatibility-env',
    },
  };
}

function validateCompatibilityEnv(ctx: AuthContext): AuthValidationResult {
  const hasLegacyAuth = fs.existsSync(getLegacyCodexAuthPath());
  if (hasLegacyAuth) {
    return { ok: true };
  }

  const secrets = readEnvFile(['OPENAI_API_KEY']);
  if (secrets.OPENAI_API_KEY) {
    return { ok: true };
  }

  return {
    ok: false,
    errors: ['No compatible auth material found for codex runtime.'],
  };
}

function prepareCompatibilityFile(): AuthMaterial {
  const files: AuthMaterial['files'] = [];
  const authPath = getLegacyCodexAuthPath();
  const configPath = getLegacyCodexConfigPath();

  if (fs.existsSync(authPath)) {
    files.push({
      path: 'auth.json',
      content: fs.readFileSync(authPath, 'utf-8'),
      mode: 0o600,
    });
  }

  if (fs.existsSync(configPath)) {
    files.push({
      path: 'config.toml',
      content: fs.readFileSync(configPath, 'utf-8'),
      mode: 0o600,
    });
  }

  return {
    files,
    metadata: files.length > 0 ? { source: 'compatibility-file' } : undefined,
  };
}

function validateCompatibilityFile(): AuthValidationResult {
  const authPath = getLegacyCodexAuthPath();
  if (!fs.existsSync(authPath)) {
    return { ok: true };
  }

  return {
    ok: true,
    warnings: [
      'Using transitional host file auth import for codex; migrate to a provider-neutral backend later.',
    ],
  };
}

const compatibilityFileBackend: ProviderAuthBackend = {
  id: 'compatibility-file',
  supports(runtime) {
    return runtime === 'codex';
  },

  prepare() {
    return Promise.resolve(prepareCompatibilityFile());
  },

  validate() {
    return Promise.resolve(validateCompatibilityFile());
  },
};

const secretManagerBackend: ProviderAuthBackend = {
  id: 'secret-manager',
  supports() {
    return false;
  },
  async prepare() {
    return {
      metadata: {
        status: 'todo',
      },
    };
  },
  async validate() {
    return {
      ok: false,
      warnings: [
        'Secret-manager auth backend is not implemented yet.',
      ],
    };
  },
};

const gatewayProxyBackend: ProviderAuthBackend = {
  id: 'gateway-proxy',
  supports() {
    return false;
  },
  async prepare() {
    return {
      metadata: {
        status: 'todo',
      },
    };
  },
  async validate() {
    return {
      ok: false,
      warnings: [
        'Gateway/proxy auth backend is not implemented yet.',
      ],
    };
  },
};

const AUTH_BACKENDS: ProviderAuthBackend[] = [
  compatibilityFileBackend,
  compatibilityEnvBackend,
  secretManagerBackend,
  gatewayProxyBackend,
];

export function getProviderAuthBackends(runtime: string): ProviderAuthBackend[] {
  return AUTH_BACKENDS.filter((backend) => backend.supports(runtime));
}

export async function validateProviderAuth(
  ctx: AuthContext,
): Promise<AuthValidationResult> {
  return validateProviderAuthSync(ctx);
}

export function validateProviderAuthSync(
  ctx: AuthContext,
): AuthValidationResult {
  const backends = getProviderAuthBackends(ctx.runtime);
  if (backends.length === 0) {
    return { ok: true };
  }

  const warnings: string[] = [];
  const errors: string[] = [];

  for (const backend of backends) {
    const result = validateBackendSync(backend, ctx);
    if (!result) continue;
    if (result.warnings) warnings.push(...result.warnings);
    if (result.errors) errors.push(...result.errors);
  }

  return {
    ok: errors.length === 0,
    warnings: warnings.length > 0 ? warnings : undefined,
    errors: errors.length > 0 ? errors : undefined,
  };
}

export async function prepareProviderAuth(
  ctx: AuthContext,
): Promise<AuthMaterial> {
  return prepareProviderAuthSync(ctx);
}

export function prepareProviderAuthSync(
  ctx: AuthContext,
): AuthMaterial {
  const backends = getProviderAuthBackends(ctx.runtime);
  const merged: AuthMaterial = {};

  for (const backend of backends) {
    const material = prepareBackendSync(backend, ctx);

    if (material.env && Object.keys(material.env).length > 0) {
      merged.env = { ...(merged.env || {}), ...material.env };
    }

    if (material.files && material.files.length > 0) {
      merged.files = [...(merged.files || []), ...material.files];
    }

    if (material.mounts && material.mounts.length > 0) {
      merged.mounts = [...(merged.mounts || []), ...material.mounts];
    }

    if (material.metadata && Object.keys(material.metadata).length > 0) {
      merged.metadata = { ...(merged.metadata || {}), ...material.metadata };
    }
  }

  return merged;
}

function validateBackendSync(
  backend: ProviderAuthBackend,
  ctx: AuthContext,
): AuthValidationResult | undefined {
  switch (backend.id) {
    case 'compatibility-env':
      return validateCompatibilityEnv(ctx);
    case 'compatibility-file':
      return validateCompatibilityFile();
    default:
      return undefined;
  }
}

function prepareBackendSync(
  backend: ProviderAuthBackend,
  ctx: AuthContext,
): AuthMaterial {
  switch (backend.id) {
    case 'compatibility-env':
      return prepareCompatibilityEnv(ctx);
    case 'compatibility-file':
      return prepareCompatibilityFile();
    default:
      return {};
  }
}

export function materializeAuthMaterial(
  authDir: string,
  material: AuthMaterial,
): void {
  if (material.files) {
    fs.mkdirSync(authDir, { recursive: true });
    for (const file of material.files) {
      const dst = path.join(authDir, file.path);
      fs.mkdirSync(path.dirname(dst), { recursive: true });
      fs.writeFileSync(dst, file.content, file.mode ? { mode: file.mode } : undefined);
    }
  }

  if (material.mounts && material.mounts.length > 0) {
    // TODO: Teach container-runner to enforce auth-specific mounts when we wire
    // enterprise secret managers or gateway proxies through the launch policy.
    logger.debug(
      { authDir, mounts: material.mounts.map((mount) => mount.containerPath) },
      'Auth material includes deferred mounts',
    );
  }
}

export const _testing = {
  getLegacyCodexAuthPath,
  getLegacyCodexConfigPath,
  getCodexBaseUrl,
  prepareCompatibilityEnv,
  validateCompatibilityEnv,
  prepareCompatibilityFile,
  validateCompatibilityFile,
};
