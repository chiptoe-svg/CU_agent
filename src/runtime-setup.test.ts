import { describe, it, expect, vi, beforeEach } from 'vitest';

// Hoist the fs mock so it's available to vi.mock factories
const fsMock = vi.hoisted(() => ({
  existsSync: vi.fn(() => false),
  mkdirSync: vi.fn(),
  writeFileSync: vi.fn(),
  readFileSync: vi.fn(() => ''),
  readdirSync: vi.fn((): string[] => []),
  statSync: vi.fn(() => ({ isDirectory: () => false, mtimeMs: 0 })),
  copyFileSync: vi.fn(),
  cpSync: vi.fn(),
}));

vi.mock('fs', () => ({ default: fsMock, ...fsMock }));

vi.mock('./config.js', () => ({
  CREDENTIAL_PROXY_PORT: 3001,
}));

vi.mock('./container-runtime.js', () => ({
  CONTAINER_HOST_GATEWAY: 'host.docker.internal',
}));

vi.mock('./credential-proxy.js', () => ({
  detectAuthMode: vi.fn(() => 'api-key'),
}));

vi.mock('./env.js', () => ({
  readEnvFile: vi.fn(() => ({})),
}));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import { getRuntimeSetup, type RuntimeSetupContext } from './runtime-setup.js';
import { detectAuthMode } from './credential-proxy.js';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';

function makeCtx(
  overrides?: Partial<RuntimeSetupContext>,
): RuntimeSetupContext {
  return {
    group: {
      name: 'Test',
      folder: 'test-group',
      trigger: '@Andy',
      added_at: new Date().toISOString(),
    },
    runtime: 'claude',
    groupSessionsBase: '/data/sessions/test-group',
    projectRoot: '/app',
    ...overrides,
  };
}

describe('runtime-setup', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fsMock.existsSync.mockReturnValue(false);
    fsMock.readdirSync.mockReturnValue([]);
  });

  // --- Claude ---

  describe('claude', () => {
    it('prepareHome creates .claude dir with settings and skills', () => {
      const setup = getRuntimeSetup('claude');
      // skills source dir exists with one skill
      fsMock.existsSync.mockImplementation(((p: string) => {
        if (p.includes('container/skills')) return true;
        return false;
      }) as typeof fsMock.existsSync);
      fsMock.readdirSync.mockReturnValue(['wiki'] as unknown as ReturnType<
        typeof fsMock.readdirSync
      >);
      fsMock.statSync.mockReturnValue({
        isDirectory: () => true,
        mtimeMs: 0,
      } as unknown as ReturnType<typeof fsMock.statSync>);

      const mount = setup.prepareHome(makeCtx({ runtime: 'claude' }));

      expect(mount.hostPath).toBe('/data/sessions/test-group/.claude');
      expect(mount.containerPath).toBe('/home/node/.claude');
      // Created dir
      expect(fsMock.mkdirSync).toHaveBeenCalledWith(
        '/data/sessions/test-group/.claude',
        { recursive: true },
      );
      // Wrote settings.json
      expect(fsMock.writeFileSync).toHaveBeenCalledWith(
        '/data/sessions/test-group/.claude/settings.json',
        expect.stringContaining('CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS'),
      );
      // Synced skills
      expect(fsMock.cpSync).toHaveBeenCalledWith(
        '/app/container/skills/wiki',
        '/data/sessions/test-group/.claude/skills/wiki',
        { recursive: true },
      );
    });

    it('does not overwrite existing settings.json', () => {
      const setup = getRuntimeSetup('claude');
      fsMock.existsSync.mockImplementation(((p: string) =>
        p.endsWith('settings.json')) as typeof fsMock.existsSync);

      setup.prepareHome(makeCtx({ runtime: 'claude' }));

      const settingsWrites = fsMock.writeFileSync.mock.calls.filter((args) =>
        String(args[0]).includes('settings.json'),
      );
      expect(settingsWrites).toHaveLength(0);
    });

    it('getCredentialEnv returns proxy URL + api-key placeholder', () => {
      vi.mocked(detectAuthMode).mockReturnValue('api-key');
      const setup = getRuntimeSetup('claude');
      const env = setup.getCredentialEnv(makeCtx({ runtime: 'claude' }));

      expect(env.ANTHROPIC_BASE_URL).toBe('http://host.docker.internal:3001');
      expect(env.ANTHROPIC_API_KEY).toBe('placeholder');
      expect(env.CLAUDE_CODE_OAUTH_TOKEN).toBeUndefined();
    });

    it('getCredentialEnv returns oauth placeholder when oauth mode', () => {
      vi.mocked(detectAuthMode).mockReturnValue('oauth');
      const setup = getRuntimeSetup('claude');
      const env = setup.getCredentialEnv(makeCtx({ runtime: 'claude' }));

      expect(env.CLAUDE_CODE_OAUTH_TOKEN).toBe('placeholder');
      expect(env.ANTHROPIC_API_KEY).toBeUndefined();
    });
  });

  // --- Codex ---

  describe('codex', () => {
    it('prepareHome copies auth + config when host .codex exists', () => {
      const setup = getRuntimeSetup('codex');
      fsMock.existsSync.mockImplementation(((p: string) => {
        if (p.includes('.codex')) return true;
        if (p.includes('container/skills')) return true;
        return false;
      }) as typeof fsMock.existsSync);
      fsMock.readdirSync.mockReturnValue([]);

      const mount = setup.prepareHome(makeCtx({ runtime: 'codex' }));

      expect(mount.containerPath).toBe('/home/node/.codex');
      expect(fsMock.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('/data/sessions/test-group/.codex/auth.json'),
        expect.any(String),
        expect.objectContaining({ mode: 0o600 }),
      );
      expect(fsMock.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('/data/sessions/test-group/.codex/config.toml'),
        expect.any(String),
        expect.objectContaining({ mode: 0o600 }),
      );
    });

    it('prepareHome falls back to minimal home when no host .codex', () => {
      const setup = getRuntimeSetup('codex');
      fsMock.existsSync.mockReturnValue(false);

      const mount = setup.prepareHome(makeCtx({ runtime: 'codex' }));

      expect(mount.hostPath).toBe('/data/sessions/test-group/.codex');
      expect(mount.containerPath).toBe('/home/node/.codex');
      const codexWrites = fsMock.writeFileSync.mock.calls
        .filter((args) => String(args[0]).includes('/data/sessions/test-group/.codex/'));
      expect(codexWrites).toHaveLength(0);
    });

    it('getCredentialEnv injects API key when no subscription auth', () => {
      const setup = getRuntimeSetup('codex');
      fsMock.existsSync.mockReturnValue(false); // no auth.json
      vi.mocked(readEnvFile).mockReturnValue({ OPENAI_API_KEY: 'sk-test' });

      const env = setup.getCredentialEnv(makeCtx({ runtime: 'codex' }));

      expect(env.OPENAI_API_KEY).toBe('sk-test');
    });

    it('getCredentialEnv skips API key when subscription auth exists', () => {
      const setup = getRuntimeSetup('codex');
      fsMock.existsSync.mockImplementation(((p: string) =>
        p.includes('auth.json')) as typeof fsMock.existsSync);

      const env = setup.getCredentialEnv(makeCtx({ runtime: 'codex' }));

      expect(env.OPENAI_API_KEY).toBeUndefined();
    });

    it('getCredentialEnv uses group baseUrl over env', () => {
      const setup = getRuntimeSetup('codex');
      fsMock.existsSync.mockReturnValue(false);
      vi.mocked(readEnvFile).mockReturnValue({
        OPENAI_BASE_URL: 'http://global',
      });

      const ctx = makeCtx({ runtime: 'codex' });
      ctx.group.containerConfig = { baseUrl: 'http://per-group' };
      const env = setup.getCredentialEnv(ctx);

      expect(env.OPENAI_BASE_URL).toBe('http://per-group');
    });
  });

  // --- Gemini ---

  describe('gemini', () => {
    it('prepareHome creates .gemini dir with skills', () => {
      const setup = getRuntimeSetup('gemini');
      fsMock.existsSync.mockImplementation(((p: string) =>
        p.includes('container/skills')) as typeof fsMock.existsSync);
      fsMock.readdirSync.mockReturnValue([]);

      const mount = setup.prepareHome(makeCtx({ runtime: 'gemini' }));

      expect(mount.hostPath).toBe('/data/sessions/test-group/.gemini');
      expect(mount.containerPath).toBe('/home/node/.gemini');
    });

    it('getCredentialEnv injects API key from GEMINI_API_KEY', () => {
      const setup = getRuntimeSetup('gemini');
      vi.mocked(readEnvFile).mockReturnValue({ GEMINI_API_KEY: 'gk-test' });

      const env = setup.getCredentialEnv(makeCtx({ runtime: 'gemini' }));

      expect(env.GEMINI_API_KEY).toBe('gk-test');
      expect(env.GOOGLE_API_KEY).toBe('gk-test');
    });

    it('getCredentialEnv falls back to GOOGLE_API_KEY', () => {
      const setup = getRuntimeSetup('gemini');
      vi.mocked(readEnvFile).mockReturnValue({ GOOGLE_API_KEY: 'gak-test' });

      const env = setup.getCredentialEnv(makeCtx({ runtime: 'gemini' }));

      expect(env.GEMINI_API_KEY).toBe('gak-test');
    });

    it('getCredentialEnv returns empty when no key configured', () => {
      const setup = getRuntimeSetup('gemini');
      vi.mocked(readEnvFile).mockReturnValue({});

      const env = setup.getCredentialEnv(makeCtx({ runtime: 'gemini' }));

      expect(Object.keys(env)).toHaveLength(0);
    });
  });

  // --- Fallback / unknown runtime ---

  describe('fallback (unknown runtime)', () => {
    it('prepareHome creates minimal home and logs warning', () => {
      const setup = getRuntimeSetup('google-adk');

      const mount = setup.prepareHome(makeCtx({ runtime: 'google-adk' }));

      expect(mount.hostPath).toBe('/data/sessions/test-group/home');
      expect(mount.containerPath).toBe('/home/node');
      expect(logger.warn).toHaveBeenCalledWith(
        expect.objectContaining({ runtime: 'google-adk' }),
        expect.stringContaining('Unknown runtime'),
      );
    });

    it('getCredentialEnv returns empty and logs warning', () => {
      const setup = getRuntimeSetup('unknown-runtime');

      const env = setup.getCredentialEnv(
        makeCtx({ runtime: 'unknown-runtime' }),
      );

      expect(env).toEqual({});
      expect(logger.warn).toHaveBeenCalledWith(
        expect.objectContaining({ runtime: 'unknown-runtime' }),
        expect.stringContaining('No credential injection'),
      );
    });

    it('empty string runtime uses fallback', () => {
      const setup = getRuntimeSetup('');
      const mount = setup.prepareHome(makeCtx({ runtime: '' }));

      expect(mount.containerPath).toBe('/home/node');
    });
  });
});
