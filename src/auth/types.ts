import type { RegisteredGroup } from '../types.js';

export interface AuthMaterial {
  env?: Record<string, string>;
  files?: Array<{
    path: string;
    content: string;
    mode?: number;
  }>;
  mounts?: Array<{
    hostPath: string;
    containerPath: string;
  }>;
  metadata?: Record<string, string>;
}

export interface AuthValidationResult {
  ok: boolean;
  warnings?: string[];
  errors?: string[];
}

export interface AuthContext {
  group: RegisteredGroup;
  runtime: string;
  groupSessionsBase: string;
  projectRoot: string;
  runtimeOptions?: Record<string, unknown>;
}

export interface ProviderAuthBackend {
  id: string;
  supports(runtime: string): boolean;
  prepare(ctx: AuthContext): Promise<AuthMaterial>;
  validate?(ctx: AuthContext): Promise<AuthValidationResult>;
}
