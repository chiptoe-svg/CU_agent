#!/usr/bin/env tsx
/**
 * Generic provider login script.
 *
 * Usage:
 *   npm run provider-login          # Lists available providers
 *   npm run provider-login ms365    # Runs MS365 OAuth login
 *   npm run provider-login gws      # Runs GWS OAuth login
 *
 * Reads provider configs from ~/.nanoclaw/providers/ (or container/providers/
 * as fallback) and executes the auth.loginCommand.
 */
import { execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

interface ProviderConfig {
  id: string;
  name: string;
  auth: {
    loginCommand: string;
    postLogin?: string;
    description: string;
  };
}

function loadProviders(): ProviderConfig[] {
  const dirs = [
    path.join(os.homedir(), '.nanoclaw', 'providers'),
    path.join(process.cwd(), 'container', 'providers'),
  ];

  for (const dir of dirs) {
    if (!fs.existsSync(dir)) continue;
    const providers: ProviderConfig[] = [];
    for (const file of fs.readdirSync(dir)) {
      if (!file.endsWith('.json')) continue;
      try {
        const config = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf-8'));
        if (config.id && config.auth?.loginCommand) {
          providers.push(config);
        }
      } catch {
        // skip invalid files
      }
    }
    if (providers.length > 0) return providers;
  }

  return [];
}

const providerId = process.argv[2];
const providers = loadProviders();

if (!providerId) {
  console.log('Available providers:\n');
  for (const p of providers) {
    console.log(`  ${p.id.padEnd(12)} ${p.name} — ${p.auth.description}`);
  }
  console.log(`\nUsage: npm run provider-login <provider-id>`);
  process.exit(0);
}

const provider = providers.find((p) => p.id === providerId);
if (!provider) {
  console.error(`Unknown provider: ${providerId}`);
  console.error(`Available: ${providers.map((p) => p.id).join(', ')}`);
  process.exit(1);
}

console.log(`Logging in to ${provider.name}...`);
console.log(`${provider.auth.description}\n`);

try {
  execSync(provider.auth.loginCommand, { stdio: 'inherit' });

  if (provider.auth.postLogin) {
    console.log('\nRunning post-login setup...');
    execSync(provider.auth.postLogin, { stdio: 'inherit' });
  }

  console.log(`\n✓ ${provider.name} login complete.`);
} catch (err) {
  console.error(`\nLogin failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}
