/**
 * Google Workspace CLI init hook.
 *
 * Extracted from shared.ts setupGwsCredentials(). Sets up gws CLI
 * credentials inside the container by copying tokens from the mounted
 * directory and configuring plain-text credential storage.
 *
 * Self-registers with the provider init hook registry.
 */
import fs from 'fs';
import path from 'path';

import { log } from '../shared.js';
import { registerProviderInit } from '../provider-registry.js';

function setupGwsCredentials(): void {
  const gwsCredsSource = '/workspace/.gws-tokens/credentials.json';
  if (!fs.existsSync(gwsCredsSource)) return;

  const gwsConfigDir = path.join(
    process.env.HOME || '/home/node',
    '.config',
    'gws',
  );
  fs.mkdirSync(gwsConfigDir, { recursive: true });

  // Copy credentials as plain-text (container has no keyring)
  const destPath = path.join(gwsConfigDir, 'credentials.json');
  fs.copyFileSync(gwsCredsSource, destPath);

  // Also copy client_secret.json if present
  const clientSecretSource = '/workspace/.gws-tokens/client_secret.json';
  if (fs.existsSync(clientSecretSource)) {
    fs.copyFileSync(
      clientSecretSource,
      path.join(gwsConfigDir, 'client_secret.json'),
    );
  }

  // Tell gws to use plain-text storage (no keyring)
  process.env.GWS_CREDENTIAL_STORE = 'plaintext';
  const bashrc = path.join(process.env.HOME || '/home/node', '.bashrc');
  fs.appendFileSync(bashrc, '\nexport GWS_CREDENTIAL_STORE=plaintext\n');

  log('Configured Google Workspace CLI credentials');
}

// Self-register with provider registry
registerProviderInit('gws-init', setupGwsCredentials);
