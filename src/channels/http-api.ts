/**
 * HTTP API channel for NanoClaw.
 *
 * Lightweight REST endpoint for programmatic access (iOS voice app, scripts, etc.).
 * POST /api/message with { text, apiKey } → waits for agent response → returns { response }.
 *
 * JID format: http:<group-folder> (e.g., "http:telegram_main")
 * Self-registers with the channel registry.
 */
import crypto from 'crypto';
import http from 'http';

import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, type ChannelOpts } from './registry.js';
import type {
  Channel,
  OnInboundMessage,
  OnChatMetadata,
  RegisteredGroup,
} from '../types.js';

const envSecrets = readEnvFile([
  'HTTP_API_KEY',
  'HTTP_API_PORT',
  'HTTP_API_BIND',
]);
const HTTP_API_PORT = parseInt(envSecrets.HTTP_API_PORT || '3100', 10);
const HTTP_API_KEY = envSecrets.HTTP_API_KEY || '';
// Defaults to loopback. Expose on the LAN/tailnet by setting HTTP_API_BIND=0.0.0.0
// in .env, but only after fronting NanoClaw with TLS (e.g. `tailscale serve`).
const HTTP_API_BIND = envSecrets.HTTP_API_BIND || '127.0.0.1';

interface PendingResponse {
  resolve: (text: string) => void;
  timer: ReturnType<typeof setTimeout>;
}

function createHttpApiChannel(opts: ChannelOpts): Channel | null {
  if (!HTTP_API_KEY) {
    return null; // Not configured — skip
  }

  const onMessage = opts.onMessage;
  const registeredGroups = opts.registeredGroups;

  // Map of request ID → pending response resolver
  const pending = new Map<string, PendingResponse>();

  const server = http.createServer(async (req, res) => {
    // No CORS headers by design. Native clients (NanoVoice) don't enforce CORS,
    // so advertising Access-Control-Allow-Origin:* only widens the browser-side
    // attack surface (DNS rebinding, malicious pages probing the local network).
    // If a browser client is ever needed, add an explicit allowlist here.

    if (req.method !== 'POST' || req.url !== '/api/message') {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found. Use POST /api/message' }));
      return;
    }

    // Read body
    let body = '';
    for await (const chunk of req) body += chunk;

    let parsed: { text?: string; group?: string };
    try {
      parsed = JSON.parse(body);
    } catch {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid JSON' }));
      return;
    }

    // Auth check: Authorization: Bearer <token> only.
    // The key is deliberately NOT accepted in the JSON body — request bodies
    // are more likely to end up in access logs, proxies, and tcpdump captures
    // than header-only auth.
    const authHeader = req.headers.authorization;
    const apiKey = authHeader?.startsWith('Bearer ')
      ? authHeader.slice('Bearer '.length).trim()
      : undefined;
    if (!apiKey || apiKey !== HTTP_API_KEY) {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid API key' }));
      return;
    }

    const text = parsed.text?.trim();
    if (!text) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing "text" field' }));
      return;
    }

    // Find the target group — default to main group
    const groups = registeredGroups();
    const targetFolder =
      parsed.group || Object.values(groups).find((g) => g.isMain)?.folder;
    if (!targetFolder) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          error: 'No group found. Set "group" field or register a main group.',
        }),
      );
      return;
    }

    const group = Object.values(groups).find((g) => g.folder === targetFolder);
    if (!group) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `Group "${targetFolder}" not found` }));
      return;
    }

    // Find the actual JID for this group (e.g., "tg:8731035088")
    const groupEntry = Object.entries(groups).find(
      ([, g]) => g.folder === targetFolder,
    );
    if (!groupEntry) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          error: `No registered JID for group "${targetFolder}"`,
        }),
      );
      return;
    }
    const jid = groupEntry[0]; // The actual JID from the registered group

    const requestId = `http-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;

    // Set up response promise with timeout
    const responsePromise = new Promise<string>((resolve) => {
      const timer = setTimeout(
        () => {
          pending.delete(jid);
          resolve('(Request timed out — the agent took too long to respond.)');
        },
        5 * 60 * 1000,
      ); // 5 minute timeout

      pending.set(jid, { resolve, timer });
    });

    // Deliver as an inbound message using the group's real JID
    const message = {
      id: requestId,
      chat_jid: jid,
      sender: 'http-user',
      sender_name: 'Voice',
      content: text,
      timestamp: new Date().toISOString(),
      is_from_me: false,
    };

    logger.info(
      { requestId, jid, text: text.slice(0, 80) },
      'HTTP API message received',
    );

    // Deliver the message (uses the real JID so it routes through the normal pipeline)
    onMessage(jid, message);

    // Wait for agent response
    const response = await responsePromise;

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ response, requestId }));
  });

  return {
    name: 'http-api',

    async connect() {
      return new Promise<void>((resolve) => {
        server.listen(HTTP_API_PORT, HTTP_API_BIND, () => {
          logger.info(
            { host: HTTP_API_BIND, port: HTTP_API_PORT },
            'HTTP API channel listening',
          );
          resolve();
        });
      });
    },

    async sendMessage(jid: string, text: string) {
      const entry = pending.get(jid);
      if (entry) {
        clearTimeout(entry.timer);
        pending.delete(jid);
        entry.resolve(text);
        logger.info({ jid, length: text.length }, 'HTTP API response sent');
      }
      // If no pending request, ignore — Telegram or other channel will handle it
    },

    isConnected() {
      return server.listening;
    },

    ownsJid(jid: string) {
      // Only claim ownership when there's a pending HTTP request for this JID.
      // This prevents stealing responses from Telegram for normal messages.
      return pending.has(jid);
    },

    async disconnect() {
      return new Promise<void>((resolve) => {
        server.close(() => {
          logger.info('HTTP API channel stopped');
          resolve();
        });
      });
    },
  };
}

// Self-register
registerChannel('http-api', (opts) => createHttpApiChannel(opts));
