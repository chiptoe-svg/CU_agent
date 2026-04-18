/**
 * Microsoft 365 To-Do reconciliation poll.
 *
 * Symmetric with reminders-reconciler.ts but for the MS365 provider. Every
 * MS365_RECONCILER_INTERVAL seconds (default 300 = 5 min, slower than the
 * Apple poll because MS Graph is rate-limited), asks Graph which to-do
 * tasks were completed in the last 24h. For any completed task whose body
 * parses as email-triage metadata, enqueues a one-shot agent task to file
 * the email via the MS365 MCP tools running in-container.
 *
 * Token handling: reads the @azure/msal-node cache that ms-365-mcp-server
 * maintains at ~/.nanoclaw/.ms365-tokens/.token-cache.json, and refreshes
 * via the OAuth refresh-token grant directly against login.microsoftonline.com.
 * No msal-node host dep needed. We do NOT write back to the cache — leaving
 * MSAL ownership intact so we don't race with the in-container MCP.
 *
 * Self-disabling: returns silently each tick if the token cache is missing,
 * the provider config is missing, or no main group is registered. Safe to
 * leave running on installs that haven't set up MS365.
 */

import fs from 'fs';
import os from 'os';
import path from 'path';

import { DATA_DIR } from './config.js';
import { createTask, getTaskById } from './db.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

const TOKEN_CACHE_PATH = path.join(
  os.homedir(),
  '.nanoclaw',
  '.ms365-tokens',
  '.token-cache.json',
);
const PROVIDER_CONFIG_PATH = path.join(
  os.homedir(),
  '.nanoclaw',
  'providers',
  'ms365.json',
);
const STATE_FILE = path.join(DATA_DIR, 'ms365-reconciled.json');

const DEFAULT_INTERVAL_S = 300;
const SEEN_TTL_MS = 48 * 60 * 60 * 1000;
const SEEN_MAX = 2000;
const LOOKBACK_HOURS = 24;

interface TodoTask {
  id: string;
  title: string;
  body?: { content?: string; contentType?: string };
  status: string;
  completedDateTime?: { dateTime: string; timeZone?: string };
}

interface EmailMetadata {
  email_id: string;
  account: string;
  from?: string;
  subject?: string;
  folder?: string;
}

interface SeenEntry {
  id: string;
  at: number;
}

export interface Ms365ReconcilerDeps {
  registeredGroups: () => Record<string, RegisteredGroup>;
  onTaskCreated: () => void;
}

function getIntervalMs(): number {
  const s = parseInt(process.env.MS365_RECONCILER_INTERVAL || '', 10);
  return (Number.isFinite(s) && s > 0 ? s : DEFAULT_INTERVAL_S) * 1000;
}

function getClientAndTenant(): { clientId: string; tenantId: string } | null {
  try {
    const raw = fs.readFileSync(PROVIDER_CONFIG_PATH, 'utf-8');
    const config = JSON.parse(raw) as {
      mcp?: {
        env?: { MS365_MCP_CLIENT_ID?: string; MS365_MCP_TENANT_ID?: string };
      };
    };
    const env = config?.mcp?.env;
    if (env?.MS365_MCP_CLIENT_ID && env?.MS365_MCP_TENANT_ID) {
      return {
        clientId: env.MS365_MCP_CLIENT_ID,
        tenantId: env.MS365_MCP_TENANT_ID,
      };
    }
  } catch {
    /* missing or unparseable — caller treats as "not configured" */
  }
  return null;
}

interface MsalCache {
  AccessToken?: Record<string, { secret: string; expiresOn?: string }>;
  RefreshToken?: Record<string, { secret: string }>;
}

async function getAccessToken(): Promise<string | null> {
  const conf = getClientAndTenant();
  if (!conf) return null;

  let cache: MsalCache;
  try {
    cache = JSON.parse(fs.readFileSync(TOKEN_CACHE_PATH, 'utf-8')) as MsalCache;
  } catch {
    return null;
  }

  // Use a non-expired access token if one exists (1 min safety buffer).
  const now = Date.now();
  for (const t of Object.values(cache.AccessToken || {})) {
    if (!t.expiresOn) continue;
    const expires = parseInt(t.expiresOn, 10) * 1000;
    if (expires > now + 60_000) return t.secret;
  }

  // Otherwise refresh against login.microsoftonline.com.
  const refresh = Object.values(cache.RefreshToken || {})[0];
  if (!refresh?.secret) return null;

  try {
    const resp = await fetch(
      `https://login.microsoftonline.com/${conf.tenantId}/oauth2/v2.0/token`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'refresh_token',
          client_id: conf.clientId,
          refresh_token: refresh.secret,
          scope: 'https://graph.microsoft.com/Tasks.Read',
        }).toString(),
      },
    );
    if (!resp.ok) {
      logger.debug(
        { status: resp.status },
        'ms365 reconciler: token refresh failed',
      );
      return null;
    }
    const data = (await resp.json()) as { access_token?: string };
    return data.access_token ?? null;
  } catch (err) {
    logger.debug({ err }, 'ms365 reconciler: token endpoint unreachable');
    return null;
  }
}

async function fetchRecentlyCompleted(token: string): Promise<TodoTask[]> {
  const cutoff = new Date(
    Date.now() - LOOKBACK_HOURS * 60 * 60 * 1000,
  ).toISOString();
  const headers = { Authorization: `Bearer ${token}` };

  let lists: Array<{ id: string }> = [];
  try {
    const r = await fetch('https://graph.microsoft.com/v1.0/me/todo/lists', {
      headers,
    });
    if (!r.ok) return [];
    lists = ((await r.json()) as { value: Array<{ id: string }> }).value || [];
  } catch {
    return [];
  }

  const all: TodoTask[] = [];
  const filter = `status eq 'completed' and completedDateTime/dateTime ge '${cutoff}'`;
  for (const list of lists) {
    try {
      const url =
        `https://graph.microsoft.com/v1.0/me/todo/lists/${list.id}/tasks` +
        `?$filter=${encodeURIComponent(filter)}&$top=50`;
      const r = await fetch(url, { headers });
      if (!r.ok) continue;
      const tasks = ((await r.json()) as { value: TodoTask[] }).value || [];
      all.push(...tasks);
    } catch {
      /* per-list failure shouldn't kill the tick */
    }
  }
  return all;
}

function tryParseEmailMeta(body: TodoTask['body']): EmailMetadata | null {
  const content = body?.content?.trim();
  if (!content) return null;
  // MS To-Do strips the JSON of formatting if you save it through the UI.
  // Try parsing as-is first, then strip HTML if that fails.
  const candidates = [content, content.replace(/<[^>]*>/g, '').trim()];
  for (const c of candidates) {
    try {
      const parsed = JSON.parse(c) as Record<string, unknown>;
      if (
        typeof parsed.email_id === 'string' &&
        typeof parsed.account === 'string'
      ) {
        return {
          email_id: parsed.email_id,
          account: parsed.account,
          from: typeof parsed.from === 'string' ? parsed.from : undefined,
          subject:
            typeof parsed.subject === 'string' ? parsed.subject : undefined,
          folder: typeof parsed.folder === 'string' ? parsed.folder : undefined,
        };
      }
    } catch {
      /* try next candidate */
    }
  }
  return null;
}

function loadSeen(): SeenEntry[] {
  try {
    const raw = fs.readFileSync(STATE_FILE, 'utf-8');
    const parsed = JSON.parse(raw) as SeenEntry[];
    if (Array.isArray(parsed)) return parsed;
  } catch {
    /* empty start */
  }
  return [];
}

function saveSeen(entries: SeenEntry[]): void {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(STATE_FILE, JSON.stringify(entries));
  } catch (err) {
    logger.warn({ err }, 'ms365 reconciler: failed to persist seen set');
  }
}

function pruneSeen(entries: SeenEntry[]): SeenEntry[] {
  const cutoff = Date.now() - SEEN_TTL_MS;
  return entries.filter((e) => e.at >= cutoff).slice(-SEEN_MAX);
}

function findMainGroup(
  groups: Record<string, RegisteredGroup>,
): { jid: string; group: RegisteredGroup } | null {
  for (const [jid, group] of Object.entries(groups)) {
    if (group.isMain) return { jid, group };
  }
  return null;
}

function buildFilingPrompt(task: TodoTask, meta: EmailMetadata): string {
  return [
    `The user tap-completed a Microsoft 365 to-do that was created by email-triage.`,
    `File the associated email silently — do not send a chat message. Append the filing record to /workspace/group/email-triage/state/filed.jsonl.`,
    ``,
    `Task:`,
    `- id: ${task.id}`,
    `- title: ${task.title}`,
    `- completed_at: ${task.completedDateTime?.dateTime ?? 'unknown'}`,
    ``,
    `Email metadata (from task body):`,
    `- email_id: ${meta.email_id}`,
    `- account: ${meta.account}`,
    meta.from ? `- from: ${meta.from}` : '',
    meta.subject ? `- subject: ${meta.subject}` : '',
    meta.folder
      ? `- target folder: ${meta.folder}`
      : `- target folder: (not specified — look up from email-archive/rules.yaml)`,
    ``,
    `Use the appropriate MCP tool for the account (mcp__ms365__move-mail-message, mcp__gws_mcp__*, or the legacy gws CLI) to move the message out of the inbox into the target folder. If the email is already filed, log that as the outcome.`,
  ]
    .filter(Boolean)
    .join('\n');
}

export function startMs365Reconciler(deps: Ms365ReconcilerDeps): void {
  const seenList = pruneSeen(loadSeen());
  const seen = new Set(seenList.map((e) => e.id));

  const intervalMs = getIntervalMs();
  logger.info({ intervalMs }, 'ms365 reconciler starting');

  const tick = async (): Promise<void> => {
    try {
      const main = findMainGroup(deps.registeredGroups());
      if (!main) return;

      const token = await getAccessToken();
      if (!token) return; // MS365 not set up, or refresh failed — try again next tick

      const tasks = await fetchRecentlyCompleted(token);
      if (tasks.length === 0) return;

      let filed = 0;
      for (const t of tasks) {
        if (t.status !== 'completed' || seen.has(t.id)) continue;

        const meta = tryParseEmailMeta(t.body);
        seen.add(t.id);
        seenList.push({ id: t.id, at: Date.now() });
        if (!meta) continue;

        const taskId = `ms365-reconcile-${t.id}`;
        if (getTaskById(taskId)) continue;

        try {
          createTask({
            id: taskId,
            group_folder: main.group.folder,
            chat_jid: main.jid,
            prompt: buildFilingPrompt(t, meta),
            script: null,
            schedule_type: 'once',
            schedule_value: new Date().toISOString(),
            context_mode: 'isolated',
            next_run: new Date().toISOString(),
            status: 'active',
            created_at: new Date().toISOString(),
          });
          filed += 1;
          logger.info(
            {
              ms365TaskId: t.id,
              emailId: meta.email_id,
              account: meta.account,
            },
            'enqueued email-filing task from completed MS365 to-do',
          );
        } catch (err) {
          logger.warn(
            { err, ms365TaskId: t.id },
            'failed to enqueue MS365 filing task — will retry next tick',
          );
          seen.delete(t.id);
          seenList.pop();
        }
      }

      if (filed > 0) {
        deps.onTaskCreated();
        const pruned = pruneSeen(seenList);
        seenList.length = 0;
        seenList.push(...pruned);
        saveSeen(seenList);
      }
    } catch (err) {
      logger.warn({ err }, 'ms365 reconciler tick failed');
    } finally {
      setTimeout(tick, intervalMs);
    }
  };

  setTimeout(tick, 10_000);
}
