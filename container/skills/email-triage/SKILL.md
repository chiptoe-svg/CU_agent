---
name: email-triage
description: Scan inbox for actionable emails, create MS365 To Do tasks (or Apple Reminders as fallback) for things needing response or action, and file emails when the task is completed. Use /email-triage to scan now, /email-triage status for pending items.
---

# /email-triage — Email Triage

Scan inbox emails, identify actionable items, create MS365 To Do tasks for them (with Apple Reminders and the deprecated file-backed `todo_*` as fallbacks). Filing happens when the task is completed — either by the agent on user request ("mark it done") or by the user tap-completing on their iPhone / in Outlook / in Microsoft To Do (picked up by the host-side reconciliation polls for both surfaces).

## Modes

Parse the user's command:
- `/email-triage` or `/email-triage scan` → **Scan mode**
- `/email-triage status` → **Status mode**
- `/email-triage file <task-id>` → **File mode** (complete task + file email)

If this is a scheduled task (message starts with `[SCHEDULED TASK`), run scan mode directly.

## Prerequisites

```bash
test -f /workspace/group/email-accounts.yaml && echo "ACCOUNTS_OK" || echo "NO_ACCOUNTS"
test -f /workspace/group/email-archive/config.yaml && echo "ARCHIVE_OK" || echo "NO_ARCHIVE"
test -f /workspace/group/email-triage/config.yaml && echo "TRIAGE_OK" || echo "NO_TRIAGE"
```

If `NO_ACCOUNTS`: "No email accounts. Run `/add-email-account` first."
If `NO_ARCHIVE`: "No archive config. Run `/add-email-archive` first (need taxonomy and rules)."
If `NO_TRIAGE`: "Triage not configured. Run `/add-email-triage` to set up."

## Load Configuration

```bash
cat /workspace/group/email-triage/config.yaml
cat /workspace/group/email-archive/config.yaml
cat /workspace/group/email-archive/rules.yaml
cat /workspace/group/email-accounts.yaml
```

## Scan Mode

### Delegation — REQUIRED

**When invoked from a user message (not a scheduled task), ALWAYS delegate to a background task.** Do NOT run inline — scanning blocks the conversation.

1. Acknowledge: "Scanning inbox for new actionable emails..."
2. Schedule a one-time immediate task:
   - `schedule_type`: "once"
   - `schedule_value`: 1 minute from now
   - `prompt`: "/email-triage scan"
3. Return control to the user

### Scan Pipeline (when running as scheduled task)

1. **Load state** — read `email-triage/state/progress.yaml` for `last_scan_date` per account
2. **Fetch new inbox emails** per account:

   **gws:**
   ```bash
   GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --query "in:inbox newer_than:1d" --max 50 --format json
   ```

   **ms365:**
   ```
   Call mcp__ms365__list-mail-messages — filter to messages received since last scan.
   ```

3. **For each email, classify:**

   a. **Check sender rules** from `email-archive/rules.yaml`:
      - If sender matches a rule for a non-actionable category (Newsletters, Accounts, Notifications, To Delete) → **skip** (leave in inbox for archive run)
      - If sender matches a rule for an actionable category (Work, Personal) → proceed to step (b)
      - If no rule matches → proceed to step (b)

   b. **Read email content** (subject + snippet/preview is usually enough for triage):
      - Look for signals: question marks, requests ("please", "can you", "need"), deadlines ("by Friday", "due", "deadline"), direct addressing (To: vs CC:)
      - Assess: does this need a response or action from the user?

   c. **Decision:**
      - **Clearly actionable** → create reminder (step 4)
      - **Uncertain** → add to uncertain list (reported in summary for user to decide)
      - **Clearly not actionable** → skip (leave in inbox)

4. **Create a to-do item** for actionable emails. Preferred surface: **MS365 To Do** (tasks sync to iOS Reminders via the Exchange account and the user works out of MS365). Call `mcp__ms365__create-todo-task` with:
   - `title`: concise action (e.g., "Reply to Dr. Smith re: budget meeting")
   - `body`: JSON string with email metadata (same convention as the legacy todo path):
     ```json
     {"email_id": "MSG_ID", "account": "gmail", "from": "smith@clemson.edu", "subject": "Budget meeting", "folder": "Sorted/Work"}
     ```
   - `dueDateTime`: extracted from email content if present (e.g., "by Friday", "due April 18", "deadline tomorrow"), otherwise next business day (skip weekends — Friday defaults to Monday, Saturday/Sunday default to Monday). Provide as an object with `dateTime` and `timeZone` per Graph's requirements.
   - `importance`: `high` if urgent signals, `normal` otherwise
   - List: the user's default To Do list (named `Tasks` on this install). Call `mcp__ms365__list-todo-lists` once at setup time to get the `listId`.
   - On MS365 failure (token expired, Graph 5xx), fall back in order:
     1. `mcp__reminders__reminder_create` into an "Email Actions" Apple Reminders list (call `reminder_list_available` / `reminder_list_create` if the list doesn't exist). Same metadata convention but in `notes` instead of `body`.
     2. `mcp__nanoclaw__todo_create` (deprecated file-backed fallback) so triage doesn't lose items.
   - Surface the underlying error to the user whenever you fall back, so they can fix the upstream cause.

5. **Update state** — save `last_scan_date` per account

6. **Send summary** via `mcp__nanoclaw__send_message`:
   ```
   📬 Email Triage Scan

   Scanned: N new emails (N Gmail, N Outlook)
   New action items: N
   Skipped (known non-actionable): N

   Uncertain — want reminders for any of these?
   • "Re: Q3 budget projections" from jane@clemson.edu
   • "Meeting notes from Tuesday" from dept-list@clemson.edu

   Pending reminders: N total (N overdue)

   View all: /email-triage status
   ```

   Only include the "Uncertain" section if there are uncertain emails. If no new emails found, send a brief "No new emails since last scan."

## File Mode

When the user says something like "mark the Smith email as done" or `/email-triage file <task-id>`, or when a reconciliation poll detects a completion:

- **MS365 reconciler** (`src/ms365-reconciler.ts`) polls Graph for `status eq 'completed'` tasks and fires this mode with the task's body parsed from JSON.
- **Apple Reminders reconciler** (`src/reminders-reconciler.ts`) polls EventKit's `recently_completed` reminders and fires this mode with the reminder's notes parsed from JSON.

Steps:

1. **Look up the to-do item** only if invoked manually — the reconcilers pass the parsed metadata directly. For manual invocation:
   - MS365: `mcp__ms365__list-todo-tasks` (filter by id)
   - Apple: `mcp__reminders__reminder_list` (list: "Email Actions", include_notes: true)
2. **Parse the metadata** to get `email_id`, `account`, and proposed `folder`
3. **Move the email** to the folder:

   **gws_mcp** (preferred — structured MCP tools):
   ```
   Call mcp__gws_mcp__move_gmail_message (or similar) with the message ID and destination label.
   ```

   **gws** (legacy CLI fallback):
   ```bash
   GWS_CREDENTIAL_STORE=plaintext gws gmail users messages modify --params '{"id":"MSG_ID"}' --json '{"addLabelIds":["LABEL_ID"],"removeLabelIds":["INBOX"]}'
   ```

   **ms365:**
   ```
   Call mcp__ms365__move-mail-message with the message ID and destination folder ID.
   ```

   Look up folder/label IDs from `email-archive/config.yaml` (`archive_accounts[].folder_ids`).

4. **Mark the to-do item complete** — skip if the user already tap-completed on their phone (that's how the reconciler saw it). Only needed when the agent is closing the loop itself:
   - MS365: `mcp__ms365__update-todo-task` with `status: 'completed'`
   - Apple: `mcp__reminders__reminder_complete`
5. **Log the filing** — append to `email-triage/state/filed.jsonl`:
   ```json
   {"timestamp": "ISO", "email_id": "...", "account": "...", "folder": "...", "task_id": "..."}
   ```
6. **Confirm** only if the user triggered it manually: "Filed 'Budget meeting' → Sorted/Work. Task completed." Reconciler-triggered filings should run silently (no chat message).

## Status Mode

1. **List pending tasks** on whichever surface was used during scan:
   - MS365 (default): `mcp__ms365__list-todo-tasks` with `status eq 'notStarted' or status eq 'inProgress'`, scoped to the `Tasks` list
   - Apple Reminders (fallback): `mcp__reminders__reminder_list` (list: "Email Actions", status: "pending")
   - Legacy: `mcp__nanoclaw__todo_list`
2. **Read state** for scan stats
3. **Report:**

   ```
   📊 Email Triage Status

   Pending actions (MS365 Tasks):
   [ ] Reply to Dr. Smith re: budget (due Apr 14) ⚠️ overdue
   [ ] Review contract from Legal (due Apr 16)
   [ ] Submit expense report (due Apr 18)

   Last scan: 45min ago
   Filed today: N emails

   Complete an item: /email-triage file <task-id>
   Or tell me: "mark the Smith email as done"
   ```

## Constraints

- **Triage NEVER auto-files emails** — only on explicit todo completion
- **Non-actionable emails stay in inbox** — handled by `/email-archive` runs
- **Agent can complete reminders** when the user instructs (e.g., "mark it done", "I replied to Smith"). The agent can also leave the reminder alone and wait for the user to tap-complete on their iPhone — the reconciliation poll will catch it.
- **NEVER auto-delete emails**
- **NEVER send emails** on the user's behalf without explicit instruction
- **Provider-agnostic** — all email operations use provider reference from `/add-email-account`
- **Save state after each operation** — crash-safe
