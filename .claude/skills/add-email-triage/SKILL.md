---
name: add-email-triage
description: Set up email triage — hourly inbox scanning that creates MS365 To Do tasks (or Apple Reminders, or file-backed todos as a last resort) for actionable emails. Filing happens when the task is completed. Requires email accounts (/add-email-account) and archive config (/add-email-archive).
---

# Add Email Triage

Interactive setup for the email triage system. Configures hourly inbox scanning that identifies actionable emails and creates a to-do item for each one. When the user checks off the item (on iPhone via the Reminders app, in Microsoft To Do, in Outlook — anywhere), the scheduler's reconciliation poll picks it up and files the associated email.

**Prerequisites:**
- At least one email account registered (run `/add-email-account`)
- Email archive configured with taxonomy and rules (run `/add-email-archive`)
- One of the following to-do surfaces set up (triage uses the first one that's available, in this order):
  1. **MS365 Tasks** (preferred) — set up via `/add-email-account` if the email account is MS365. Tasks sync to iOS Reminders via the Exchange account.
  2. **Apple Reminders** — optional, installed via `/add-apple-reminders`. Gives you Siri-creatable items visible to the agent and a different reconciler path.
  3. **Legacy file-backed `todo_*` tools** — automatic fallback when neither of the above is reachable; kept alive with deprecation warnings.

## Tool surface

Email triage prefers **MS365 To Do** for new items via `mcp__ms365__create-todo-task` into the user's default task list (typically named `Tasks`). Each task's `body` field stores the email metadata as JSON — message id, account, sender, subject, proposed folder — so the filing step has everything it needs without re-querying the inbox. When the task is tap-completed on any Microsoft surface (Outlook, MS To Do, iOS Reminders Exchange list), `src/ms365-reconciler.ts` picks it up within ~5 min and enqueues the filing action.

If MS365 isn't set up on this install but Apple Reminders is, use `mcp__reminders__reminder_create` into a dedicated list (default: "Email Actions"); `src/reminders-reconciler.ts` handles completions via EventKit. Same JSON convention goes in the reminder's `notes` field.

If neither is reachable, fall back to the deprecated `mcp__nanoclaw__todo_create` so triage doesn't break. Log the fallback so the user knows to re-run `todo_migrate_to_reminders` (Apple path) or re-enable MS365 once things are back.

## Phase 1: Prerequisites

### Check if already configured

```bash
MAIN_FOLDER=$(sqlite3 store/nanoclaw.db "SELECT folder FROM registered_groups WHERE is_main = 1 LIMIT 1;" 2>/dev/null)
echo "MAIN_FOLDER=${MAIN_FOLDER:-unknown}"
test -f "groups/${MAIN_FOLDER}/email-triage/config.yaml" && echo "CONFIGURED" || echo "NOT_CONFIGURED"
```

If `CONFIGURED`, show current config and ask whether to reconfigure or leave as-is.

### Check email accounts

```bash
test -f "groups/${MAIN_FOLDER}/email-accounts.yaml" && cat "groups/${MAIN_FOLDER}/email-accounts.yaml" || echo "NO_ACCOUNTS"
```

If `NO_ACCOUNTS`:
> No email accounts registered. Run `/add-email-account` first.

Stop here.

### Check archive config

```bash
test -f "groups/${MAIN_FOLDER}/email-archive/config.yaml" && echo "ARCHIVE_OK" || echo "NO_ARCHIVE"
test -f "groups/${MAIN_FOLDER}/email-archive/rules.yaml" && echo "RULES_OK" || echo "NO_RULES"
```

If either missing:
> Email archive not configured. Run `/add-email-archive` first — triage reuses the archive's taxonomy and sender rules.

Stop here.

## Phase 2: Configure

### Create directory structure

```bash
mkdir -p "groups/${MAIN_FOLDER}/email-triage/state"
```

### Choose accounts

Read the accounts from `email-accounts.yaml` and present:

> Which accounts should triage scan?
>
> 1. gmail (gws) — tonkin@g.clemson.edu
> 2. outlook (ms365) — tonkin@clemson.edu
>
> Default: all. Or pick specific ones.

Use `AskUserQuestion`.

### Configure settings

Ask the user to confirm or adjust:

> **Triage settings:**
>
> - **Scan frequency:** every hour (`0 * * * *`)
> - **Default due date:** 3 days from email receipt (for items without explicit deadlines)
> - **Todo list name:** "Email Actions"
>
> Want to change any of these?

Use `AskUserQuestion`.

### Write config

Write `groups/${MAIN_FOLDER}/email-triage/config.yaml`:

```yaml
# Email Triage Configuration
# Scans inbox hourly, creates todos for actionable emails
# Filing happens when todos are completed
# Shares taxonomy and rules with email-archive/

accounts:
  - gmail
  - outlook

schedule:
  scan_cron: "0 * * * *"
  timezone: "America/New_York"

reminders:
  list: "Email Actions"
  default_due_days: 1
  overdue_nag: true

classification:
  rules_path: "../email-archive/rules.yaml"

state_path: "state/"
```

### Initialize state

Write `groups/${MAIN_FOLDER}/email-triage/state/progress.yaml`:

```yaml
accounts:
  gmail:
    last_scan_date: null
    total_scanned: 0
    total_todos_created: 0
  outlook:
    last_scan_date: null
    total_scanned: 0
    total_todos_created: 0
stats:
  total_filed: 0
  total_skipped: 0
```

## Phase 3: Schedule

Set up the hourly scan:

```
Use schedule_task:
  schedule_type: "cron"
  schedule_value: "0 * * * *"
  context_mode: "isolated"
  prompt: "[SCHEDULED TASK] /email-triage scan"
```

## Phase 4: Rebuild and Done

```bash
./container/build.sh
```

> Email triage configured for N account(s).
>
> - Inbox scan: every hour
> - Todo list: "Email Actions"
> - Filing: manual (complete the todo to file the email)
>
> Commands:
> - `/email-triage` — scan new emails now
> - `/email-triage status` — view pending action items
> - `/email-triage file <id>` — complete a todo and file the email
>
> The first hourly scan will run at the next hour mark.
> Or send `/email-triage` now to scan immediately.
