---
name: add-email-archive
description: Set up email archive system — interactive calibration builds sender rules, creates folder taxonomy, and configures batch processing to reach inbox zero. Requires at least one email account registered via /add-email-account. Run this before using /email-archive in your main chat.
---

# Add Email Archive

Interactive setup for the email archive system. Builds sender-based classification rules from your actual email, creates a folder taxonomy, and configures the container skill for batch processing.

**Prerequisite:** At least one email account must be registered. Run `/add-email-account` first.

## Phase 1: Prerequisites

### Check if already configured

```bash
MAIN_FOLDER=$(sqlite3 store/nanoclaw.db "SELECT folder FROM registered_groups WHERE is_main = 1 LIMIT 1;" 2>/dev/null)
echo "MAIN_FOLDER=${MAIN_FOLDER:-unknown}"
test -d "groups/${MAIN_FOLDER}/email-archive" && echo "CONFIGURED" || echo "NOT_CONFIGURED"
```

If `CONFIGURED`, ask the user whether to recalibrate or skip to Phase 3.

If `MAIN_FOLDER` is empty, ask which group folder to use.

### Check email accounts

```bash
test -f "groups/${MAIN_FOLDER}/email-accounts.yaml" && cat "groups/${MAIN_FOLDER}/email-accounts.yaml" || echo "NO_ACCOUNTS"
```

If `NO_ACCOUNTS` or no enabled accounts:
> No email accounts registered. Run `/add-email-account` first to connect your email.

Stop here.

If accounts exist, list them:
> Found N registered email account(s):
> - gmail (gws) — tonkin@g.clemson.edu
> - outlook (ms365) — tonkin@clemson.edu
>
> Which accounts should be included in the archive? (default: all)

Use `AskUserQuestion` to confirm.

### Create directory structure

```bash
mkdir -p "groups/${MAIN_FOLDER}/email-archive/calibration"
mkdir -p "groups/${MAIN_FOLDER}/email-archive/state"
touch "groups/${MAIN_FOLDER}/email-archive/overrides.jsonl"
```

## Phase 2: Calibration

The goal is to build sender rules that cover 80%+ of email volume so the LLM only handles the long tail.

### Step 2a: Initial taxonomy

Present the default categories:

> Here's a starting folder taxonomy:
>
> - `Archive/Work` — work-related email
> - `Archive/Personal` — personal correspondence
> - `Archive/Accounts` — service notifications, password resets, confirmations
> - `Archive/Newsletters` — subscriptions, digests, mailing lists worth keeping
> - `To Delete` — spam, marketing, expired notifications
>
> Want to customize? We can add subcategories later after seeing your actual email patterns.

Use `AskUserQuestion` to confirm or collect changes.

### Step 2b: Sample and scan

For each selected account, pull up to 200 inbox messages using the provider operations from `/add-email-account`'s reference table. Look up the account's `type` in `email-accounts.yaml` and use the corresponding **list inbox** command.

Parse results. For each email, extract `from` address and domain. Group by domain, then by address within domain. Sort by frequency.

Save raw samples to `groups/${MAIN_FOLDER}/email-archive/calibration/samples-{account_id}.json`.

### Step 2c: Interactive rule building

Merge sender frequencies across all selected accounts. Present top senders grouped by domain:

```
Top email senders (all accounts):

1. noreply@github.com (23 emails)
   Subjects: "PR #123 merged", "Issue #456 opened"
   → Which folder? [1-5, or new name]

2. *@clemson.edu (15 emails from 8 addresses)
   Subjects: "Budget meeting", "Student worker hours"
   → Which folder? [1-5, or new name]

Categories:
  1. Archive/Work
  2. Archive/Personal
  3. Archive/Accounts
  4. Archive/Newsletters
  5. To Delete
  6. [Type a new category name]
  7. Skip (let LLM decide per email)
```

Use `AskUserQuestion` for each sender. Let the user batch answers (e.g., "1:5, 2:1, 3:3").

Each answer becomes a rule in `rules.yaml`:

```yaml
rules:
  - match:
      from_domain: "github.com"
    folder: "To Delete"
    source: "calibration"
    created: "2026-04-11"
    applied_count: 0

  - match:
      from_address: "no-reply@accounts.google.com"
    folder: "Archive/Accounts"
    source: "calibration"
    created: "2026-04-11"
    applied_count: 0
```

Continue until the user says stop or all senders with 3+ emails are covered.

### Step 2d: Coverage check

Re-scan the sample against the rules:

```
Coverage report:
  Rules match: 164/200 emails (82%)
  Unmatched: 36 emails from 12 unique senders

  Unmatched senders (by frequency):
  - random-person@gmail.com (3 emails)
  - support@someservice.com (2 emails)
```

If coverage < 70%, suggest adding rules for remaining high-frequency senders.
If coverage >= 80%, declare calibration successful.

### Step 2e: Model selection and cost estimate

The unmatched emails need LLM classification. Estimate costs:

```
Estimated total inbox: ~N emails
Rule-matched (N%): ~N emails — FREE
LLM-classified (N%): ~N emails

The group's configured model handles LLM classification.
Current group model: [read from DB or .env]

Recommendation: Start with the current model. If accuracy is poor
in interactive review, switch to a stronger model via /model.
```

Use `AskUserQuestion` to confirm.

### Step 2f: Create folders

For each selected account, create the taxonomy folders/labels using the account's provider operations. Look up the account type and use the corresponding **create label/folder** command.

Store the resulting folder/label IDs in the archive config (see Phase 3).

## Phase 3: Deploy

### Write config.yaml

Write `groups/${MAIN_FOLDER}/email-archive/config.yaml`:

```yaml
# Email Archive Configuration
# Accounts are registered in email-accounts.yaml (via /add-email-account).
# This file stores archive-specific settings per account.

archive_accounts:
  - id: gmail               # must match an id in email-accounts.yaml
    folder_ids:              # provider-specific folder/label IDs
      "Archive/Work": "Label_abc123"
      "Archive/Personal": "Label_def456"
      "Archive/Accounts": "Label_ghi789"
      "Archive/Newsletters": "Label_jkl012"
      "To Delete": "Label_mno345"
  - id: outlook
    folder_ids:
      "Archive/Work": "AAMkAD..."
      "Archive/Personal": "AAMkAE..."
      "Archive/Accounts": "AAMkAF..."
      "Archive/Newsletters": "AAMkAG..."
      "To Delete": "AAMkAH..."

batch_size: 25
mode: interactive  # interactive | supervised | autonomous

taxonomy:
  - Archive/Work
  - Archive/Personal
  - Archive/Accounts
  - Archive/Newsletters
  - To Delete

thresholds:
  llm_confidence: 0.8
  auto_promote_after: 5
```

### Write initial rules.yaml

Write the rules built during calibration.

### Initialize state

Write `groups/${MAIN_FOLDER}/email-archive/state/progress.yaml` with one entry per selected account:

```yaml
accounts:
  gmail:
    last_processed_date: null
    total_processed: 0
    status: not_started
  outlook:
    last_processed_date: null
    total_processed: 0
    status: not_started
stats:
  rule_matched: 0
  llm_classified: 0
  corrections: 0
  rules_auto_promoted: 0
```

### Rebuild container

```bash
./container/build.sh
```

### Done

> Email archive is configured with N account(s). Next steps:
>
> 1. Send `/email-archive` in your main chat to process the first batch
> 2. Review the classifications — correct any mistakes
> 3. After a few batches, switch mode from `interactive` to `supervised`
> 4. The system learns from your corrections and auto-generates rules
>
> Commands: `/email-archive`, `/email-archive status`, `/email-archive review`, `/email-archive recalibrate`
