---
name: email-archive
description: Batch-classify old emails into organized folders, working toward inbox zero. Uses sender rules (free, fast) with LLM fallback for unknown senders. Supports interactive, supervised, and autonomous modes. Provider-agnostic — works with any email account configured in config.yaml. Use /email-archive to process a batch, /email-archive status for progress.
---

# /email-archive — Email Archive

Classify and file old emails in batches. Rule-based classification handles known senders (no LLM cost). Unknown senders get LLM classification with confidence scoring.

## Modes

Parse the user's command:
- `/email-archive` or `/email-archive run` → **Batch mode**
- `/email-archive status` → **Status mode**
- `/email-archive review` → **Review mode**
- `/email-archive recalibrate` → **Recalibrate mode**

If this is a scheduled task (message starts with `[SCHEDULED TASK`), run in batch mode directly.

### Delegation — REQUIRED for batch mode

**When invoked from a user message (not a scheduled task), ALWAYS delegate batch mode to a background task.** Do NOT run email processing inline — it blocks the main conversation for minutes.

Instead:
1. Acknowledge immediately: "Starting email archive batch in the background."
2. Schedule a one-time immediate task using `schedule_task`:
   - `schedule_type`: "once"
   - `schedule_value`: 1 minute from now
   - `prompt`: "/email-archive run"
3. Return control to the user

The scheduled task runs in its own container and sends results via `send_message` when done. Status and review modes are fast and CAN run inline.

## Setup Check

```bash
test -f /workspace/group/email-archive/config.yaml && echo "CONFIG_OK" || echo "NO_CONFIG"
```

If `NO_CONFIG`:
> Email archive is not configured. Run `/add-email-archive` to set up.

## Load Configuration

```bash
cat /workspace/group/email-accounts.yaml
cat /workspace/group/email-archive/config.yaml
cat /workspace/group/email-archive/rules.yaml
cat /workspace/group/email-archive/state/progress.yaml
```

Parse into working variables:
- From `email-accounts.yaml`: account id, type, address, enabled status
- From `config.yaml`: archive_accounts (with folder_ids), batch_size, mode, taxonomy, taxonomy_context, thresholds, age_rules
- From `rules.yaml`: sender classification rules + llm_hints
- From `progress.yaml`: per-account bookmark and stats

Join accounts: for each entry in `config.yaml`'s `archive_accounts`, look up the account type from `email-accounts.yaml`.

### Current Taxonomy

The taxonomy is defined in `config.yaml`. Each category has context in `taxonomy_context` to guide LLM classification:

| Category | Description |
|----------|-------------|
| Sorted/SI-Development | Sonoco Institute development projects, equipment, systems |
| Sorted/SI-Operations | Sonoco Institute operations, scheduling, tours, events |
| Sorted/SI-Other | Sonoco Institute other/general |
| Sorted/GC-Development | Graphic Communications department development |
| Sorted/GC-Operations | Graphic Communications operations, admin |
| Sorted/GC-Curriculum | Graphic Communications curriculum, course design |
| Sorted/GC-Student | Graphic Communications student-related (internships, advising, registration) |
| Sorted/Classes | Course/class related (teaching, assignments, grades) |
| Sorted/Research | Research projects, papers, grants |
| Sorted/Industry | Industry contacts, partnerships, vendors, conferences |
| Sorted/University-Partners | Other universities, academic collaborations |
| Sorted/Personal | Personal correspondence |
| Sorted/Accounts | Service notifications, password resets, account confirmations |
| Sorted/Newsletters | Subscriptions, digests, mailing lists worth keeping |
| To Delete | Spam, marketing, expired notifications, transient alerts |

All categories except "To Delete" live under the `Sorted/` parent folder/label. In Gmail this creates a nested label hierarchy. In Outlook this creates a `Sorted` folder with subfolders.

---

## Provider Operations

Each account in config.yaml has a `type` field. Use the matching operations:

### Type: `gws`

| Operation | Command |
|-----------|---------|
| List inbox | `GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --query "in:inbox" --max N --format json` |
| List with date filter | `GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --query "in:inbox before:YYYY/MM/DD" --max N --format json` |
| Read message | `GWS_CREDENTIAL_STORE=plaintext gws gmail +read --id MSG_ID --headers --format json` |
| Move to folder | `GWS_CREDENTIAL_STORE=plaintext gws gmail users messages modify --params '{"id":"MSG_ID"}' --json '{"addLabelIds":["LABEL_ID"],"removeLabelIds":["INBOX"]}'` |
| Batch move | `GWS_CREDENTIAL_STORE=plaintext gws gmail users messages batchModify --json '{"ids":["ID1","ID2"],"addLabelIds":["LABEL_ID"],"removeLabelIds":["INBOX"]}'` |

### Type: `ms365`

| Operation | Tool |
|-----------|------|
| List inbox | `mcp__ms365__list-mail-messages` |
| Read message | `mcp__ms365__get-mail-message` with message ID |
| Move to folder | `mcp__ms365__move-mail-message` with message ID and destination folder ID |

When processing an account, look up its type and use the corresponding operations. The folder/label IDs for each account are in `config.yaml` under `accounts[].folder_ids`.

---

## Batch Mode

Process the next batch of emails toward inbox zero.

### Step 1: Select account and fetch batch

Pick the next account to process. Strategy: alternate between enabled accounts, or process whichever has the most remaining.

Use the account's **list inbox** operation to fetch `batch_size` messages.

If there's a bookmark (`last_processed_date` in progress.yaml), use the **list with date filter** operation to resume from where we left off.

If the result is empty, set this account's status to `completed` and try the next account. If all accounts are empty:
> Inbox zero reached! All emails have been processed.

Update status and self-pause if this is a scheduled task (`mcp__nanoclaw__pause_task`).

### Step 2: Classify each email

For each email in the batch:

#### 2a. Check age rules

Load `age_rules` from `config.yaml`. For each email, check if any age rule matches:

- Match by `from_domain` or `from_address`
- If the email is older than `max_age_days`, classify as "To Delete"
- If `condition: "read_or_old"`, also classify as "To Delete" if the email has been read
- Age rules override sender rules — they catch time-expired notifications

Current age rules:
- `adobesign@adobesign.com` → To Delete if read or older than 14 days
- `drive-shares-dm-noreply@google.com` → To Delete if older than 30 days
- `comments-noreply@docs.google.com` → To Delete if older than 30 days

#### 2b. Check sender rules

Load rules from `rules.yaml`. Check in priority order:

1. **from_address** — exact sender match (e.g., `quarantine@messaging.microsoft.com`)
2. **from_domain** — domain match (e.g., `apriva.com`)
3. **subject_contains** — keyword in subject line

If a rule matches:
- Assign the rule's folder
- Set confidence to 1.0, method to "rule"
- Increment the rule's `applied_count`
- No LLM needed

#### 2c. LLM classification (no rule matched)

For unmatched emails, classify using your own judgment. Consider:
- Sender address and domain
- Subject line and body content
- The `taxonomy_context` from config.yaml (descriptions of what each category means)
- The `llm_hints` from rules.yaml (domain-level guidance)
- The existing rules as examples of classification patterns

LLM hints from rules.yaml:
- Non-Clemson `.edu` addresses → likely `University-Partners`
- `g.clemson.edu` / `clemson.edu` → classify by content (could be SI, GC, Classes, Research, Student — read the subject/body to decide)

Produce a classification:
```
folder: "Sorted/GC-Student"
confidence: 0.85
reason: "From g.clemson.edu student address, subject about internship situation"
```

If confidence < `thresholds.llm_confidence`: mark as `needs_review`.

### Step 3: Present or apply

Based on `mode` in config:

#### Interactive mode

Present ALL classifications for approval via `mcp__nanoclaw__send_message`:

```
*Email Archive — Batch #N* (BATCH_SIZE emails, ACCOUNT_ID)

*By rule* (N emails, no LLM cost):
  To Delete: N (quarantine, bitly, miro, etc.)
  Sorted/Industry: N (flexography, bobst)

*By age rule* (N emails):
  To Delete: N (expired google notifications, old adobesign)

*By LLM* (N emails):
  Sorted/GC-Student: N — "Current Internship Situation" from dynastg@g.clemson.edu
  Sorted/SI-Operations: N — "Tour signup request" from ocarr@g.clemson.edu
  Sorted/GC-Curriculum: N — "GC 3800 Registration" from sbaier@g.clemson.edu
  Sorted/University-Partners: N — "Clemson Travel" from ctwomey@calpoly.edu

*Needs your call* (N emails):
  1. "Re: Graduate Student Applications" from santigomez82@gmail.com → Sorted/GC-Student? (0.65)
  2. "Budget meeting" from emweise@clemson.edu → Sorted/GC-Operations? (0.72)

Reply *approve* to apply all, or correct items:
  *1:Sorted/GC-Operations* to override item 1
  *reject* to skip this batch
```

Wait for user response. Parse corrections. Log corrections to `overrides.jsonl`.

#### Supervised mode

Apply rule-based classifications immediately (they're trusted). Present only LLM-classified emails for review. Same format as interactive but without the rule-matched section.

#### Autonomous mode

Apply all classifications. Send a brief summary report. Flag `needs_review` items in the report but still apply the best-guess classification.

### Step 4: Execute moves

After approval (or immediately in autonomous mode):

Group emails by target folder. For each group, use the account's **move** or **batch move** operation with the folder ID from `accounts[].folder_ids[folder_name]`.

### Step 5: Save state

Update `state/progress.yaml`:
- Increment total_processed for this account
- Update last_processed_date to the oldest email in this batch
- Update stats (rule_matched, llm_classified, corrections)

Append batch record to `state/history.jsonl`:
```json
{"batch":12,"timestamp":"2026-04-11T14:30:00Z","account":"gmail","count":25,"rule":19,"llm":6,"needs_review":1,"corrections":0}
```

Update `rules.yaml` if any rule's `applied_count` changed.

### Step 6: Auto-promote rules

Check recent LLM classifications (from history + overrides). If a sender has been classified the same way `thresholds.auto_promote_after` times with zero corrections:

1. Generate a new rule in `rules.yaml`
2. Report to user:
   ```
   New rule auto-generated: *@store.com → To Delete (5 consistent classifications, 0 corrections)
   ```

### Step 7: Progressive taxonomy check

Every 100 emails (check total stats), scan for overcrowded categories:

If any category has 30+ emails and one sender domain accounts for 40%+ of that category:
```
Sorted/GC-Student has 45 emails. Top senders:
  - *@g.clemson.edu: 38 emails
  - *@gmail.com: 7 emails

Suggested subcategories:
  - Sorted/GC-Student/Internships
  - Sorted/GC-Student/Advising

Want to create these? (Existing emails won't be re-filed — applies to future batches.)
```

Only suggest in interactive or supervised mode. If approved, create new folders/labels using the account's provider operations and update config.yaml with the new IDs.

---

## Status Mode

```bash
cat /workspace/group/email-archive/state/progress.yaml
cat /workspace/group/email-archive/rules.yaml
```

Report:

```
*Email Archive Status*

*Accounts:*
  gmail (gws): N processed (status)
  outlook (ms365): N processed (status)

*Classification:*
  Rule-matched: N (N% of total)
  Age-rule matched: N
  LLM-classified: N
  Corrections: N (N% error rate)

*Rules:* N active (N from calibration, N auto-promoted)
*Mode:* interactive / supervised / autonomous

*Top folders:*
  To Delete: N
  Sorted/GC-Student: N
  Sorted/SI-Operations: N
  Sorted/Industry: N
```

---

## Review Mode

```bash
cat /workspace/group/email-archive/state/history.jsonl | tail -5
cat /workspace/group/email-archive/overrides.jsonl | tail -10
```

Show recent batches and any flagged items. If there are pending `needs_review` items that were auto-applied (autonomous mode), list them for the user to confirm or correct.

---

## Recalibrate Mode

Analyze `overrides.jsonl` for patterns:

```bash
cat /workspace/group/email-archive/overrides.jsonl
```

Group corrections by sender domain. If 2+ corrections point the same direction:

```
Correction patterns found:

  *@newsletter.example.com: 3 corrections, all → Sorted/Newsletters
    Currently: no rule (LLM was classifying as To Delete)
    → Create rule?

  john@partner.com: 2 corrections, all → Archive/Work
    Currently: no rule
    → Create rule?
```

Present to user. For each approved rule, add to `rules.yaml`.

Also check for rules that are getting overridden frequently — those may need updating or removal.

---

## Important Constraints

- **NEVER auto-delete emails.** "To Delete" is a staging folder. Only the user deletes.
- **NEVER send emails** on the user's behalf.
- **Preserve email state** — only move/label, never modify content.
- **Save state after each batch** — crashes must not lose progress.
- **Respect container timeout** — if approaching timeout, save state and exit cleanly.
- **Rules first, LLM second** — always check rules before using LLM classification.
- **Idempotent moves** — labeling a message that already has the label is safe. Moving a message already in the target folder is safe. Interrupted batches can be safely re-run.
- **Provider-agnostic logic** — classification, rules, state, and reporting are the same regardless of email provider. Only the fetch/move operations differ per account type.
