---
name: add-email-account
description: Register an email account for use by email skills (archive, triage, etc.). Supports Google Workspace (Gmail), Microsoft 365 (Outlook/Exchange), and IMAP. Handles authentication and stores account config. Run /add-email-account to add or manage accounts.
---

# Add Email Account

Register email accounts that other skills (`/email-archive`, `/email-triage`, etc.) can use. Each account has a provider type, credentials, and a verified connection.

## Detect Group

```bash
MAIN_FOLDER=$(sqlite3 store/nanoclaw.db "SELECT folder FROM registered_groups WHERE is_main = 1 LIMIT 1;" 2>/dev/null)
echo "MAIN_FOLDER=${MAIN_FOLDER:-unknown}"
```

If empty, ask:
> Which group folder should I configure email accounts for?

## Check Existing Accounts

```bash
test -f "groups/${MAIN_FOLDER}/email-accounts.yaml" && cat "groups/${MAIN_FOLDER}/email-accounts.yaml" || echo "NO_ACCOUNTS"
```

If accounts exist, show them:

> Registered email accounts:
> 1. gmail — Google Workspace (tonkin@g.clemson.edu) ✓
> 2. outlook — Microsoft 365 (tonkin@clemson.edu) ✓
>
> What would you like to do?
> - **Add** another account
> - **Remove** an existing account
> - **Test** an existing account
> - **Done**

If no accounts exist, proceed to add one.

## Add Account

### Step 1: Choose provider

> What type of email account?
>
> 1. **Google Workspace / Gmail** — uses `gws` CLI, OAuth authentication
> 2. **Microsoft 365 / Outlook / Exchange** — uses MS365 MCP, OAuth authentication
> 3. **IMAP** — any standard mail server (iCloud, university, self-hosted)

Use `AskUserQuestion` to get the choice.

### Step 2: Authenticate (provider-specific)

#### Type: `gws` (Google Workspace / Gmail)

Check if already authenticated:
```bash
GWS_CREDENTIAL_STORE=plaintext gws gmail users getProfile --format json 2>&1
```

If this returns a profile with an email address, authentication is already set up. Extract the email address.

If not authenticated:
> Google Workspace CLI needs authentication. Please run:
> ```
> ! gws auth login
> ```
> This will open a browser for OAuth consent. Once complete, tell me to continue.

After auth, verify:
```bash
GWS_CREDENTIAL_STORE=plaintext gws gmail users getProfile --format json
```

Extract the email address from the profile response.

Check that gws tokens are in the right place for container mounting:
```bash
test -d ~/.nanoclaw/.gws-tokens && echo "TOKENS_OK" || echo "TOKENS_MISSING"
```

If missing, copy credentials:
```bash
mkdir -p ~/.nanoclaw/.gws-tokens
cp ~/.config/gws/credentials.json ~/.nanoclaw/.gws-tokens/ 2>/dev/null
cp ~/.config/gws/client_secret.json ~/.nanoclaw/.gws-tokens/ 2>/dev/null
```

#### Type: `ms365` (Microsoft 365 / Outlook / Exchange)

Check if already authenticated:
```bash
test -f ~/.nanoclaw/.ms365-tokens/.token-cache.json && echo "MS365_OK" || echo "MS365_NOT_FOUND"
```

If not authenticated:
> Microsoft 365 needs authentication. Please run:
> ```
> ! npm run ms365-login
> ```
> This will open a browser for OAuth consent. Once complete, tell me to continue.

After auth, verify by checking the token file exists and ask the user for their email address:

> What email address is this account for? (e.g., user@clemson.edu)

#### Type: `imap` (Standard IMAP)

Collect connection details via `AskUserQuestion`:

> IMAP server details:
> 1. **Email address** (e.g., user@icloud.com)
> 2. **IMAP server** (e.g., imap.mail.me.com)
> 3. **Port** (default: 993)
> 4. **Security** (SSL/TLS — default, or STARTTLS)
> 5. **Username** (usually the email address)
> 6. **App password** (NOT your regular password — generate one in your email provider's security settings)

For common providers, auto-fill server details:
- iCloud: `imap.mail.me.com:993`
- Yahoo: `imap.mail.yahoo.com:993`
- Fastmail: `imap.fastmail.com:993`
- ProtonMail: requires ProtonMail Bridge (`127.0.0.1:1143`)

Store credentials securely:
```bash
mkdir -p ~/.nanoclaw/.imap-tokens
cat > ~/.nanoclaw/.imap-tokens/${ACCOUNT_ID}.json << 'EOF'
{
  "host": "imap.mail.me.com",
  "port": 993,
  "security": "ssl",
  "username": "user@icloud.com",
  "password": "xxxx-xxxx-xxxx-xxxx"
}
EOF
chmod 600 ~/.nanoclaw/.imap-tokens/${ACCOUNT_ID}.json
```

**Note:** IMAP support requires an IMAP MCP server or CLI tool in the container. If one is not yet available, tell the user:
> IMAP accounts are registered but not yet operational — an IMAP tool needs to be added to the container. The `gws` and `ms365` providers are ready to use now.

### Step 3: Choose an account ID

Suggest a short ID based on the provider and email:
> I'll call this account `gmail` (or `outlook`, `icloud`, etc.). Want a different name?

The ID must be unique, lowercase, alphanumeric + hyphens.

### Step 4: Verify connection

Test that the account can list inbox messages.

**gws:**
```bash
GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --max 3 --format json
```

**ms365:**
```
Call mcp__ms365__list-mail-messages to get 3 recent messages.
```

**imap:** Skip verification for now if no IMAP tool is available.

Report:
> ✓ Connected to tonkin@g.clemson.edu (Gmail)
> Found 3 recent messages — account is working.

### Step 5: Save account

Write or update `groups/${MAIN_FOLDER}/email-accounts.yaml`:

```yaml
# Registered email accounts
# Used by: /email-archive, /email-triage, and other email skills
# Provider types: gws, ms365, imap

accounts:
  - id: gmail
    type: gws
    address: tonkin@g.clemson.edu
    enabled: true
    added: "2026-04-11"

  - id: outlook
    type: ms365
    address: tonkin@clemson.edu
    enabled: true
    added: "2026-04-11"
```

If the file already exists, append the new account (don't overwrite existing entries).

### Step 6: Done

> Account `ACCOUNT_ID` registered (TYPE — ADDRESS).
>
> This account is now available to email skills:
> - `/add-email-archive` — bulk sort old emails toward inbox zero
> - `/email-triage` — daily inbox management (coming soon)
>
> Add another account? Or type **done** to finish.

If the user wants to add another, loop back to Step 1.

---

## Remove Account

If the user chose "Remove":

> Which account to remove?

List accounts with numbers. On selection, remove the entry from `email-accounts.yaml`. Don't delete credentials — just disable the account.

---

## Test Account

If the user chose "Test":

> Which account to test?

List accounts. Run the verification step (Step 4) for the selected account. Report success or failure.

---

## Provider Operations Reference

This is the canonical reference for email operations by provider type. Other skills (`/email-archive`, `/email-triage`) should reference this.

### Type: `gws` (Google Workspace)

| Operation | Command |
|-----------|---------|
| List inbox | `GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --query "in:inbox" --max N --format json` |
| List with date filter | `GWS_CREDENTIAL_STORE=plaintext gws gmail +triage --query "in:inbox before:YYYY/MM/DD" --max N --format json` |
| Read message | `GWS_CREDENTIAL_STORE=plaintext gws gmail +read --id MSG_ID --headers --format json` |
| List labels | `GWS_CREDENTIAL_STORE=plaintext gws gmail users labels list --format json` |
| Create label | `GWS_CREDENTIAL_STORE=plaintext gws gmail users labels create --json '{"name":"NAME","labelListVisibility":"labelShow","messageListVisibility":"show"}' --format json` |
| Move (label + archive) | `GWS_CREDENTIAL_STORE=plaintext gws gmail users messages modify --params '{"id":"MSG_ID"}' --json '{"addLabelIds":["LABEL_ID"],"removeLabelIds":["INBOX"]}'` |
| Batch move | `GWS_CREDENTIAL_STORE=plaintext gws gmail users messages batchModify --json '{"ids":["ID1","ID2"],"addLabelIds":["LABEL_ID"],"removeLabelIds":["INBOX"]}'` |

### Type: `ms365` (Microsoft 365)

| Operation | Tool |
|-----------|------|
| List inbox | `mcp__ms365__list-mail-messages` |
| Read message | `mcp__ms365__get-mail-message` with message ID |
| List folders | `mcp__ms365__list-mail-folders` |
| Create folder | `mcp__ms365__create-mail-folder` (parent before child for nested) |
| Move message | `mcp__ms365__move-mail-message` with message ID and destination folder ID |

### Type: `imap` (Standard IMAP)

| Operation | Status |
|-----------|--------|
| All operations | Pending — requires IMAP MCP server or CLI tool in container |

To add IMAP support: install an IMAP MCP server, register it in `container/agent-runner/src/shared.ts`, add tool names to `allowedTools`, and fill in this table.

### Adding a new provider

1. Add a type entry to this reference table
2. Add detection + auth flow in Step 2 above
3. Update the container skill's provider dispatch
4. No changes needed to classification, rules, or state logic in consuming skills
