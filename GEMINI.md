# NanoClaw FlexAgents

Multi-runtime personal assistant built on NanoClaw. Supports Claude, Codex (OpenAI), and Gemini agent SDKs.

## Architecture

Four-layer system:
1. **App Shell** — channels, state, scheduling, IPC (`src/index.ts`)
2. **Runtime Boundary** — provider-neutral host adapters with `runtimeOptions` (`src/runtime/`)
3. **Runtime Setup + Container Launch** — credential resolution, provider mounts, container spawning
4. **In-Container Agent Runner** — SDK-specific agent loops with provider-driven MCP/tools/docs

All SDKs run inside the same container image. The agent-runner detects the runtime from `ContainerInput.runtime` and uses the appropriate SDK. SDKs self-register via a registry pattern (same as channels).

External services (MS365, Google Workspace, IMAP) are configured as provider JSON files — no code changes needed to add or remove a provider. Provider tokens are only mounted for authorized groups.

## Key Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Orchestrator: state, message loop, runtime invocation |
| `src/runtime/types.ts` | AgentRuntime, ContainerManager interfaces |
| `src/runtime/registry.ts` | SDK self-registration registry |
| `src/runtime/claude-runtime.ts` | Claude host adapter |
| `src/runtime/codex-runtime.ts` | Codex host adapter |
| `src/runtime/codex-policy.ts` | Codex-specific option resolution |
| `src/runtime/gemini-runtime.ts` | Gemini host adapter |
| `src/runtime/container-manager.ts` | Container lifecycle management |
| `src/container-runner.ts` | Container spawning, mounts, credential injection |
| `src/runtime-setup.ts` | Runtime home preparation, skill sync |
| `src/provider-registry.ts` | Host-side provider plugin loader (token mounts, startup copy) |
| `src/auth/types.ts` | Neutral auth backend contracts |
| `src/auth/backends.ts` | Compatibility env/file auth backends |
| `src/channels/registry.ts` | Channel registry (self-registration at startup) |
| `src/ipc.ts` | IPC watcher and task processing |
| `src/config.ts` | Config: runtime, model, trigger, paths, intervals |
| `src/credential-proxy.ts` | Anthropic credential proxy (Claude runtime) |
| `src/task-scheduler.ts` | Runs scheduled tasks via AgentRuntime |
| `container/providers/` | Provider JSON configs (ms365, gws, gws-mcp, imap) |
| `scripts/gws-mcp-login.sh` | workspace-mcp OAuth login wrapper (invoked via provider-login) |
| `container/agent-runner/src/index.ts` | In-container shared agent loop |
| `container/agent-runner/src/runtime-registry.ts` | Container-side SDK dispatch |
| `container/agent-runner/src/provider-registry.ts` | Container-side provider discovery (MCP, tools, init hooks, docs) |
| `container/agent-runner/src/shared.ts` | Shared container plumbing (IO, IPC, MessageStream) |
| `container/agent-runner/src/runtimes/claude.ts` | Claude SDK query loop |
| `container/agent-runner/src/runtimes/codex.ts` | Codex SDK query loop |
| `container/agent-runner/src/runtimes/gemini.ts` | Gemini ADK runtime |
| `container/agent-runner/src/providers/gws-init.ts` | GWS credential init hook |
| `container/agent-runner/adk/nanoclaw_agent/` | ADK agent definition (Python) |
| `container/agent-runner/src/specialist-runner.ts` | Specialist subagent dispatch |
| `container/agent-runner/src/ipc-mcp-stdio.ts` | MCP server for NanoClaw IPC tools |
| `container/skills/` | Skills loaded inside agent containers |
| `groups/{name}/AGENT.md` | Per-group agent persona (runtime-agnostic) |
| `groups/global/AGENT.md` | Global persona shared across all groups |
| `groups/{name}/memory/` | Persistent memory (user profile, knowledge) |

## Runtime Configuration

Default runtime and model set in `.env`:
```
DEFAULT_RUNTIME=codex
OPENAI_MODEL=gpt-5.4-mini
```

Per-group override via `containerConfig` in the database:
```sql
UPDATE registered_groups SET container_config = '{"runtime":"claude","model":"claude-sonnet-4-6"}' WHERE jid = '...';
```

Telegram commands:
- `/model` — view/switch model for this group
- `/auth` — view/switch auth mode
- `/ping` — bot status
- `/chatid` — get chat registration ID

## Credentials

**Codex (OpenAI):** Subscription auth via `codex auth login`. Credentials in `~/.codex/auth.json` synced to containers. Falls back to `OPENAI_API_KEY` in `.env`.

**Claude:** OAuth token via `claude setup-token` stored in `.env` as `CLAUDE_CODE_OAUTH_TOKEN`. Credential proxy on port 3001 injects into containers.

**Gemini:** API key from https://aistudio.google.com/apikey stored as `GEMINI_API_KEY` in `.env`. Free tier: 60 req/min.

## Providers

External services are configured as JSON files in `container/providers/`. On startup, defaults are copied to `~/.nanoclaw/providers/`. Each provider declares:
- Token paths (host and container mount points). `requiredFile` may be an exact name or a glob (`*.json`).
- MCP server config (or null for CLI-based providers like `gws`)
- Allowed tools (e.g., `mcp__ms365__*`)
- Init hooks (e.g., GWS credential setup)
- `containerEnv` — host env var names to forward from `.env` into the container (for BYO OAuth client credentials, API keys, etc.). In `mcp.env`, reference them with `${env.VAR_NAME}`.
- Agent docs (injected into AGENT.md at runtime)
- Auth flow (login command for `npm run provider-login`)

Shipped provider configs: `ms365` (Outlook/Calendar/Tasks), `gws` (Gmail/Drive/Calendar/Sheets/Docs via CLI), `gws_mcp` (same services via `workspace-mcp`, BYO OAuth), `imap` (placeholder). These are definitions only — run `/add-email-account` or `npm run provider-login <id>` to authenticate and activate a provider.

Provider tokens are only mounted for authorized groups (main group by default). The container-side provider registry only enables MCP servers whose token files are actually present.

## Agent Persona (AGENT.md)

`AGENT.md` is the canonical persona file. It's runtime-agnostic.

Inside the container, the agent-runner assembles the final instructions:
- **Codex:** concatenates `global/AGENT.md` + `group/AGENT.md` → writes `AGENTS.md`
- **Claude:** copies `AGENT.md` → `CLAUDE.md` for SDK discovery, injects global via system prompt
- **Gemini (ADK):** reads `AGENT.md` directly, parses specialist sub-agents from `## Specialists` section

## Skills

Container skills in `container/skills/` are synced to `.claude/skills/`, `.codex/skills/`, and `.gemini/` per group. Same SKILL.md format works with all SDKs.

## Development

```bash
npm run dev          # Run with hot reload
npm run build        # Compile TypeScript
./container/build.sh # Rebuild agent container
```

Service management (macOS):
```bash
launchctl load ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl kickstart -k gui/$(id -u)/com.nanoclaw  # restart
```

## Container Build Cache

The container buildkit caches aggressively. `--no-cache` alone does NOT invalidate COPY steps. To force a truly clean rebuild, prune the builder then re-run `./container/build.sh`.

---

# Gemini CLI / ADK — Developer Guide

## Response style
- Be concise and direct
- Lead with the answer, not the reasoning
- Use structured output when presenting options or comparisons

## Tool usage
- Use built-in tools: `read_file`, `write_file`, `replace`, `glob`, `grep_search`
- Use `run_shell_command` for system commands and git operations
- Use `google_web_search` for current information

## File reading
- Use `read_file` with line range parameters for large files
- Use `grep_search` for content search across files

## Memory
- Use `save_memory` tool to persist important context
- Check saved memories at session start for project context

## Extensions
- MCP servers configured in `.gemini/settings.json`
- Extensions in `.gemini/extensions/`
