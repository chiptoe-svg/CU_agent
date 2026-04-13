<p align="center">
  <img src="assets/nanoclaw-logo.png" alt="NanoClaw FlexAgents" width="400">
</p>

<p align="center">
  A multi-runtime AI assistant that runs agents securely in containers. Choose your agent SDK — Claude, Codex, or Gemini — and customize everything.
</p>

<p align="center">
  <a href="https://nanoclaw.dev">nanoclaw.dev</a>&nbsp; • &nbsp;
  <a href="https://docs.nanoclaw.dev">docs</a>&nbsp; • &nbsp;
  <a href="https://discord.gg/VDdww8qS42"><img src="https://img.shields.io/discord/1470188214710046894?label=Discord&logo=discord&v=2" alt="Discord" valign="middle"></a>
</p>

---

## Why FlexAgents

<!-- DRAFT — narrative for your review/editing -->

[NanoClaw](https://github.com/qwibitai/nanoclaw) is a brilliant piece of software — a personal AI assistant that's small enough to understand, secure by design, and built to be customized. But it's built entirely on the Claude Agent SDK, which means you need an Anthropic subscription, you can't use local models for privacy-sensitive work, and you're locked to one provider.

FlexAgents keeps NanoClaw's philosophy — one process, a handful of files, skills over features — but abstracts the agent SDK into a modular layer. You choose which SDK to install during setup, the same way you choose which messaging channels to add. The base system has no SDK-specific code at all.

Three agent SDKs are supported:

- **Codex (OpenAI)** — ChatGPT subscription or API key. Supports local models via OMLX/Ollama. Open source (Rust).
- **Claude (Anthropic)** — Claude subscription OAuth or API key. Best built-in tools. Agent teams/swarms.
- **Gemini (Google)** — Free tier available (60 req/min). Google ADK (Agent Development Kit) with native sub-agents, session persistence, and A2A protocol support. Open source (Python).

You can run one SDK or all three simultaneously — different groups can use different SDKs and models. Your main chat might use Codex with GPT-5.4, a code review group uses Claude Opus, and a research group uses Gemini Flash on the free tier. Each agent gets its own container with its own SDK, persona, memory, and skills. Switch models instantly with `/model` in Telegram.

### SDK Comparison

<table>
<tr>
  <th>Feature</th>
  <th>Claude</th>
  <th>Codex</th>
  <th>Gemini</th>
</tr>
<tr>
  <td><b>Runtime</b></td>
  <td><code>@anthropic-ai/claude-agent-sdk</code></td>
  <td><code>codex app-server</code> — JSON-RPC over stdio</td>
  <td>Google ADK — Python agent framework</td>
</tr>
<tr>
  <td><b>Agent loop</b></td>
  <td><code>query()</code> — async iterable</td>
  <td><code>initialize</code> → <code>thread/start</code> → <code>turn/start</code> → stream notifications</td>
  <td><code>GeminiChat</code> + <code>Scheduler</code> — event-driven</td>
</tr>
<tr>
  <td><b>Built-in tools</b></td>
  <td>18 tools: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch, Task, Teams</td>
  <td>Shell, apply_patch, web search + 5 MCP file tools (read, write, edit, glob, grep)</td>
  <td>15+ tools: shell, read_file, write_file, replace, glob, grep_search, web_search, web_fetch</td>
</tr>
<tr>
  <td><b>MCP support</b></td>
  <td>Via <code>query()</code> options</td>
  <td>Via <code>config.toml</code> — app-server reads on startup</td>
  <td>Via <code>.gemini/settings.json</code> + extensions</td>
</tr>
<tr>
  <td><b>Skills</b></td>
  <td><code>.claude/skills/</code> on-demand via Skill tool</td>
  <td><code>.codex/skills/</code> native discovery via <code>skills/list</code></td>
  <td>Agent skills via <code>activate_skill</code></td>
</tr>
<tr>
  <td><b>Session persist</b></td>
  <td>JSONL with resume by UUID</td>
  <td>Thread persistence + <code>thread/resume</code> by ID, <code>thread/read</code> for inspection</td>
  <td>Chat recording with checkpoint resume</td>
</tr>
<tr>
  <td><b>Compaction</b></td>
  <td>Native auto-compact with pre-compact hook</td>
  <td>Native <code>thread/compact/start</code> triggered at 40K tokens</td>
  <td>Chat compression with checkpoints</td>
</tr>
<tr>
  <td><b>Thread lifecycle</b></td>
  <td>Resume by session ID</td>
  <td><code>thread/start</code>, <code>thread/resume</code>, <code>thread/fork</code>, <code>thread/rollback</code>, <code>thread/read</code></td>
  <td>Session checkpoints</td>
</tr>
<tr>
  <td><b>Multi-agent</b></td>
  <td>Agent teams — TeamCreate / SendMessage. Shared context.</td>
  <td>Host-level via IPC + scheduled tasks</td>
  <td>Subagent tools + A2A protocol + remote invocation</td>
</tr>
<tr>
  <td><b>Tool approval</b></td>
  <td><code>permissionMode: bypassPermissions</code></td>
  <td><code>approvalPolicy: never</code> + auto-approve notifications</td>
  <td>PolicyEngine with TOML rules</td>
</tr>
<tr>
  <td><b>Auth</b></td>
  <td>OAuth token or API key</td>
  <td>ChatGPT subscription or API key</td>
  <td>API key (free tier: 60 req/min)</td>
</tr>
<tr>
  <td><b>Local models</b></td>
  <td>Not possible</td>
  <td>OMLX, Ollama, LiteLLM via <code>baseUrl</code></td>
  <td>Not confirmed</td>
</tr>
<tr>
  <td><b>Open source</b></td>
  <td>No</td>
  <td>Yes (Rust)</td>
  <td>Yes (TypeScript / Python)</td>
</tr>
<tr>
  <td><b>Models</b></td>
  <td>Opus 4.6, Sonnet 4.6/4.5, Haiku 4.5</td>
  <td>GPT-5.4, 5.4-mini, 5.3-codex, 5.2</td>
  <td>Gemini 2.5 Pro, Flash, Flash Lite</td>
</tr>
</table>

For the full comparison with status badges, see [docs/sdk-comparison.html](https://htmlpreview.github.io/?https://github.com/chiptoe-svg/CUagent/blob/main/docs/sdk-comparison.html).

<!-- END DRAFT -->

## Quick Start

```bash
gh repo fork chiptoe-svg/nanoclaw_flexagents --clone
cd nanoclaw_flexagents
```

Then open your preferred development tool:

| Dev tool | Command | Persona file |
|----------|---------|-------------|
| Claude Code | `claude` | Reads `CLAUDE.md` |
| Codex CLI | `codex` | Reads `AGENTS.md` |
| Gemini CLI | `gemini` | Reads `GEMINI.md` (uses Google ADK in containers) |

Run `/setup` inside the CLI. It handles everything: dependencies, container runtime, agent SDK selection, authentication, channels, and service configuration.

> **Note:** Commands prefixed with `/` (like `/setup`, `/add-agentSDK-codex`) are CLI skills. Type them inside the agent CLI prompt, not in your regular terminal.

<details>
<summary>Without GitHub CLI</summary>

1. Fork [chiptoe-svg/nanoclaw_flexagents](https://github.com/chiptoe-svg/nanoclaw_flexagents) on GitHub
2. `git clone https://github.com/<your-username>/nanoclaw_flexagents.git`
3. `cd nanoclaw_flexagents`
4. Open `claude`, `codex`, or `gemini`
5. Run `/setup`

</details>

## What It Supports

- **Multi-runtime agents** — Choose Claude, Codex, or Gemini as your agent SDK. Install via `/add-agentSDK-codex`, `/add-agentSDK-claude`, or `/add-agentSDK-gemini`. Run one or all three.
- **Per-group model selection** — `/model` switches models instantly. Each group can use a different SDK and model.
- **Custom model endpoints** — Use local models (OMLX, Ollama) or any third-party provider (Together AI, Groq, HuggingFace, self-hosted vLLM). Set up with `/add-custom-models` and `/add-model-endpoint`. Uses Codex SDK as the bridge to any OpenAI-compatible endpoint.
- **Multi-channel messaging** — WhatsApp, Telegram, Discord, Slack, Gmail. Add with skills like `/add-telegram`.
- **Isolated group context** — Each group has its own `AGENT.md` persona, memory, filesystem, and container sandbox.
- **Scheduled tasks** — Recurring jobs with optional pre-check scripts to minimize API usage.
- **Web access** — Search and fetch content from the web.
- **Container isolation** — Agents sandboxed in Docker or Apple Container. Only mounted directories accessible.
- **Credential security** — Claude uses a credential proxy (containers see placeholders). Codex mounts subscription auth. Gemini uses API key injection. Secrets never exposed to agents.
- **Agent teams** — Claude SDK supports multi-agent orchestration via TeamCreate/TeamDelete. Gemini ADK supports native sub-agents (SequentialAgent, ParallelAgent, LoopAgent). All runtimes support specialist delegation via MCP tool.
- **Skills system** — Add capabilities with `/add-*` skills. All SDKs load skills on-demand from their respective directories.
- **Provider plugins** — External services (MS365, Google Workspace, IMAP) described as JSON config files — add or remove a provider without code changes. Token mounts, MCP servers, allowed tools, and agent docs are all declared in the config. Providers are definitions only — run `/add-email-account` to authenticate and activate.
- **Email management** — Register email accounts (`/add-email-account`), calibrate sender rules (`/add-email-archive`), and batch-classify emails toward inbox zero (`/email-archive`). Provider-agnostic — works with any configured email account.

## Usage

Talk to your assistant with the trigger word (default: `@Linda` or whatever you set during `/setup`):

```
@Linda send an overview of the sales pipeline every weekday morning at 9am
@Linda review the git history for the past week each Friday and update the README
@Linda every Monday at 8am, compile news on AI developments and message me a briefing
```

Telegram commands:
```
/model              — view/switch model for this group
/model gpt-5.4     — switch to a specific model
/auth               — view credential status
/ping               — check if the bot is online
/chatid             — get chat registration ID
```

## Architecture

NanoClaw FlexAgents is organized as four layers. The goal is to keep the app shell provider-neutral, push provider-specific behavior into runtime modules, and keep the in-container SDK loops isolated from the host process.

### Layer 1: App Shell

The app shell is the long-running Node.js process. It owns channels, SQLite, scheduling, message routing, per-group queues, and container lifecycle orchestration.

Flow at this layer:

1. A channel receives a message or scheduled task fires.
2. The app shell loads the group config and decides which runtime to use.
3. The selected runtime adapter gets a neutral `AgentRuntimeConfig`.
4. Results stream back through the queue and are sent to the channel.

Core files:
- `src/index.ts` — main orchestrator and message loop
- `src/channels/*` — channel adapters
- `src/task-scheduler.ts` — scheduled task execution
- `src/group-queue.ts` — per-group serialization and process tracking

### Layer 2: Runtime Boundary

This layer is the host-side abstraction over provider SDKs. The shared boundary is intentionally generic:

- `AgentRuntime` exposes `run()`, optional `preflight()`, and optional `capabilities()`
- `ContainerInput` carries neutral fields plus `runtimeOptions`
- provider-specific normalization happens inside the runtime, not in the app shell

This is where each runtime can:

- resolve provider-specific options from group config
- validate auth before launch
- declare capabilities honestly, such as resume support or manual delegation

Core files:
- `src/runtime/types.ts` — neutral runtime interfaces
- `src/runtime/registry.ts` — runtime self-registration
- `src/runtime/claude-runtime.ts` — Claude host adapter
- `src/runtime/codex-runtime.ts` — Codex host adapter
- `src/runtime/gemini-runtime.ts` — Gemini host adapter
- `src/runtime/codex-policy.ts` — Codex-specific option resolution

### Layer 3: Runtime Setup and Container Launch

This layer prepares the container environment without teaching the app shell about a specific provider’s credential format.

Responsibilities:

- choose the runtime-specific home layout
- sync skills into the per-group runtime home
- resolve auth material through provider-neutral auth backends
- load provider definitions from `~/.nanoclaw/providers/`
- mount the group workspace, IPC directory, and runtime home into the container
- selectively mount provider token directories when a container is allowed to use them

The important distinction is:

- the framework knows how to prepare a runtime home and launch a container
- only a runtime or auth backend knows what credentials/options its provider needs
- provider definitions are data-driven JSON files, but credential exposure is still controlled by the host launcher

Codex has one extra wrinkle: it may use an inner sandbox based on user namespaces and bubblewrap inside the container. Because of that, the host launcher applies a small Codex-only container compatibility adjustment when needed. Claude and Gemini rely only on the outer container sandbox and do not receive the same relaxation.

Core files:
- `src/container-runner.ts` — container spawning, mounts, env injection
- `src/runtime-setup.ts` — runtime home preparation
- `src/provider-registry.ts` — provider plugin loader (token mounts, MCP servers, tools)
- `src/auth/types.ts` — neutral auth backend contracts
- `src/auth/backends.ts` — compatibility env/file backends plus future stubs
- `container/providers/` — provider JSON configs (ms365, gws, imap)

### Layer 4: In-Container Agent Runner

Inside the container, a shared agent-runner process dispatches to the correct SDK module. All runtimes share the same container image, IPC protocol, and basic filesystem layout, but each runtime module owns its own SDK semantics.

Flow at this layer:

1. The container agent-runner reads `ContainerInput`.
2. The runtime registry selects the in-container runtime implementation.
3. The container-side provider registry enables only the provider MCP servers, docs, and init hooks whose token files are actually mounted into that container.
4. The runtime module talks to its SDK, streams output, and writes structured results back over stdout/IPC.

Core files:
- `container/agent-runner/src/index.ts` — shared container entrypoint
- `container/agent-runner/src/runtime-registry.ts` — in-container dispatch
- `container/agent-runner/src/provider-registry.ts` — container-side provider discovery
- `container/agent-runner/src/shared.ts` — shared IPC/output plumbing
- `container/agent-runner/src/runtimes/claude.ts` — Claude SDK loop
- `container/agent-runner/src/runtimes/codex.ts` — Codex SDK loop
- `container/agent-runner/src/runtimes/gemini.ts` — Gemini ADK loop

### Personas, Skills, and Group Isolation

Each group is isolated by folder, persona, memory, IPC namespace, and runtime home.

- `groups/*/AGENT.md` — runtime-agnostic persona
- `groups/*/memory/` — persistent memory and notes
- `container/skills/` — shared skill source copied into runtime-specific homes

Provider integrations follow the same pattern:

- `container/providers/*.json` — shipped provider definitions (ms365, gws, imap)
- `~/.nanoclaw/providers/*.json` — active host-side provider configs (copied from defaults on first startup)
- providers are definitions only — `/add-email-account` handles authentication and activation
- provider tokens stay on the host and are mounted into containers only when allowed by the launcher
- the in-container provider registry turns mounted credentials into MCP servers, init hooks, allowed tools, and appended agent docs

At startup or launch time, the system assembles the provider-specific instruction file a runtime expects:

- Codex reads `AGENTS.md`
- Claude reads `CLAUDE.md`
- Gemini reads `GEMINI.md`

### End-to-End Flow

```text
Channel / Scheduler
  -> App Shell (routing, queueing, state)
  -> AgentRuntime (provider-neutral host adapter)
  -> Runtime Setup + Container Runner
  -> Container Agent Runner
  -> Provider SDK loop
  -> Structured output / IPC
  -> App Shell
  -> Channel response
```

## Philosophy

**Small enough to understand.** One process, a few source files, no microservices.

**Secure by isolation.** Agents run in Linux containers and can only see what's explicitly mounted.

**Runtime-agnostic at the core.** The app shell stays provider-neutral, while SDK-specific behavior lives in runtime and container modules.

**Provider-driven integrations.** External services are described as provider configs, but the host process still controls when credentials are mounted and which containers can see them.

**Built for the individual.** Fork it, customize it, make it yours. The codebase is small enough to modify safely.

**Skills over features.** Don't add features to core — add skills that transform your fork.

**AI-native.** No installation wizard; your agent CLI guides setup. No dashboards; ask the agent.

## Customizing

Tell your development tool what you want:

- "Change the trigger word to @Bob"
- "Add a custom greeting when I say good morning"
- "Store conversation summaries weekly"

Or run `/customize` for guided changes.

## Contributing

**Don't add features. Add skills.**

Fork, make changes on a branch, open a PR. We'll create a `skill/*` branch from your PR that other users can merge into their fork.

### RFS (Request for Skills)

Skills we'd like to see:

- `/add-signal` — Signal as a channel
- `/add-agentSDK-*` — Additional agent SDK adapters
- IMAP MCP server — Enable IMAP email accounts (provider JSON ready at `container/providers/imap.json`)
- `/add-email-triage` — Daily inbox management and action item tracking

## Requirements

- macOS, Linux, or Windows (via WSL2)
- Node.js 20+
- One of: [Claude Code](https://claude.ai/download), [Codex CLI](https://www.npmjs.com/package/@openai/codex), or [Gemini CLI](https://www.npmjs.com/package/@google/gemini-cli) (for development; containers use Google ADK)
- [Docker](https://docker.com/products/docker-desktop) or [Apple Container](https://github.com/apple/container) (macOS)

## FAQ

**Which agent SDK should I choose?**

- **Codex** if you have a ChatGPT subscription and want local model support
- **Claude** if you have a Claude subscription and want the best built-in tools
- **Gemini** if you want a free tier or prefer Google's ecosystem
- You can install multiple SDKs and switch per-group

**Can I use local or third-party models?**

Yes. Run `/add-custom-models` to enable, then `/add-model-endpoint` to connect providers — OMLX (Apple Silicon optimized), Ollama, Together AI, Groq, HuggingFace, or any OpenAI-compatible URL. Uses Codex SDK as the bridge (installed automatically if needed). Switch between cloud and custom models with `/model` in Telegram. Works alongside any primary SDK — your main groups stay on Claude or Gemini while specific groups use local models.

**Is this secure?**

Agents run in containers, not behind application-level permission checks. They can only access explicitly mounted directories. Credentials are injected via proxy (Claude) or mounted auth files (Codex) — raw API keys never enter the container.

**Can I use a different development tool?**

Yes. The project has persona files for all three: `CLAUDE.md` (Claude Code), `AGENTS.md` (Codex CLI), `GEMINI.md` (Gemini CLI). All generated from a shared `PROJECT.md`. Run `bash scripts/regenerate-persona.sh` after editing project context.

**How do I add a new external service (email provider, API, etc.)?**

Add a provider JSON file under `container/providers/`. On startup, NanoClaw copies shipped provider configs into `~/.nanoclaw/providers/` if they are missing. Each provider file declares token paths, MCP server config, allowed tools, init hooks, and agent docs — but the provider is inactive until you authenticate via `/add-email-account` or `npm run provider-login`. See `container/providers/ms365.json` as an example.

**How do I debug issues?**

Ask your development tool. "Why isn't the scheduler running?" "What's in the recent logs?" Or run `/debug` for guided troubleshooting.

## Community

Questions? Ideas? [Join the Discord](https://discord.gg/VDdww8qS42).

## License

MIT
