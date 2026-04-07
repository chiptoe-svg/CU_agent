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
- **Gemini (Google)** — Free tier available (60 req/min). 15+ built-in tools. Open source (TypeScript).

You can run one SDK or all three simultaneously — different groups can use different SDKs and models. Your main chat might use Codex with GPT-5.4, a code review group uses Claude Opus, and a research group uses Gemini Flash on the free tier. Each agent gets its own container with its own SDK, persona, memory, and skills. Switch models instantly with `/model` in Telegram.

For the detailed feature comparison, see [docs/sdk-comparison.html](docs/sdk-comparison.html).

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
| Gemini CLI | `gemini` | Reads `GEMINI.md` |

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
- **Agent teams** — Claude SDK supports multi-agent orchestration via TeamCreate/TeamDelete (Codex/Gemini: host-level via IPC).
- **Skills system** — Add capabilities with `/add-*` skills. Both SDKs load skills on-demand from their respective directories.

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

```
Channels → SQLite → Polling loop → AgentRuntime → Container (SDK agent loop) → Response
                                        ↓
                              SDK Registry selects:
                              • ClaudeRuntime  → Claude Agent SDK query()
                              • CodexRuntime   → Codex SDK thread.runStreamed()
                              • GeminiRuntime  → Gemini CLI subprocess
```

Single Node.js process. Agent SDKs self-register via a modular registry (same pattern as channels). All SDKs share one container image — the agent-runner detects the runtime from config and dispatches to the appropriate SDK module. Per-group message queue with concurrency control. IPC via filesystem.

Key files:
- `src/index.ts` — Orchestrator: state, message loop, runtime invocation
- `src/runtime/registry.ts` — SDK self-registration registry
- `src/runtime/claude-runtime.ts` — Claude adapter
- `src/runtime/codex-runtime.ts` — Codex adapter
- `src/runtime/gemini-runtime.ts` — Gemini adapter
- `src/container-runner.ts` — Container spawning, mounts, credential injection
- `src/channels/registry.ts` — Channel registry
- `container/agent-runner/src/runtimes/` — SDK-specific agent loops
- `groups/*/AGENT.md` — Per-group agent persona (runtime-agnostic)

## Philosophy

**Small enough to understand.** One process, a few source files, no microservices.

**Secure by isolation.** Agents run in Linux containers and can only see what's explicitly mounted.

**Runtime-agnostic.** The base system has no SDK-specific code. Agent SDKs are installed as modular skills.

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

## Requirements

- macOS, Linux, or Windows (via WSL2)
- Node.js 20+
- One of: [Claude Code](https://claude.ai/download), [Codex CLI](https://www.npmjs.com/package/@openai/codex), or [Gemini CLI](https://www.npmjs.com/package/@google/gemini-cli)
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

**How do I debug issues?**

Ask your development tool. "Why isn't the scheduler running?" "What's in the recent logs?" Or run `/debug` for guided troubleshooting.

## Community

Questions? Ideas? [Join the Discord](https://discord.gg/VDdww8qS42).

## License

MIT
