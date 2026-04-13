# NanoClaw FlexAgents Specification

Multi-runtime personal assistant based on upstream NanoClaw. The host app manages channels, persistence, scheduling, and security boundaries; containers run the selected agent runtime for each group.

---

## Table of Contents

1. Architecture
2. Core Flow
3. Runtime Layer
4. Channel Layer
5. Storage and State
6. Scheduling and IPC
7. Container Model
8. Credentials and Security
9. Folder Structure
10. Current Implementation Notes

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          HOST (Node.js)                              │
├──────────────────────────────────────────────────────────────────────┤
│ Channels ──▶ SQLite ──▶ Message Loop ──▶ GroupQueue ──▶ Runtime      │
│    ▲            ▲              │                │           Adapter   │
│    │            │              │                │              │      │
│    │            │              └──── Scheduler ─┘              │      │
│    │            │                                               ▼      │
│    │            └────────────── IPC Watcher ◀────────── ContainerMgr   │
│    │                                                           │       │
│    └──────────────────────────── outbound routing ──────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       CONTAINER (per active group)                    │
├──────────────────────────────────────────────────────────────────────┤
│ Shared agent-runner                                                   │
│   ├─ Claude runtime  -> Claude Agent SDK                              │
│   ├─ Codex runtime   -> codex app-server (JSON-RPC over stdio)        │
│   └─ Gemini runtime  -> Google ADK (Python sidecar)                   │
│                                                                      │
│ Mounted group workspace + per-group runtime home + IPC + extras      │
│ MCP bridge exposes NanoClaw tools back to the host via filesystem    │
└──────────────────────────────────────────────────────────────────────┘
```

### Design Summary

- One host process coordinates everything.
- Channels only ingest and send messages; the host decides when an agent runs.
- Each registered group has isolated filesystem state and a per-group session.
- Runtime selection is per group via `containerConfig.runtime`.
- Containers are the trust boundary; only mounted paths are visible to agents.

---

## Core Flow

### Inbound message flow

1. A channel adapter receives a platform event.
2. The adapter stores chat metadata and, for registered groups, stores the message in SQLite.
3. The host polling loop in `src/index.ts` reads new messages from the database.
4. Messages are grouped by chat and handed to `GroupQueue`.
5. If a container for that group is already running, the message is piped into the active session via IPC.
6. Otherwise the host creates a runtime adapter and starts a new container session.

### Agent execution flow

1. `src/index.ts` resolves the runtime for the group.
2. The runtime adapter delegates to `DefaultContainerManager`.
3. `src/container-runner.ts` prepares mounts, credentials, image selection, and IPC.
4. `container/agent-runner/src/index.ts` dispatches to the container-side runtime handler.
5. The selected runtime emits streamed results wrapped in sentinel markers.
6. The host forwards visible output to the owning channel and updates session state.

### Idle and follow-up flow

- Active containers stay alive temporarily after output.
- Follow-up messages are written into `data/ipc/<group>/input/`.
- The agent-runner drains IPC messages and continues the same session.
- A `_close` sentinel ends the session gracefully.

---

## Runtime Layer

The major fork-specific change from upstream `qwibitai/nanoclaw` is the runtime abstraction in `src/runtime/`.

### Host-side runtime registry

Files:

- `src/runtime/index.ts`
- `src/runtime/registry.ts`
- `src/runtime/types.ts`
- `src/runtime/claude-runtime.ts`
- `src/runtime/codex-runtime.ts`
- `src/runtime/gemini-runtime.ts`

The host imports all runtime adapters at startup. Each runtime self-registers the same way channels do. `src/index.ts` chooses the runtime using:

- `group.containerConfig?.runtime`
- or `DEFAULT_RUNTIME` from `.env`

### Runtime responsibilities

Each runtime adapter:

- identifies itself with `id`
- starts a container session through the shared container manager
- translates streamed container output into neutral `AgentEvent` values
- knows when an error means the saved session should be cleared

The host does not import SDK-specific types directly; it only deals with the neutral runtime interfaces from `src/runtime/types.ts`.

### Container-side runtime registry

Files:

- `container/agent-runner/src/index.ts`
- `container/agent-runner/src/runtime-registry.ts`
- `container/agent-runner/src/runtimes/claude.ts`
- `container/agent-runner/src/runtimes/codex.ts`
- `container/agent-runner/src/runtimes/gemini.ts`
- `container/agent-runner/src/shared.ts`

The shared runner:

- reads `ContainerInput` from stdin
- resolves the runtime handler
- sets up prompt/IPC/script handling
- loops until the session is closed

#### Claude runtime

- Uses `@anthropic-ai/claude-agent-sdk`
- Synthesizes `CLAUDE.md` from `AGENT.md` when needed
- Loads global instructions as appended system prompt
- Registers NanoClaw MCP tools via stdio
- Supports streamed results and conversation archiving on compaction

#### Codex runtime

- Uses `codex app-server` over JSON-RPC stdio
- Synthesizes `AGENTS.md` from global and group agent files, then passes it as `baseInstructions`
- Writes NanoClaw MCP config into `.codex/config.toml`
- Archives conversations with tool-call context
- Supports custom `baseUrl` for OpenAI-compatible endpoints
- Supports worker-thread style delegation through Codex-native tools rather than a framework-owned shared-session abstraction

#### Gemini runtime

- Uses Google ADK (Agent Development Kit) as a Python FastAPI sidecar
- ADK agent reads `AGENT.md` directly and parses specialist sub-agents from `## Specialists` section
- MCP tools configured via `McpToolset` with stdio connection to the NanoClaw MCP server
- Sessions persist in SQLite (`/workspace/group/.adk-sessions.db`) for conversation continuity
- Supports native sub-agents (SequentialAgent, ParallelAgent, LoopAgent) and A2A protocol
- API key injected via `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variables

---

## Channel Layer

Files:

- `src/channels/index.ts`
- `src/channels/registry.ts`
- `src/channels/telegram.ts`
- `src/types.ts`

The channel system still uses the upstream self-registration pattern, but this fork currently wires in Telegram.

### Channel responsibilities

- connect to the external platform
- convert platform events into `NewMessage`
- store chat metadata for discovery
- send outbound text
- optionally expose platform-specific UX like typing indicators

### Telegram specifics

`src/channels/telegram.ts` is both the current messaging adapter and the operational control surface.

It handles:

- normal message ingestion
- reply context capture
- media download into group attachments
- `/chatid`
- `/ping`
- `/auth`
- `/model`

That means Telegram is not just an input/output channel here; it also exposes runtime-aware management for auth mode and per-group model selection.

---

## Storage and State

Primary file:

- `src/db.ts`

SQLite is the persistent coordination layer. The host intentionally polls the database rather than relying on in-memory message delivery.

### Main tables

- `chats`: known chat metadata and last activity
- `messages`: stored inbound and bot messages
- `registered_groups`: enabled groups plus container config
- `sessions`: saved runtime session IDs keyed by group folder
- `scheduled_tasks`: recurring and one-shot tasks
- `task_run_logs`: execution history
- `router_state`: polling cursors

### Important state concepts

- `last_timestamp`: newest message seen by the global poller
- `last_agent_timestamp`: newest message consumed by each group agent
- `sessions[groupFolder]`: persisted runtime session/thread/conversation handle

### Group registration

Registered groups map a chat JID to:

- display name
- folder
- trigger
- trigger requirement
- main-group status
- optional `containerConfig`

`containerConfig` is where runtime selection and common compatibility settings live:

- `runtime`
- `model`
- `baseUrl`
- `timeout`
- `additionalMounts`

At the runtime boundary, runtimes translate provider-specific fields into neutral `runtimeOptions` before crossing into the shared container interface.

---

## Scheduling and IPC

Files:

- `src/task-scheduler.ts`
- `src/ipc.ts`
- `container/agent-runner/src/ipc-mcp-stdio.ts`

### Scheduler

The scheduler runs on the host and checks for due tasks every minute.

Task execution:

1. Load due tasks from SQLite.
2. Resolve the owning group.
3. Snapshot task metadata to the group IPC directory.
4. Create the correct runtime for that group.
5. Run the task in the group context or isolated context.
6. Persist result, next run, and log entry.

This fork adds task scripts (`scheduled_tasks.script`), which can decide whether a task should wake the agent.

### IPC model

IPC is filesystem-based and namespaced per group under `data/ipc/<group>/`.

Subdirectories:

- `messages/`
- `tasks/`
- `input/`

Containers use MCP tools to write structured JSON files. The host watcher authorizes and applies those actions.

Supported IPC actions include:

- send a message
- schedule, pause, resume, cancel, or update a task
- refresh available groups
- register a group from the main chat

Authorization is enforced by directory ownership:

- main group may act broadly
- non-main groups may only operate on their own group resources

---

## Container Model

Files:

- `src/container-runner.ts`
- `src/runtime/container-manager.ts`
- `src/container-runtime.ts`
- `src/mount-security.ts`

### Container images

`getContainerImage(runtime)` chooses the image. All runtimes can share one image, but the config supports runtime-specific images.

### Standard mounts

Depending on group type and runtime, the container may receive:

- `groups/<group>` -> `/workspace/group`
- `groups/global` -> `/workspace/global` for non-main groups
- project root read-only for the main group
- `store/` writable for the main group
- per-group runtime home:
  - `.claude`
  - `.codex`
  - `.gemini`
- `data/ipc/<group>` -> `/workspace/ipc`
- validated additional mounts -> `/workspace/extra/*`
- per-group writable copy of `container/agent-runner/src` -> `/app/src`

### Queueing and concurrency

`src/group-queue.ts` ensures:

- one active container per group
- bounded global concurrency
- tasks are prioritized over queued message work
- retries use exponential backoff

---

## Credentials and Security

### Credential handling

Files:

- `src/credential-proxy.ts`
- `src/auth-switch.ts`
- `src/container-runner.ts`
- `src/mount-security.ts`

#### Claude

- Containers talk to a host-side credential proxy.
- The proxy injects either API-key or OAuth auth.
- Real Claude credentials never need to be visible inside the container.

#### Codex

- Prefers mounted per-group `.codex` auth copied from host auth state.
- Falls back to `OPENAI_API_KEY` if subscription auth is unavailable.
- Supports custom OpenAI-compatible endpoints via `baseUrl`.

#### Gemini

- Injects `GEMINI_API_KEY` or `GOOGLE_API_KEY` from `.env`.

### Mount security

Additional mounts are validated against an allowlist stored outside the repo:

- `~/.config/nanoclaw/mount-allowlist.json`

This prevents agent-controlled code from loosening mount policy.

### Runtime isolation

Security still relies primarily on container isolation:

- agents only see mounted paths
- IPC is group-scoped
- the main group has elevated access, other groups do not

---

## Folder Structure

```
CU_agent/
├── src/
│   ├── index.ts
│   ├── db.ts
│   ├── router.ts
│   ├── ipc.ts
│   ├── task-scheduler.ts
│   ├── group-queue.ts
│   ├── container-runner.ts
│   ├── container-runtime.ts
│   ├── credential-proxy.ts
│   ├── runtime-setup.ts
│   ├── remote-control.ts
│   ├── channels/
│   │   ├── index.ts
│   │   ├── registry.ts
│   │   └── telegram.ts
│   └── runtime/
│       ├── index.ts
│       ├── registry.ts
│       ├── types.ts
│       ├── container-manager.ts
│       ├── claude-runtime.ts
│       ├── codex-runtime.ts
│       └── gemini-runtime.ts
├── container/
│   ├── Dockerfile
│   ├── build.sh
│   ├── agent-runner/
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── shared.ts
│   │   │   ├── runtime-registry.ts
│   │   │   ├── ipc-mcp-stdio.ts
│   │   │   ├── specialist-runner.ts
│   │   │   └── runtimes/
│   │   │       ├── claude.ts
│   │   │       ├── codex.ts
│   │   │       └── gemini.ts
│   │   └── adk/
│   │       ├── requirements.txt
│   │       └── nanoclaw_agent/
│   │           └── __init__.py
│   └── skills/
├── groups/
│   ├── global/
│   │   ├── AGENT.md
│   │   └── CLAUDE.md
│   └── main/
├── data/
│   ├── ipc/
│   └── sessions/
├── store/
│   └── messages.db
├── setup/
└── docs/
```

---

## Current Implementation Notes

- This spec describes the current fork in `/Users/tonkin/CU_agent`, not upstream `qwibitai/nanoclaw`.
- Upstream documentation assumptions still visible in some files may be stale.
- The code currently reflects a Telegram-first operational setup.
- The runtime abstraction is real and implemented, but some docs and comments still reference earlier Claude-only behavior.
- Persona files are in transition: this repo uses `AGENT.md`, `CLAUDE.md`, and `GEMINI.md` surfaces depending on runtime and development tool.
