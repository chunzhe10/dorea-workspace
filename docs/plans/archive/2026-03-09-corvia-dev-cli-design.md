# corvia-dev CLI — Redesigning Workspace Management

> **Status:** Superseded — Python CLI abandoned in favor of Rust CLI (`corvia workspace`)

**Date**: 2026-03-09
**Replaces**: `corvia-workspace.sh` (390-line bash script), `corvia-supervisor.sh` (79-line bash daemon), `extension.js` (818-line VS Code extension)

## Problem

The current dev environment management is spread across three fragile scripts:

1. **corvia-workspace.sh** — service toggles, config mutation via `sed`, health checks, auto-healing, binary rebuilds. 390 lines of bash with no structured output.
2. **corvia-supervisor.sh** — process restart daemon with exponential backoff. Uses PID files in `/tmp`.
3. **extension.js** — VS Code dashboard with 481 lines of inline HTML/CSS/JS, 6 parallel data sources (2 CLI text parsers, 2 HTTP checks, 1 agent list, 1 log tail).

Key problems:
- Config mutation via `sed` on `corvia.toml` is fragile and bypasses the Rust `CorviaConfig` type system.
- The VS Code extension parses unstructured CLI text output — breaks when output format changes.
- No JSON output for machine consumption.
- Supervisor state scattered across PID files in `/tmp`.
- Tight coupling between extension and multiple data sources.

## Decision: Approach B — Unified CLI + Thin Extension

A Python CLI (`corvia-dev`) owns all logic. The VS Code extension becomes a thin webview that polls `corvia-dev status --json` and sends commands back. One JSON contract is the only interface between them.

### Why Python (not Rust, not bash)

- **Not Rust**: This is a dev tool owned by the workspace repo, not the corvia product. No compile step needed. Faster iteration.
- **Not bash**: Config mutation needs proper TOML parsing. Process supervision needs structured state. JSON output needs serialization. Bash can't do these reliably.
- **Python**: Pre-installed in devcontainers. `pydantic` for strict typing. `tomli`/`tomli_w` for lossless TOML round-tripping. `click` for CLI. No runtime to install.

### Why this repo (not corvia repo)

`corvia workspace` (Rust) is core product — multi-repo knowledge management for end users. `corvia-dev` is dev environment orchestration — service lifecycle, provider switching, dashboards. Different audiences, different release cadences.

## CLI Structure

```
corvia-dev
├── up [--foreground]          # Start all enabled services, supervise them
├── down                       # Stop all managed services
├── status [--json]            # Health of all services + config summary
├── enable <service>           # Start + persist: coding-llm, surrealdb, postgres
├── disable <service>          # Stop + persist
├── use <provider>             # Switch corvia's backend: ollama, corvia-inference, vllm
├── logs [service] [--tail N]  # Supervisor and service logs
├── restart [service]          # Restart a specific service or all
└── config                     # Show current corvia.toml config (read-only view)
```

### Command Semantics

- **`enable/disable`** — additive services (coding-llm, surrealdb, postgres). Persisted to flags file. Respected by `up`.
- **`use`** — switches corvia's provider. Mutates `corvia.toml` programmatically. `use ollama` sets both `embedding.provider` and `merge.provider`. `use corvia-inference` switches back.
- **`up`** — starts tier 0 (corvia-server + configured provider) plus all enabled services. Supervises with restart-on-crash. Foreground by default (like `tilt up`), `--no-foreground` for devcontainer `post-start.sh`.
- **`status --json`** — single data source for the VS Code extension.

## Service Model

### Tiers

| Tier | Purpose | Failure Impact |
|------|---------|----------------|
| 0 | Core (always runs) | Blocks everything |
| 1 | Provider (replaces defaults) | Blocks corvia-server if active |
| 2 | Additive (extra capability) | Warning only, doesn't block |

### Service Registry

| Service | Tier | Port | Health | Depends On | Exclusive Group |
|---------|------|------|--------|------------|-----------------|
| `corvia-inference` | 0 | 8030 | `/health` | — | `embedding`* |
| `corvia-server` | 0 | 8020 | `/health` | active embedding provider | — |
| `ollama` | 1 | 11434 | `/api/tags` | — | `embedding`* |
| `vllm` | 1 | TBD | TBD | — | `embedding`* |
| `surrealdb` | 1 | 8000 | `/health` | docker | `storage` |
| `postgres` | 1 | 5432 | pg_isready | docker | `storage` |
| `coding-llm` | 2 | — | — | `ollama` | — |

*`exclusive_group` only applies for corvia's configured provider. Ollama can run alongside corvia-inference when serving coding-llm.

### Dependency Resolution

The key insight: **embedding provider and merge/chat provider are independently configurable**, and **Continue (coding-llm) cannot use corvia-inference's gRPC API** — it requires ollama or an OpenAI-compatible HTTP API.

Valid concurrent configurations:
- `corvia-inference` for corvia embeddings + `ollama` for coding-llm (both run)
- `ollama` for everything (corvia embeddings + chat + coding-llm)
- `corvia-inference` for corvia + no coding-llm (ollama not needed)

### Startup Order (on `dev up`)

1. Resolve active embedding/storage provider from `corvia.toml`
2. Start active embedding provider → wait for health
3. Start active storage if non-lite (surrealdb/postgres) → wait for health
4. Start corvia-server → wait for health
5. Start additive services (ollama for coding-llm if not already running)
6. Post-start hooks (pull models, write Continue config)

### Graceful Degradation

- Health checks: HTTP with 3s timeout, 3 retries, 1s between
- Tier 2 failure: log warning, don't block
- Tier 0/1 failure: retry with exponential backoff (1s → 2s → 4s → ... → 60s cap)
- `status --json` always returns, even if everything is down

## Config Mutation

### How `use <provider>` works

Parse-modify-serialize with field preservation:

1. Load `corvia.toml` as raw dict via `tomli`
2. Validate only sections we touch via Pydantic models
3. Mutate dict in-place
4. Write back via `tomli_w` — preserves all untouched sections

### What `use` changes vs preserves

**Changes:** `embedding.provider`, `embedding.url`, `merge.provider`, `merge.url`
**Preserves:** `embedding.model`, `embedding.dimensions`, `storage.*`, `server.*`, `workspace.*`, everything else

### Validation before write

- Check provider value is valid
- Check target service reachable (warning if not, write anyway)
- Warn if running server needs restart after switch

## Process Supervision

### ProcessManager

Single Python process manages all children. No PID files in `/tmp`.

State held in-memory, written to `/tmp/corvia-dev-state.json` every health check cycle:

```json
{
  "manager": {"pid": 123, "uptime_s": 3600, "state": "running"},
  "services": [
    {"name": "corvia-inference", "state": "healthy", "port": 8030, "pid": 456, "uptime_s": 3595},
    {"name": "corvia-server", "state": "healthy", "port": 8020, "pid": 789, "uptime_s": 3590},
    {"name": "ollama", "state": "stopped", "port": null, "pid": null, "reason": "not enabled"}
  ],
  "config": {
    "embedding_provider": "corvia",
    "merge_provider": "corvia",
    "storage": "lite",
    "workspace": "corvia-workspace"
  },
  "enabled_services": ["coding-llm"],
  "logs": ["2026-03-09T02:30:00 corvia-server started (pid 789)", "...last 20 lines"]
}
```

### Why state file over Unix socket

- Simpler — no async server in the manager
- `status` works even if manager is crashed (reports stale data with warning)
- VS Code extension can read file directly for instant updates

### Supervisor behavior

- `Ctrl+C` → SIGTERM to all children, clean shutdown
- Child crash → restart with exponential backoff (1s → 60s, reset after 5 min stable)
- `dev down` → SIGTERM to manager, cascades to children

## VS Code Extension (Thin Skin)

### Responsibilities (~200 lines)

1. **Poll** — `corvia-dev status --json` every 10s
2. **Display** — render JSON into webview dashboard
3. **Command** — send user actions as `corvia-dev <command>` in terminal

### Status bar

- Green: all tier 0 services healthy
- Yellow: tier 0 healthy, some tier 1/2 down
- Red: tier 0 down

### JSON contract is the interface

Extension never shells out to `corvia workspace status`, never tails logs, never does HTTP health checks. All aggregated by `corvia-dev status --json`.

## File Layout

### New files

```
corvia-workspace/
├── tools/
│   └── corvia-dev/
│       ├── pyproject.toml          # deps: pydantic, tomli, tomli_w, click
│       ├── corvia_dev/
│       │   ├── __init__.py
│       │   ├── cli.py              # click CLI entry point
│       │   ├── config.py           # CorviaToml parse/mutate/write
│       │   ├── services.py         # Service registry & definitions
│       │   ├── manager.py          # ProcessManager, health checks, supervision
│       │   └── models.py           # Pydantic models (StatusResponse, Service, etc.)
│       └── tests/
│           ├── test_config.py
│           ├── test_services.py
│           └── test_manager.py
├── .devcontainer/
│   └── extensions/
│       └── corvia-services/
│           ├── extension.js        # Rewritten: ~200 lines, thin skin
│           └── package.json
```

### What gets deleted

| Old File | Replaced By |
|----------|-------------|
| `.devcontainer/scripts/corvia-workspace.sh` (390 lines) | `tools/corvia-dev/` |
| `.devcontainer/scripts/corvia-supervisor.sh` (79 lines) | `corvia_dev/manager.py` |
| `.devcontainer/extensions/corvia-services/extension.js` (818 lines) | Rewritten (~200 lines) |

### What stays but changes

| File | Change |
|------|--------|
| `.devcontainer/scripts/post-start.sh` | Calls `corvia-dev up --no-foreground` instead of supervisor script |
| `.devcontainer/scripts/post-create.sh` | Adds `pip install -e tools/corvia-dev` |

### Installation

`pip install -e tools/corvia-dev` in post-create, gives `corvia-dev` on PATH.

---

## Implementation Decisions (post-design)

Decisions made during and after initial implementation that refine the design above.

### Health Check Protocols (2026-03-09)

**Problem**: corvia-inference serves gRPC (HTTP/2 only), not HTTP/1.1. A plain `GET /health` always fails because the server rejects HTTP/1.1 requests.

**Decision**: Add a `health_proto` field to `ServiceDefinition` with values `"http"` (default), `"grpc"`, `"tcp"`, or `"none"`.

**gRPC check implementation**: TCP connect → send HTTP/2 connection preface (`PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`) → read response → validate that byte[3] == 0x04 (HTTP/2 SETTINGS frame). This confirms the server is alive and speaking HTTP/2 without needing a gRPC client library.

| Service | health_proto | Check Method |
|---------|-------------|--------------|
| corvia-server | http | GET /health |
| corvia-inference | grpc | HTTP/2 preface handshake |
| ollama | http | GET /api/tags |
| postgres | tcp | TCP connect (port 5432) |
| coding-llm | none | Virtual service, no check |

### Dashboard UX: Contextual Actions Over Command Palettes (2026-03-09)

**Problem**: The initial extension dashboard had a "Quick Actions" grid — six CLI commands dumped as buttons (Status, Restart, Use Ollama, Use CI, Config, Logs). No hierarchy, no context. "Use Ollama" has nothing to do with "Restart" but they sit side by side.

**Decision**: Kill the Quick Actions section entirely. Every action belongs inline where it's contextually meaningful:

| Action | Where It Lives | Why |
|--------|---------------|-----|
| Restart service | Button on each health card | You restart *this* service, not "services in general" |
| Restart all | Header toolbar (danger-styled) | Global action, lives at the top |
| Switch provider | Pill buttons in Configuration card, next to the Embedding row | Provider switching is a config action |
| Enable/disable | Toggle switch on each service row | Per-service control |
| View full logs | Icon button on log panel header | Contextual to the log panel |
| Start services | Offline banner CTA | Only shown when services are down |

**Layout**: Two-column middle section (Services | Configuration) instead of stacked. Health banner at top. Collapsible log panel pinned at bottom.

### Service Log Capture (2026-03-09)

**Problem**: The ProcessManager piped subprocess stdout/stderr to `asyncio.subprocess.PIPE` but never read from them. This meant: (1) no logs visible anywhere, (2) potential deadlock if pipe buffer fills up.

**Decision**: Write subprocess output to per-service log files at `/tmp/corvia-dev-logs/<service>.log` in append mode.

**JSON contract extension**: Added `service_logs` field to `StatusResponse`:

```json
{
  "service_logs": {
    "corvia-server": ["line1", "line2", "...last 30 lines"],
    "corvia-inference": ["..."],
    "supervisor": ["...legacy supervisor log if present"]
  }
}
```

The existing `logs` field continues to carry manager lifecycle events ("corvia-server started pid 123", "now healthy", etc.). `service_logs` carries actual service output.

**Dashboard**: Tabbed log viewer at the bottom of the dashboard. One tab per service with log output. Tabs show line count badges. Active tab persists across poll refreshes. Auto-scrolls to bottom. Collapsible (open by default, state survives re-renders since the log panel is outside the `#content` div that gets replaced on each poll).

**CLI**: `corvia-dev logs` shows all service logs. `corvia-dev logs corvia-server` shows one. Reads directly from log files (works even without a running manager).

**Legacy fallback**: When running without the manager (services started by old supervisor), the status command also checks `/tmp/corvia-supervisor.log` and includes it as a "supervisor" tab.

### TOML Serialization Side Effect (2026-03-09)

`tomli_w` normalizes TOML syntax when round-tripping. Specifically, `[[workspace.repos]]` (array of tables) becomes inline `repos = [{...}]`. This is semantically identical but looks different in the file. Accepted as a trade-off for reliable parse-modify-serialize over fragile `sed`.
