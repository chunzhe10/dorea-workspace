# dorea-workspace

> Corvia-powered workspace for [dorea](repos/dorea) — automated underwater video
> AI editing pipeline.

This file follows the [AGENTS.md standard](https://agents.md/).

## Workspace Layout

```
dorea-workspace/
├── AGENTS.md                # Cross-platform AI agent instructions (this file)
├── CLAUDE.md                # Claude Code wrapper (imports AGENTS.md)
├── corvia.toml              # Workspace config (repos, embedding, server, docs)
├── .agents/                 # Agent-agnostic skills & reference docs
│   └── skills/              # Reusable patterns for any AI assistant
├── .mcp.json                # MCP server config (corvia, resolve-mcp, playwright)
├── repos/
│   └── dorea/               # Pipeline: Python scripts, config, LUTs, templates
├── .corvia/                 # Local knowledge store (LiteStore)
├── footage/                 # Raw + flattened dive footage (gitignored)
├── working/                 # Ephemeral AI outputs: masks, depth, keyframes (gitignored)
├── models/                  # AI model weights: SAM2, Depth Anything (gitignored)
└── docs/
    ├── decisions/           # Workspace-level architectural decisions
    ├── learnings/           # Captured knowledge and patterns
    └── plans/               # Active implementation plans
```

## Quick Reference

```bash
corvia workspace status          # Check workspace + service health
corvia search "query"            # Search ingested knowledge
corvia workspace ingest          # Index dorea repo
corvia workspace ingest --fresh  # Re-index from scratch
corvia serve &                   # Start server (auto-started by devcontainer)
```

## Service Ports (container-internal)

| Port | Service | Description |
|------|---------|-------------|
| 8020 | API server | REST + MCP protocol |
| 8021 | Dashboard | Knowledge browser, system health |
| 8030 | Inference | gRPC embedding + chat (CPU mode) |

Host-forwarded ports are offset +100 (8120, 8121, 8130) to coexist with corvia-workspace.

## MCP Servers

| Server | URL | Description |
|--------|-----|-------------|
| corvia | `http://127.0.0.1:8020/mcp` | Organizational memory (container-internal) |
| davinci-resolve-mcp | `http://host.docker.internal:9090/mcp` | DaVinci Resolve API bridge (host) |
| playwright | `http://127.0.0.1:8050/mcp` | Browser automation (container-internal) |

Available corvia MCP tools (use `scope_id: "dorea"` for all calls):
- `corvia_search` — semantic search across ingested knowledge
- `corvia_write` — write knowledge entries (requires agent identity)
- `corvia_history` — entry supersession history
- `corvia_graph` — graph edges for an entry
- `corvia_reason` — run health checks on a scope
- `corvia_agent_status` — agent contribution summary
- `corvia_context` — retrieve assembled context (RAG retrieval only)
- `corvia_ask` — full RAG: question → AI-generated answer from knowledge
- `corvia_system_status` — system status (entry counts, agents, sessions, queue)
- `corvia_config_get` — read config section as JSON
- `corvia_config_set` — update hot-reloadable config value (requires confirmation)
- `corvia_adapters_list` — discovered adapter binaries
- `corvia_agents_list` — all registered agents
- `corvia_gc_run` — trigger garbage collection (requires confirmation)
- `corvia_rebuild_index` — rebuild HNSW vector index (requires confirmation)
- `corvia_agent_suspend` — suspend an agent (requires confirmation)
- `corvia_merge_retry` — retry failed merge entries (requires confirmation)
- `corvia_merge_queue` — inspect merge queue status

**IMPORTANT:** The `scope_id` for this workspace is `"dorea"` (defined in `corvia.toml`).
Do NOT use `"corvia"`, `"dorea-workspace"`, or any other variant.

## Hybrid Tool Usage (corvia MCP + native tools)

**IMPORTANT: Always call corvia MCP tools FIRST before using native tools for any
development task or question.** corvia is the project's knowledge base — skipping it
means you risk re-discovering decisions that were already made or contradicting
established patterns. This applies to ALL agents (Claude Code, Codex, etc.).

### When to use corvia MCP tools (ALWAYS do this first)

- **Starting ANY task**: Call `corvia_search` or `corvia_ask` first to find prior decisions,
  design context, or patterns relevant to the work. **This is mandatory, not optional.**
- **Answering ANY question about the project**: Call `corvia_ask` before searching code.
- **Understanding "why"**: Use `corvia_ask` for questions about architecture, rationale,
  or past discussions.
- **Recording decisions**: Use `corvia_write` to persist design decisions, architectural
  context, or implementation notes that future sessions should know.

### When to use native tools

- **Reading/editing specific files** — corvia doesn't replace file access.
- **Searching for code patterns** — precise text/regex matching in source code.
- **Running commands** — Python scripts, tests, git, CLI tools.
- **File discovery** — finding files by name or extension.

### Rule of thumb

> **corvia = project knowledge & context. Native tools = source code & execution.**
> **Always check corvia first.**

## Pipeline Overview

Dorea automates underwater video post-production in 6 phases:

| Phase | Script | Runs in | GPU |
|-------|--------|---------|-----|
| 0 | `00_generate_lut.py` | Container | No (CPU) |
| 1 | `01_extract_frames.py` | Container | No (ffmpeg CPU) |
| 2 | `02_claude_scene_analysis.py` | Container | No (API call) |
| 3 | `03_run_sam2.py` | Container | Yes (~3GB VRAM) |
| 4 | `04_run_depth.py` | Container | Yes (~1.5GB VRAM) |
| 5 | `05_resolve_setup.py` | **Host** | No (Resolve IPC) |

**Critical constraint:** 6GB VRAM. Only one GPU model loaded at a time. Sequential
processing enforced by pipeline scripts. Corvia inference runs on CPU.

## Development

- **Language**: Python
- **Package manager**: pip (venv at `/opt/dorea-venv`)
- **AI models**: SAM2-small, Depth Anything V2 Small, Claude API (sonnet)
- **Video tools**: ffmpeg, DaVinci Resolve Studio (host)
- **Storage**: corvia LiteStore — data in `.corvia/`
- **Embedding**: corvia-inference server at `http://127.0.0.1:8030` (CPU mode)
- **API server**: `http://127.0.0.1:8020` (REST + MCP)
- **Config**: `corvia.toml` at workspace root, `repos/dorea/config.yaml` for pipeline
