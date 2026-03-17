# corvia-workspace

> Multi-repo workspace for [corvia](repos/corvia) — organizational memory for AI agents.

This file follows the [AGENTS.md standard](https://agents.md/).

## Workspace Layout

```
corvia-workspace/
├── AGENTS.md                # Cross-platform AI agent instructions (this file)
├── CLAUDE.md                # Claude Code wrapper (imports AGENTS.md)
├── corvia.toml              # Workspace config (repos, embedding, server, docs)
├── .agents/                 # Agent-agnostic skills & reference docs
│   └── skills/              # Reusable patterns for any AI assistant
├── .mcp.json                # MCP server config (Claude Code, Codex, etc.)
├── repos/
│   └── corvia/              # Core: kernel, server, CLI, adapters (Rust, AGPL-3.0)
├── .corvia/                 # Local knowledge store (LiteStore)
└── docs/
    ├── decisions/           # Workspace-level architectural decisions
    ├── learnings/           # Captured knowledge and patterns
    ├── marketing/           # LinkedIn carousels, brand assets
    └── plans/               # Active implementation plans
```

## Quick Reference

```bash
corvia workspace status          # Check workspace + service health
corvia search "query"            # Search ingested knowledge
corvia workspace ingest          # Index all repos
corvia workspace ingest --fresh  # Re-index from scratch
corvia serve &                   # Start server (auto-started by devcontainer)
corvia workspace init-hooks      # Generate doc-placement hooks from config
```

## Service Ports

| Port | Service | Description |
|------|---------|-------------|
| 8020 | API server | REST + MCP protocol |
| 8021 | Dashboard | Knowledge browser, system health |
| 8030 | Inference | gRPC embedding + chat (ONNX Runtime) |

## MCP Server (Dogfooding)

This workspace uses corvia's own MCP server at `http://localhost:8020/mcp`.
Any MCP-compatible AI tool can connect to it. The server is started automatically
by the devcontainer's `post-start.sh`.

Available MCP tools (use `scope_id: "corvia"` for all calls):
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

**IMPORTANT:** The `scope_id` for this workspace is `"corvia"` (defined in `corvia.toml`).
Do NOT use `"corvia-workspace"`, `"corvia-demo"`, or any other variant.

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
  or past discussions (e.g., "why does LiteStore use JSON files?").
- **Exploring relationships**: Use `corvia_graph` to understand how concepts, components,
  or entries relate to each other.
- **Checking history**: Use `corvia_history` to see how a piece of knowledge evolved.
- **Recording decisions**: Use `corvia_write` to persist design decisions, architectural
  context, or implementation notes that future sessions should know.
- **Health checks**: Use `corvia_reason` to validate knowledge consistency in a scope.

### When to use native tools

- **Reading/editing specific files** — corvia doesn't replace file access.
- **Searching for code patterns** — precise text/regex matching in source code.
- **Running commands** — builds, tests, git, CLI tools.
- **File discovery** — finding files by name or extension.

### Hybrid patterns

| Task | corvia first | Then native tools |
|------|-------------|-------------------|
| Start a feature | `corvia_search` for prior art/decisions | Read relevant files, implement |
| Debug an issue | `corvia_ask` "how does X work?" | Search code, read files, fix |
| Explore unfamiliar area | `corvia_search` for high-level context | Search/read for code details |
| Make a design decision | `corvia_ask` for existing patterns | Write design doc, `corvia_write` to record |
| Review a PR or change | `corvia_context` for relevant knowledge | Read changed files, search for impact |

### Rule of thumb

> **corvia = project knowledge & context. Native tools = source code & execution.**
> **Always check corvia first** — it's fast and prevents re-discovering things that
> were already decided. Do NOT jump straight to file reads or code search without
> checking corvia for relevant context first.

## AI Development Learnings

This workspace incorporates proven patterns from community best practices.
See [.agents/skills/ai-assisted-development.md](.agents/skills/ai-assisted-development.md)
for the full reference.

Key principles applied here:
- **Context engineering > prompt engineering** — AGENTS.md is essential infrastructure
- **Verify explicitly** — give pass/fail criteria, run tests before claiming success
- **Guard context** — delegate research to subagents, compact proactively, fresh sessions per task
- **Record decisions** — use `corvia_write` to persist learnings (dogfood the product)

## Production Agent BKMs

Best Known Methods for building production-grade AI agents, adapted from
[agents-towards-production](https://github.com/NirDiamant/agents-towards-production).

### Architecture

- **Graph-based orchestration**: Use directed graph architectures with explicit state
  transitions for multi-step workflows. Avoid linear chains for anything non-trivial.
- **Layered separation of concerns**: Keep orchestration, memory, tools, security, and
  evaluation as distinct layers. Do not mix tool-calling logic with reasoning logic.
- **Protocol-first integration**: Adopt MCP for tool integration and A2A for multi-agent
  communication. Protocol-based design makes agents composable and replaceable.

### Memory Systems

- **Dual-memory architecture**: Short-term (session/conversation context) + long-term
  (persistent knowledge with semantic search — this is what corvia provides).
- **Self-improving memory**: Design memory that evolves through interaction — automatic
  insight extraction, conflict resolution, and knowledge consolidation across sessions.

### Security (Defense-in-Depth)

- **Three-layer guardrails**: Input validation (prompt injection prevention), behavioral
  constraints (during execution), and output filtering (before delivery to user).
- **Tool access control**: Restrict which tools an agent can invoke based on user context
  and permissions. Never give agents unrestricted access to external tools.
- **User isolation**: Prevent cross-user data leakage in multi-user deployments.

### Observability

- **Trace every decision point**: Capture the full reasoning chain — which tools were
  called, what the LLM decided, timing data for each step.
- **Instrument from day one**: Do not bolt on observability later. Traces are essential
  for debugging, performance analysis, and evaluation.
- **Monitor cost, latency, accuracy** continuously, not just during development.

### Evaluation & Testing

- **Domain-specific test suites**: Build evaluation sets tailored to your domain.
  Generic benchmarks are insufficient.
- **Multi-dimensional metrics**: Evaluate beyond accuracy — include cost per interaction,
  latency, safety compliance, and tool-use correctness.
- **Iterative improvement cycles**: Evaluation should produce actionable insights that
  feed back into agent refinement.

### Deployment Strategy

- **Containerize everything**: Docker for portability and environment consistency.
- **Start stateless, migrate to persistent**: Prototype without memory, then layer in
  persistence once the workflow is stable.
- **Production readiness progression**: Prototype → Functional (add memory, auth, tracing)
  → Production (guardrails, evaluation, observability) → Scaled (multi-agent, GPU, fine-tuning).

## Repo-Specific Instructions

For detailed build/test/architecture guidance, see:
- [repos/corvia/AGENTS.md](repos/corvia/AGENTS.md) — kernel, server, CLI, adapters

## Development

- **Language**: Rust workspace (cargo)
- **Storage**: LiteStore (default, zero-Docker) — data in `.corvia/`
- **Embedding**: corvia-inference server at `http://127.0.0.1:8030` (default: nomic-embed-text-v1.5 768d; also supports all-MiniLM-L6-v2 384d)
- **API server**: `http://127.0.0.1:8020` (REST + MCP)
- **Config**: `corvia.toml` at workspace root
