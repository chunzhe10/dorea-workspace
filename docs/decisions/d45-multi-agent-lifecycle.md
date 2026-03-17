# D45: Multi-Agent Lifecycle

**Date:** 2026-02-27
**Status:** Shipped (v0.2.0)
**Supersedes:** None (builds on D16 concurrency model, D43 staging hybrid, D44 capability scope)

## Decision

Multi-layer agent identity with session isolation, staging branches, LLM-assisted merge,
and crash recovery. Agents are identified across four tiers: Registered (internal, full
staging access), McpClient (external via MCP `_meta.agent_id`), ActorToken (future IETF
standard), and Anonymous (read-only). Sessions are ephemeral work units that survive
crashes through a five-step idempotent commit flow.

## Context

Multiple concurrent AI agents need to write to the same knowledge base without conflicts
or data loss. No standard for agent-level identity exists (as of 2026-03-01) -- MCP
identifies applications, not agents. The system needed optimistic concurrency (agents
write freely, conflicts resolved at merge time) to maximize throughput, plus crash
recovery so that no work is lost when agents or the server restart unexpectedly.

## Consequences

- `AgentCoordinator` manages the full session lifecycle: Created -> Active (heartbeat) ->
  Committing -> Merging -> Closed, with Stale (5min) -> Orphaned (20min) -> Recoverable
  recovery path.
- Staging hybrid architecture: each agent writes to `.corvia/staging/{agent-id}/` with a
  dedicated git branch `{agent-id}/session-{uuid}`. HNSW index and Redb remain shared
  (no per-agent duplication). Disk overhead ~10KB per session vs ~50-100MB for full
  git worktrees.
- Merge worker with conflict detection (semantic similarity) and LLM-assisted resolution.
  Non-conflicting writes merge automatically. Failed LLM merges retry with exponential
  backoff (max 3), then stay in queue for human intervention.
- Garbage collection: orphaned sessions > 24h auto-rollback, closed sessions > 7d
  cleaned from Redb, inactive agents > 30d suspended. All thresholds configurable via
  `[agent_lifecycle]` in `corvia.toml`.
- OpenTelemetry spans and structured metrics designed in at lifecycle definition time,
  not bolted on later.
- Key files: `agent_coordinator.rs`, `staging.rs`, `process_adapter.rs`.

**Source:** `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md` (D43-D45)
