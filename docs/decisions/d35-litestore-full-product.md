# D35: LiteStore as Full Product

**Date:** 2026-02-27
**Status:** Shipped (v0.2.0)
**Supersedes:** D13 (pgvector-required architecture), D15 (SurrealDB as sole queryable store)

## Decision

LiteStore (hnsw_rs + petgraph + Redb + Git-tracked JSON) is the full product, not a
lite fallback. It provides vector search, graph traversal, multi-agent coordination,
LLM-assisted merge, bi-temporal queries, and the complete namespace system -- all with
zero Docker dependency. FullStore (SurrealDB) and PostgresStore are opt-in upgrades for
advanced graph SQL and scale.

## Context

The original design (D13, D15) assumed SurrealDB in Docker was the real store and
embedded storage was a future possibility. After further analysis, it became clear that
most users want zero-dependency setup. Docker was originally justified by environment
isolation requirements, but correctness isolation (preventing cross-scope data pollution)
can be enforced at the application level without OS-level boundaries. The `QueryableStore`
trait boundary made this a natural evolution rather than a rewrite.

## Consequences

- Three-tier storage behind `QueryableStore` trait: LiteStore (default, zero Docker) ->
  SurrealStore (opt-in, Docker) -> PostgresStore (opt-in, feature-gated).
- LiteStore is the default for `corvia init`. FullStore requires `corvia init --full`.
- Knowledge files stored as individual JSON in `.corvia/knowledge/`, Git-trackable.
  Embeddings included in JSON to avoid re-embedding on rebuild.
- Resource footprint: ~50-150MB RAM, <1s startup vs FullStore's ~200-500MB RAM and ~30s
  first run.
- LiteStore components: `hnsw_rs` (vector search), `petgraph` (graph traversal), `Redb`
  (metadata, coordination, temporal index, edge storage), Git (source of truth).
- `corvia rebuild` reconstructs all indexes from Git-tracked knowledge files.

**Source:** `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md` (D35-D42)
