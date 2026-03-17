# D72: JSONL IPC Adapter Protocol

**Date:** 2026-03-03
**Status:** Shipped (v0.3.4)
**Supersedes:** None (formalizes the adapter pattern from D10 core-vs-complement)

## Decision

Cross-language ingestion adapters communicate with the corvia kernel via a JSONL protocol
over stdin/stdout. Adapters are standalone binaries discovered at runtime through PATH
scanning and `~/.config/corvia/adapters/`. The kernel never touches the filesystem for
source discovery -- adapters handle all domain-specific file walking and content
extraction.

## Context

Adapters need to work across languages (Rust, Python, future community adapters) without
requiring shared memory or FFI. A standard protocol enables third-party adapters without
forking the core. The bottom-up approach (IPC layer first, test with adapter-basic, then
migrate repos) ensured protocol correctness before committing to the architecture.

## Consequences

- Protocol types in `corvia-kernel/src/adapter_protocol.rs`: `AdapterMetadata` (returned
  by `--corvia-metadata`), `AdapterRequest` (Ingest, Chunk), `AdapterResponse`
  (SourceFile, ChunkResult, Done, Error).
- Discovery in `adapter_discovery.rs`: scans `~/.config/corvia/adapters/` then `$PATH`
  for `corvia-adapter-*` executables, spawns each with `--corvia-metadata` to get
  capabilities, caches results for session lifetime.
- `ProcessAdapter` in `process_adapter.rs`: IPC wrapper managing adapter child process
  lifecycle (spawn -> ingest -> chunk -> shutdown). One process per adapter per session.
- Three-tier chunking priority preserved: (1) adapter override via IPC (e.g., `.rs` ->
  AstChunker), (2) kernel defaults (MarkdownChunker, ConfigChunker), (3) fallback
  (FallbackChunker line-split).
- Two first-party adapters in the monorepo: `corvia-adapter-git` (tree-sitter code
  ingestion) and `corvia-adapter-basic` (filesystem walk). External
  `corvia-adapter-git` repo archived.
- Config extensions: optional `[adapters]` and `[[sources]]` sections in `corvia.toml`.
  Zero config required -- auto-detection handles everything by default.

**Source:** `repos/corvia/docs/rfcs/2026-03-03-adapter-plugin-system-design.md`
