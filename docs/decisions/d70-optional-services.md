# D70: Optional Devcontainer Services

**Date:** 2026-03-03
**Status:** Shipped (v0.3.3)
**Supersedes:** None (refines D52 devcontainer integration)

## Decision

Light default devcontainer -- no Ollama, no SurrealDB by default. The container ships
with Rust toolchain + corvia-inference (fastembed/ONNX for CPU embeddings) + LiteStore
only. Ollama and SurrealDB are independently toggleable after entering the container via
a `corvia-workspace enable/disable` shell script.

## Context

The original devcontainer unconditionally installed Ollama (~500MB), pulled an embedding
model (~274MB), and auto-started both Ollama and the corvia server. This was heavy for
development and code review where full experimentation was not needed. Most dev work only
requires LiteStore + corvia-inference, making the heavy defaults wasteful of both time
and RAM.

## Consequences

- Light mode default: ~1.5GB image (Rust toolchain + build deps). Ollama adds ~500MB.
  SurrealDB runs as a separate Docker container.
- Toggle script at `.devcontainer/scripts/corvia-workspace.sh`, installed to PATH during
  `post-create.sh`. Commands: `enable/disable ollama|surrealdb`, `status`, `rebuild`.
- State persisted in `.devcontainer/.corvia-workspace-flags` (gitignored). `post-start.sh`
  reads flags on container restart and re-starts previously enabled services.
- `enable ollama` installs binary, starts serve, pulls model, updates `corvia.toml`
  embedding section. `disable ollama` reverts to corvia-inference provider.
- `enable surrealdb` starts Docker container, updates `corvia.toml` to `store_type =
  "surrealdb"`. `disable surrealdb` reverts to `store_type = "lite"`.
- `corvia.toml` defaults: `provider = "corvia"`, `store_type = "lite"`.

**Source:** `docs/decisions/2026-03-03-optional-devcontainer-services-design.md`
