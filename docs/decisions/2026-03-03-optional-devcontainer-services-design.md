# Optional Devcontainer Services — Light Default with Toggle

> **Status:** Shipped (v0.3.3)

**Date**: 2026-03-03
**Decision**: D70 (in `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md`)
**Refines**: D52 (Devcontainer as Integration Surface)

## Problem

The devcontainer unconditionally installs Ollama (~500MB), pulls a model (~274MB), and auto-starts both Ollama and the Corvia server. This is heavy for development and code review where full experimentation isn't needed.

## Design

Default to a **light mode** — Rust toolchain + `corvia-inference` (fastembed/ONNX for CPU embeddings) + LiteStore. Ollama and SurrealDB become independently toggleable after entering the container.

### Two Modes

| Aspect | Light (default) | With Ollama | With SurrealDB |
|--------|----------------|-------------|----------------|
| Storage | LiteStore (redb + hnsw_rs) | LiteStore | SurrealStore (SurrealDB v3) |
| Embeddings | corvia-inference (fastembed/ONNX) | Ollama (nomic-embed-text) | unchanged |
| Docker needed | No | No | Yes (docker-compose) |
| Image size | ~1.5GB (Rust toolchain + build deps) | +500MB (Ollama binary) | N/A (container) |

Ollama and SurrealDB are independent checkboxes — any combination is valid.

### Toggle Mechanism

`corvia-workspace` shell script at `.devcontainer/scripts/corvia-workspace.sh`, installed to PATH during `post-create.sh`.

```
corvia-workspace enable ollama       # install + pull model + start + update corvia.toml
corvia-workspace disable ollama      # stop + revert corvia.toml
corvia-workspace enable surrealdb   # start container + update corvia.toml
corvia-workspace disable surrealdb  # stop container + revert corvia.toml
corvia-workspace status             # show enabled/running state
```

### State Persistence

File: `.devcontainer/.corvia-workspace-flags` (gitignored)

```
ollama=disabled
surrealdb=disabled
```

`post-start.sh` reads this file on container restart and re-starts previously enabled services.

### Files Changed

1. **`.devcontainer/Dockerfile`** — Remove Ollama install
2. **`.devcontainer/scripts/post-create.sh`** — Remove Ollama model pull, add corvia-workspace install
3. **`.devcontainer/scripts/post-start.sh`** — Only start services from flags file
4. **`corvia.toml`** — Default to `provider = "corvia"`, `store_type = "lite"`
5. **New: `.devcontainer/scripts/corvia-workspace.sh`** — Toggle script
6. **New: `.devcontainer/.corvia-workspace-flags`** — State file (gitignored)

### Config Switching

`corvia-workspace enable/disable` updates `corvia.toml` via targeted `sed` replacements:

| Action | corvia.toml change |
|--------|-------------------|
| `enable ollama` | `provider = "ollama"`, `model = "nomic-embed-text"`, `url = "http://127.0.0.1:11434"` |
| `disable ollama` | `provider = "corvia"`, `model = "nomic-embed-text-v1.5"`, `url = "http://127.0.0.1:8030"` |
| `enable surrealdb` | `store_type = "surrealdb"` |
| `disable surrealdb` | `store_type = "lite"` |

### Enable Ollama Flow

1. Check if Ollama binary exists → install via `curl -fsSL https://ollama.com/install.sh | sh` if not
2. Start `ollama serve` in background
3. Wait for readiness (poll `/api/tags` endpoint)
4. Pull `nomic-embed-text` model if not present
5. Update `corvia.toml` embedding section
6. Write `ollama=enabled` to flags file

### Enable SurrealDB Flow

1. Run `docker compose -f repos/corvia/docker/docker-compose.yml up -d`
2. Wait for readiness (poll port 8000)
3. Update `corvia.toml` storage section
4. Write `surrealdb=enabled` to flags file

### User Flow

```
1. Open devcontainer → light mode, CPU embeddings, instant startup
2. corvia-workspace status → shows ollama=off, surrealdb=off
3. corvia-workspace enable ollama → ~2-3 min first time, instant after
4. corvia-workspace enable surrealdb → ~10s (container pull cached)
5. Container restart → both auto-start (flags persist)
6. corvia-workspace disable ollama → stops, reverts to CPU embeddings
```
