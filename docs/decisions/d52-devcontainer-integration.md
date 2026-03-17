# D52: Devcontainer as Integration Surface

**Date:** 2026-03-01
**Status:** Shipped (v0.2.1)
**Supersedes:** D4 (VS Code extension primary form factor from v0.1.0)

## Decision

The devcontainer is the primary integration surface for corvia workspaces. It serves a
triple role: template for scaffolding new workspaces (`corvia workspace create --template`),
one-click demo for evaluators (clone and open in Codespaces/DevPod), and the development
environment used to build corvia itself. The workspace model (purpose-built directory with
`corvia.toml` and `.corvia/` data store) works identically in native and containerized
modes.

## Context

Corvia needed a reproducible development environment that works across GitHub Codespaces,
VS Code Dev Containers, and DevPod, while also serving as the showcase for multi-repo
knowledge management. The workspace concept (D46) separates corvia artifacts from target
repos -- repos live under `repos/` and are never polluted with corvia files. The
devcontainer packages this into a portable, zero-setup experience.

## Consequences

- `.devcontainer/` directory with `devcontainer.json`, Dockerfile, and lifecycle scripts:
  `post-create.sh` (clone repos, install binaries) and `post-start.sh` (start corvia
  server in background).
- Pre-ingested knowledge (`.corvia/knowledge/` + `hnsw_index/`) ships with the repo for
  instant search without waiting for embedding on first open.
- `corvia.toml` with `[workspace]` section declares repos, storage, embedding, and server
  config. Detection: if `[workspace]` section present, directory is a workspace.
- Workspace CLI surface: `corvia workspace create|init|status|add|remove|ingest`.
- Two operating modes: Mode 1 (native, local Ollama) and Mode 2 (devcontainer, services
  in container). Both use the same workspace layout and config.
- Port 8020 forwarded for the REST + MCP API server.

**Source:** `repos/corvia/docs/rfcs/2026-03-01-m2.1-workspace-devcontainer-design.md`
