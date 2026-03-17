# D71: Prebuilt Binaries via GitHub Releases

**Date:** 2026-03-03
**Status:** Shipped (v0.3.3)
**Supersedes:** None (extends D70 optional services)

## Decision

GitHub releases provide prebuilt binaries (`corvia-cli-linux-amd64`,
`corvia-inference-linux-amd64`) for fast container startup. `post-create.sh` downloads
binaries first (~5 seconds) and falls back to `cargo install` from source (~5 minutes)
if the download fails or no release exists.

## Context

`post-create.sh` compiled corvia-cli and corvia-inference from source (~600 crates, ~5
minutes) on every devcontainer creation. This ran every time the container was rebuilt,
creating significant friction during development. Prebuilt binaries eliminate this
compile step for the common case while preserving the source-build path as a fallback.

## Consequences

- GitHub Actions release workflow at `repos/corvia/.github/workflows/release.yml`,
  triggered by `v*` tags. Builds four binaries: `corvia-cli`, `corvia-inference`,
  `corvia-adapter-basic`, `corvia-adapter-git`.
- `post-create.sh` uses `curl` to download from the `latest` release. On failure, falls
  back to `cargo install --path` from the local source checkout.
- `corvia-workspace rebuild` command added to the workspace script for local
  recompilation when developing against source changes in `repos/corvia/`.
- Version strategy: always downloads `latest` release, no pinning. Appropriate for
  pre-1.0 development.
- Normal container open: ~5s binary download. No-release/network-failure: ~5min cargo
  build. Both paths end with `corvia workspace init`.

**Source:** `docs/decisions/2026-03-03-prebuilt-binaries-design.md`
