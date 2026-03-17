# Prebuilt Binaries via GitHub Releases

> **Status:** Shipped (v0.3.3)

**Date**: 2026-03-03
**Decision**: D71 (in `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md`)
**Extends**: D70 (Optional Devcontainer Services)

## Problem

`post-create.sh` compiles corvia-cli and corvia-inference from source (~600 crates, ~5 minutes). This runs every time the devcontainer is created.

## Design

Download prebuilt binaries from GitHub releases (~5 seconds). Fall back to source build if unavailable. Allow local rebuild when developing against source changes.

### Components

#### 1. GitHub Actions Release Workflow

**File**: `repos/corvia/.github/workflows/release.yml`
**Trigger**: Push of tags matching `v*` (e.g., `v0.2.0`)

Steps:
1. Checkout code
2. Install system deps (`pkg-config libssl-dev cmake protobuf-compiler`)
3. `cargo build --release -p corvia-cli -p corvia-inference`
4. Upload as release assets: `corvia-cli-linux-amd64`, `corvia-inference-linux-amd64`

#### 2. Updated post-create.sh

Download-first with source fallback:

```bash
RELEASE_URL="https://github.com/chunzhe10/corvia/releases/latest/download"
if curl -fsSL -o /usr/local/bin/corvia "$RELEASE_URL/corvia-cli-linux-amd64" && \
   curl -fsSL -o /usr/local/bin/corvia-inference "$RELEASE_URL/corvia-inference-linux-amd64"; then
    chmod +x /usr/local/bin/corvia /usr/local/bin/corvia-inference
else
    # Fall back to source build
    cd "$WORKSPACE_ROOT/repos/corvia"
    cargo install --path crates/corvia-cli
    cargo install --path crates/corvia-inference
    cd "$WORKSPACE_ROOT"
fi
```

#### 3. `corvia-workspace rebuild` command

New subcommand in `corvia-workspace.sh` for local recompilation:

```bash
corvia-workspace rebuild    # builds both binaries from repos/corvia/ source
```

Runs:
```bash
cargo install --path "$WORKSPACE_ROOT/repos/corvia/crates/corvia-cli"
cargo install --path "$WORKSPACE_ROOT/repos/corvia/crates/corvia-inference"
```

This overwrites the prebuilt binaries with locally compiled versions, useful when developing against source changes in `repos/corvia/`.

### Files Changed

1. **New: `repos/corvia/.github/workflows/release.yml`** — Release workflow
2. **Modify: `.devcontainer/scripts/post-create.sh`** — Download-first with fallback
3. **Modify: `.devcontainer/scripts/corvia-workspace.sh`** — Add `rebuild` command

### Version Strategy

Always download `latest` release. No pinning. Simple pre-1.0.

### User Flows

**Normal devcontainer open (has release):**
```
post-create.sh → downloads 2 binaries (~5s) → corvia workspace init → done
```

**No release yet / network failure:**
```
post-create.sh → download fails → cargo install from source (~5min) → corvia workspace init → done
```

**Developing against local source:**
```
# Edit code in repos/corvia/
corvia-workspace rebuild    # recompiles from local source
```

**Cutting a new release:**
```
cd repos/corvia
git tag v0.2.0
git push origin v0.2.0     # triggers workflow, uploads binaries
```
