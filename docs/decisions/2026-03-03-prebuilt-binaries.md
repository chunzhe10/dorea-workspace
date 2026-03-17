# Prebuilt Binaries Implementation Plan

> **Status:** Shipped (v0.3.3)

**Goal:** Speed up devcontainer creation by downloading prebuilt corvia-cli and corvia-inference binaries from GitHub releases (~5s vs ~5min), with source fallback and local rebuild.

**Architecture:** GitHub Actions workflow in `chunzhe10/corvia` builds binaries on tag push. `post-create.sh` downloads from latest release, falls back to cargo build. `corvia-workspace rebuild` recompiles from local source.

**Tech Stack:** GitHub Actions, bash, curl

**Design doc:** `docs/decisions/2026-03-03-prebuilt-binaries-design.md`
**Decision:** D71 in `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md`

---

### Task 1: Create GitHub Actions release workflow

**Files:**
- Create: `repos/corvia/.github/workflows/release.yml`

**Step 1: Create the workflow directory**

Run: `mkdir -p repos/corvia/.github/workflows`

**Step 2: Write the release workflow**

Create `repos/corvia/.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y pkg-config libssl-dev cmake protobuf-compiler

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo registry and build
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-release-${{ hashFiles('**/Cargo.lock') }}

      - name: Build release binaries
        run: cargo build --release -p corvia-cli -p corvia-inference

      - name: Prepare release assets
        run: |
          cp target/release/corvia corvia-cli-linux-amd64
          cp target/release/corvia-inference corvia-inference-linux-amd64
          chmod +x corvia-cli-linux-amd64 corvia-inference-linux-amd64

      - name: Create GitHub release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            corvia-cli-linux-amd64
            corvia-inference-linux-amd64
          generate_release_notes: true
```

Note: The corvia-cli binary is named `corvia` in `target/release/` (the `[[bin]]` name in `crates/corvia-cli/Cargo.toml` is `corvia`). The corvia-inference binary is named `corvia-inference`.

**Step 3: Verify the workflow YAML is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('repos/corvia/.github/workflows/release.yml')); print('Valid YAML')"` (if pyyaml available) or just verify visually.

**Step 4: Commit**

```bash
cd repos/corvia
git add .github/workflows/release.yml
git commit -m "ci: add release workflow for prebuilt binaries (D71)"
```

---

### Task 2: Update post-create.sh with download-first logic

**Files:**
- Modify: `.devcontainer/scripts/post-create.sh`

**Step 1: Replace the script contents**

The current script (lines 9-19) tries `corvia workspace init` then falls back to building from source. Replace with download-first logic:

```bash
#!/bin/bash
set -e

echo "=== Corvia Workspace: Post-Create Setup ==="

# Capture workspace root (where devcontainer mounts the workspace)
WORKSPACE_ROOT="$(pwd)"

# Install Corvia binaries (prebuilt from GitHub release, or build from source)
RELEASE_URL="https://github.com/chunzhe10/corvia/releases/latest/download"

echo "Downloading prebuilt binaries..."
if curl -fsSL -o /usr/local/bin/corvia "$RELEASE_URL/corvia-cli-linux-amd64" && \
   curl -fsSL -o /usr/local/bin/corvia-inference "$RELEASE_URL/corvia-inference-linux-amd64"; then
    chmod +x /usr/local/bin/corvia /usr/local/bin/corvia-inference
    echo "  Binaries installed from latest release."
else
    echo "  Download failed — building from source..."
    git clone https://github.com/chunzhe10/corvia repos/corvia 2>/dev/null || true
    cd "$WORKSPACE_ROOT/repos/corvia"
    cargo install --path crates/corvia-cli
    cargo install --path crates/corvia-inference
    cd "$WORKSPACE_ROOT"
fi

# Initialize workspace
echo "Initializing workspace..."
corvia workspace init

# Install corvia-workspace toggle command
chmod +x .devcontainer/scripts/corvia-workspace.sh
ln -sf "$WORKSPACE_ROOT/.devcontainer/scripts/corvia-workspace.sh" /usr/local/bin/corvia-workspace

echo "=== Post-Create Complete ==="
echo "Run 'corvia-workspace status' to see available services."
echo "Run 'corvia-workspace enable ollama' for Ollama embeddings."
echo "Run 'corvia-workspace enable surrealdb' for SurrealDB FullStore."
echo "Run 'corvia-workspace rebuild' to recompile from local source."
```

Key changes from current:
- Downloads prebuilt binaries first (fast path ~5s)
- Falls back to source build if download fails (no release yet, network issue)
- `corvia workspace init` runs unconditionally after either install path (no longer inside a fallback block)
- Added rebuild hint in output

**Step 2: Verify syntax**

Run: `bash -n .devcontainer/scripts/post-create.sh`
Expected: No output (no syntax errors).

**Step 3: Commit**

```bash
git add .devcontainer/scripts/post-create.sh
git commit -m "feat: download prebuilt binaries in post-create, fallback to source (D71)"
```

---

### Task 3: Add `rebuild` command to corvia-workspace.sh

**Files:**
- Modify: `.devcontainer/scripts/corvia-workspace.sh`

**Step 1: Add the rebuild function**

Insert before the `# --- Main ---` comment (before line 180):

```bash
# --- Rebuild ---

rebuild_binaries() {
    echo "Rebuilding Corvia binaries from local source..."

    local corvia_src="$WORKSPACE_ROOT/repos/corvia"

    if [ ! -d "$corvia_src" ]; then
        echo "  Error: corvia source not found at $corvia_src"
        exit 1
    fi

    cd "$corvia_src"
    cargo install --path crates/corvia-cli
    cargo install --path crates/corvia-inference
    cd "$WORKSPACE_ROOT"

    echo "  Binaries rebuilt from local source."
}
```

**Step 2: Add the `rebuild` case to the main dispatch**

In the main `case` block (line 182), add after the `status)` case and before the `*)` default:

```bash
    rebuild)
        rebuild_binaries
        ;;
```

**Step 3: Update the help text**

In the `*)` default case, add the rebuild line to the usage output:

```bash
    *)
        echo "corvia-workspace — toggle optional devcontainer services"
        echo ""
        echo "Usage:"
        echo "  corvia-workspace enable {ollama|surrealdb}"
        echo "  corvia-workspace disable {ollama|surrealdb}"
        echo "  corvia-workspace status"
        echo "  corvia-workspace rebuild"
        exit 1
        ;;
```

**Step 4: Verify syntax**

Run: `bash -n .devcontainer/scripts/corvia-workspace.sh`
Expected: No output.

**Step 5: Test help output**

Run: `bash .devcontainer/scripts/corvia-workspace.sh`
Expected: Shows usage including the new `rebuild` line.

**Step 6: Commit**

```bash
git add .devcontainer/scripts/corvia-workspace.sh
git commit -m "feat: add corvia-workspace rebuild command (D71)"
```

---

### Task 4: Verify the binary name in corvia-cli

**Files:**
- Read only: `repos/corvia/crates/corvia-cli/Cargo.toml`

**Step 1: Confirm the binary output name**

Run: `grep -A2 '^\[\[bin\]\]' repos/corvia/crates/corvia-cli/Cargo.toml`

Expected: The binary name is `corvia` (not `corvia-cli`). This confirms the release workflow correctly copies `target/release/corvia` (not `target/release/corvia-cli`) to `corvia-cli-linux-amd64`.

If the binary name is different, update the `release.yml` `Prepare release assets` step accordingly.

**Step 2: Confirm corvia-inference binary name**

Run: `grep -A2 '^\[\[bin\]\]' repos/corvia/crates/corvia-inference/Cargo.toml`

Expected: The binary name is `corvia-inference`.

No commit needed — this is a verification step.

---

### Task 5: Smoke test

**Step 1: Verify all script syntax**

Run: `bash -n .devcontainer/scripts/post-create.sh && echo "post-create: OK" && bash -n .devcontainer/scripts/corvia-workspace.sh && echo "corvia-workspace: OK"`
Expected: Both OK.

**Step 2: Test corvia-workspace help shows rebuild**

Run: `bash .devcontainer/scripts/corvia-workspace.sh 2>&1 || true`
Expected: Usage output includes `corvia-workspace rebuild`.

**Step 3: Test download path (will fail gracefully since no release exists yet)**

Run: `curl -fsSL -o /dev/null "https://github.com/chunzhe10/corvia/releases/latest/download/corvia-cli-linux-amd64" 2>&1; echo "exit: $?"`
Expected: Non-zero exit (no release yet). This confirms the fallback path would activate.

**Step 4: Verify workflow YAML exists**

Run: `cat repos/corvia/.github/workflows/release.yml | head -5`
Expected: Shows the workflow name and trigger.

**Step 5: Commit if needed, push**

```bash
git add -A
git status  # verify only expected files
git push
```

Also push the corvia repo workflow:

```bash
cd repos/corvia
git push
```
