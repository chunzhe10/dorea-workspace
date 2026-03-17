# Optional Devcontainer Services Implementation Plan

> **Status:** Shipped (v0.3.3)

**Goal:** Make the devcontainer default to a light mode (LiteStore + corvia-inference/ONNX) with Ollama and SurrealDB as independent opt-in toggles via `corvia-workspace enable/disable`.

**Architecture:** Remove Ollama from Dockerfile and lifecycle scripts. Change `corvia.toml` default to `provider = "corvia"` (fastembed/ONNX via gRPC on port 8030). Add a `corvia-workspace` shell script that manages enable/disable/status for Ollama and SurrealDB, persisting state in a flags file that `post-start.sh` reads on container restart.

**Tech Stack:** Bash (toggle script), Docker (Dockerfile), corvia-inference (fastembed/ONNX gRPC server on port 8030)

**Design doc:** `docs/decisions/2026-03-03-optional-devcontainer-services-design.md`
**Decision:** D70 in `repos/corvia/docs/rfcs/2026-02-27-corvia-v0.2.0-brainstorm.md`

---

### Task 1: Slim down the Dockerfile

**Files:**
- Modify: `.devcontainer/Dockerfile`

**Step 1: Remove the Ollama install from the Dockerfile**

Replace the entire file with:

```dockerfile
FROM rust:1.88-bookworm

RUN apt-get update && apt-get install -y \
    curl git build-essential pkg-config libssl-dev cmake protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
```

**Step 2: Verify the Dockerfile is valid**

Run: `docker build -f /root/corvia-project/corvia-workspace/.devcontainer/Dockerfile /root/corvia-project/corvia-workspace/.devcontainer/ --no-cache --progress=plain 2>&1 | tail -5`
Expected: Build succeeds, no Ollama references in output.

**Step 3: Commit**

```bash
git add .devcontainer/Dockerfile
git commit -m "chore: remove Ollama from Dockerfile for light default (D70)"
```

---

### Task 2: Switch corvia.toml to corvia-inference provider

**Files:**
- Modify: `corvia.toml`

**Step 1: Update the embedding section**

Change the `[embedding]` section from:

```toml
[embedding]
provider = "ollama"
model = "nomic-embed-text"
url = "http://127.0.0.1:11434"
dimensions = 768
```

To:

```toml
[embedding]
provider = "corvia"
model = "nomic-embed-text-v1.5"
url = "http://127.0.0.1:8030"
dimensions = 768
```

This ensures `corvia workspace init` skips Ollama provisioning (the Rust code checks `config.embedding.provider == InferenceProvider::Ollama` at `crates/corvia-cli/src/workspace.rs:249`). The `corvia serve` command will auto-provision corvia-inference when it sees `provider = "corvia"` (`crates/corvia-cli/src/main.rs:332-337`).

**Step 2: Commit**

```bash
git add corvia.toml
git commit -m "chore: default embedding provider to corvia-inference (D70)"
```

---

### Task 3: Update post-create.sh

**Files:**
- Modify: `.devcontainer/scripts/post-create.sh`

**Step 1: Replace the entire script**

```bash
#!/bin/bash
set -e

echo "=== Corvia Workspace: Post-Create Setup ==="

# Capture workspace root (where devcontainer mounts the workspace)
WORKSPACE_ROOT="$(pwd)"

# Build Corvia from source (repos cloned by workspace init)
echo "Initializing workspace..."
corvia workspace init 2>/dev/null || {
    echo "Corvia not installed — building from source..."
    git clone https://github.com/chunzhe10/corvia repos/corvia 2>/dev/null || true
    cd "$WORKSPACE_ROOT/repos/corvia"
    cargo install --path crates/corvia-cli
    cargo install --path crates/corvia-inference
    cd "$WORKSPACE_ROOT"
    corvia workspace init
}

# Install corvia-workspace toggle command
chmod +x .devcontainer/scripts/corvia-workspace.sh
ln -sf "$WORKSPACE_ROOT/.devcontainer/scripts/corvia-workspace.sh" /usr/local/bin/corvia-workspace

echo "=== Post-Create Complete ==="
echo "Run 'corvia-workspace status' to see available services."
echo "Run 'corvia-workspace enable ollama' for Ollama embeddings."
echo "Run 'corvia-workspace enable surrealdb' for SurrealDB FullStore."
```

Key changes:
- Removed `ollama serve &`, `ollama pull nomic-embed-text`, and the kill logic (lines 20-26 of original)
- Added `cargo install --path crates/corvia-inference` so the fastembed binary is available
- Added symlink install for `corvia-workspace` toggle command

**Step 2: Verify the script is syntactically valid**

Run: `bash -n /root/corvia-project/corvia-workspace/.devcontainer/scripts/post-create.sh`
Expected: No output (no syntax errors).

**Step 3: Commit**

```bash
git add .devcontainer/scripts/post-create.sh
git commit -m "chore: remove Ollama from post-create, add corvia-workspace install (D70)"
```

---

### Task 4: Update post-start.sh

**Files:**
- Modify: `.devcontainer/scripts/post-start.sh`

**Step 1: Replace the entire script**

```bash
#!/bin/bash
set -e

echo "=== Corvia Workspace: Starting Services ==="

WORKSPACE_ROOT="$(pwd)"
FLAGS_FILE="$WORKSPACE_ROOT/.devcontainer/.corvia-workspace-flags"

# Start Corvia server (always — uses corvia-inference automatically when provider=corvia)
corvia serve &
echo "Corvia server running on http://127.0.0.1:8020"

# Re-start any previously enabled optional services
if [ -f "$FLAGS_FILE" ]; then
    if grep -q "ollama=enabled" "$FLAGS_FILE"; then
        echo "Starting Ollama (previously enabled)..."
        ollama serve &
        for i in $(seq 1 30); do
            curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
            sleep 1
        done
    fi
    if grep -q "surrealdb=enabled" "$FLAGS_FILE"; then
        echo "Starting SurrealDB (previously enabled)..."
        docker compose -f "$WORKSPACE_ROOT/repos/corvia/docker/docker-compose.yml" up -d
    fi
fi

echo "Run 'corvia-workspace status' to check services."
```

Key changes:
- Removed unconditional `ollama serve &` and `sleep 2`
- `corvia serve` stays — it auto-provisions corvia-inference when `provider = "corvia"` (MCP always enabled)
- Added flags-file-driven restart for previously enabled optional services
- Uses readiness polling instead of `sleep` for Ollama (from devcontainer-improvements.md item 4)

**Step 2: Verify the script is syntactically valid**

Run: `bash -n /root/corvia-project/corvia-workspace/.devcontainer/scripts/post-start.sh`
Expected: No output (no syntax errors).

**Step 3: Commit**

```bash
git add .devcontainer/scripts/post-start.sh
git commit -m "chore: make post-start conditional on flags file (D70)"
```

---

### Task 5: Create the corvia-workspace toggle script

**Files:**
- Create: `.devcontainer/scripts/corvia-workspace.sh`

**Step 1: Write the toggle script**

```bash
#!/bin/bash
set -e

# corvia-workspace — toggle optional devcontainer services
# Usage: corvia-workspace {enable|disable|status} [ollama|surrealdb]

WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLAGS_FILE="$WORKSPACE_ROOT/.devcontainer/.corvia-workspace-flags"
CORVIA_TOML="$WORKSPACE_ROOT/corvia.toml"

# Ensure flags file exists with defaults
init_flags() {
    if [ ! -f "$FLAGS_FILE" ]; then
        cat > "$FLAGS_FILE" <<'EOF'
ollama=disabled
surrealdb=disabled
EOF
    fi
}

get_flag() {
    init_flags
    grep "^$1=" "$FLAGS_FILE" | cut -d= -f2
}

set_flag() {
    init_flags
    if grep -q "^$1=" "$FLAGS_FILE"; then
        sed -i "s/^$1=.*/$1=$2/" "$FLAGS_FILE"
    else
        echo "$1=$2" >> "$FLAGS_FILE"
    fi
}

# --- Ollama ---

enable_ollama() {
    echo "Enabling Ollama..."

    # Install if not present
    if ! command -v ollama &>/dev/null; then
        echo "  Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Start server if not running
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "  Starting Ollama server..."
        ollama serve &
        for i in $(seq 1 30); do
            curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
            sleep 1
        done
    fi

    # Pull model if needed
    if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo "  Pulling nomic-embed-text model..."
        ollama pull nomic-embed-text
    fi

    # Update corvia.toml
    sed -i 's/^provider = "corvia"/provider = "ollama"/' "$CORVIA_TOML"
    sed -i 's/^model = "nomic-embed-text-v1.5"/model = "nomic-embed-text"/' "$CORVIA_TOML"
    sed -i 's|^url = "http://127.0.0.1:8030"|url = "http://127.0.0.1:11434"|' "$CORVIA_TOML"

    set_flag ollama enabled
    echo "  Ollama enabled. Embedding provider switched to Ollama."
}

disable_ollama() {
    echo "Disabling Ollama..."

    # Stop server
    pkill -f "ollama serve" 2>/dev/null || true

    # Revert corvia.toml
    sed -i 's/^provider = "ollama"/provider = "corvia"/' "$CORVIA_TOML"
    sed -i 's/^model = "nomic-embed-text"$/model = "nomic-embed-text-v1.5"/' "$CORVIA_TOML"
    sed -i 's|^url = "http://127.0.0.1:11434"|url = "http://127.0.0.1:8030"|' "$CORVIA_TOML"

    set_flag ollama disabled
    echo "  Ollama disabled. Embedding provider switched to corvia-inference (ONNX)."
}

# --- SurrealDB ---

enable_surrealdb() {
    echo "Enabling SurrealDB..."

    local compose_file="$WORKSPACE_ROOT/repos/corvia/docker/docker-compose.yml"

    if [ ! -f "$compose_file" ]; then
        echo "  Error: docker-compose.yml not found at $compose_file"
        exit 1
    fi

    docker compose -f "$compose_file" up -d

    # Wait for readiness
    echo "  Waiting for SurrealDB..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
        sleep 1
    done

    # Update corvia.toml
    sed -i 's/^store_type = "lite"/store_type = "surrealdb"/' "$CORVIA_TOML"

    set_flag surrealdb enabled
    echo "  SurrealDB enabled. Storage switched to SurrealStore."
}

disable_surrealdb() {
    echo "Disabling SurrealDB..."

    local compose_file="$WORKSPACE_ROOT/repos/corvia/docker/docker-compose.yml"

    if [ -f "$compose_file" ]; then
        docker compose -f "$compose_file" down 2>/dev/null || true
    fi

    # Revert corvia.toml
    sed -i 's/^store_type = "surrealdb"/store_type = "lite"/' "$CORVIA_TOML"

    set_flag surrealdb disabled
    echo "  SurrealDB disabled. Storage switched to LiteStore."
}

# --- Status ---

show_status() {
    init_flags
    echo "=== Corvia Workspace Services ==="
    echo ""

    # Ollama
    local ollama_flag
    ollama_flag=$(get_flag ollama)
    local ollama_running="no"
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        ollama_running="yes"
    fi
    printf "  %-12s enabled=%-8s running=%s\n" "ollama" "$ollama_flag" "$ollama_running"

    # SurrealDB
    local surreal_flag
    surreal_flag=$(get_flag surrealdb)
    local surreal_running="no"
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        surreal_running="yes"
    fi
    printf "  %-12s enabled=%-8s running=%s\n" "surrealdb" "$surreal_flag" "$surreal_running"

    # Corvia server
    local corvia_running="no"
    if curl -sf http://localhost:8020/health >/dev/null 2>&1; then
        corvia_running="yes"
    fi
    printf "  %-12s enabled=%-8s running=%s\n" "corvia" "always" "$corvia_running"

    echo ""

    # Show current config
    local provider model store_type
    provider=$(grep '^provider' "$CORVIA_TOML" | head -1 | cut -d'"' -f2)
    model=$(grep '^model' "$CORVIA_TOML" | head -1 | cut -d'"' -f2)
    store_type=$(grep '^store_type' "$CORVIA_TOML" | head -1 | cut -d'"' -f2)
    echo "  Config: provider=$provider model=$model store=$store_type"
}

# --- Main ---

case "${1:-}" in
    enable)
        case "${2:-}" in
            ollama)    enable_ollama ;;
            surrealdb) enable_surrealdb ;;
            *)         echo "Usage: corvia-workspace enable {ollama|surrealdb}"; exit 1 ;;
        esac
        ;;
    disable)
        case "${2:-}" in
            ollama)    disable_ollama ;;
            surrealdb) disable_surrealdb ;;
            *)         echo "Usage: corvia-workspace disable {ollama|surrealdb}"; exit 1 ;;
        esac
        ;;
    status)
        show_status
        ;;
    *)
        echo "corvia-workspace — toggle optional devcontainer services"
        echo ""
        echo "Usage:"
        echo "  corvia-workspace enable {ollama|surrealdb}"
        echo "  corvia-workspace disable {ollama|surrealdb}"
        echo "  corvia-workspace status"
        exit 1
        ;;
esac
```

**Step 2: Make it executable**

Run: `chmod +x .devcontainer/scripts/corvia-workspace.sh`

**Step 3: Verify syntax**

Run: `bash -n /root/corvia-project/corvia-workspace/.devcontainer/scripts/corvia-workspace.sh`
Expected: No output (no syntax errors).

**Step 4: Test the help output**

Run: `bash /root/corvia-project/corvia-workspace/.devcontainer/scripts/corvia-workspace.sh`
Expected:
```
corvia-workspace — toggle optional devcontainer services

Usage:
  corvia-workspace enable {ollama|surrealdb}
  corvia-workspace disable {ollama|surrealdb}
  corvia-workspace status
```

**Step 5: Test the status command**

Run: `bash /root/corvia-project/corvia-workspace/.devcontainer/scripts/corvia-workspace.sh status`
Expected: Shows all services with enabled=disabled, running=no, and current config.

**Step 6: Commit**

```bash
git add .devcontainer/scripts/corvia-workspace.sh
git commit -m "feat: add corvia-workspace toggle for optional services (D70)"
```

---

### Task 6: Add flags file to .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add the flags file pattern**

Append to `.gitignore`:

```
# Optional service state (per-user, not shared)
.devcontainer/.corvia-workspace-flags
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore corvia-workspace flags file"
```

---

### Task 7: Smoke test the full flow

**Step 1: Run status to verify clean state**

Run: `cd /root/corvia-project/corvia-workspace && bash .devcontainer/scripts/corvia-workspace.sh status`
Expected: All services disabled, config shows `provider=corvia model=nomic-embed-text-v1.5 store=lite`.

**Step 2: Verify post-create.sh syntax**

Run: `bash -n .devcontainer/scripts/post-create.sh`
Expected: No errors.

**Step 3: Verify post-start.sh syntax**

Run: `bash -n .devcontainer/scripts/post-start.sh`
Expected: No errors.

**Step 4: Verify Dockerfile builds**

Run: `docker build -f .devcontainer/Dockerfile .devcontainer/ --no-cache 2>&1 | tail -3`
Expected: Build succeeds, no Ollama in output.

**Step 5: Final commit (squash if desired)**

If all tasks were committed individually, no action needed. Otherwise:

```bash
git add -A
git commit -m "feat: optional devcontainer services — light default with toggle (D70)

Ollama and SurrealDB are now opt-in via 'corvia-workspace enable/disable'.
Default uses corvia-inference (fastembed/ONNX) for CPU embeddings."
```
