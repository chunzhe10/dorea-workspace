#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# Prevent duplicate runs when multiple VS Code clients connect simultaneously.
LOCK_FILE="/tmp/corvia-post-create.lock"
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo "unknown")
DONE_MARKER="/tmp/corvia-post-create.done"

if [ -f "$DONE_MARKER" ] && [ "$(cat "$DONE_MARKER" 2>/dev/null)" = "$BOOT_ID" ]; then
    echo "post-create.sh already completed this boot. Skipping."
    exit 0
fi
if ! mkdir "$LOCK_FILE" 2>/dev/null; then
    echo "post-create.sh is already running (lock: $LOCK_FILE). Skipping."
    exit 0
fi
trap 'rmdir "$LOCK_FILE" 2>/dev/null' EXIT

step() { printf " => %s\n" "$*"; }

echo "=== Corvia Workspace: post-create ==="

step "Waiting for network"
wait_for_network || exit 1

# Forward gh auth first — binary download uses gh if available
step "Forwarding GitHub credentials"
forward_gh_auth

step "Installing Corvia binaries"
retry 3 install_binaries

step "Initializing workspace"
init_workspace

step "Ingesting workspace into knowledge base"
spin "Ingesting repos..." corvia workspace ingest \
    || echo "    ingest failed — run 'corvia workspace ingest' manually to populate knowledge base"

step "Ensuring tooling"
ensure_tooling

step "Installing VS Code extensions"
EXT_DIR="$WORKSPACE_ROOT/.devcontainer/extensions/corvia-services"
VSIX="$EXT_DIR/corvia-services-$(python3 -c "import json; print(json.load(open('$EXT_DIR/package.json'))['version'])").vsix"
if [ -f "$VSIX" ]; then
    install_vsix_direct "$VSIX"
elif [ -f "$EXT_DIR/package.json" ] && command -v vsce >/dev/null 2>&1; then
    printf "    building extension"
    if (cd "$EXT_DIR" && vsce package --no-dependencies) >/dev/null 2>&1; then
        echo " done"
        VSIX=$(ls -t "$EXT_DIR"/*.vsix 2>/dev/null | head -1)
        [ -n "$VSIX" ] && install_vsix_direct "$VSIX"
    else
        echo " FAILED"
    fi
else
    echo "    no .vsix found — build with: cd $EXT_DIR && vsce package --no-dependencies"
fi

echo "$BOOT_ID" > "$DONE_MARKER"

echo ""
echo "=== post-create complete ==="
echo "Run 'corvia-dev status' to see available services."
echo "Run 'corvia-dev use ollama' to switch to Ollama embeddings."
echo "Run 'corvia-dev enable coding-llm' to enable local coding LLM."
