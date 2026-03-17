#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# Prevent duplicate/concurrent runs.
# The lock dir prevents concurrent execution; the done marker prevents re-runs
# after a successful completion (VS Code fires postStartCommand on each connect).
LOCK_FILE="/tmp/corvia-post-start.lock"
# Use boot ID so the done marker is invalidated on container restart
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo "unknown")
DONE_MARKER="/tmp/corvia-post-start.done"

if [ -f "$DONE_MARKER" ] && [ "$(cat "$DONE_MARKER" 2>/dev/null)" = "$BOOT_ID" ]; then
    echo "post-start.sh already completed this boot. Skipping."
    exit 0
fi
if ! mkdir "$LOCK_FILE" 2>/dev/null; then
    echo "post-start.sh is already running (lock: $LOCK_FILE). Skipping."
    exit 0
fi
trap 'rmdir "$LOCK_FILE" 2>/dev/null' EXIT

step() { printf " => %s\n" "$*"; }
done_msg() { printf "    ... done\n"; }
skip_msg() { printf "    ... skipped (%s)\n" "$*"; }
fail_msg() { printf "    ... FAILED (%s)\n" "$*" >&2; }

FLAGS_FILE="$WORKSPACE_ROOT/.devcontainer/.corvia-workspace-flags"

export TZ=Asia/Kuala_Lumpur

echo "=== Corvia Workspace: post-start ==="

# ── 0/5 ───────────────────────────────────────────────────────────────
# Intel iGPU: ensure /dev/dri/by-path symlinks exist.
# The NEO compute runtime discovers GPUs by scanning by-path.
# Docker device passthrough creates /dev/dri but may not populate by-path
# for integrated GPUs. Without this, OpenCL/OpenVINO see 0 platforms.
if [ -d /dev/dri ] && [ -d /dev/dri/by-path ]; then
    for card_dir in /sys/class/drm/card*/device; do
        card_name="$(basename "$(dirname "$card_dir")")"
        vendor="$(cat "$card_dir/vendor" 2>/dev/null || true)"
        [ "$vendor" = "0x8086" ] || continue  # Intel only

        pci_slot="$(cat "$card_dir/uevent" 2>/dev/null | grep PCI_SLOT_NAME | cut -d= -f2 || true)"
        [ -n "$pci_slot" ] || continue

        # Find the renderD node for this card
        render_node=""
        for rd in /sys/class/drm/renderD*/device; do
            rd_vendor="$(cat "$rd/vendor" 2>/dev/null || true)"
            rd_device="$(cat "$rd/device" 2>/dev/null || true)"
            card_device="$(cat "$card_dir/device" 2>/dev/null || true)"
            if [ "$rd_vendor" = "$vendor" ] && [ "$rd_device" = "$card_device" ]; then
                render_node="$(basename "$(dirname "$rd")")"
                break
            fi
        done

        # Create by-path symlinks if missing
        card_link="/dev/dri/by-path/pci-${pci_slot}-card"
        render_link="/dev/dri/by-path/pci-${pci_slot}-render"
        if [ ! -L "$card_link" ] && [ -e "/dev/dri/$card_name" ]; then
            ln -sf "../$card_name" "$card_link"
        fi
        if [ -n "$render_node" ] && [ ! -L "$render_link" ] && [ -e "/dev/dri/$render_node" ]; then
            ln -sf "../$render_node" "$render_link"
        fi
    done
fi

# ── 1/5 ───────────────────────────────────────────────────────────────
step "Forwarding host authentication"
forward_host_auth

# ── 2/5 ───────────────────────────────────────────────────────────────
step "Ensuring GPU provider libraries"
ensure_ort_provider_libs

# ── 3/5 ───────────────────────────────────────────────────────────────
step "Starting corvia-dev services"
if ! command -v corvia-dev >/dev/null 2>&1; then
    echo "    corvia-dev not found — running ensure_tooling"
    ensure_tooling
fi
if command -v corvia-dev >/dev/null 2>&1; then
    if corvia-dev status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('manager',{}).get('state')=='running' else 1)" 2>/dev/null; then
        echo "    manager already running"
    else
        corvia-dev up --no-foreground
        echo "    manager started"
    fi
else
    fail_msg "corvia-dev not available — run ensure_tooling manually"
fi

printf "    waiting for MCP server (port 8020)"
mcp_ready=false
for _attempt in $(seq 1 30); do
    if curl -sf --max-time 2 -o /dev/null http://127.0.0.1:8020/mcp 2>/dev/null; then
        done_msg
        mcp_ready=true
        break
    fi
    printf "."
    sleep 2
done
if [ "$mcp_ready" = false ]; then
    fail_msg "not ready after 60s — check 'corvia-dev logs corvia-server'"
fi

# Ensure dashboard dependencies are installed
DASHBOARD_DIR="$WORKSPACE_ROOT/tools/corvia-dashboard"
if [ -f "$DASHBOARD_DIR/package.json" ] && [ ! -d "$DASHBOARD_DIR/node_modules" ]; then
    printf "    installing dashboard dependencies"
    (cd "$DASHBOARD_DIR" && npm install --no-fund --no-audit) >/dev/null 2>&1 && done_msg || fail_msg "npm install"
fi

printf "    waiting for dashboard (port 8021)"
dash_ready=false
for _attempt in $(seq 1 15); do
    if curl -sf --max-time 2 -o /dev/null http://127.0.0.1:8021/ 2>/dev/null; then
        done_msg
        dash_ready=true
        break
    fi
    printf "."
    sleep 2
done
if [ "$dash_ready" = false ]; then
    fail_msg "not ready after 30s — check 'corvia-dev logs corvia-dashboard'"
fi

# ── 4/5 ───────────────────────────────────────────────────────────────
step "Claude Code integration"
# MCP server is configured via .mcp.json in the workspace root (checked into git).
# No need to call 'claude mcp add' — Claude Code reads .mcp.json directly.
MCP_JSON="$WORKSPACE_ROOT/.mcp.json"
if [ -f "$MCP_JSON" ] && python3 -c "import json; d=json.load(open('$MCP_JSON')); assert 'corvia' in d.get('mcpServers',{})" 2>/dev/null; then
    echo "    MCP server configured via .mcp.json"
else
    echo "    writing .mcp.json"
    python3 -c "
import json, os
p = '$MCP_JSON'
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault('mcpServers', {})['corvia'] = {'type': 'http', 'url': 'http://127.0.0.1:8020/mcp'}
json.dump(d, open(p, 'w'), indent=2)
print('    MCP server added to .mcp.json')
"
fi

# Ensure Claude Code workspace settings enable MCP servers from .mcp.json.
# settings.local.json is gitignored — recreate it if missing or incomplete.
SETTINGS_LOCAL="$WORKSPACE_ROOT/.claude/settings.local.json"
if [ ! -f "$SETTINGS_LOCAL" ] || ! python3 -c "import json; d=json.load(open('$SETTINGS_LOCAL')); assert 'enabledMcpjsonServers' in d" 2>/dev/null; then
    echo "    writing .claude/settings.local.json"
    python3 -c "
import json, os
p = '$SETTINGS_LOCAL'
d = json.load(open(p)) if os.path.exists(p) else {}
d['enabledMcpjsonServers'] = ['corvia', 'playwright', 'davinci-resolve-mcp']
json.dump(d, open(p, 'w'), indent=2)
"
else
    echo "    settings.local.json already configured"
fi

# Install superpowers plugin via direct git clone (no claude CLI dependency)
printf "    superpowers plugin: "
install_claude_plugin "https://github.com/obra/superpowers.git" superpowers claude-plugins-official \
    || fail_msg "git clone failed — check network connectivity"

# Activate dorea Python venv for all shells
if [ -d /opt/dorea-venv ]; then
    grep -q 'dorea-venv' /root/.bashrc 2>/dev/null || \
        echo 'source /opt/dorea-venv/bin/activate' >> /root/.bashrc
fi

# ── 5/5 ───────────────────────────────────────────────────────────────
step "Optional services"
if [ -f "$FLAGS_FILE" ]; then
    if grep -q "coding-llm=enabled" "$FLAGS_FILE"; then
        # coding-llm depends on ollama (corvia-dev starts it via dependency chain).
        # Here we only pull the coding model that corvia-dev doesn't handle.
        if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            if ! ollama list 2>/dev/null | grep -q "qwen2.5-coder:7b"; then
                printf "    pulling qwen2.5-coder:7b (first run only)"
                ollama pull qwen2.5-coder:7b >/dev/null 2>&1 && done_msg || fail_msg "pull failed"
            else
                echo "    ollama: qwen2.5-coder:7b already present"
            fi
        else
            fail_msg "ollama not reachable on :11434 — check 'corvia-dev logs ollama'"
        fi
    fi
    if ! grep -qE "(coding-llm)=enabled" "$FLAGS_FILE"; then
        echo "    none enabled"
    fi
else
    echo "    none enabled"
fi

echo "$BOOT_ID" > "$DONE_MARKER"

echo ""
echo "Ready. Run 'corvia-dev status' to check services."
