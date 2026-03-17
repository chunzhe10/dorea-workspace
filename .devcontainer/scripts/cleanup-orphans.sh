#!/bin/bash
# TEMPORARY WORKAROUND for Claude Code memory leak via orphaned processes in WSL.
# Upstream issue: https://github.com/anthropics/claude-code/issues
# Remove this script once the upstream fix lands.
#
# Called automatically by Claude Code SessionEnd hook, or manually:
#   bash .devcontainer/scripts/cleanup-orphans.sh
#
# Throttled: only runs once per 10 minutes to avoid redundant work.
# Only targets processes reparented to init (PID 1) — truly orphaned.

set -euo pipefail

QUIET="${1:-}"
log() { [ "$QUIET" = "--quiet" ] || echo "$*"; }

# ── Throttle: skip if we ran less than 10 minutes ago ──────────────
THROTTLE_FILE="/tmp/corvia-cleanup-orphans.last"
THROTTLE_SECONDS=600
if [ -f "$THROTTLE_FILE" ]; then
    last_run=$(cat "$THROTTLE_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    elapsed=$((now - last_run))
    if [ "$elapsed" -lt "$THROTTLE_SECONDS" ]; then
        log "cleanup: throttled (ran ${elapsed}s ago, next in $((THROTTLE_SECONDS - elapsed))s)"
        exit 0
    fi
fi
date +%s > "$THROTTLE_FILE"

killed=0

# Helper: kill a process if it's orphaned (PPID=1) and old enough
kill_orphan() {
    local pid="$1" min_age="${2:-600}" label="${3:-process}"
    local ppid etimes cmdline
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$ppid" = "1" ] || return 1
    etimes=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$etimes" ] && [ "$etimes" -ge "$min_age" ] || return 1
    cmdline=$(ps -o args= -p "$pid" 2>/dev/null || true)
    log "  killing orphaned $label: pid=$pid (uptime=${etimes}s)"
    log "    cmd: ${cmdline:0:120}"
    kill "$pid" 2>/dev/null && killed=$((killed + 1)) || true
}

# ── 1. Orphaned node processes from Claude Code (reparented to init) ──
# Only targets node processes whose parent is PID 1 (parent died = orphaned)
# and whose command line contains "claude". Skips anything under 10 minutes
# to avoid killing active subagents.
while IFS= read -r pid; do
    kill_orphan "$pid" 600 "claude node"
done < <(pgrep -f 'node.*claude' 2>/dev/null || true)

# ── 1b. Orphaned Vite dev servers ──
# corvia-dev restarts can leave stale vite processes that hold file watchers
# and memory. Only kill if orphaned (PPID=1) and older than 5 minutes.
while IFS= read -r pid; do
    kill_orphan "$pid" 300 "vite dev server"
done < <(pgrep -f 'node.*vite' 2>/dev/null || true)

# ── 1c. Stale debug inference processes ──
# Debug builds of corvia-inference can linger after test runs. Kill if
# orphaned and older than 10 minutes.
while IFS= read -r pid; do
    kill_orphan "$pid" 600 "debug inference"
done < <(pgrep -f 'target/debug/corvia-inference' 2>/dev/null || true)

# ── 1d. Orphaned Ollama processes ──
# corvia-dev restarts or post-start re-runs can leave stale ollama serve
# processes. Kill if orphaned (PPID=1) and older than 10 minutes.
while IFS= read -r pid; do
    kill_orphan "$pid" 600 "ollama serve"
done < <(pgrep -f 'ollama serve' 2>/dev/null || true)

# ── 2. Drop filesystem caches under memory pressure (WSL only) ──
# WSL's memory management benefits from explicit cache drops; native Linux does not.
if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
    mem_available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 1)
    if [ "$mem_total" -gt 0 ]; then
        pct_available=$((mem_available * 100 / mem_total))
        if [ "$pct_available" -lt 15 ]; then
            log "  memory pressure detected (${pct_available}% available) — dropping caches"
            sync
            echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
        fi
    fi
fi

if [ "$killed" -gt 0 ]; then
    log "cleaned up $killed orphaned process(es)"
else
    log "no orphaned processes found"
fi
