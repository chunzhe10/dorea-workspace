#!/usr/bin/env bash
# Display-only reminder for agent identity selection.
# Runs on Claude Code SessionStart -- does NOT require interaction.
# See: docs/plans/2026-03-13-dashboard-ux-overhaul-design.md Section 2.2

CORVIA_API="${CORVIA_API:-http://localhost:8020}"

if [ -n "$CORVIA_AGENT_ID" ]; then
    echo "Connected as: $CORVIA_AGENT_ID"
    exit 0
fi

# Check for reconnectable agents (fail silently if server is down)
COUNT=$(curl -s --max-time 2 "$CORVIA_API/api/dashboard/agents/reconnectable" 2>/dev/null | jq 'length' 2>/dev/null)

if [ "$COUNT" -gt "0" ] 2>/dev/null; then
    echo "Run 'corvia agent connect' to select an identity ($COUNT agents available)"
else
    echo "No agent identity set. Run 'corvia agent connect' to register."
fi
