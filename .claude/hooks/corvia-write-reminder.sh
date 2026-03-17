#!/bin/bash
# PostToolUse hook: remind to write decisions/learnings to corvia after git commits.
# Reads JSON from stdin (Claude Code hook protocol).

# Suppress all stderr — Claude Code treats any stderr as a hook error
exec 2>/dev/null

# Use jq to extract both fields in a single stdin read (avoids echo of large JSON)
eval "$(jq -r '
  @sh "COMMAND=\(.tool_input.command // "")",
  @sh "EXIT_CODE=\(.tool_response.exitCode // .tool_response.exit_code // 0)"
')" || exit 0

# Only trigger on successful git commit commands
case "$COMMAND" in
    *"git commit "*)
        [ "$EXIT_CODE" != "0" ] && exit 0
        echo "REMINDER: You just committed code. If this commit contains a design decision, architectural change, or notable learning, persist it with corvia_write (scope_id: corvia, agent_id: claude-code). Skip if the commit is trivial (typo fix, formatting, etc.)."
        ;;
esac

exit 0
