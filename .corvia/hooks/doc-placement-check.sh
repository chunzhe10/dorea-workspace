#!/usr/bin/env bash
set -euo pipefail
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.file_path // empty')
[ -z "$FILE_PATH" ] && exit 0
case "$FILE_PATH" in *.md|*.mdx|*.rst) ;; *) exit 0 ;; esac
case "$FILE_PATH" in
  docs/superpowers/*)
    echo "BLOCKED: '$FILE_PATH' matches blocked path 'docs/superpowers/*'." >&2
    exit 2
    ;;
  repos/*/docs/*|docs/decisions/*|docs/learnings/*|docs/marketing/*|docs/plans/*)
    exit 0 ;;
esac
echo "NOTE: '$FILE_PATH' is in an unusual location for docs." >&2
exit 0
