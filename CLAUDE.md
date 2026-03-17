# corvia-workspace — Claude Code

## STOP — Read this FIRST before doing ANYTHING

**Your FIRST tool call in EVERY conversation and EVERY new task MUST be a corvia MCP
tool** (`corvia_search`, `corvia_ask`, or `corvia_context`). This overrides your
default behavior of reaching for Grep/Glob/Read first.

**DO NOT use Grep, Glob, or code search as your first action.** Always query corvia
for project context before touching code. The order is:

1. `corvia_search` or `corvia_ask` — get project knowledge and prior decisions
2. THEN use Grep/Glob/Read for code-level details

This is the single most important instruction in this file. If you skip corvia and
go straight to code search, you are violating this project's workflow.

@AGENTS.md

## corvia MCP tool usage (detailed)

- Before writing or modifying code: `corvia_search` for prior decisions and patterns
- Before answering any question about the project: `corvia_ask` first
- Before designing a feature: `corvia_search` + `corvia_context` for existing context
- After making a design decision: `corvia_write` to record it for future sessions
- When exploring unfamiliar areas: `corvia_ask` before diving into code

**Do NOT skip corvia lookups to save time.** The knowledge base exists to prevent
re-discovering things that were already decided. Always check corvia first, then
use native tools (file read, grep, bash) for code-level details.

## Known workarounds (Claude Code specific)

### WSL memory leak from orphaned processes

Claude Code leaks memory in WSL via orphaned node processes that persist after
sessions close. A `SessionEnd` hook in `.claude/settings.json` auto-runs
`.devcontainer/scripts/cleanup-orphans.sh` to kill these orphans on exit.

- **Scope**: Claude Code on WSL only — not a corvia product concern
- **Script**: `.devcontainer/scripts/cleanup-orphans.sh` (throttled to once per 10min)
- **Manual run**: `bash .devcontainer/scripts/cleanup-orphans.sh`
- **Upstream**: https://github.com/anthropics/claude-code/issues
- **Remove when**: upstream fix lands in Claude Code

## Documentation Save Locations

- Product-specific designs and RFCs → `repos/corvia/docs/rfcs/`
- Workspace-level decisions → `docs/decisions/`
- Implementation plans → alongside their design doc in the repo
- Learnings → `docs/learnings/`
- Marketing content → `docs/marketing/`

Do NOT create `docs/superpowers/` — that path is blocked by enforcement hooks.

## Recording Decisions

Use `corvia_write` with `content_role` and `source_origin` params:
- corvia product decisions: `source_origin = "repo:corvia"`
- Workspace decisions: `source_origin = "workspace"`
