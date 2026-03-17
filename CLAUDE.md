# dorea-workspace — Claude Code

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

## Dorea-Specific Context

### Project
Dorea is an automated underwater video AI editing pipeline. See
`repos/dorea/README.md` for the full overview.

### Architecture Document
The pipeline architecture is defined in `underwater_pipeline_architecture.docx`.
Always consult this document (via corvia — it should be ingested) before making
pipeline design decisions.

### GPU Constraint
This workstation has an RTX 3060 with **6GB VRAM**. Only one AI model may be loaded
at a time. Pipeline scripts enforce sequential GPU scheduling. Corvia inference runs
on CPU (`inference.device = "cpu"` in corvia.toml) to avoid VRAM contention.

### DaVinci Resolve
Resolve runs on the **host machine**, not in the devcontainer. The
`davinci-resolve-mcp` server bridges Claude Desktop to Resolve via
`host.docker.internal:9090`. Script `05_resolve_setup.py` must run on the host
(it imports `fusionscript` which requires a running Resolve instance).

## Known workarounds (Claude Code specific)

### WSL memory leak from orphaned processes

Claude Code leaks memory in WSL via orphaned node processes that persist after
sessions close. A `SessionEnd` hook in `.claude/settings.json` auto-runs
`.devcontainer/scripts/cleanup-orphans.sh` to kill these orphans on exit.

- **Scope**: Claude Code on WSL only — not a dorea concern
- **Script**: `.devcontainer/scripts/cleanup-orphans.sh` (throttled to once per 10min)
- **Remove when**: upstream fix lands in Claude Code

## Documentation Save Locations

- Pipeline designs and decisions → `docs/decisions/`
- Implementation plans → `docs/plans/`
- Learnings → `docs/learnings/`

Do NOT create `docs/superpowers/` — that path is blocked by enforcement hooks.

## Recording Decisions

Use `corvia_write` with `content_role` and `source_origin` params:
- Pipeline decisions: `source_origin = "repo:dorea"`
- Workspace decisions: `source_origin = "workspace"`
