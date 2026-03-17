# Docs Workflow & Knowledge Dashboard

**Date:** 2026-03-13
**Status:** Active
**Supersedes:** None

## Decision

Multi-repo doc ownership with corvia as the aggregation layer and the dashboard as the
primary human read interface. Each repo owns its product-specific docs (stored in the
repo's `docs/` directory). The workspace owns cross-cutting decisions (in `docs/decisions/`
and `docs/learnings/`). corvia aggregates all sources into a unified, searchable,
graph-navigable knowledge base with `content_role` and `source_origin` metadata on every
entry.

## Context

94+ doc files scattered across 5+ directories (`docs/plans/`, `repos/corvia/docs/rfcs/`,
`.agents/skills/`, Claude memory, workspace docs) with no unified management. 3,581
knowledge entries stored as UUID-named JSON -- unreadable by humans. No way to filter
search by content type (design vs code vs memory) or origin (which repo). AI tools
frequently saved docs to wrong locations (e.g., `docs/superpowers/` instead of the
relevant repo's `docs/` directory).

## Consequences

- `EntryMetadata` extended with `content_role` (design, decision, plan, code, memory,
  finding, instruction, learning) and `source_origin` (repo:corvia, workspace, memory).
- `corvia_write` and `corvia_search` MCP tools accept optional `content_role` and
  `source_origin` filter parameters. Dashboard filter dropdowns map to these API params.
- Dashboard redesigned as an interactive graphed document reader (Heptabase + Neo4j Bloom
  patterns): split-panel with graph navigation on the left and content reader on the
  right. Nodes styled by role (color), connection count (size), and origin (shape).
- `corvia docs check` CLI command extends the Reasoner with three new check types:
  `MisplacedDoc`, `TemporalContradiction`, and `CoverageGap`.
- Defense in depth: PreToolUse hooks block wrong write paths (real-time), CLAUDE.md
  guides correct behavior (soft), aggregator detects drift (periodic).
- `DocsConfig` and `DocsRulesConfig` structs in `corvia-common/src/config.rs`,
  configured via `[workspace.docs]` section in `corvia.toml`.
- Phased delivery: metadata extensions -> dashboard reader -> reasoner checks -> hooks.

**Source:** `docs/decisions/2026-03-13-docs-workflow-design.md`
