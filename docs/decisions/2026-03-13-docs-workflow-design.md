# Docs Workflow & Interactive Knowledge Dashboard — Design Spec

**Date:** 2026-03-13
**Status:** Approved design
**Scope:** corvia product feature + workspace configuration
**Approach:** B (Full Loop) — end-to-end write→store→dashboard read loop

---

## Problem Statement

AI coding tools (superpowers, Claude Code memory) and human developers create documentation
in 8+ scattered locations with no unified read or management layer. corvia ingests source
code but treats all content uniformly — no content-role awareness, no repo ownership tracking,
no human-readable interface beyond file browsing.

**Current state:**
- 75+ doc files across `docs/plans/`, `docs/superpowers/`, `repos/corvia/docs/rfcs/`,
  `.agents/skills/`, Claude memory (`~/.claude/projects/`)
- 3,581 knowledge entries in `.corvia/` as UUID-named JSON — unreadable by humans
- No way to filter search by "show me only design docs" or "show me only corvia repo knowledge"
- No detection of stale, misplaced, or duplicate docs across sources

## Design Principles

1. **Each repo owns its docs.** Product decisions live with the product repo. The workspace
   has its own docs for cross-cutting decisions. corvia aggregates all of them.
2. **Dashboard is the primary human read interface.** Humans don't browse `.corvia/` JSON or
   30+ RFC files. The dashboard renders knowledge as readable documents with graph navigation.
3. **Git is truth.** Markdown files in repos are the source of truth for curated docs. The
   knowledge store is a derived index that corvia rebuilds from files.
4. **Defense in depth.** Hooks prevent wrong writes (real-time). CLAUDE.md guides correct
   behavior (soft). Aggregator detects drift (periodic).

## Architecture

```
WRITE PATH (knowledge goes in)
─────────────────────────────────────────────────
Agent decision     → corvia_write(source_origin: "repo:corvia") → knowledge store
Human edits docs   → git commit → post-commit hook → incremental re-index
Code changes       → corvia ingest → chunked + embedded → knowledge store
Superpowers spec   → hook enforces path → saved to repo docs/ → ingested

READ PATH (knowledge comes out)
─────────────────────────────────────────────────
Quick orientation  → repo's README.md / ARCHITECTURE.md (curated, few files)
Deep exploration   → corvia dashboard (graphed document reader)
Agent queries      → corvia_search / corvia_ask via MCP (with filters)
Periodic health    → corvia docs check (Aggregator findings in dashboard)
```

### Multi-Repo Doc Ownership Model

```
┌─────────────────────────────────────────────────────┐
│                 corvia (aggregator)                  │
│   unified search · cross-repo linking · reasoner    │
│   "show me all auth decisions across every repo"    │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
    ┌──────▼──────┐            ┌──────▼──────┐
    │ repos/corvia│            │ repos/foo   │
    │ docs/       │            │ docs/       │
    │ (product    │            │ (product    │
    │  decisions) │            │  decisions) │
    └─────────────┘            └─────────────┘
           ▲                          ▲
    ┌──────┴──────┐            ┌──────┴──────┐
    │  workspace  │            │  .memory/   │
    │  docs/      │            │  (Claude    │
    │  (cross-repo│            │   memory,   │
    │   decisions)│            │   git-tracked)
    └─────────────┘            └─────────────┘
```

---

## Section 1: Dashboard — Interactive Graphed Document Reader

### UX Model

Based on research across 12 knowledge management tools (Obsidian, Roam, Logseq, Neo4j Bloom,
Heptabase, Scrintal, TheBrain, Kumu, Graphistry, GitHub, Dendron, Foam). Full research at
`docs/decisions/graph-document-ux-research.md`.

**Target zone:** Heptabase pattern (graph as first-class navigation with seamless content
access via side panel), enhanced with Neo4j Bloom's inspector pattern (navigable neighbor
cards in the detail panel).

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ corvia                    [search ________] [repo ▾] [role ▾]   │
├──────────────────────────────────────────────────────────────────┤
│                           │                                      │
│   Interactive Graph       │   Document Reader (side panel)       │
│                           │                                      │
│      ●──────●             │   ┌─ Adapter Plugin System ────────┐│
│     /│\    /              │   │ repo:corvia · design · Mar 3   ││
│    ● ● ●  ●              │   │                                 ││
│    │   │ /│\              │   │ We chose JSONL over gRPC for   ││
│    ●   ● ● ●             │   │ adapter IPC because adapters    ││
│                           │   │ can be written in any language..││
│   [depth: ●───○ 2 hops]  │   │                                 ││
│                           │   ├─ Connected Knowledge ──────────┤│
│   Legend:                 │   │ ● adapter_protocol.rs    code  ││
│   ● design  ● code       │   │ ● process_adapter.rs     code  ││
│   ● plan    ● memory     │   │ ● M3.3 chunking impl     plan  ││
│   ● finding               │   │ ● D72-D79 decisions   decision ││
│                           │   └────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Core Interactions

| Interaction | Behavior | Source pattern |
|------------|----------|---------------|
| Click node | Content loads in right panel, graph re-centers on node | Heptabase |
| Hover node | Connected nodes highlight, unrelated nodes fade to low opacity | Obsidian |
| Click neighbor card in panel | Navigates graph + updates panel (graph traversal from reader) | Neo4j Bloom |
| Depth slider | Controls graph expansion radius (1-3 hops) | Obsidian |
| Search | Highlights matching nodes in graph, lists results in panel | Neo4j Bloom |
| Filter dropdowns | Filter by repo (`source_origin`) and type (`content_role`) | Custom |

### Node Styling (data-driven)

| Visual property | Driven by | Example |
|----------------|-----------|---------|
| Color | `content_role` | design=blue, code=green, plan=orange, memory=purple, finding=red |
| Size | Connection count | More connections = larger node |
| Shape | `source_origin` | Circle=repo, diamond=workspace, triangle=memory |
| Edge color | Relation type | imports=gray, references=blue, contains=green |

### Anti-Patterns Avoided

- Click node NEVER replaces graph view (Logseq failure mode)
- Graph is navigable, not decorative (Obsidian complaint)
- Nodes visually differentiated by role (not uniform circles)
- Graph state persists across navigation (no "lost my place")
- Hover preview before committing to navigation

### Replaces

The current `GraphView.tsx` (754 lines, cluster-based force simulation) is replaced by a
new split-panel graphed document reader. This is closer to a rewrite than an evolution —
the current component groups nodes into clusters and has no individual-entry interaction.

**Technical approach:**
- **New components:** `DocumentReader.tsx` (side panel), `EntryGraph.tsx` (node-level graph),
  `NeighborCards.tsx` (clickable related entries), `GraphControls.tsx` (depth slider, filters)
- **Cluster view:** Preserved as the initial zoom level. Clicking into a cluster transitions
  to the entry-level graph for that cluster's entries.
- **API endpoints needed:**
  - `GET /v1/entries/{id}` — fetch full entry content for the reader panel (new)
  - `GET /v1/entries/{id}/neighbors` — fetch graph-expanded neighbor entries (new, wraps existing `corvia_graph`)
  - Existing `GET /v1/search` — extended with `content_role` and `source_origin` filter params
- **Estimated scope:** 4-5 new React components, 2 new REST endpoints, refactored graph
  rendering from cluster-level to entry-level with cluster as a zoom mode.

---

## Section 2: Metadata Extensions & Repo-Aware `corvia_write`

### Extended `EntryMetadata`

```rust
pub struct EntryMetadata {
    // Existing fields (unchanged)
    pub source_file: Option<String>,
    pub language: Option<String>,
    pub chunk_type: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,

    // New — docs workflow
    #[serde(default)]
    pub content_role: Option<String>,    // "design", "decision", "plan", "code",
                                         // "memory", "finding", "instruction",
                                         // "learning"
    #[serde(default)]
    pub source_origin: Option<String>,   // "repo:corvia", "repo:foo",
                                         // "workspace", "memory"
}
```

**Serde compatibility:** `#[serde(default)]` on the new fields ensures existing serialized
entries in `.corvia/knowledge/` deserialize correctly (missing fields default to `None`).

**`content_role` values:**
- `"design"` — full design specification or RFC
- `"decision"` — point-in-time architectural choice (lighter than a full design doc)
- `"plan"` — implementation plan or task breakdown
- `"code"` — source code chunk
- `"memory"` — agent session context (Claude memory, etc.)
- `"finding"` — Reasoner-generated health check finding
- `"instruction"` — AGENTS.md, CLAUDE.md, configuration docs
- `"learning"` — operational knowledge, troubleshooting notes

Arbitrary strings are accepted for forward compatibility, but the above values are
the recognized set used for dashboard filtering and node styling.

**DTOs requiring update:** The following response types must thread through the new fields:
- `SearchResultDto` in `crates/corvia-server/src/rest.rs`
- MCP search response formatting in `crates/corvia-server/src/mcp.rs`
- MCP write response (should echo back stored `content_role` and `source_origin`)
```

### Population Rules

| Write path | `content_role` | `source_origin` |
|-----------|---------------|-----------------|
| `corvia ingest` (code files) | Inferred: `.rs`/`.py`/`.ts` → `"code"` | From repo path: `"repo:<name>"` |
| `corvia ingest` (markdown in repo docs/) | Inferred from dir: `rfcs/` → `"design"`, `plans/` → `"plan"` | `"repo:<name>"` |
| `corvia ingest` (workspace docs/) | Inferred from dir: `decisions/` → `"design"`, `learnings/` → `"learning"` | `"workspace"` |
| `corvia ingest` (.memory/) | `"memory"` | `"memory"` |
| `corvia_write` MCP tool | New optional param, agent provides | New optional param, agent provides |
| Reasoner findings | `"finding"` | `"workspace"` |

### `corvia_write` MCP Changes

New optional parameters (backward compatible):

```json
{
  "name": "corvia_write",
  "arguments": {
    "content": "We chose JSONL over gRPC for adapter IPC because...",
    "scope_id": "corvia",
    "content_role": "decision",
    "source_origin": "repo:corvia"
  }
}
```

Defaults: `content_role` = null (unclassified), `source_origin` = `"workspace"`.

### `corvia_search` Filtering

New optional filter parameters on search and context MCP tools:

```json
{
  "name": "corvia_search",
  "arguments": {
    "query": "adapter IPC protocol",
    "scope_id": "corvia",
    "content_role": "design",
    "source_origin": "repo:corvia"
  }
}
```

Dashboard filter dropdowns map directly to these API params.

### `QueryableStore` Trait Change

The existing `search` method signature:
```rust
async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize) -> Result<Vec<SearchResult>>;
```

Must be extended to accept optional metadata filters. Two approaches:

**Option A (recommended): Post-filter.** Keep the trait signature unchanged. After vector
search returns top-N results, filter by `content_role` and `source_origin` in the calling
code. Simple, no backend changes required. Trade-off: may return fewer than `limit` results
if many are filtered out. Mitigate by over-fetching (request `limit * 3`, then filter).

**Option B: Backend-native filtering.** Add a `SearchFilter` struct to the trait:
```rust
pub struct SearchFilter {
    pub content_role: Option<String>,
    pub source_origin: Option<String>,
}
async fn search(&self, embedding: &[f32], scope_id: &str, limit: usize,
                filter: Option<SearchFilter>) -> Result<Vec<SearchResult>>;
```
Requires changes to all three backends (LiteStore, SurrealStore, PostgresStore).
SurrealDB/Postgres can push filters into the query. LiteStore falls back to post-filter.

Phase 1 should implement Option A (post-filter) for speed. Option B can be added later
as a performance optimization if filter selectivity causes result count issues.

The `RetrievalOpts` struct in `rag_types.rs` also needs the filter fields for RAG pipeline
passthrough.

---

## Section 3: Git-Triggered Re-Indexing

### Mechanism

Git post-commit hook triggers incremental re-indexing of changed files.

```bash
# repos/<repo>/.git/hooks/post-commit
# Generated by: corvia workspace init --hooks
changed=$(git diff-tree --no-commit-id --name-only -r HEAD)
docs_changed=$(echo "$changed" | grep -E '\.(md|toml|yaml|json|rs|py|ts|js)$')
if [ -n "$docs_changed" ]; then
    corvia ingest --incremental --files "$docs_changed" &
fi
```

### New CLI Flag

`corvia ingest --incremental --files <file1> [<file2> ...]`

Accepts one or more file paths as positional arguments after `--files`. The post-commit
hook passes newline-separated `git diff-tree` output which the shell splits naturally.

- Re-indexes only the listed files instead of full directory walk
- Finds existing entries where `source_file` matches
- Sets `valid_to` on old entries, creates new entries with `superseded_by` link
- Re-embeds only changed content (skips unchanged)
- Temporal supersession chain preserved — `corvia history` shows the edit trail

### Multi-Repo

`corvia workspace init --hooks` installs post-commit hooks in all configured repos from
`corvia.toml`. Each repo triggers re-indexing independently.

### Fallback

If hooks aren't installed or fail silently, `corvia workspace ingest` (full batch) works
as today. Hooks are an optimization, not a requirement.

---

## Section 4: Hook Enforcement & `corvia docs check`

### Layer 1: PreToolUse Hook (Real-Time Prevention)

Generated by `corvia workspace init --hooks`. Intercepts `Write` and `Edit` tool calls
in Claude Code.

```json
{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "command",
    "command": "bash .corvia/hooks/doc-placement-check.sh"
  }]
}
```

The validation script reads the file path from stdin and applies rules from `corvia.toml`:

| Path pattern | Action | Reason |
|-------------|--------|--------|
| `docs/superpowers/*` | **Block** (exit 2) | Superpowers default path — redirect to repo docs |
| `repos/*/docs/*` | Allow | Correct repo-owned location |
| `docs/decisions/*`, `docs/learnings/*`, `docs/marketing/*` | Allow | Valid workspace locations |
| Other `*.md` in unexpected location | **Warn** (additionalContext) | Soft guidance |

### Layer 2: `corvia docs check` (Periodic Aggregator)

New CLI command extending the existing Reasoner:

```bash
$ corvia docs check
```

#### New Reasoner Check Types

| Check | Detects | Method |
|-------|---------|--------|
| `MisplacedDoc` | Doc in wrong directory for its `content_role` | Compare path against `corvia.toml` rules |
| `TemporalContradiction` | Two entries with conflicting claims at overlapping valid time ranges | Extension of existing `Contradiction` check: same high-similarity filter, but additionally requires overlapping `valid_from`/`valid_to` windows AND different `source_origin` values. The existing `Contradiction` compares `source_version`; this compares temporal ranges across origins. |
| `CoverageGap` | Topic has agent memory but no formal design doc | Cluster by embedding, flag clusters with only `memory` role |

These extend existing checks (`StaleEntry`, `BrokenChain`, `Contradiction`, `SemanticGap`)
using the same `Finding` → `KnowledgeEntry` pipeline. Dashboard surfaces findings in a
dedicated view.

#### Optional Auto-Fix

`corvia docs check --fix` proposes moves for misplaced files and requires explicit `--yes`
to execute. Safety invariants:
- Refuses to run if the git working tree has uncommitted changes (clean tree required)
- Skips moves where target path already has a file (reports conflict, does not overwrite)
- Each move creates a separate git commit with message: `docs(migrate): move <file> to <target>`
- Prints a dry-run summary first, waits for `--yes` confirmation

---

## Section 5: Configuration

### `corvia.toml` Additions

```toml
[workspace.docs]
memory_dir = ".memory"
workspace_docs = "docs"
allowed_workspace_subdirs = ["decisions", "learnings", "marketing"]

[workspace.docs.rules]
blocked_paths = ["docs/superpowers/*"]
repo_docs_pattern = "docs/"
```

**Rust struct needed** (new in `crates/corvia-common/src/config.rs`):
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsConfig {
    #[serde(default)]
    pub memory_dir: Option<String>,
    #[serde(default)]
    pub workspace_docs: Option<String>,
    #[serde(default)]
    pub allowed_workspace_subdirs: Vec<String>,
    #[serde(default)]
    pub rules: Option<DocsRulesConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocsRulesConfig {
    #[serde(default)]
    pub blocked_paths: Vec<String>,
    #[serde(default)]
    pub repo_docs_pattern: Option<String>,
}
```

Added as `pub docs: Option<DocsConfig>` on `WorkspaceConfig`. The hook generator reads
`blocked_paths` globs and converts them to shell case patterns in the validation script.

### `CLAUDE.md` Additions

```markdown
## Documentation Save Locations

When using superpowers brainstorming or writing-plans skills:
- Product-specific designs → relevant repo's docs/ directory
- Workspace-level decisions → docs/decisions/
- Implementation plans → alongside their design doc in the repo
- Learnings → docs/learnings/

Do NOT save to docs/superpowers/specs/ or docs/superpowers/plans/.

## Recording Decisions

Use corvia_write with content_role and source_origin params.
- corvia product decisions: source_origin = "repo:corvia"
- Workspace decisions: source_origin = "workspace"
```

### `.claude/settings.json` Changes

```json
{
  "autoMemoryDirectory": "/workspaces/corvia-workspace/.memory",
  "hooks": {
    "PreToolUse": [
      { "matcher": "Grep|Glob", "hooks": [{"type": "command", "command": "...existing..."}] },
      { "matcher": "Write|Edit", "hooks": [{"type": "command", "command": "bash .corvia/hooks/doc-placement-check.sh"}] }
    ]
  }
}
```

### Generated by `corvia workspace init --hooks`

| File | Purpose |
|------|---------|
| `.corvia/hooks/doc-placement-check.sh` | PreToolUse validation script |
| `repos/*/.git/hooks/post-commit` | Git-triggered incremental re-index |
| `.claude/settings.json` (merged) | Hook registration + memory directory |

One command generates all enforcement from `corvia.toml` config.

---

## Testing Strategy

| Component | Approach |
|-----------|----------|
| `EntryMetadata` serde | Unit: roundtrip serialize/deserialize with and without new fields, verify backward compat with existing JSON entries |
| `content_role` / `source_origin` population | Unit: ingestion of test files from known directories, assert correct metadata |
| Filtered search (post-filter) | Integration: insert entries with mixed roles, search with filter, verify only matching entries returned |
| `DocsConfig` / `DocsRulesConfig` | Unit: TOML parse roundtrip, verify default handling |
| `--incremental` ingest | Integration: ingest file, modify, re-ingest with `--incremental`, verify supersession chain |
| New Reasoner checks | Unit: `MisplacedDoc`, `TemporalContradiction`, `CoverageGap` with synthetic entries |
| `corvia docs check` CLI | Integration: set up workspace with misplaced/stale files, run check, verify findings |
| Dashboard graph + reader | Manual + Playwright: click node → panel renders content, hover → fade, neighbor card → navigate |
| Hook validation script | Unit: feed various file paths to script, verify block/allow/warn responses |
| Post-commit hook | Integration: commit a changed .md file, verify incremental re-index triggers |

---

## Phased Delivery

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Metadata extensions (`content_role`, `source_origin`) + repo-aware `corvia_write` + filtered search | Foundation |
| 2 | Dashboard graphed document reader (Heptabase + Neo4j Bloom patterns) | Core UX |
| 3 | `corvia docs check` + new Reasoner checks + dashboard findings view | Intelligence |
| 4 | `corvia workspace init --hooks` + git post-commit hooks + incremental ingest | Enforcement |

## Decisions Log

- **D1:** Multi-repo doc ownership — repos own product docs, workspace owns cross-cutting
- **D2:** Approach B (Full Loop) selected over Aggregator-First (A) and Intelligence-Led (C)
- **D3:** Dashboard is primary human read interface, not file browsing
- **D4:** Interactive graphed document reader (Heptabase + Neo4j Bloom patterns)
- **D5:** `content_role` + `source_origin` metadata on `EntryMetadata`
- **D6:** Git post-commit hooks trigger incremental re-indexing with supersession
- **D7:** PreToolUse hooks block wrong paths, generated from `corvia.toml` rules
- **D8:** Claude memory relocated to `.memory/` via `autoMemoryDirectory` (git-tracked)
- **D9:** Superpowers save path overrides via CLAUDE.md instruction priority
- **D10:** Aggregator (`corvia docs check`) extends Reasoner with doc-specific checks

## References

- UX research: `docs/decisions/graph-document-ux-research.md`
- Original brainstorm: `repos/corvia/docs/rfcs/2026-02-25-corvia-brainstorm.md` (Flow 2: Living Docs)
- Existing Reasoner: `crates/corvia-kernel/src/reasoner.rs`
- Adapter protocol: `crates/corvia-kernel/src/adapter_protocol.rs`
- Dashboard: `tools/corvia-dashboard/src/components/GraphView.tsx`
- corvia knowledge entry: `019ce5eb-0930-7f11-9d97-34349917316b`
