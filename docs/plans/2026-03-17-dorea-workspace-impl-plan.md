# Dorea Workspace Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up a corvia-workspace for Dorea (underwater video AI editing pipeline) by templating from corvia-workspace, adapting configs, and scaffolding the pipeline project.

**Architecture:** Create dorea-workspace from corvia-workspace GitHub template, swap repos/corvia for repos/dorea, adapt all configs/docs for dorea's Python pipeline, and scaffold the 6 pipeline script stubs with supporting files. Devcontainer adapted for Python/CUDA workload with port offset +100 for coexistence.

**Tech Stack:** Python, PyTorch, SAM2, Depth Anything V2, DaVinci Resolve API, corvia (organizational memory), Docker devcontainer

**Spec:** `docs/plans/2026-03-17-dorea-underwater-pipeline-design.md`

---

## File Map

### Files to create (new)

| File | Responsibility |
|------|---------------|
| `.gitattributes` | Merge strategy for upstream sync (corvia.toml merge=ours) |
| `repos/dorea/scripts/00_generate_lut.py` | Stub: reference images → .cube LUT |
| `repos/dorea/scripts/01_extract_frames.py` | Stub: ffmpeg keyframe extraction |
| `repos/dorea/scripts/02_claude_scene_analysis.py` | Stub: Claude API scene scanning → JSON |
| `repos/dorea/scripts/03_run_sam2.py` | Stub: SAM2 subject tracking |
| `repos/dorea/scripts/04_run_depth.py` | Stub: Depth Anything V2 depth maps |
| `repos/dorea/scripts/05_resolve_setup.py` | Stub: Resolve API import + DRX + mattes (HOST ONLY) |
| `repos/dorea/scripts/run_all.sh` | Master overnight batch script |
| `repos/dorea/config.yaml` | Pipeline configuration |
| `repos/dorea/requirements.txt` | Python dependencies |
| `repos/dorea/README.md` | Project overview and quick start |
| `repos/dorea/.gitignore` | Ignore patterns for dorea repo |

### Files to modify (from template)

| File | What changes |
|------|-------------|
| `corvia.toml` | project.name, scope_id, repo entry (name, url, namespace), inference.device → cpu |
| `.mcp.json` | Add davinci-resolve-mcp server entry |
| `.gitignore` | Replace repos/corvia/ with repos/dorea/, add footage/ working/ models/ |
| `CLAUDE.md` | Adapt for dorea: scope_id, pipeline context, script 05 host note |
| `AGENTS.md` | Full rewrite for dorea's Python pipeline stack |
| `README.md` | Full rewrite for dorea project |
| `.claude/settings.json` | Update scope_id in reminder hook |
| `.claude/hooks/corvia-write-reminder.sh` | Update scope_id from corvia to dorea |
| `.devcontainer/devcontainer.json` | Name, workspaceFolder, forwardPorts, extensions |
| `.devcontainer/Dockerfile` | Add Python venv, PyTorch+CUDA, SAM2, ffmpeg |
| `.devcontainer/docker-compose.yml` | Workspace mount path, volume names, extra_hosts |
| `.devcontainer/scripts/post-start.sh` | settings.local.json MCP list, add venv activation |
| `.devcontainer/scripts/lib.sh` | Review for corvia-specific path references |
| `.devcontainer/scripts/init-host.sh` | Review for corvia-specific references |

### Files to delete (from template)

| File | Reason |
|------|--------|
| `TASK.md` | Corvia-specific parallel agent task |
| `repos/corvia/` | Replaced by repos/dorea |
| `.github/workflows/release.yml` | Builds corvia-specific artifacts (VS Code ext + dashboard dist) |

---

## Chunk 1: GitHub Setup & Local Clone

### Task 1: Mark corvia-workspace as GitHub Template

**Files:** None (GitHub API call)

- [ ] **Step 1: Mark repo as template**

```bash
gh api -X PATCH repos/chunzhe10/corvia-workspace -f is_template=true
```

Expected: `"is_template": true` in response JSON.

- [ ] **Step 2: Verify template status**

```bash
gh api repos/chunzhe10/corvia-workspace --jq '.is_template'
```

Expected: `true`

### Task 2: Create dorea-workspace from template

**Files:** None (GitHub operations)

- [ ] **Step 1: Create dorea-workspace from template**

```bash
gh repo create chunzhe10/dorea-workspace --template chunzhe10/corvia-workspace --public --clone=false
```

Expected: Repository created message.

- [ ] **Step 2: Create dorea repo (empty)**

```bash
gh repo create chunzhe10/dorea --public --clone=false
```

Expected: Repository created message.

- [ ] **Step 3: Clone dorea-workspace locally**

```bash
cd /home/chunzhe/corvia-project
git clone https://github.com/chunzhe10/dorea-workspace.git
cd dorea-workspace
```

- [ ] **Step 4: Add upstream remote**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git remote add upstream https://github.com/chunzhe10/corvia-workspace.git
git fetch upstream
```

- [ ] **Step 5: Verify remotes**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git remote -v
```

Expected: Both `origin` (dorea-workspace) and `upstream` (corvia-workspace).

- [ ] **Step 6: Remove repos/corvia if present**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
rm -rf repos/corvia 2>/dev/null
# Remove any .gitkeep or placeholder in repos/
ls -la repos/
```

- [ ] **Step 7: Remove corvia-specific files**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
rm -f TASK.md
rm -rf .github/workflows/release.yml
```

TASK.md is a corvia parallel agent task. release.yml builds corvia-specific artifacts (VS Code extension + dashboard dist) that don't apply to dorea.

- [ ] **Step 8: Commit cleanup**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add TASK.md repos/ .github/
git commit -m "chore: clean up corvia-specific files from template

Remove repos/corvia references, TASK.md (corvia parallel agent task),
and release.yml (corvia-specific CI artifacts).
Preparing workspace for dorea pipeline project."
```

---

## Chunk 2: Workspace Configuration Files

### Task 3: Update corvia.toml

**Files:**
- Modify: `corvia.toml`

- [ ] **Step 1: Update corvia.toml for dorea**

Replace the full content of `corvia.toml` with:

```toml
[project]
name = "dorea"
scope_id = "dorea"

[storage]
store_type = "lite"
data_dir = ".corvia"

[embedding]
provider = "corvia"
model = "nomic-embed-text-v1.5"
url = "http://127.0.0.1:8030"
dimensions = 768

[server]
host = "127.0.0.1"
port = 8020

[agent_lifecycle]
heartbeat_interval_secs = 30
stale_timeout_secs = 300
orphan_grace_secs = 1200
gc_orphan_after_secs = 86400
gc_closed_session_after_secs = 604800
gc_inactive_agent_after_secs = 2592000

[merge]
provider = "corvia"
model = "qwen3"
similarity_threshold = 0.8500000238418579
max_retries = 3

[workspace]
repos_dir = "repos"

[[workspace.repos]]
name = "dorea"
url = "https://github.com/chunzhe10/dorea"
namespace = "pipeline"

[rag]
default_limit = 10
graph_expand = true
graph_depth = 2
graph_alpha = 0.30000001192092896
reserve_for_answer = 0.20000000298023224
max_context_tokens = 0
system_prompt = ""
graph_oversample_factor = 3
skills_enabled = false
skills_dirs = ["skills"]
max_skills = 3
skill_threshold = 0.30000001192092896
reserve_for_skills = 0.15000000596046448

[workspace.docs]
memory_dir = ".memory"
workspace_docs = "docs"
allowed_workspace_subdirs = ["decisions", "learnings", "plans"]

[workspace.docs.rules]
blocked_paths = ["docs/superpowers/*"]
repo_docs_pattern = "docs/"

[chunking]
max_tokens = 512
min_tokens = 32
overlap_tokens = 64
strategy = "auto"

[telemetry]
exporter = "stdout"
otlp_endpoint = ""
otlp_protocol = "grpc"
service_name = "dorea"
log_format = "json"
metrics_enabled = true

[inference]
device = "cpu"
kv_quant = "q8"
flash_attention = true

[inference.chat_models."llama3.2"]
repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

[inference.chat_models.qwen3]
repo = "bartowski/Qwen_Qwen3-8B-GGUF"
filename = "Qwen_Qwen3-8B-Q4_K_M.gguf"

[inference.chat_models."llama3.2:1b"]
repo = "bartowski/Llama-3.2-1B-Instruct-GGUF"
filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

Key changes from template:
- `project.name` → `"dorea"`
- `scope_id` → `"dorea"`
- `workspace.repos` → dorea URL, namespace `"pipeline"`
- `telemetry.service_name` → `"dorea"`
- `inference.device` → `"cpu"` (frees 6GB VRAM for SAM2/Depth Anything)
- Removed `"marketing"` from `allowed_workspace_subdirs` (not needed for dorea)

- [ ] **Step 2: Verify TOML is valid**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
python3 -c "import tomllib; tomllib.load(open('corvia.toml','rb')); print('valid')"
```

Expected: `valid`

### Task 4: Update .mcp.json

**Files:**
- Modify: `.mcp.json`

- [ ] **Step 1: Update .mcp.json with three servers**

```json
{
  "mcpServers": {
    "corvia": {
      "type": "http",
      "url": "http://127.0.0.1:8020/mcp"
    },
    "davinci-resolve-mcp": {
      "type": "http",
      "url": "http://host.docker.internal:9090/mcp"
    },
    "playwright": {
      "type": "http",
      "url": "http://127.0.0.1:8050/mcp"
    }
  }
}
```

Note: `davinci-resolve-mcp` uses `host.docker.internal` because it runs on the host machine alongside DaVinci Resolve.

### Task 5: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Replace .gitignore content**

```gitignore
# Cloned repos (created by corvia workspace init)
repos/dorea/

# Dorea data directories (large binary, not version controlled)
footage/
working/
models/

# Derived data (rebuilt at runtime)
.corvia/hnsw/
.corvia/hnsw_index/
.corvia/lite_store.redb
.corvia/coordination.redb
.corvia/coordination.redb.lock
.corvia/staging/

# Knowledge files are tracked — no source to re-ingest from
# .corvia/knowledge/ is NOT ignored

# Local working files (per-user, not shared)
.local/
.devcontainer/.corvia-workspace-flags
.devcontainer/docker-compose.override.yml

# Claude Code local settings (session-specific permissions)
.claude/settings.local.json

# Editor / IDE state (per-user)
.vscode/
.continue/

# Built VS Code extension artifacts (built from source by post-start.sh or CI)
*.vsix

# Python build artifacts
__pycache__/
*.egg-info/

# Node dependencies (install via package.json)
node_modules/

# Playwright MCP runtime logs
.playwright-mcp/

# Model caches (large, downloaded on demand)
.fastembed_cache/
.worktrees/
.superpowers/
```

### Task 6: Create .gitattributes

**Files:**
- Create: `.gitattributes`

- [ ] **Step 1: Create .gitattributes for upstream merge protection**

```
# Prevent upstream (corvia-workspace) merges from overwriting dorea-specific config
corvia.toml merge=ours
```

### Task 7: Commit workspace config changes

- [ ] **Step 1: Stage and commit**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add corvia.toml .mcp.json .gitignore .gitattributes
git commit -m "feat: configure workspace for dorea pipeline

- corvia.toml: scope_id=dorea, namespace=pipeline, inference on CPU
- .mcp.json: add davinci-resolve-mcp via host.docker.internal
- .gitignore: swap repos/corvia for repos/dorea, add footage/working/models
- .gitattributes: merge=ours on corvia.toml for upstream sync safety"
```

---

## Chunk 3: Documentation Files

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace CLAUDE.md content**

```markdown
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
```

### Task 9: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Replace AGENTS.md content**

```markdown
# dorea-workspace

> Corvia-powered workspace for [dorea](repos/dorea) — automated underwater video
> AI editing pipeline.

This file follows the [AGENTS.md standard](https://agents.md/).

## Workspace Layout

```
dorea-workspace/
├── AGENTS.md                # Cross-platform AI agent instructions (this file)
├── CLAUDE.md                # Claude Code wrapper (imports AGENTS.md)
├── corvia.toml              # Workspace config (repos, embedding, server, docs)
├── .agents/                 # Agent-agnostic skills & reference docs
│   └── skills/              # Reusable patterns for any AI assistant
├── .mcp.json                # MCP server config (corvia, resolve-mcp, playwright)
├── repos/
│   └── dorea/               # Pipeline: Python scripts, config, LUTs, templates
├── .corvia/                 # Local knowledge store (LiteStore)
├── footage/                 # Raw + flattened dive footage (gitignored)
├── working/                 # Ephemeral AI outputs: masks, depth, keyframes (gitignored)
├── models/                  # AI model weights: SAM2, Depth Anything (gitignored)
└── docs/
    ├── decisions/           # Workspace-level architectural decisions
    ├── learnings/           # Captured knowledge and patterns
    └── plans/               # Active implementation plans
```

## Quick Reference

```bash
corvia workspace status          # Check workspace + service health
corvia search "query"            # Search ingested knowledge
corvia workspace ingest          # Index dorea repo
corvia workspace ingest --fresh  # Re-index from scratch
corvia serve &                   # Start server (auto-started by devcontainer)
```

## Service Ports (container-internal)

| Port | Service | Description |
|------|---------|-------------|
| 8020 | API server | REST + MCP protocol |
| 8021 | Dashboard | Knowledge browser, system health |
| 8030 | Inference | gRPC embedding + chat (CPU mode) |

Host-forwarded ports are offset +100 (8120, 8121, 8130) to coexist with corvia-workspace.

## MCP Servers

| Server | URL | Description |
|--------|-----|-------------|
| corvia | `http://127.0.0.1:8020/mcp` | Organizational memory (container-internal) |
| davinci-resolve-mcp | `http://host.docker.internal:9090/mcp` | DaVinci Resolve API bridge (host) |
| playwright | `http://127.0.0.1:8050/mcp` | Browser automation (container-internal) |

Available corvia MCP tools (use `scope_id: "dorea"` for all calls):
- `corvia_search` — semantic search across ingested knowledge
- `corvia_write` — write knowledge entries (requires agent identity)
- `corvia_history` — entry supersession history
- `corvia_graph` — graph edges for an entry
- `corvia_reason` — run health checks on a scope
- `corvia_agent_status` — agent contribution summary
- `corvia_context` — retrieve assembled context (RAG retrieval only)
- `corvia_ask` — full RAG: question → AI-generated answer from knowledge
- `corvia_system_status` — system status (entry counts, agents, sessions, queue)
- `corvia_config_get` — read config section as JSON
- `corvia_config_set` — update hot-reloadable config value (requires confirmation)
- `corvia_adapters_list` — discovered adapter binaries
- `corvia_agents_list` — all registered agents
- `corvia_gc_run` — trigger garbage collection (requires confirmation)
- `corvia_rebuild_index` — rebuild HNSW vector index (requires confirmation)
- `corvia_agent_suspend` — suspend an agent (requires confirmation)
- `corvia_merge_retry` — retry failed merge entries (requires confirmation)
- `corvia_merge_queue` — inspect merge queue status

**IMPORTANT:** The `scope_id` for this workspace is `"dorea"` (defined in `corvia.toml`).
Do NOT use `"corvia"`, `"dorea-workspace"`, or any other variant.

## Hybrid Tool Usage (corvia MCP + native tools)

**IMPORTANT: Always call corvia MCP tools FIRST before using native tools for any
development task or question.** corvia is the project's knowledge base — skipping it
means you risk re-discovering decisions that were already made or contradicting
established patterns. This applies to ALL agents (Claude Code, Codex, etc.).

### When to use corvia MCP tools (ALWAYS do this first)

- **Starting ANY task**: Call `corvia_search` or `corvia_ask` first to find prior decisions,
  design context, or patterns relevant to the work. **This is mandatory, not optional.**
- **Answering ANY question about the project**: Call `corvia_ask` before searching code.
- **Understanding "why"**: Use `corvia_ask` for questions about architecture, rationale,
  or past discussions.
- **Recording decisions**: Use `corvia_write` to persist design decisions, architectural
  context, or implementation notes that future sessions should know.

### When to use native tools

- **Reading/editing specific files** — corvia doesn't replace file access.
- **Searching for code patterns** — precise text/regex matching in source code.
- **Running commands** — Python scripts, tests, git, CLI tools.
- **File discovery** — finding files by name or extension.

### Rule of thumb

> **corvia = project knowledge & context. Native tools = source code & execution.**
> **Always check corvia first.**

## Pipeline Overview

Dorea automates underwater video post-production in 6 phases:

| Phase | Script | Runs in | GPU |
|-------|--------|---------|-----|
| 0 | `00_generate_lut.py` | Container | No (CPU) |
| 1 | `01_extract_frames.py` | Container | No (ffmpeg CPU) |
| 2 | `02_claude_scene_analysis.py` | Container | No (API call) |
| 3 | `03_run_sam2.py` | Container | Yes (~3GB VRAM) |
| 4 | `04_run_depth.py` | Container | Yes (~1.5GB VRAM) |
| 5 | `05_resolve_setup.py` | **Host** | No (Resolve IPC) |

**Critical constraint:** 6GB VRAM. Only one GPU model loaded at a time. Sequential
processing enforced by pipeline scripts. Corvia inference runs on CPU.

## Development

- **Language**: Python
- **Package manager**: pip (venv at `/opt/dorea-venv`)
- **AI models**: SAM2-small, Depth Anything V2 Small, Claude API (sonnet)
- **Video tools**: ffmpeg, DaVinci Resolve Studio (host)
- **Storage**: corvia LiteStore — data in `.corvia/`
- **Embedding**: corvia-inference server at `http://127.0.0.1:8030` (CPU mode)
- **API server**: `http://127.0.0.1:8020` (REST + MCP)
- **Config**: `corvia.toml` at workspace root, `repos/dorea/config.yaml` for pipeline
```

### Task 10: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md content**

```markdown
# dorea development workspace

Corvia-powered workspace for developing [dorea](repos/dorea) — an automated
underwater video AI editing pipeline. This workspace uses corvia's MCP server
and knowledge store for organizational memory during development.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| **API server** | `http://localhost:8120` | REST + MCP protocol server (host port) |
| **Dashboard** | `http://localhost:8121` | Knowledge browser and system health |
| **Inference** | `http://localhost:8130` | gRPC embedding + chat (CPU mode) |

All services start automatically in the devcontainer via `post-start.sh`.

## Quick start

### Option 1: Devcontainer (recommended)

Open in VS Code Dev Containers or DevPod. Everything is pre-configured — services
start automatically.

### Option 2: Local

```bash
git clone https://github.com/chunzhe10/dorea-workspace
cd dorea-workspace
corvia workspace init          # clones repos, sets up config
corvia workspace ingest        # indexes dorea repo
corvia serve &                 # start API + MCP server
```

## What's inside

- **[dorea](repos/dorea)** (namespace: `pipeline`) — underwater video AI editing
  pipeline using SAM2, Depth Anything V2, Claude API, and DaVinci Resolve

## Pipeline

Dorea automates underwater video post-production:

1. **Frame extraction** — ffmpeg pulls keyframes from dive footage
2. **Scene analysis** — Claude API identifies subjects (fish, divers, coral)
3. **Subject tracking** — SAM2 generates per-subject mask sequences
4. **Depth estimation** — Depth Anything V2 creates depth maps
5. **Resolve setup** — Python API imports footage, deploys DRX template, attaches mattes
6. **Creative grading** — Human editor grades in Resolve with Claude Desktop for consultation

## Upstream sync

This workspace was created from the [corvia-workspace](https://github.com/chunzhe10/corvia-workspace) template. To pull upstream updates:

```bash
git fetch upstream
git merge upstream/main
```
```

### Task 11: Commit documentation changes

- [ ] **Step 1: Stage and commit**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add CLAUDE.md AGENTS.md README.md
git commit -m "docs: adapt CLAUDE.md, AGENTS.md, README.md for dorea pipeline

- scope_id: dorea (not corvia)
- Pipeline-specific context: GPU constraints, Resolve on host, script phases
- Python stack replaces Rust references
- Port offset documentation for coexistence"
```

---

## Chunk 4: Claude Code Integration

### Task 12: Update .claude/settings.json

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Update settings.json**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "_comment": "After git commits, remind to persist decisions/learnings to corvia knowledge base.",
        "hooks": [
          {
            "command": "bash .claude/hooks/corvia-write-reminder.sh",
            "timeout": 10000,
            "type": "command"
          }
        ],
        "matcher": "Bash"
      }
    ],
    "PreToolUse": [
      {
        "hooks": [
          {
            "command": "echo 'REMINDER: Have you called corvia_search or corvia_ask first? Per CLAUDE.md, you MUST query corvia MCP tools (scope_id: dorea) before using Grep/Glob for any new task or question. If you already called corvia this session, proceed.'",
            "type": "command"
          }
        ],
        "matcher": "Grep|Glob"
      },
      {
        "hooks": [
          {
            "command": "bash .corvia/hooks/doc-placement-check.sh",
            "type": "command"
          }
        ],
        "matcher": "Write|Edit"
      }
    ],
    "SessionEnd": [
      {
        "_comment": "WORKAROUND: Claude Code leaks memory via orphaned node processes in WSL. Kills orphans on session exit. Remove when upstream fix lands.",
        "hooks": [
          {
            "command": "bash \"$CORVIA_WORKSPACE/.devcontainer/scripts/cleanup-orphans.sh\" --quiet 2>/dev/null || true",
            "type": "command"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "_comment": "Display-only reminder for agent identity selection.",
        "hooks": [
          {
            "command": "bash .claude/hooks/agent-check.sh",
            "timeout": 5000,
            "type": "command"
          }
        ]
      }
    ]
  }
}
```

Key change: PreToolUse Grep|Glob reminder now says `scope_id: dorea`.

### Task 13: Update corvia-write-reminder.sh

**Files:**
- Modify: `.claude/hooks/corvia-write-reminder.sh`

- [ ] **Step 1: Update scope_id in reminder output**

Change the echo line in the script from:

```
persist it with corvia_write (scope_id: corvia, agent_id: claude-code)
```

to:

```
persist it with corvia_write (scope_id: dorea, agent_id: claude-code)
```

### Task 14: Commit Claude Code integration changes

- [ ] **Step 1: Stage and commit**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add .claude/settings.json .claude/hooks/corvia-write-reminder.sh
git commit -m "chore: update Claude Code hooks for dorea scope_id

- PreToolUse reminder references scope_id: dorea
- corvia-write-reminder.sh uses scope_id: dorea"
```

---

## Chunk 5: Devcontainer Adaptations

### Task 15: Update devcontainer.json

**Files:**
- Modify: `.devcontainer/devcontainer.json`

- [ ] **Step 1: Update devcontainer.json**

Changes to make:
1. `"name"` → `"Dorea Workspace"`
2. `"workspaceFolder"` → `"/workspaces/dorea-workspace"`
3. `"forwardPorts"` → `[8120, 8121, 8130, 11534]` (offset +100, note: add portsAttributes if supported)
4. `"customizations.vscode.extensions"` → remove `"rust-lang.rust-analyzer"`, add `"ms-python.python"`
5. Remove `rust-analyzer` settings from `customizations.vscode.settings`
6. Keep `corvia.serverUrl` as `http://localhost:8020` (container-internal, unchanged)

Full content:

```jsonc
{
    "name": "Dorea Workspace",
    "dockerComposeFile": ["docker-compose.yml", "docker-compose.override.yml"],
    "service": "app",
    "workspaceFolder": "/workspaces/dorea-workspace",
    "initializeCommand": ".devcontainer/scripts/init-host.sh",
    "postCreateCommand": ".devcontainer/scripts/post-create.sh",
    "postStartCommand": ".devcontainer/scripts/post-start.sh",
    // 8120-8130: corvia services (host-forwarded +100), 11534: ollama (host-forwarded +100)
    // 8050 (playwright-mcp) is container-internal only — not forwarded.
    "forwardPorts": [8120, 8121, 8130, 11534],
    "portsAttributes": {
        "8120": {"label": "Corvia API", "onAutoForward": "notify"},
        "8121": {"label": "Corvia Dashboard", "onAutoForward": "notify"},
        "8130": {"label": "Corvia Inference", "onAutoForward": "notify"},
        "11534": {"label": "Ollama", "onAutoForward": "ignore"}
    },
    "mounts": [
        "source=${localEnv:HOME}${localEnv:USERPROFILE}/.config/gh,target=/root/.config/gh-host,type=bind,consistency=cached,readonly",
        "source=${localEnv:HOME}${localEnv:USERPROFILE}/.claude,target=/root/.claude,type=bind,consistency=cached"
    ],
    "remoteUser": "root",
    "containerEnv": {
        "CORVIA_WORKSPACE": "${containerWorkspaceFolder}"
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "tamasfe.even-better-toml",
                "Continue.continue",
                "ms-vscode.live-server"
            ],
            "settings": {
                "corvia.serverUrl": "http://localhost:8020",
                "files.watcherExclude": {
                    "**/.corvia/**": true,
                    "**/node_modules/**": true,
                    "**/footage/**": true,
                    "**/working/**": true,
                    "**/models/**": true
                }
            }
        }
    }
}
```

### Task 16: Update docker-compose.yml

**Files:**
- Modify: `.devcontainer/docker-compose.yml`

- [ ] **Step 1: Update docker-compose.yml**

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    stdin_open: true
    tty: true
    volumes:
      - ..:/workspaces/dorea-workspace:cached
      - dorea-pip-cache:/root/.cache/pip
      - /var/run/docker.sock:/var/run/docker.sock
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  dorea-pip-cache:
```

Changes from template:
- Workspace mount: `dorea-workspace` (was `corvia-workspace`)
- Removed `corvia-cargo-registry` and `corvia-cargo-git` volumes (no Rust)
- Added `dorea-pip-cache` for Python package caching
- Added `extra_hosts` for `host.docker.internal` (Resolve MCP connection)

### Task 16.5: Add dorea Python environment to Dockerfile

**Files:**
- Modify: `.devcontainer/Dockerfile`

- [ ] **Step 1: Add dorea Python venv and dependencies**

Append the following to the end of the Dockerfile (after the existing Rust/CUDA/Ollama setup):

```dockerfile
# === Dorea Pipeline Python Environment ===
# Isolated venv prevents CUDA runtime conflicts with corvia's ONNX inference

RUN python3 -m venv /opt/dorea-venv

# Install pipeline dependencies with CUDA 12.1 wheels
RUN /opt/dorea-venv/bin/pip install --no-cache-dir \
    anthropic colour-science numpy Pillow opencv-python pyyaml \
    transformers

RUN /opt/dorea-venv/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM2 — canonical install from GitHub (not on PyPI)
RUN /opt/dorea-venv/bin/pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam2.git

# ffmpeg for frame extraction (Phase 1)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

Key decisions:
- Separate `pip install` commands for caching: base deps, PyTorch (large), SAM2 (from git)
- CUDA 12.1 wheels match the CUDA runtime already in the image
- Isolated venv (`/opt/dorea-venv`) prevents conflicts with corvia's system-level pip packages
- `pyyaml` included (needed by pipeline scripts for config.yaml parsing)

- [ ] **Step 2: Verify Dockerfile syntax**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
docker build --check -f .devcontainer/Dockerfile .devcontainer/ 2>/dev/null || \
  echo "Note: --check may not be supported. Visual review is sufficient."
```

### Task 16.6: Review lib.sh and init-host.sh

**Files:**
- Review: `.devcontainer/scripts/lib.sh`
- Review: `.devcontainer/scripts/init-host.sh`

- [ ] **Step 1: Review lib.sh for corvia-specific hardcoded paths**

Check `lib.sh` for any hardcoded references to "corvia-workspace" or "corvia" paths that need updating. Key functions to check:
- `install_binaries` — downloads from GitHub releases (may reference chunzhe10/corvia-workspace)
- `pre_clone_repos` — reads corvia.toml for repo list (should work with updated toml)
- `init_workspace` — calls `corvia workspace init`
- `WORKSPACE_ROOT` — should use `$CORVIA_WORKSPACE` env var (not hardcoded)

Update any hardcoded "corvia-workspace" strings to use the `$CORVIA_WORKSPACE` env var instead.

- [ ] **Step 2: Review init-host.sh for corvia-specific references**

Check `init-host.sh` for hardcoded workspace names. This script runs on the host before container creation. Key areas:
- Container name cleanup (may reference "corvia")
- Override file generation (generic, should work as-is)

Update any corvia-specific container name patterns.

### Task 17: Update post-start.sh for dorea

**Files:**
- Modify: `.devcontainer/scripts/post-start.sh`

- [ ] **Step 1: Update settings.local.json generation**

Find the section in post-start.sh that writes `.claude/settings.local.json` and update the `enabledMcpjsonServers` list to include `davinci-resolve-mcp`:

Change from:

```bash
"enabledMcpjsonServers": ["corvia", "playwright"]
```

to:

```bash
"enabledMcpjsonServers": ["corvia", "playwright", "davinci-resolve-mcp"]
```

- [ ] **Step 2: Keep corvia-dev service management**

`corvia-dev` is the mechanism that starts the corvia server, inference server, and dashboard — all of which still run in dorea-workspace. Do NOT remove `corvia-dev` references. The services are needed for corvia's organizational memory functionality.

- [ ] **Step 3: Add Python venv activation**

Add the following near the end of post-start.sh (after service startup, before optional services section), so all devcontainer terminals have the dorea venv on PATH:

```bash
# Activate dorea Python venv for all shells
if [ -d /opt/dorea-venv ]; then
    echo 'source /opt/dorea-venv/bin/activate' >> /root/.bashrc
fi
```

- [ ] **Step 4: Verify post-start.sh syntax**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
bash -n .devcontainer/scripts/post-start.sh
```

Expected: no output (valid syntax).

### Task 18: Commit devcontainer changes

- [ ] **Step 1: Stage and commit**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add .devcontainer/
git commit -m "feat: adapt devcontainer for dorea workspace

- Dockerfile: add dorea Python venv, PyTorch+CUDA, SAM2, ffmpeg
- devcontainer.json: Dorea Workspace, port offset +100, Python extension
- docker-compose.yml: dorea-workspace mount, pip cache, host.docker.internal
- post-start.sh: add davinci-resolve-mcp to MCP list, dorea venv activation
- Review lib.sh and init-host.sh for corvia-specific references"
```

---

## Chunk 6: Pipeline Scaffolding (repos/dorea)

### Task 19: Initialize dorea repo

**Files:**
- Create: `repos/dorea/` directory structure

- [ ] **Step 1: Clone dorea repo**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git clone https://github.com/chunzhe10/dorea.git repos/dorea
```

- [ ] **Step 2: Create directory structure**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace/repos/dorea
mkdir -p scripts luts templates references
```

### Task 20: Create pipeline script stubs

**Files:**
- Create: `repos/dorea/scripts/00_generate_lut.py`
- Create: `repos/dorea/scripts/01_extract_frames.py`
- Create: `repos/dorea/scripts/02_claude_scene_analysis.py`
- Create: `repos/dorea/scripts/03_run_sam2.py`
- Create: `repos/dorea/scripts/04_run_depth.py`
- Create: `repos/dorea/scripts/05_resolve_setup.py`

- [ ] **Step 1: Create 00_generate_lut.py**

```python
"""Phase 0: Reference LUT Generation (One-Time Setup)

Analyses reference underwater images and generates a 33x33x33 3D .cube LUT
that captures the target colour look. This LUT is applied as Node 1 on every
clip in DaVinci Resolve.

Run once per look. Re-run when developing a new visual aesthetic.

Usage:
    python 00_generate_lut.py --references /path/to/reference/images --output /path/to/output.cube

Inputs:
    - 20-30 underwater reference images in references/ directory
    - Images should cover: different depths, lighting, subject types

Outputs:
    - luts/underwater_base.cube (33x33x33 3D LUT)

Dependencies:
    - colour-science, numpy, Pillow
    - No GPU required (CPU only)

Architecture doc: Section 4.0
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate underwater reference LUT from images")
    parser.add_argument("--references", required=True, help="Directory of reference images")
    parser.add_argument("--output", default="luts/underwater_base.cube", help="Output .cube file path")
    args = parser.parse_args()

    # TODO: Implement LUT generation
    # 1. Load reference images from --references directory
    # 2. Analyse colour characteristics: mean RGB per zone (shadow/midtone/highlight)
    # 3. Compute hue distribution, red channel falloff with depth, saturation targets
    # 4. Generate 33x33x33 3D LUT mapping D-Log M → target look
    # 5. Write .cube file to --output path

    print(f"LUT generation not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create 01_extract_frames.py**

```python
"""Phase 1: Footage Ingest & Frame Extraction (~5-10 min per dive session)

Extracts keyframes from raw footage at 2-second intervals for AI analysis.
Handles both DJI Action 4 (D-Log M) and Insta360 X5 (pre-flattened) clips.

Usage:
    python 01_extract_frames.py --date 2026-03-17

Inputs:
    - footage/raw/YYYY-MM-DD/ — DJI Action 4 clips
    - footage/flat/YYYY-MM-DD/ — Insta360 X5 flattened clips

Outputs:
    - working/keyframes/{clip_id}/frame_NNNNNN.jpg (1280px wide, JPEG 85%)

Dependencies:
    - ffmpeg (system package, CPU only, no GPU required)

Architecture doc: Section 4.1
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from dive footage")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement frame extraction
    # 1. Scan footage/raw/{date}/ and footage/flat/{date}/ for video files
    # 2. For each clip, generate clip_id from filename
    # 3. Run: ffmpeg -i {clip} -vf fps=0.5,scale=1280:-1 -q:v 5 working/keyframes/{clip_id}/frame_%06d.jpg
    # 4. Log clip count and frame count

    print(f"Frame extraction not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create 02_claude_scene_analysis.py**

```python
"""Phase 2: Claude Scene Analysis — Subject Detection (~2-5 min per clip, API cost)

Sends keyframe batches to Claude API for temporal scene analysis. Claude identifies
subjects (divers, fish species, coral), their bounding boxes, and first-appearance
frames. Uses two-pass sampling for frame-accurate detection.

Usage:
    python 02_claude_scene_analysis.py --date 2026-03-17

Inputs:
    - working/keyframes/{clip_id}/ — extracted keyframes from Phase 1

Outputs:
    - working/scene_analysis/{clip_id}.json — per-clip scene analysis
      Schema: {"clip_id": str, "frames": {frame_id: {subjects: [], new_subjects: [{label, bbox_normalised, first_appearance, confidence}]}}}

Dependencies:
    - anthropic SDK
    - No GPU required (API call)
    - Estimated cost: <$2 per dive session

Architecture doc: Section 4.2
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Claude scene analysis on keyframes")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement Claude scene analysis
    # Pass 1: Sample every 2s — Claude identifies approximate appearance windows
    # 1. Load keyframes in batches of 12 (config: claude_batch_size)
    # 2. Encode as base64, send to Claude API (claude-sonnet-4-6)
    # 3. Claude returns structured JSON with subjects, bboxes, timestamps
    #
    # Pass 2: Around each detected appearance, sample every 10 frames
    # 4. Pinpoint exact first-appearance frame per subject
    #
    # 5. Write scene_analysis JSON per clip

    print(f"Scene analysis not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create 03_run_sam2.py**

```python
"""Phase 3: SAM2 Subject Tracking (~1-3 min per clip, local GPU)

Generates per-frame mask sequences for each tracked subject using SAM2.
Initialised at the exact first-appearance frame from Phase 2 scene analysis,
using the bounding box as the starting prompt.

Usage:
    python 03_run_sam2.py --date 2026-03-17

Inputs:
    - working/scene_analysis/{clip_id}.json — from Phase 2
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/masks/{clip_id}/{subject_label}/frame_NNNNNN.png
      (binary alpha mask, 0 or 255 per pixel)

Dependencies:
    - SAM2 (facebookresearch/sam2)
    - PyTorch with CUDA
    - VRAM: ~3GB (SAM2-small). Only one model loaded at a time.
    - Preferred: UWSAM variant if available (better underwater performance)

Architecture doc: Section 4.3
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 subject tracking from scene analysis")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement SAM2 tracking
    # 1. Load SAM2-small model (or UWSAM if available)
    # 2. For each clip, load scene_analysis JSON
    # 3. For each subject in the clip:
    #    a. Seek to first-appearance frame
    #    b. Initialise SAM2 predictor with bounding box
    #    c. Propagate mask forward through all frames
    #    d. Export mask as PNG sequence (binary alpha)
    #    e. Handle subject exit: pad with empty masks
    #    f. Handle re-entry: new SAM2 session after 30+ frame gap
    # 4. Unload SAM2 model before next phase

    print(f"SAM2 tracking not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create 04_run_depth.py**

```python
"""Phase 4: Depth Anything V2 — Depth Map Generation (~1-2 min per clip, local GPU)

Performs monocular depth estimation on every frame. Output is a per-frame depth
map used as a luminance matte in Resolve to drive depth-dependent colour correction.

Bright pixels = close to camera. Dark pixels = far from camera.

Usage:
    python 04_run_depth.py --date 2026-03-17

Inputs:
    - Raw footage files (for full-resolution frame access)

Outputs:
    - working/depth/{clip_id}/frame_NNNNNN.png
      (16-bit grayscale PNG, resolution matches source)

Dependencies:
    - Depth Anything V2 Small (via transformers)
    - PyTorch with CUDA
    - VRAM: ~1.5GB
    - Note: Trained on terrestrial footage. Reduced accuracy in turbid water.

Architecture doc: Section 4.4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 depth estimation")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement depth estimation
    # 1. Load Depth Anything V2 Small model
    # 2. For each clip, extract every frame at source resolution
    # 3. Run depth estimation per frame
    # 4. Save as 16-bit grayscale PNG (preserves precision for soft mattes)
    # 5. Unload model before next phase

    print(f"Depth estimation not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create 05_resolve_setup.py**

```python
"""Phase 5: DaVinci Resolve Import & Setup (HOST ONLY)

WARNING: This script runs on the HOST MACHINE, not in the devcontainer.
It requires DaVinci Resolve Studio to be running and imports fusionscript.

Imports footage into Resolve, deploys the DRX template, applies the base LUT,
and attaches mask/depth sequences as Fusion mattes.

Usage:
    python 05_resolve_setup.py --date 2026-03-17

Inputs:
    - Raw footage files
    - working/masks/{clip_id}/{subject}/ — SAM2 mask sequences from Phase 3
    - working/depth/{clip_id}/ — depth map sequences from Phase 4
    - repos/dorea/luts/underwater_base.cube — reference LUT from Phase 0
    - repos/dorea/templates/underwater_grade_v1.drx — DRX node template

Outputs:
    - DaVinci Resolve project with timeline ready for creative grading
    - Each clip has 8-node structure: Base LUT, Neutral Balance, Depth Grade,
      Foreground Pop, Diver, Marine Life, Creative Look, Output

Dependencies:
    - DaVinci Resolve Studio (running on host)
    - fusionscript.so (Resolve Python API)
    - Environment vars: RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH
    - No GPU required (Resolve uses GPU independently)

Architecture doc: Sections 5 and 7
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Set up DaVinci Resolve project with mattes")
    parser.add_argument("--date", required=True, help="Dive date YYYY-MM-DD")
    args = parser.parse_args()

    # TODO: Implement Resolve setup
    # Requires: RESOLVE_SCRIPT_API, RESOLVE_SCRIPT_LIB, PYTHONPATH set
    # import DaVinciResolveScript as dvr
    #
    # 1. Connect to running Resolve instance
    # 2. Create or open project (config: resolve_project_name)
    # 3. Import all footage to Media Pool
    # 4. Create timeline from imported clips
    # 5. For each clip:
    #    a. Apply DRX template (8-node structure)
    #    b. SetLUT(1, underwater_base.cube) on Node 1
    #    c. Import mask sequences as image sequences to Media Pool
    #    d. Import depth sequences as image sequences to Media Pool
    #    e. Connect mattes to corresponding nodes via Fusion page
    # 6. Queue initial optimised media render

    print(f"Resolve setup not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

### Task 21: Create run_all.sh

**Files:**
- Create: `repos/dorea/scripts/run_all.sh`

- [ ] **Step 1: Create run_all.sh**

```bash
#!/bin/bash
# Master overnight batch script for Dorea pipeline
# Runs Phases 1-4 inside the devcontainer, then prompts for Phase 5 on host.
#
# Usage: bash scripts/run_all.sh
#
# Requires:
#   - Python venv activated (source /opt/dorea-venv/bin/activate)
#   - config.yaml configured with correct paths
#   - Footage dumped to footage/raw/ and/or footage/flat/
#
# Architecture doc: Section 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y-%m-%d)

echo "=== Dorea Pipeline — $DATE ==="
echo ""

# Phase 0 (00_generate_lut.py) is NOT included here — it is a one-time setup
# that runs once per visual look, not per dive session.
# Run manually: python scripts/00_generate_lut.py --references references/look_v1/ --output luts/underwater_base.cube

echo "=== Phase 1: Extract keyframes ==="
python "$SCRIPT_DIR/01_extract_frames.py" --date "$DATE"

echo "=== Phase 2: Claude scene analysis ==="
python "$SCRIPT_DIR/02_claude_scene_analysis.py" --date "$DATE"

echo "=== Phase 3: SAM2 subject tracking ==="
python "$SCRIPT_DIR/03_run_sam2.py" --date "$DATE"

echo "=== Phase 4: Depth estimation ==="
python "$SCRIPT_DIR/04_run_depth.py" --date "$DATE"

echo ""
echo "=== Phases 1-4 complete ==="
echo ""
echo "Phase 5 must run on the HOST (requires DaVinci Resolve)."
echo "Open a terminal on the host and run:"
echo ""
echo "  cd $(dirname "$SCRIPT_DIR")"
echo "  python scripts/05_resolve_setup.py --date $DATE"
echo ""
echo "Then open Resolve — your timeline is ready for creative grading."
```

- [ ] **Step 2: Make executable**

```bash
chmod +x repos/dorea/scripts/run_all.sh
```

### Task 22: Create config.yaml

**Files:**
- Create: `repos/dorea/config.yaml`

- [ ] **Step 1: Create config.yaml**

```yaml
# Dorea Pipeline Configuration
# Paths resolve relative to workspace root ($CORVIA_WORKSPACE)

# Footage paths
footage_raw: footage/raw
footage_flat: footage/flat
working_dir: working
pipeline_dir: repos/dorea

# Claude API (for overnight batch scene analysis only)
# Set ANTHROPIC_API_KEY env var or replace below
anthropic_api_key: ${ANTHROPIC_API_KEY}
claude_model: claude-sonnet-4-6
frame_sample_rate_seconds: 2
claude_batch_size: 12

# SAM2
sam2_model: sam2_small  # options: tiny, small, base_plus, large
sam2_weights: models/sam2/sam2.1_hiera_small.pt

# Depth Anything V2
depth_model: depth_anything_v2_small
depth_weights: models/depth_anything_v2_small/
depth_output_format: png16  # 16-bit PNG for matte precision

# DaVinci Resolve (host paths — used by 05_resolve_setup.py on host)
resolve_lut_path: repos/dorea/luts/underwater_base.cube
resolve_drx_path: repos/dorea/templates/underwater_grade_v1.drx
resolve_project_name: Dive_2026

# Processing
gpu_device: cuda:0
clear_working_after_import: false  # set true to save disk space
```

### Task 23: Create requirements.txt

**Files:**
- Create: `repos/dorea/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
# Dorea pipeline dependencies
# Install: pip install -r requirements.txt

# Core
anthropic
colour-science
numpy
Pillow
opencv-python
pyyaml

# AI models (pinned to CUDA 12.1 wheels)
--extra-index-url https://download.pytorch.org/whl/cu121
torch
torchvision
transformers

# SAM2 — install from GitHub after pip install:
# pip install git+https://github.com/facebookresearch/sam2.git

# Video
# ffmpeg must be installed as a system package (apt install ffmpeg)
```

### Task 24: Create dorea README.md and .gitignore

**Files:**
- Create: `repos/dorea/README.md`
- Create: `repos/dorea/.gitignore`

- [ ] **Step 1: Create repos/dorea/README.md**

```markdown
# Dorea

Automated underwater video AI editing pipeline. Named after the Dorado
constellation (the golden fish).

## What it does

Dorea automates the technically repetitive stages of underwater video
post-production while preserving full creative control for the human editor.

### Pipeline phases

| # | Script | What it does | GPU |
|---|--------|-------------|-----|
| 0 | `00_generate_lut.py` | Reference images → .cube LUT | No |
| 1 | `01_extract_frames.py` | ffmpeg keyframe extraction | No |
| 2 | `02_claude_scene_analysis.py` | Claude API scene + subject detection | No |
| 3 | `03_run_sam2.py` | SAM2 per-subject mask tracking | ~3GB |
| 4 | `04_run_depth.py` | Depth Anything V2 depth maps | ~1.5GB |
| 5 | `05_resolve_setup.py` | Resolve API: import, DRX, mattes | No |

### Hardware requirements

- Linux workstation
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- 64GB RAM recommended
- DaVinci Resolve Studio (one-time purchase)

### Camera support

- DJI Action 4 (D-Log M)
- Insta360 X5 (pre-flattened via Insta360 Studio)

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# 2. Download model weights
# SAM2: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
# Depth Anything V2: auto-downloads from HuggingFace on first run

# 3. Configure
# Edit config.yaml with your paths and API key

# 4. One-time: generate reference LUT
python scripts/00_generate_lut.py --references references/look_v1/ --output luts/underwater_base.cube

# 5. After each dive: run overnight batch
bash scripts/run_all.sh

# 6. Morning: open Resolve — timeline is ready for creative grading
```

## Architecture

See `underwater_pipeline_architecture.docx` for the full architecture document.
```

- [ ] **Step 2: Create repos/dorea/.gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.egg-info/
.venv/

# Note: model weights live at workspace root (dorea-workspace/models/),
# not inside this repo. The workspace .gitignore handles them.

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

### Task 25: Commit pipeline scaffolding and push dorea repo

- [ ] **Step 1: Commit and push dorea repo**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace/repos/dorea
git add scripts/ config.yaml requirements.txt README.md .gitignore luts/ templates/ references/
git commit -m "feat: scaffold dorea pipeline project

Script stubs (00-05), overnight batch runner, config.yaml,
requirements.txt, and directory structure.

Dorea is an automated underwater video AI editing pipeline using
SAM2, Depth Anything V2, Claude API, and DaVinci Resolve."
git push -u origin main
```

---

## Chunk 7: Data Directories, Transfer Docs & Final Verification

### Task 26: Create workspace data directories

**Files:**
- Create: directory placeholders

- [ ] **Step 1: Create gitignored data directories with .gitkeep**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
mkdir -p footage/raw footage/flat
mkdir -p working/keyframes working/scene_analysis working/masks working/depth
mkdir -p models/sam2 models/depth_anything_v2_small
```

Note: These directories are gitignored. They exist on the local filesystem only.

### Task 27: Transfer design spec and plan

**Files:**
- Move from temp to workspace

- [ ] **Step 1: Copy docs from temp directory**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
mkdir -p docs/plans
cp /home/chunzhe/dorea-temp/docs/plans/2026-03-17-dorea-underwater-pipeline-design.md docs/plans/
cp /home/chunzhe/dorea-temp/docs/plans/2026-03-17-dorea-workspace-impl-plan.md docs/plans/
```

- [ ] **Step 2: Copy architecture document**

```bash
cp /home/chunzhe/Downloads/underwater_pipeline_architecture.docx /home/chunzhe/corvia-project/dorea-workspace/docs/
```

- [ ] **Step 3: Commit docs**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git add docs/
git commit -m "docs: add design spec, implementation plan, and architecture document"
```

### Task 28: Final commit and push

- [ ] **Step 1: Check for any uncommitted changes**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git status
```

- [ ] **Step 2: Stage any remaining changes and commit**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git status  # review what's untracked/modified
# Stage only the files that belong — review before adding
git add <specific files shown by git status>
git diff --cached --stat  # review what's staged
git commit -m "feat: dorea workspace setup complete

Corvia-powered workspace for the Dorea underwater video AI editing pipeline.
Created from corvia-workspace template with upstream tracking."
```

- [ ] **Step 3: Push to remote**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git push -u origin main
```

- [ ] **Step 4: Verify upstream remote works**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git fetch upstream
git log --oneline upstream/main -5
```

Expected: shows recent corvia-workspace commits.

- [ ] **Step 5: Clean up temp directory**

```bash
rm -rf /home/chunzhe/dorea-temp
```

### Task 29: Verification checklist

- [ ] **Step 1: Verify repo structure**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
echo "=== Top level ==="
ls -la
echo "=== repos/dorea ==="
ls -la repos/dorea/
echo "=== repos/dorea/scripts ==="
ls -la repos/dorea/scripts/
echo "=== docs ==="
ls -la docs/
```

Expected: all files present, no repos/corvia, dorea scripts visible.

- [ ] **Step 2: Verify corvia.toml scope**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
grep scope_id corvia.toml
grep 'name = "dorea"' corvia.toml
```

Expected: `scope_id = "dorea"` and repo name `"dorea"`.

- [ ] **Step 3: Verify .mcp.json has all three servers**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
python3 -c "import json; d=json.load(open('.mcp.json')); print(list(d['mcpServers'].keys()))"
```

Expected: `['corvia', 'davinci-resolve-mcp', 'playwright']`

- [ ] **Step 4: Verify git remotes**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
git remote -v
```

Expected: `origin` → dorea-workspace, `upstream` → corvia-workspace.

- [ ] **Step 5: Verify no corvia references remain in key files**

```bash
cd /home/chunzhe/corvia-project/dorea-workspace
grep -r "scope_id.*corvia" CLAUDE.md AGENTS.md corvia.toml .claude/settings.json .claude/hooks/ || echo "Clean — no stale corvia scope_id references"
```

Expected: "Clean — no stale corvia scope_id references"
