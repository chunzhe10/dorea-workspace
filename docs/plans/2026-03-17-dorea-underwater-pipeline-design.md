# Dorea — Underwater Video AI Editing Pipeline

**Date**: 2026-03-17
**Status**: Design
**Author**: Chunzhe + Claude

## Overview

Dorea (named after the Dorado constellation — the golden fish) is an automated AI-assisted underwater video editing and colour grading pipeline. It integrates DaVinci Resolve Studio, local AI models (SAM2, Depth Anything V2), and Claude API to automate the technically repetitive stages of post-production while preserving full creative control for the human editor.

The system targets two camera systems — DJI Action 4 (D-Log M) and Insta360 X5 — running overnight batch processing on a Linux workstation with an RTX 3060 GPU (6GB VRAM) and 64GB DDR5 RAM. All AI inference runs locally. Claude API is called sparingly for high-reasoning vision tasks only.

## Architecture Source

The full pipeline architecture is documented in `underwater_pipeline_architecture.docx`. This spec covers the **workspace setup and scaffolding** — not the pipeline implementation itself.

---

## Section 1: Repos & Setup Flow

### GitHub Repos

| Repo | Purpose |
|------|---------|
| `chunzhe10/corvia-workspace` | Upstream template (marked as GitHub Template) |
| `chunzhe10/dorea-workspace` | Created from template, hosts the pipeline workspace |
| `chunzhe10/dorea` | The pipeline project code itself |

### Setup Sequence

1. Mark `chunzhe10/corvia-workspace` as a GitHub Template Repository (Settings → Template repository)
2. Create `chunzhe10/dorea-workspace` using "Use this template" on GitHub
3. Create `chunzhe10/dorea` as a new empty repo on GitHub
4. Clone dorea-workspace locally to `/home/chunzhe/corvia-project/dorea-workspace/`
5. Add upstream remote: `git remote add upstream https://github.com/chunzhe10/corvia-workspace.git`
6. Remove `repos/corvia` references:
   - Delete `repos/corvia/` directory (if template included it — may be gitignored)
   - Remove any `.gitkeep` or placeholder files in `repos/corvia/`
   - Update `.gitignore`: replace `repos/corvia/` with `repos/dorea/`
   - Update `corvia.toml` repo entry (see Section 2)
7. Clone `chunzhe10/dorea` into `repos/dorea`
8. Commit the workspace adaptations (see Section 2)

### Upstream Sync

To pull corvia-workspace updates into dorea-workspace:

```bash
git fetch upstream
git merge upstream/main
```

Conflicts will primarily occur in `corvia.toml` (repo entries) and `.gitignore` (repo paths). To prevent upstream merges from overwriting dorea-specific config, add a `.gitattributes`:

```
corvia.toml merge=ours
```

This tells git to keep dorea's version of `corvia.toml` during upstream merges. Workspace infrastructure (docs structure, .agents/, devcontainer base) merges cleanly.

---

## Section 2: Workspace Adaptations

### corvia.toml

Update the workspace configuration for dorea:

```toml
[project]
name = "dorea"
scope_id = "dorea"

[[workspace.repos]]
name = "dorea"
url = "https://github.com/chunzhe10/dorea"
namespace = "pipeline"
```

- `scope_id = "dorea"` — all MCP calls use this scope (not "corvia")
- `namespace = "pipeline"` — distinguishes dorea's knowledge from other repos
- Keep all other settings unchanged: storage, embedding, server, merge, rag, chunking, telemetry
- **VRAM note**: Set `[inference] device = "cpu"` — corvia-inference must not compete with SAM2/Depth Anything for the 6GB VRAM. The overnight pipeline needs full GPU access. Corvia embedding on CPU is acceptable for a single-repo workspace.

### .mcp.json

Three MCP servers. Container-internal ports stay at base values (8020, 8050). The `davinci-resolve-mcp` connects to the host.

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

**Note on davinci-resolve-mcp**: This is `github.com/samuelgursky/davinci-resolve-mcp` — an open-source MCP server that exposes 342 DaVinci Resolve API methods. It runs on the **host** (not in the container) because it needs direct IPC with Resolve. The container reaches it via `host.docker.internal`. The host-side setup is:

```bash
# On host — install and run the MCP server
git clone https://github.com/samuelgursky/davinci-resolve-mcp.git
cd davinci-resolve-mcp
python install.py --clients claude-desktop
# Server starts when Resolve is running, listens on port 9090
```

### .claude/settings.json

Adapt hooks for dorea context:

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
            "command": "echo 'REMINDER: Have you called corvia_search or corvia_ask first? Per CLAUDE.md, you MUST query corvia MCP tools (scope_id: dorea) before using Grep/Glob for any new task or question.'",
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
        "_comment": "WORKAROUND: Claude Code orphan cleanup",
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

Key change: PreToolUse reminder references `scope_id: dorea` instead of `corvia`.

### .claude/settings.local.json (generated by post-start.sh)

```json
{
  "enabledMcpjsonServers": ["corvia", "playwright", "davinci-resolve-mcp"]
}
```

### CLAUDE.md

Adapt the knowledge-first workflow for dorea:

- Retain corvia-first rule: always `corvia_search` / `corvia_ask` before coding
- **Change scope_id references from `"corvia"` to `"dorea"`**
- Add pipeline-specific context: underwater video processing, DaVinci Resolve scripting API, sequential GPU scheduling (6GB VRAM constraint)
- Reference the architecture document for pipeline design decisions
- Note that DaVinci Resolve runs on the host, not in the devcontainer
- Note that `05_resolve_setup.py` runs on the **host** (not container) since it imports `fusionscript` directly

### AGENTS.md

Adapt for dorea's tech stack:

- **Project description**: Underwater video AI editing pipeline (not organizational memory for AI agents)
- **Language**: Python (not Rust)
- **Key dependencies**: PyTorch, SAM2, Depth Anything V2, anthropic SDK, colour-science, opencv, ffmpeg
- **GPU constraint**: 6GB VRAM, one model loaded at a time, sequential processing
- **scope_id**: `"dorea"` for all corvia MCP calls
- **Testing**: pipeline scripts testable with sample frames
- **Resolve API**: runs on host via `fusionscript` module — `05_resolve_setup.py` is the only script that runs outside the container
- Update workspace layout diagram to show `repos/dorea` instead of `repos/corvia`
- Remove Rust-specific guidance (cargo, rust-analyzer)
- Keep: corvia MCP usage patterns, hybrid tool usage, AI development learnings

---

## Section 3: Pipeline Scaffolding

### repos/dorea structure

```
dorea/
├── scripts/
│   ├── 00_generate_lut.py           # One-time: reference images → .cube LUT
│   ├── 01_extract_frames.py         # ffmpeg keyframe extraction per clip
│   ├── 02_claude_scene_analysis.py  # Claude API scene scanning → JSON
│   ├── 03_run_sam2.py               # SAM2 subject tracking from Claude prompts
│   ├── 04_run_depth.py              # Depth Anything V2 depth map generation
│   ├── 05_resolve_setup.py          # Resolve API: import, DRX deploy, mattes (HOST ONLY)
│   └── run_all.sh                   # Master overnight batch script
├── config.yaml                      # Paths, API keys, model choices
├── luts/                            # Generated .cube LUT files
├── templates/                       # DRX node graph templates
├── references/                      # Reference images for LUT generation
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview and quick start
└── .gitignore
```

Scripts are **stubs with docstrings** describing:
- Purpose and phase number
- Inputs and outputs (file paths, formats)
- Key logic from the architecture document
- Dependencies and VRAM requirements

**Execution model**: Scripts 00–04 run inside the devcontainer (Python + CUDA). Script 05 runs on the **host** because it imports `fusionscript` (Resolve's Python API), which requires Resolve to be running locally. `run_all.sh` handles this split — it runs 00–04 in the container, then prompts the user to run 05 on the host (or uses SSH/exec to bridge).

### Workspace-root data directories (gitignored)

```
dorea-workspace/
├── footage/
│   ├── raw/          # SD card dumps (YYYY-MM-DD subdirs)
│   └── flat/         # Insta360 Studio flattened output
├── working/          # Ephemeral AI processing outputs
│   ├── keyframes/    # Extracted frames for AI analysis
│   ├── scene_analysis/  # Claude JSON output
│   ├── masks/        # SAM2 mask PNG sequences
│   └── depth/        # Depth map PNG sequences
└── models/
    ├── sam2/         # SAM2 weights (sam2.1_hiera_small.pt)
    └── depth_anything_v2_small/  # Depth model weights
```

These directories are added to the workspace `.gitignore`:

```gitignore
# Dorea data directories (large binary, not version controlled)
footage/
working/
models/
repos/dorea/
```

### config.yaml

Based on the architecture document Section 8. Paths resolve relative to workspace root using the `CORVIA_WORKSPACE` environment variable (set in devcontainer.json's `containerEnv`):

```yaml
# Paths (resolved via $CORVIA_WORKSPACE)
footage_raw: footage/raw
footage_flat: footage/flat
working_dir: working
pipeline_dir: repos/dorea

# Claude API (for overnight batch scene analysis only)
anthropic_api_key: ${ANTHROPIC_API_KEY}
claude_model: claude-sonnet-4-6
frame_sample_rate_seconds: 2
claude_batch_size: 12

# SAM2
sam2_model: sam2_small
sam2_weights: models/sam2/sam2.1_hiera_small.pt

# Depth Anything V2
depth_model: depth_anything_v2_small
depth_weights: models/depth_anything_v2_small/
depth_output_format: png16

# DaVinci Resolve (host paths — used by 05_resolve_setup.py on host)
resolve_lut_path: repos/dorea/luts/underwater_base.cube
resolve_drx_path: repos/dorea/templates/underwater_grade_v1.drx
resolve_project_name: Dive_2026

# Processing
gpu_device: cuda:0
clear_working_after_import: false
```

---

## Section 4: Devcontainer & Environment

### Port Strategy

**Container-internal ports are unchanged** from corvia-workspace (8020, 8021, 8030, 8050). All container-internal configs (`corvia.toml`, `.mcp.json`) use the base ports. The +100 offset applies **only** to `forwardPorts` in `devcontainer.json`, which maps container ports to host ports for external access.

| Service | Container Port | Host Port | Notes |
|---------|---------------|-----------|-------|
| Corvia server | 8020 | 8120 | `forwardPorts` mapping only |
| Corvia dashboard | 8021 | 8121 | `forwardPorts` mapping only |
| Corvia inference | 8030 | 8130 | `forwardPorts` mapping only |
| Ollama | 11434 | 11534 | `forwardPorts` mapping only |

Playwright (8050) remains container-internal only — not forwarded to host (same as corvia-workspace).

### devcontainer.json changes

```jsonc
{
    "name": "Dorea Workspace",
    "workspaceFolder": "/workspaces/dorea-workspace",
    // Port offset: +100 from corvia-workspace for coexistence
    "forwardPorts": [8120, 8121, 8130, 11534],
    "portsAttributes": {
        "8120": {"label": "Corvia API", "onAutoForward": "notify"},
        "8121": {"label": "Corvia Dashboard", "onAutoForward": "notify"},
        "8130": {"label": "Corvia Inference", "onAutoForward": "notify"},
        "11534": {"label": "Ollama", "onAutoForward": "ignore"}
    },
    "containerEnv": {
        "CORVIA_WORKSPACE": "${containerWorkspaceFolder}"
    },
    "customizations": {
        "vscode": {
            "extensions": [
                // Remove rust-lang.rust-analyzer (not needed)
                // Remove tamasfe.even-better-toml (optional, keep if wanted)
                "ms-python.python",
                "Continue.continue",
                "ms-vscode.live-server"
            ],
            "settings": {
                "corvia.serverUrl": "http://localhost:8020"
            }
        }
    }
}
```

### docker-compose.yml changes

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
      # Docker-outside-of-Docker
      - /var/run/docker.sock:/var/run/docker.sock
    # Enable host.docker.internal for reaching Resolve MCP on host
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  dorea-pip-cache:
```

Key changes from corvia-workspace:
- Workspace mount: `dorea-workspace` instead of `corvia-workspace`
- Remove `corvia-cargo-registry` and `corvia-cargo-git` volumes (no Rust builds)
- Add `dorea-pip-cache` for Python package caching
- Add `extra_hosts` for `host.docker.internal` (needed for Resolve MCP connection)

### post-start.sh adaptations

Update the following in the inherited `post-start.sh`:
- Health check URLs: keep container-internal port 8020 (unchanged — health checks run inside container)
- Dashboard check: keep container-internal port 8021 (unchanged)
- MCP server URL: keep `http://127.0.0.1:8020/mcp` (unchanged — container-internal)
- `settings.local.json`: add `"davinci-resolve-mcp"` to `enabledMcpjsonServers`
- Add: Python venv activation for dorea pipeline (`source /opt/dorea-venv/bin/activate`)
- Remove: `corvia-dev` service management (not applicable to dorea)

### Dockerfile adaptations

Add to the existing Dockerfile (after the corvia/Rust build stages):

```dockerfile
# Dorea Python environment — isolated venv to avoid CUDA conflicts
RUN python3 -m venv /opt/dorea-venv
RUN /opt/dorea-venv/bin/pip install --no-cache-dir \
    anthropic colour-science numpy Pillow opencv-python \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    transformers

# SAM2 installed from GitHub (not on PyPI as segment-anything-2)
RUN /opt/dorea-venv/bin/pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam2.git

# ffmpeg for frame extraction
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

Key decisions:
- **Isolated venv** (`/opt/dorea-venv`): prevents CUDA runtime conflicts between corvia's ONNX inference and PyTorch
- **Pinned CUDA wheels** (`cu121`): matches the CUDA 12 runtime already in the image
- **SAM2 from GitHub**: the canonical install method (not a PyPI package)

### VRAM Management

With 6GB VRAM, corvia-inference and the pipeline cannot share the GPU:

- `corvia.toml` sets `[inference] device = "cpu"` for the dorea workspace
- Overnight batch scripts (00–04) get full GPU access
- Corvia embedding runs on CPU — acceptable performance for a single-repo workspace
- If GPU embedding is needed during development (not overnight), stop the pipeline first

### .agents/ directory

Retain the existing skills inherited from corvia-workspace:
- `ai-assisted-development.md` — general best practices (applicable)
- `content-verification.md` — verification patterns (applicable)
- `visual-content-playwright.md` — browser automation (applicable)

No dorea-specific skills needed at scaffolding time. Add later as pipeline patterns emerge.

### Host-Container Interaction

```
┌──────────────────────────────────────┐     ┌──────────────────────────┐
│ Devcontainer                         │     │ Host Machine             │
│                                      │     │                          │
│ Scripts 00-04 (Python, /opt/dorea-venv)    │ DaVinci Resolve Studio   │
│ Corvia server (:8020 internal)       │     │   (GPU, display)         │
│ Corvia inference (:8030 internal,CPU)│     │                          │
│ Playwright (:8050 internal)          │     │ Ollama (optional, GPU)   │
│                                      │     │                          │
│ .mcp.json ──── corvia (:8020) ───────│     │ davinci-resolve-mcp      │
│            ├── playwright (:8050) ───│     │   (:9090, Python IPC)    │
│            └── resolve-mcp ──────────│────▶│   → Resolve API          │
│                (host.docker.internal) │     │                          │
│                                      │     │ Script 05 (host Python)  │
│ Host ports: 8120,8121,8130,11534 ◄───│────▶│   → fusionscript.so     │
└──────────────────────────────────────┘     └──────────────────────────┘
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Workspace pattern | Template + upstream remote | Corvia-workspace updates flow to dorea; others get clean template button |
| Data directories | Workspace root, gitignored | Separates large binary data from code; workspace is self-documenting |
| Port allocation | +100 host offset, base container ports | Container configs unchanged; simple coexistence rule |
| Script scaffolding | Stubs with docstrings | Ready to implement without blocking workspace setup |
| Resolve location | Host machine | Requires GPU + display; MCP bridges to container |
| Script 05 execution | Host only | Needs `fusionscript.so` which requires running Resolve instance |
| Corvia inference | CPU mode | Frees 6GB VRAM for SAM2/Depth Anything; acceptable for single-repo |
| Python environment | Isolated venv `/opt/dorea-venv` | Prevents CUDA runtime conflicts with corvia's ONNX |
| Upstream merge strategy | `.gitattributes merge=ours` on corvia.toml | Prevents upstream from overwriting dorea-specific config |
| Project name | Dorea | From Dorado constellation; underwater/golden fish theme; sibling to Corvia |

## Out of Scope

- Full implementation of pipeline scripts (stubs only)
- DRX template creation (must be done manually in Resolve)
- Model weight downloads (documented in README)
- Claude API key configuration (user-specific)
- Reference image curation for LUT generation
- davinci-resolve-mcp server installation on host (documented in README)
