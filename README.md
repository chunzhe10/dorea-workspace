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
