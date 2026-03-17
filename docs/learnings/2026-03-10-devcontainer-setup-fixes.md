# Devcontainer Setup Fixes (2026-03-10)

## Problems

Three cascading failures in the devcontainer lifecycle scripts.

### 1. `uv` install fails during Docker build

**Symptom:** `post-create.sh` fails with "Neither uv nor pip available" despite
the Dockerfile installing `uv`.

**Root cause:** The Dockerfile used `curl https://astral.sh/uv/install.sh | sh`
which is unreliable in Docker builds (connection resets, rate limits). The entire
`RUN` block was chained with `|| echo "WARNING: ..."` which silently swallowed
the failure.

**Fix:** Multi-stage `COPY --from=ghcr.io/astral-sh/uv:latest` — copies the `uv`
binary directly from the official image. No network call during build, no failure
modes.

```dockerfile
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM rust:1.88-trixie
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx
```

**Additional safeguard:** `lib.sh` adds `/root/.local/bin` to PATH so `uv` is
found even if it ends up in the default install location instead of `/usr/local/bin`.

### 2. `corvia-dev` not installed (cascade from #1)

**Symptom:** `post-create.sh` → `ensure_tooling` → `install_python_editable`
fails because `uv` is missing. Then `post-start.sh` also fails trying to run
`corvia-dev up`.

**Root cause:** Same as #1. `install_python_editable` had a pip fallback that
also failed because `python3-pip` wasn't in the Dockerfile's apt packages.

**Fix:** Removed pip fallback entirely — `uv`-only. With the multi-stage uv
install, this path is reliable.

### 3. VS Code extension not appearing in devcontainer

**Symptom:** The corvia-services dashboard extension doesn't show in the status
bar. `post-start.sh` logs "code CLI not available after 30s" or "Command is only
available in WSL or inside a Visual Studio Code terminal."

**Root cause:** The `code` CLI is a wrapper that needs VS Code's IPC socket
(`VSCODE_IPC_HOOK_CLI`). During `postStartCommand`, this socket isn't ready yet.
The binary exists on disk but can't actually install extensions.

**Fix:** Moved extension installation to `post-create.sh` (runs before VS Code
opens). Uses `install_vsix_direct` which extracts the `.vsix` (a zip file)
directly into `/root/.vscode-server/extensions/` using Python's `zipfile` module.
No `code` CLI dependency at all.

```
.vsix → python3 zipfile extract → /root/.vscode-server/extensions/{publisher}.{name}-{version}/
```

## Key Takeaways

| Lesson | Detail |
|--------|--------|
| **Don't mask build failures** | `\|\| echo "WARNING"` in Dockerfiles hides real problems. Let critical steps fail loudly. |
| **Multi-stage COPY > curl install** | For tools with official Docker images, `COPY --from` is always more reliable than downloading during build. |
| **`code` CLI needs IPC** | The VS Code remote CLI doesn't work in `postStartCommand` or `postCreateCommand` — it needs the full VS Code server running. |
| **Direct disk install works** | VS Code extensions are just directories in `~/.vscode-server/extensions/`. Extracting the `.vsix` manually is perfectly valid. |
| **Cascade failures are sneaky** | One silent failure (`uv` install) cascaded into three separate symptoms across two scripts. |

## Files Changed

- `.devcontainer/Dockerfile` — multi-stage uv, split RUN layers
- `.devcontainer/scripts/lib.sh` — PATH fix, uv-only install, `install_vsix_direct`
- `.devcontainer/scripts/post-create.sh` — added extension install step
- `.devcontainer/scripts/post-start.sh` — removed extension install (now in post-create)

## Commit

`746baef` — `fix(devcontainer): reliable uv install and VS Code extension setup`
