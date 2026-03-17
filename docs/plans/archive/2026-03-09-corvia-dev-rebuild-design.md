# corvia-dev rebuild + Staleness Detection

> **Status:** Superseded — rebuild functionality moved to Rust CLI
> **Date**: 2026-03-09
> **Scope**: corvia-dev CLI (Python, `tools/corvia-dev/`)

## Problem

After building new Rust code with `cargo build`, binaries land in `target/debug/` but
running services use `/usr/local/bin/` (installed from GitHub releases at container
creation). There is no command to sync them — users must manually copy binaries and
restart services, leading to confusing "stale binary" debugging sessions.

## Solution

Two additions to `corvia-dev`:

1. **`corvia-dev rebuild`** — builds from source, installs, and restarts services
2. **Staleness detection** — warns or prompts when installed binaries are older than local builds

## New Command: `corvia-dev rebuild`

```
corvia-dev rebuild [--no-build] [--release]
```

### Steps

1. Run `cargo build` in `repos/corvia/` (skip with `--no-build`, use `--release` for optimized build)
2. Copy `target/{debug,release}/corvia` → `/usr/local/bin/corvia`
3. Copy `target/{debug,release}/corvia-inference` → `/usr/local/bin/corvia-inference`
4. Restart running services that use those binaries (`corvia-server`, `corvia-inference`)

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--no-build` | false | Skip `cargo build`, only install + restart (fails if target binary missing) |
| `--release` | false | Use `cargo build --release` and install from `target/release/` |

## Staleness Detection

### Mechanism

Compare mtime of `target/debug/corvia` vs `/usr/local/bin/corvia` (same for
`corvia-inference`). A binary is "stale" when:

- The target binary exists, AND
- The target binary's mtime is newer than the installed binary's mtime

If the target binary doesn't exist (nothing built yet), no warning is shown.

### Where it triggers

| Context | Behavior |
|---------|----------|
| `corvia-dev up` | Interactive prompt: `Newer build detected. Install and restart? [Y/n]` |
| `corvia-dev status` | Yellow warning line in output |
| `corvia-dev status --json` | `"stale_binaries": ["corvia", "corvia-inference"]` field |

### Prompt behavior on `corvia-dev up`

- If stdin is a TTY: prompt `Newer build detected for {names}. Install and restart? [Y/n]`
- If not a TTY (e.g. `--no-foreground` from post-start.sh): warn to stderr, do not prompt, continue with existing binaries

## Files Changed

| File | Change |
|------|--------|
| `tools/corvia-dev/corvia_dev/rebuild.py` | New module (~80 lines): build, install, staleness check |
| `tools/corvia-dev/corvia_dev/cli.py` | Add `rebuild` command, integrate staleness check into `up` |
| `tools/corvia-dev/corvia_dev/manager.py` | Add staleness check before service start |

## Non-Goals

- Version pinning or rollback support
- Git hash embedding in binaries
- Pulling updates from GitHub releases (`corvia-dev update`)
- Checking source file changes (only checks built artifacts)
