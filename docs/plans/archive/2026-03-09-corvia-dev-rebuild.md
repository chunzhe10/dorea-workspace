# corvia-dev rebuild Implementation Plan

> **Status:** Superseded — rebuild functionality moved to Rust CLI

**Goal:** Add a `corvia-dev rebuild` command that builds Rust binaries from source, installs them, and restarts services — plus staleness detection that prompts on `corvia-dev up` and warns on `corvia-dev status`.

**Architecture:** New `rebuild.py` module handles build/install/staleness logic. `cli.py` gets a `rebuild` command and integrates staleness checks into `up` and `status`. No changes to `manager.py` — staleness is checked before the manager starts.

**Tech Stack:** Python (click, subprocess, pathlib, shutil). Tests with pytest.

---

### Task 1: Create `rebuild.py` — staleness check

**Files:**
- Create: `tools/corvia-dev/corvia_dev/rebuild.py`
- Test: `tools/corvia-dev/tests/test_rebuild.py`

**Step 1: Write the failing test**

Create `tools/corvia-dev/tests/test_rebuild.py`:

```python
"""Tests for rebuild module."""

from __future__ import annotations

import os
from pathlib import Path

from corvia_dev.rebuild import check_staleness


def test_no_target_binary_not_stale(tmp_path: Path) -> None:
    """No target binary means nothing was built — not stale."""
    installed = tmp_path / "installed" / "corvia"
    installed.parent.mkdir()
    installed.write_text("old")

    result = check_staleness(
        workspace_root=tmp_path,
        target_dir=tmp_path / "target" / "debug",
        install_dir=installed.parent,
    )
    assert result == []


def test_target_newer_than_installed_is_stale(tmp_path: Path) -> None:
    """Target binary newer than installed → stale."""
    install_dir = tmp_path / "installed"
    install_dir.mkdir()
    target_dir = tmp_path / "target" / "debug"
    target_dir.mkdir(parents=True)

    for name in ("corvia", "corvia-inference"):
        installed = install_dir / name
        installed.write_text("old")
        os.utime(installed, (1000, 1000))

        target = target_dir / name
        target.write_text("new")
        os.utime(target, (2000, 2000))

    result = check_staleness(
        workspace_root=tmp_path,
        target_dir=target_dir,
        install_dir=install_dir,
    )
    assert sorted(result) == ["corvia", "corvia-inference"]


def test_installed_newer_than_target_not_stale(tmp_path: Path) -> None:
    """Installed binary newer than target → not stale."""
    install_dir = tmp_path / "installed"
    install_dir.mkdir()
    target_dir = tmp_path / "target" / "debug"
    target_dir.mkdir(parents=True)

    for name in ("corvia", "corvia-inference"):
        target = target_dir / name
        target.write_text("old")
        os.utime(target, (1000, 1000))

        installed = install_dir / name
        installed.write_text("new")
        os.utime(installed, (2000, 2000))

    result = check_staleness(
        workspace_root=tmp_path,
        target_dir=target_dir,
        install_dir=install_dir,
    )
    assert result == []
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py -v`
Expected: FAIL with `ImportError: cannot import name 'check_staleness'`

**Step 3: Write minimal implementation**

Create `tools/corvia-dev/corvia_dev/rebuild.py`:

```python
"""Build, install, and staleness detection for corvia binaries."""

from __future__ import annotations

from pathlib import Path

BINARY_NAMES = ["corvia", "corvia-inference"]
DEFAULT_INSTALL_DIR = Path("/usr/local/bin")


def check_staleness(
    workspace_root: Path,
    target_dir: Path | None = None,
    install_dir: Path = DEFAULT_INSTALL_DIR,
) -> list[str]:
    """Check which installed binaries are older than their local build.

    Returns a list of binary names that are stale (target is newer than installed).
    Returns empty list if no target binaries exist or all are up to date.
    """
    if target_dir is None:
        target_dir = workspace_root / "repos" / "corvia" / "target" / "debug"

    stale: list[str] = []
    for name in BINARY_NAMES:
        target = target_dir / name
        installed = install_dir / name
        if not target.exists() or not installed.exists():
            continue
        if target.stat().st_mtime > installed.stat().st_mtime:
            stale.append(name)
    return stale
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/rebuild.py tools/corvia-dev/tests/test_rebuild.py
git commit -m "feat(corvia-dev): add staleness check for installed binaries"
```

---

### Task 2: Add `install_binaries` to `rebuild.py`

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/rebuild.py`
- Test: `tools/corvia-dev/tests/test_rebuild.py`

**Step 1: Write the failing test**

Append to `tools/corvia-dev/tests/test_rebuild.py`:

```python
from corvia_dev.rebuild import install_binaries


def test_install_copies_binaries(tmp_path: Path) -> None:
    """install_binaries copies from target to install dir."""
    target_dir = tmp_path / "target" / "debug"
    target_dir.mkdir(parents=True)
    install_dir = tmp_path / "installed"
    install_dir.mkdir()

    for name in ("corvia", "corvia-inference"):
        (target_dir / name).write_bytes(b"binary-content-" + name.encode())

    installed = install_binaries(target_dir=target_dir, install_dir=install_dir)
    assert sorted(installed) == ["corvia", "corvia-inference"]
    for name in ("corvia", "corvia-inference"):
        assert (install_dir / name).read_bytes() == b"binary-content-" + name.encode()


def test_install_skips_missing_target(tmp_path: Path) -> None:
    """install_binaries skips binaries that don't exist in target."""
    target_dir = tmp_path / "target" / "debug"
    target_dir.mkdir(parents=True)
    install_dir = tmp_path / "installed"
    install_dir.mkdir()

    # Only create corvia, not corvia-inference
    (target_dir / "corvia").write_bytes(b"binary")

    installed = install_binaries(target_dir=target_dir, install_dir=install_dir)
    assert installed == ["corvia"]
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py::test_install_copies_binaries -v`
Expected: FAIL with `ImportError: cannot import name 'install_binaries'`

**Step 3: Write minimal implementation**

Add to `tools/corvia-dev/corvia_dev/rebuild.py`:

```python
import shutil
import stat


def install_binaries(
    target_dir: Path,
    install_dir: Path = DEFAULT_INSTALL_DIR,
) -> list[str]:
    """Copy built binaries from target_dir to install_dir.

    Returns list of binary names that were installed.
    """
    installed: list[str] = []
    for name in BINARY_NAMES:
        src = target_dir / name
        if not src.exists():
            continue
        dst = install_dir / name
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
        installed.append(name)
    return installed
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/rebuild.py tools/corvia-dev/tests/test_rebuild.py
git commit -m "feat(corvia-dev): add install_binaries to rebuild module"
```

---

### Task 3: Add `cargo_build` to `rebuild.py`

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/rebuild.py`
- Test: `tools/corvia-dev/tests/test_rebuild.py`

**Step 1: Write the failing test**

Append to `tools/corvia-dev/tests/test_rebuild.py`:

```python
from unittest.mock import patch, MagicMock
from corvia_dev.rebuild import cargo_build


def test_cargo_build_runs_debug(tmp_path: Path) -> None:
    """cargo_build calls cargo build in the corvia repo."""
    with patch("corvia_dev.rebuild.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = cargo_build(workspace_root=tmp_path, release=False)
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ["cargo", "build", "-p", "corvia-cli", "-p", "corvia-inference"]
        assert args[1]["cwd"] == tmp_path / "repos" / "corvia"


def test_cargo_build_release_flag(tmp_path: Path) -> None:
    """cargo_build passes --release when requested."""
    with patch("corvia_dev.rebuild.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        cargo_build(workspace_root=tmp_path, release=True)
        cmd = mock_run.call_args[0][0]
        assert "--release" in cmd


def test_cargo_build_returns_false_on_failure(tmp_path: Path) -> None:
    """cargo_build returns False when cargo fails."""
    with patch("corvia_dev.rebuild.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        result = cargo_build(workspace_root=tmp_path, release=False)
        assert result is False
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py::test_cargo_build_runs_debug -v`
Expected: FAIL with `ImportError: cannot import name 'cargo_build'`

**Step 3: Write minimal implementation**

Add to `tools/corvia-dev/corvia_dev/rebuild.py` (add `import subprocess` at top):

```python
import subprocess


def cargo_build(workspace_root: Path, release: bool = False) -> bool:
    """Run cargo build for corvia binaries.

    Returns True on success, False on failure.
    """
    cmd = ["cargo", "build", "-p", "corvia-cli", "-p", "corvia-inference"]
    if release:
        cmd.append("--release")

    result = subprocess.run(
        cmd,
        cwd=workspace_root / "repos" / "corvia",
    )
    return result.returncode == 0
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/test_rebuild.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/rebuild.py tools/corvia-dev/tests/test_rebuild.py
git commit -m "feat(corvia-dev): add cargo_build to rebuild module"
```

---

### Task 4: Add `rebuild` command to CLI

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/cli.py`

**Step 1: Add the rebuild command**

Add import at top of `cli.py` (after existing imports around line 24):

```python
from corvia_dev.rebuild import cargo_build, check_staleness, install_binaries, DEFAULT_INSTALL_DIR
```

Add the command before the `if __name__` block (before line 344):

```python
@main.command()
@click.option("--no-build", is_flag=True, help="Skip cargo build, only install and restart")
@click.option("--release", is_flag=True, help="Build with --release")
def rebuild(no_build: bool, release: bool) -> None:
    """Build from source, install binaries, and restart services."""
    workspace = _workspace_root()
    profile = "release" if release else "debug"
    target_dir = workspace / "repos" / "corvia" / "target" / profile

    if not no_build:
        click.echo(f"Building corvia binaries ({profile})...")
        if not cargo_build(workspace_root=workspace, release=release):
            click.echo("Build failed.", err=True)
            raise SystemExit(1)
        click.echo("Build succeeded.")

    if not target_dir.exists():
        click.echo(f"Target directory not found: {target_dir}", err=True)
        raise SystemExit(1)

    click.echo("Installing binaries...")
    installed = install_binaries(target_dir=target_dir)
    if not installed:
        click.echo("No binaries found to install.", err=True)
        raise SystemExit(1)
    for name in installed:
        click.echo(f"  {name} -> {DEFAULT_INSTALL_DIR / name}")

    # Restart services if manager is running
    if DEFAULT_STATE_PATH.exists():
        click.echo("Restarting services...")
        try:
            data = json.loads(DEFAULT_STATE_PATH.read_text())
            resp = StatusResponse.model_validate(data)
            if resp.manager and resp.manager.pid:
                os.kill(resp.manager.pid, signal.SIGTERM)
                time.sleep(2)
                ctx = click.get_current_context()
                ctx.invoke(up, no_foreground=True)
        except (json.JSONDecodeError, ValueError, ProcessLookupError) as e:
            click.echo(f"Failed to restart: {e}", err=True)
    else:
        click.echo("No running manager. Run 'corvia-dev up' to start services.")
```

**Step 2: Test manually**

Run: `cd /workspaces/corvia-workspace && corvia-dev rebuild --help`
Expected: Shows help with `--no-build` and `--release` flags.

**Step 3: Commit**

```bash
git add tools/corvia-dev/corvia_dev/cli.py
git commit -m "feat(corvia-dev): add rebuild command"
```

---

### Task 5: Integrate staleness check into `up` command

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/cli.py`

**Step 1: Add staleness prompt to `up` command**

In `cli.py`, modify the `up` function. Insert staleness check after loading config and before creating the ProcessManager (after line 177, before line 179):

```python
    # Check for stale binaries
    stale = check_staleness(workspace_root=_workspace_root())
    if stale:
        names = ", ".join(stale)
        if sys.stdin.isatty() and not no_foreground:
            if click.confirm(
                f"Newer build detected for {names}. Install and restart?",
                default=True,
            ):
                installed = install_binaries(
                    target_dir=_workspace_root() / "repos" / "corvia" / "target" / "debug",
                )
                for name in installed:
                    click.echo(f"  Installed {name}")
        else:
            click.echo(
                f"Warning: installed binaries are older than local build: {names}",
                err=True,
            )
            click.echo(
                "  Run 'corvia-dev rebuild' to update.",
                err=True,
            )
```

**Step 2: Test manually**

Run: `corvia-dev up --help` (should still work)
Then test with a stale binary scenario if possible.

**Step 3: Commit**

```bash
git add tools/corvia-dev/corvia_dev/cli.py
git commit -m "feat(corvia-dev): prompt for stale binaries on up"
```

---

### Task 6: Integrate staleness warning into `status` command

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/cli.py`
- Modify: `tools/corvia-dev/corvia_dev/models.py`

**Step 1: Add `stale_binaries` field to `StatusResponse`**

In `models.py`, add to the `StatusResponse` class (after line 70):

```python
    stale_binaries: list[str] = []
```

**Step 2: Add staleness check to `status` command**

In `cli.py`, in the `status` function, add staleness check. After loading the state from file (around line 51, inside the `if DEFAULT_STATE_PATH.exists():` block, before returning):

```python
            # Check staleness regardless of state source
            resp.stale_binaries = check_staleness(workspace_root=_workspace_root())
```

Also add the same to the fallback path (around line 118, before the `if as_json:` check):

```python
    resp.stale_binaries = check_staleness(workspace_root=_workspace_root())
```

**Step 3: Add warning to human-readable output**

In `_print_status_human`, add after the enabled services block (after line 158):

```python
    if resp.stale_binaries:
        names = ", ".join(resp.stale_binaries)
        click.echo(f"\n  Warning: stale binaries: {names}")
        click.echo(f"  Run 'corvia-dev rebuild' to update.")
```

**Step 4: Test manually**

Run: `corvia-dev status`
Expected: If binaries are stale, shows warning. JSON output includes `stale_binaries` field.

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/cli.py tools/corvia-dev/corvia_dev/models.py
git commit -m "feat(corvia-dev): show stale binary warning in status"
```

---

### Task 7: Run full test suite and verify

**Step 1: Run all tests**

Run: `cd /workspaces/corvia-workspace/tools/corvia-dev && python -m pytest tests/ -v`
Expected: All tests pass (existing + new).

**Step 2: Manual integration test**

```bash
# Verify rebuild command works end-to-end
corvia-dev rebuild --no-build   # should install from existing target/debug
corvia-dev status               # should show no stale warning now
corvia-dev rebuild --help       # should show flags
```

**Step 3: Commit any fixes if needed**

```bash
git add -u
git commit -m "fix(corvia-dev): address test/integration issues"
```
