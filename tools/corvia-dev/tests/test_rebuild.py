"""Tests for rebuild module."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from unittest.mock import MagicMock, patch

from corvia_dev.rebuild import cargo_build, check_staleness, install_binaries


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
        assert (install_dir / name).stat().st_mode & stat.S_IEXEC


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
