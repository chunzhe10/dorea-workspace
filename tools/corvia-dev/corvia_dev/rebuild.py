"""Build, install, and staleness detection for corvia binaries."""

from __future__ import annotations

import shutil
import subprocess
import stat
from pathlib import Path

BINARY_NAMES = ["corvia", "corvia-inference"]
DEFAULT_INSTALL_DIR = Path("/usr/local/bin")

# ORT execution provider shared libraries required for GPU inference.
# libonnxruntime_providers_shared.so is the EP plugin loader — needed by ALL EPs.
# libonnxruntime_providers_openvino.so enables Intel iGPU via OpenVINO.
ORT_PROVIDER_LIBS = [
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_openvino.so",
]


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


def install_binaries(
    target_dir: Path,
    install_dir: Path = DEFAULT_INSTALL_DIR,
) -> list[str]:
    """Copy built binaries from target_dir to install_dir.

    Also installs ORT provider shared libraries required for GPU execution
    providers (CUDA, OpenVINO). Without these, ORT silently falls back to CPU.

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

    # Install ORT provider shared libraries to system lib path.
    # These are downloaded by the ort crate during cargo build and symlinked
    # into target/release/. They're required for non-CPU execution providers.
    ort_lib_dir = Path("/usr/lib/x86_64-linux-gnu")
    for lib_name in ORT_PROVIDER_LIBS:
        src = target_dir / lib_name
        if not src.exists():
            continue
        # Resolve symlinks (target/release/*.so → pyke.io cache)
        real_src = src.resolve()
        if not real_src.exists():
            continue
        dst = ort_lib_dir / lib_name
        shutil.copy2(real_src, dst)

    # Refresh the dynamic linker cache so dlopen finds the new .so files
    # without requiring a container restart.
    subprocess.run(["ldconfig"], check=False)

    return installed


def cargo_build(workspace_root: Path, release: bool = False) -> bool:
    """Run cargo build for corvia binaries.

    Returns True on success, False on failure.
    """
    # --features corvia-inference/cuda enables llama-cpp CUDA inference for chat.
    # Requires CUDA toolkit (nvcc) at build time; the runtime GPU is provided via docker passthrough.
    cmd = ["cargo", "build", "-p", "corvia-cli", "-p", "corvia-inference",
           "--features", "corvia-inference/cuda"]
    if release:
        cmd.append("--release")

    result = subprocess.run(
        cmd,
        cwd=workspace_root / "repos" / "corvia",
    )
    return result.returncode == 0
