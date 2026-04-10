# Delta Upscale Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a re-runnable benchmark that compares 9 delta-upscale methods against a RAUNE-at-4K gold standard on 3 auto-selected frames, and produces a contact sheet + metrics CSV + run report.

**Architecture:** A standalone Python package at `repos/dorea/benchmarks/upscale_bench/` with a decorator-based method registry. Methods are pure functions `(delta_proxy, orig_full, full_size) → delta_full`. A driver loads RAUNE once, computes a cached gold per frame, runs each method with warm-up + timing, scores against the gold, and emits visualizations. The driver reuses the exact OKLab transfer kernel from `raune_filter.py` so only the upscale step varies across rows.

**Tech Stack:** Python 3.13, PyTorch 2.6 (cuda 12.4), Triton (already in venv), PyAV (decode), PIL (visualize), `nvidia-vfx` (Maxine SR), optional `realesrgan`+`basicsr` (Real-ESRGAN SR), `torchmetrics` (SSIM), `gdown` (setup), git-lfs (weights).

**Spec:** `docs/decisions/2026-04-10-delta-upscale-bench-design.md`

---

## File Structure

### Files created

```
repos/dorea/
├── benchmarks/
│   └── upscale_bench/
│       ├── __init__.py
│       ├── methods.py            # @register registry + 9 method functions
│       ├── gold.py               # compute_gold with native + tiled + cache
│       ├── metrics.py            # delta/ΔE2000/SSIM/timing
│       ├── visualize.py          # summary grid + per-frame sheet + heatmaps
│       ├── frame_select.py       # heuristic auto-selection + decode
│       ├── run.py                # CLI entry point
│       ├── oklab.py              # small OKLab RGB↔Lab helpers, borrowed from raune_filter.py style
│       ├── README.md             # usage + "how to add a method"
│       └── tests/
│           ├── __init__.py
│           ├── conftest.py
│           ├── test_methods.py   # parametrized per-method smoke tests
│           └── test_e2e.py       # end-to-end CLI subprocess test
├── models/
│   └── raune_net/
│       └── models/
│           └── RAUNE-Net/
│               ├── models/           # downloaded by setup (raune_net.py, resnet.py, cbam.py, ...)
│               │   └── .gitkeep
│               └── pretrained/
│                   └── RAUNENet/
│                       └── test/
│                           └── .gitkeep        # weights_95.pth goes here via LFS
├── scripts/
│   ├── setup_bench.sh                          # pip installs + git-lfs init
│   └── download_raune_weights.sh               # gdown weights + curl model .py files
└── .gitattributes                              # LFS tracking for *.pth
```

### Files modified

```
repos/dorea/dorea.toml          # update raune_weights, raune_models_dir
repos/dorea/.gitignore          # ensure models/raune_net/**/*.pth NOT ignored (LFS takes over)
```

### Files deliberately not touched

- `repos/dorea/python/dorea_inference/raune_filter.py` — no production changes in this plan
- `repos/dorea/python/dorea_inference/raune_net.py` — wrapper kept as-is
- `repos/dorea/crates/**` — no Rust changes

---

## Task 1: Setup infrastructure — git-lfs, dependencies, bench package skeleton

**Files:**
- Create: `repos/dorea/scripts/setup_bench.sh`
- Create: `repos/dorea/.gitattributes`
- Create: `repos/dorea/benchmarks/upscale_bench/__init__.py`
- Create: `repos/dorea/benchmarks/upscale_bench/tests/__init__.py`
- Create: `repos/dorea/benchmarks/upscale_bench/tests/conftest.py`

- [ ] **Step 1: Create the bench package skeleton**

```bash
mkdir -p /workspaces/dorea-workspace/repos/dorea/benchmarks/upscale_bench/tests
touch /workspaces/dorea-workspace/repos/dorea/benchmarks/upscale_bench/__init__.py
touch /workspaces/dorea-workspace/repos/dorea/benchmarks/upscale_bench/tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures for the upscale benchmark."""
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    """Every test in this suite needs CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("upscale_bench tests require CUDA", allow_module_level=True)


@pytest.fixture
def synthetic_proxy_delta():
    """Tiny OKLab delta at proxy resolution for smoke tests."""
    return torch.randn(1, 3, 36, 64, device="cuda", dtype=torch.float32) * 0.05


@pytest.fixture
def synthetic_full_rgb():
    """Tiny full-res RGB in [0, 1] for smoke tests."""
    return torch.rand(1, 3, 72, 128, device="cuda", dtype=torch.float32)
```

- [ ] **Step 3: Write `scripts/setup_bench.sh`**

```bash
#!/bin/bash
# Idempotent setup script for the upscale benchmark.
# Safe to re-run after any devcontainer rebuild.
set -euo pipefail

VENV_PIP="/opt/dorea-venv/bin/pip"
VENV_PY="/opt/dorea-venv/bin/python"

echo "=== upscale_bench setup ==="

# 1. gdown for downloading RAUNE weights from Google Drive
if ! $VENV_PY -c "import gdown" 2>/dev/null; then
    echo " => installing gdown"
    $VENV_PIP install --quiet gdown
else
    echo " => gdown already installed"
fi

# 2. Maxine (hard-required)
if ! $VENV_PY -c "import nvvfx" 2>/dev/null; then
    echo " => installing nvidia-vfx (Maxine)"
    $VENV_PIP install nvidia-vfx
else
    echo " => nvvfx already installed"
fi

# 3. Real-ESRGAN (optional, soft-failing)
if ! $VENV_PY -c "import realesrgan" 2>/dev/null; then
    echo " => attempting Real-ESRGAN install (optional)"
    if $VENV_PIP install --quiet basicsr realesrgan 2>&1; then
        echo "    Real-ESRGAN installed"
    else
        echo "    WARNING: Real-ESRGAN install failed; sr_realesrgan will be skipped at bench time"
    fi
else
    echo " => realesrgan already installed"
fi

# 4. torchmetrics for SSIM (optional, inlined fallback exists)
if ! $VENV_PY -c "import torchmetrics" 2>/dev/null; then
    echo " => installing torchmetrics (for SSIM)"
    $VENV_PIP install --quiet torchmetrics || echo "    WARNING: torchmetrics install failed; inlined SSIM fallback will be used"
fi

# 5. git-lfs
if ! command -v git-lfs >/dev/null 2>&1; then
    echo " => installing git-lfs"
    apt-get update && apt-get install -y git-lfs
fi
git lfs install

echo "=== setup_bench.sh: complete ==="
```

Make it executable:

```bash
chmod +x /workspaces/dorea-workspace/repos/dorea/scripts/setup_bench.sh
```

- [ ] **Step 4: Write `repos/dorea/.gitattributes`**

```gitattributes
# Git LFS: RAUNE and Real-ESRGAN model weights
models/**/*.pth filter=lfs diff=lfs merge=lfs -text
models/**/*.bin filter=lfs diff=lfs merge=lfs -text
models/**/*.safetensors filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 5: Run setup script**

```bash
bash /workspaces/dorea-workspace/repos/dorea/scripts/setup_bench.sh
```

Expected: all three pip packages confirmed installed (gdown, nvidia-vfx, torchmetrics). Real-ESRGAN may or may not install cleanly — if it fails, the message says so and we continue.

- [ ] **Step 6: Verify Maxine is importable**

```bash
/opt/dorea-venv/bin/python -c "import nvvfx; print('nvvfx version:', nvvfx.get_sdk_version())"
```

Expected: `nvvfx version: 1.2.0` (or similar — any version is fine as long as import succeeds)

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/__init__.py \
        repos/dorea/benchmarks/upscale_bench/tests/__init__.py \
        repos/dorea/benchmarks/upscale_bench/tests/conftest.py \
        repos/dorea/scripts/setup_bench.sh \
        repos/dorea/.gitattributes
git commit -m "$(cat <<'EOF'
feat(bench): scaffold upscale_bench package + setup script

Adds empty bench package, pytest conftest with CUDA gate, setup_bench.sh
(idempotent installer for gdown/nvvfx/realesrgan/torchmetrics/git-lfs),
and .gitattributes for LFS tracking of model weights under models/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: RAUNE weights — download script + weights placement

**Files:**
- Create: `repos/dorea/scripts/download_raune_weights.sh`
- Create: `repos/dorea/models/raune_net/models/RAUNE-Net/models/.gitkeep`
- Create: `repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/.gitkeep`

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p /workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/models
mkdir -p /workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test
touch /workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/models/.gitkeep
touch /workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/.gitkeep
```

- [ ] **Step 2: Write `scripts/download_raune_weights.sh`**

```bash
#!/bin/bash
# Download RAUNE-Net pretrained weights and the model class files.
# Idempotent: re-runs verify and skip if present.
set -euo pipefail

VENV_PY="/opt/dorea-venv/bin/python"
BASE="/workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net"
MODELS_DIR="$BASE/models"
WEIGHTS="$BASE/pretrained/RAUNENet/test/weights_95.pth"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1pjEh6s6-a3p7qBtkONSlYLmKrfgD6rBk"
MIRROR_RAW="https://raw.githubusercontent.com/mrtycoonshrinidhi6/RAUNE-Net-Underwater-image-Enhancement-Network/main/models"
# SHA256 of weights_95.pth — update after first download
EXPECTED_SHA256=""

echo "=== download_raune_weights.sh ==="

mkdir -p "$MODELS_DIR" "$(dirname "$WEIGHTS")"

# --- Model class .py files (from mirror via raw.githubusercontent.com) ---
MODEL_FILES=(raune_net.py resnet.py cbam.py utils.py)

for f in "${MODEL_FILES[@]}"; do
    dest="$MODELS_DIR/$f"
    if [ -f "$dest" ]; then
        echo " => $f already present"
        continue
    fi
    echo " => downloading $f"
    curl -fsSL -o "$dest" "$MIRROR_RAW/$f"
done

# Ensure the directory is a Python package
if [ ! -f "$MODELS_DIR/__init__.py" ]; then
    touch "$MODELS_DIR/__init__.py"
fi

# --- Weights (from Google Drive via gdown) ---
if [ -f "$WEIGHTS" ]; then
    if [ -n "$EXPECTED_SHA256" ]; then
        actual=$(sha256sum "$WEIGHTS" | awk '{print $1}')
        if [ "$actual" = "$EXPECTED_SHA256" ]; then
            echo " => weights present and SHA256 verified: $WEIGHTS"
            exit 0
        else
            echo " => SHA mismatch (expected $EXPECTED_SHA256, got $actual) — re-downloading"
        fi
    else
        echo " => weights present (no pinned SHA yet): $WEIGHTS"
        exit 0
    fi
fi

# Ensure gdown is installed
$VENV_PY -c "import gdown" 2>/dev/null || /opt/dorea-venv/bin/pip install gdown

# Download the whole pretrained folder into a temp dir, then move only what we need
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo " => downloading RAUNE weights from Google Drive (this may take a minute)"
/opt/dorea-venv/bin/gdown --folder "$GDRIVE_FOLDER" -O "$TMP"

# The gdown folder structure contains pretrained/ with RAUNENet/test/weights_95.pth
SRC=$(find "$TMP" -name "weights_95.pth" -path "*/RAUNENet/*" | head -1)
if [ -z "$SRC" ]; then
    echo "ERROR: weights_95.pth not found in downloaded Google Drive folder" >&2
    exit 1
fi

cp "$SRC" "$WEIGHTS"
actual=$(sha256sum "$WEIGHTS" | awk '{print $1}')
echo " => downloaded: $WEIGHTS"
echo " => sha256: $actual"

if [ -z "$EXPECTED_SHA256" ]; then
    echo ""
    echo "NOTE: no pinned SHA256 yet. Update EXPECTED_SHA256 in this script to:"
    echo "   $actual"
    echo ""
elif [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: sha256 mismatch after download. Expected $EXPECTED_SHA256, got $actual" >&2
    exit 1
fi

echo "=== download_raune_weights.sh: complete ==="
```

Make it executable:

```bash
chmod +x /workspaces/dorea-workspace/repos/dorea/scripts/download_raune_weights.sh
```

- [ ] **Step 3: Run the download script**

```bash
bash /workspaces/dorea-workspace/repos/dorea/scripts/download_raune_weights.sh
```

Expected: the 4 `.py` files end up in `repos/dorea/models/raune_net/models/RAUNE-Net/models/`, and `weights_95.pth` ends up in `repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/`. The script prints the SHA256 of the downloaded weights.

- [ ] **Step 4: Record the SHA256 and update the script**

Copy the SHA256 printed by Step 3 and replace the empty `EXPECTED_SHA256=""` line in `scripts/download_raune_weights.sh` with the actual value, e.g.:

```bash
EXPECTED_SHA256="<paste SHA256 from step 3 here>"
```

- [ ] **Step 5: Re-run to verify idempotency**

```bash
bash /workspaces/dorea-workspace/repos/dorea/scripts/download_raune_weights.sh
```

Expected: prints `weights present and SHA256 verified` and exits without re-downloading.

- [ ] **Step 6: Verify weights load in PyTorch**

```bash
/opt/dorea-venv/bin/python - <<'PYEOF'
import sys
from pathlib import Path

BASE = Path("/workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net")
sys.path.insert(0, str(BASE))

import torch
from models.raune_net import RauneNet

model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2)
state = torch.load(BASE / "pretrained/RAUNENet/test/weights_95.pth", map_location="cpu", weights_only=True)
# State dict may be wrapped in a nested key — handle both
if "model" in state:
    state = state["model"]
model.load_state_dict(state)
print("OK — RauneNet loaded, params:", sum(p.numel() for p in model.parameters()))
PYEOF
```

Expected: `OK — RauneNet loaded, params: <some number>`. If `load_state_dict` fails with a key mismatch, inspect the state_dict's first key (`list(state.keys())[0]`) — some checkpoint formats prefix with `module.` which needs stripping.

- [ ] **Step 7: Commit (weights and .py files via LFS)**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/scripts/download_raune_weights.sh \
        repos/dorea/models/raune_net/models/RAUNE-Net/models/ \
        repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth
git status --short
# Verify weights_95.pth is tracked as LFS:
git lfs status
```

Expected: `git lfs status` shows `weights_95.pth` in the "Objects to be committed" list as `LFS: <sha>`.

```bash
git commit -m "$(cat <<'EOF'
feat(bench): add RAUNE weights via git-lfs + download script

Downloads weights_95.pth from the RAUNE-Net Google Drive folder and the
model class .py files (raune_net.py, resnet.py, cbam.py, utils.py) from
the mrtycoonshrinidhi6 mirror. Script is idempotent and SHA-verifies
weights on subsequent runs.

Weights live at repos/dorea/models/raune_net/models/RAUNE-Net/ to match
the layout python/dorea_inference/raune_net.py expects at sys.path
insertion time.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update dorea.toml to new weights path + verify RAUNE inference works

**Files:**
- Modify: `repos/dorea/dorea.toml`

- [ ] **Step 1: Update `dorea.toml`**

```bash
cd /workspaces/dorea-workspace
```

Edit `repos/dorea/dorea.toml` — change lines 7–8 from:

```toml
raune_weights   = "/workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
raune_models_dir = "/workspaces/dorea-workspace/working/sea_thru_poc"
```

to:

```toml
raune_weights   = "/workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
raune_models_dir = "/workspaces/dorea-workspace/repos/dorea/models/raune_net"
```

- [ ] **Step 2: Verify the inference wrapper loads the model from the new path**

```bash
/opt/dorea-venv/bin/python - <<'PYEOF'
import sys
sys.path.insert(0, "/workspaces/dorea-workspace/repos/dorea/python")

from dorea_inference.raune_net import RauneNetInference

inf = RauneNetInference(
    models_dir="/workspaces/dorea-workspace/repos/dorea/models/raune_net",
    weights="/workspaces/dorea-workspace/repos/dorea/models/raune_net/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth",
    device="cuda",
)
print("OK — RauneNetInference loaded via wrapper")
PYEOF
```

Expected: `OK — RauneNetInference loaded via wrapper`. If import chain fails because `models/` lacks `__init__.py`, ensure Step 2 of Task 2 created it (check via `ls repos/dorea/models/raune_net/models/RAUNE-Net/models/`).

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/dorea.toml
git commit -m "$(cat <<'EOF'
chore(config): point raune_weights/models_dir to committed LFS location

Updates dorea.toml to use repos/dorea/models/raune_net/ instead of the
wiped working/sea_thru_poc/ path. Paths now survive devcontainer rebuilds
and fresh clones (via git-lfs).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Method registry + shared OKLab helpers + parametrized smoke test

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/oklab.py`
- Create: `repos/dorea/benchmarks/upscale_bench/methods.py`
- Create: `repos/dorea/benchmarks/upscale_bench/tests/test_methods.py`

- [ ] **Step 1: Write `oklab.py` — reusable OKLab↔RGB helpers**

```python
"""OKLab↔sRGB converters on CUDA, for the upscale_bench.

Copied in spirit from raune_filter.py but kept self-contained here so the
bench doesn't import raune_filter.py directly (which pulls in the 3-thread
pipeline runtime). These helpers operate on (N, 3, H, W) fp32 tensors in
[0, 1] range for RGB and arbitrary range for OKLab.
"""
from __future__ import annotations

import torch


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) fp32 sRGB in [0,1] → OKLab (L, a, b)."""
    lin = srgb_to_linear(rgb)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = l.clamp_min(0.0).pow(1.0 / 3.0)
    m_ = m.clamp_min(0.0).pow(1.0 / 3.0)
    s_ = s.clamp_min(0.0).pow(1.0 / 3.0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.cat([L, a, bb], dim=1)


def oklab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) OKLab → sRGB in [0,1] fp32 (clipped)."""
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_.pow(3)
    m = m_.pow(3)
    s = s_.pow(3)
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return linear_to_srgb(torch.cat([r, g, bb], dim=1)).clamp(0, 1)
```

- [ ] **Step 2: Write `methods.py` with the registry (empty — methods added in later tasks)**

```python
"""Upscale method registry for the delta upscale benchmark.

Methods are registered via @register("name") and must conform to the
contract in docs/decisions/2026-04-10-delta-upscale-bench-design.md:

    (delta_proxy, orig_full, full_size) -> delta_full

where every tensor is (1, 3, H, W) fp32 on CUDA.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F


REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Decorator: register an upscale method under ``name``."""
    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"method {name!r} already registered")
        REGISTRY[name] = fn
        fn.__bench_name__ = name
        return fn
    return decorator


# ─── Baseline ──────────────────────────────────────────────────────────────

@register("bilinear")
def bilinear(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Current production behavior: plain bilinear upscale of the delta."""
    return F.interpolate(delta_proxy, size=full_size, mode="bilinear", align_corners=False)
```

- [ ] **Step 3: Write the parametrized smoke test**

```python
# repos/dorea/benchmarks/upscale_bench/tests/test_methods.py
"""Parametrized smoke tests for every registered upscale method."""
import pytest
import torch

from benchmarks.upscale_bench.methods import REGISTRY


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    yield


@pytest.mark.parametrize("method_name", list(REGISTRY.keys()))
def test_method_smoke(method_name, synthetic_proxy_delta, synthetic_full_rgb):
    """Every registered method must produce the expected shape/dtype/device
    on tiny synthetic input, and finite values."""
    method = REGISTRY[method_name]
    full_size = (synthetic_full_rgb.shape[-2], synthetic_full_rgb.shape[-1])
    out = method(synthetic_proxy_delta, synthetic_full_rgb, full_size=full_size)
    assert out.shape == (1, 3, *full_size), f"{method_name} wrong shape: {out.shape}"
    assert out.dtype == torch.float32, f"{method_name} wrong dtype"
    assert out.device.type == "cuda", f"{method_name} wrong device"
    assert torch.isfinite(out).all(), f"{method_name} produced non-finite values"
    # OKLab deltas for reasonable inputs should stay bounded
    assert out.abs().max() < 1.0, f"{method_name} delta too large: max={out.abs().max().item()}"
```

- [ ] **Step 4: Add conftest with sys.path fix (if needed) so `benchmarks.upscale_bench` imports work**

Append to `repos/dorea/benchmarks/upscale_bench/tests/conftest.py`:

```python
import sys
from pathlib import Path

# Add repos/dorea/ to sys.path so `from benchmarks.upscale_bench.methods import ...` works
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

- [ ] **Step 5: Run the smoke test to verify it passes on `bilinear`**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v
```

Expected: 1 test passes (`test_method_smoke[bilinear]`).

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/oklab.py \
        repos/dorea/benchmarks/upscale_bench/methods.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_methods.py \
        repos/dorea/benchmarks/upscale_bench/tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(bench): add method registry + baseline bilinear + smoke test

- OKLab↔sRGB helpers in oklab.py (self-contained, no raune_filter dep)
- @register decorator + REGISTRY dict in methods.py
- bilinear baseline registered
- parametrized smoke test covering every registered method

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Classical upscale methods — bicubic + lanczos3

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Add `bicubic` to methods.py**

Append after the `bilinear` function:

```python
@register("bicubic")
def bicubic(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Non-edge-aware bicubic upscale — one-line improvement over bilinear."""
    return F.interpolate(delta_proxy, size=full_size, mode="bicubic", align_corners=False)
```

- [ ] **Step 2: Add `lanczos3` to methods.py**

First, add the Lanczos kernel helper above the `@register` functions:

```python
def _lanczos_kernel(a: int = 3, factor: float = 2.0, device: str = "cuda") -> torch.Tensor:
    """Build a 1-D Lanczos-a kernel for integer-or-fractional upsampling factor.

    Returns a (1, 1, K) tensor that can be used with F.conv_transpose1d /
    conv1d for separable resampling. For non-integer factors we rely on the
    caller using F.conv2d with computed taps per-row.

    We keep this simple: compute taps dense enough for the factor and
    resample via F.interpolate(mode="bicubic") as the spatial engine, then
    apply a Lanczos-shaped 1-D sharpen pass. This is the "good enough"
    approach that matches Lanczos in the frequency domain without a full
    per-pixel tap implementation.
    """
    # Placeholder hook — the real implementation is inline in the lanczos3
    # function below via separable conv2d with precomputed taps.
    raise NotImplementedError
```

Wait — the above is a placeholder stub. Replace it with the real implementation. Add these helpers instead:

```python
def _lanczos_weight(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    """Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0."""
    pi_x = torch.pi * x
    pi_x_over_a = pi_x / a
    # sinc(x) = sin(pi*x) / (pi*x), with limit 1 at x=0
    sinc_x = torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(pi_x) / pi_x)
    sinc_x_a = torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(pi_x_over_a) / pi_x_over_a)
    weight = sinc_x * sinc_x_a
    return torch.where(x.abs() < a, weight, torch.zeros_like(weight))


def _lanczos_resample_1d(
    x: torch.Tensor,
    out_size: int,
    dim: int,
    a: int = 3,
) -> torch.Tensor:
    """Resample ``x`` along ``dim`` to ``out_size`` using Lanczos-a.

    Args:
        x: tensor, any number of dims
        out_size: target size along ``dim``
        dim: axis to resample
        a: Lanczos window radius (3 for lanczos3)
    """
    in_size = x.shape[dim]
    scale = out_size / in_size
    device = x.device
    dtype = x.dtype
    # For each output pixel, find the contributing input pixels and weights
    out_coords = torch.arange(out_size, device=device, dtype=dtype) + 0.5
    in_coords = out_coords / scale - 0.5  # center of input pixel
    # Support radius in input pixels
    support = a if scale >= 1.0 else a / scale
    support_int = int(torch.ceil(torch.tensor(support)).item())
    offsets = torch.arange(-support_int + 1, support_int + 1, device=device, dtype=dtype)
    # shape: (out_size, 2*support_int) — for each output, which input indices contribute
    centers = in_coords.unsqueeze(1)  # (out_size, 1)
    sample_coords = torch.floor(centers).unsqueeze(1).squeeze(-1) + offsets  # (out_size, 2*support_int)
    # Clamp indices to [0, in_size - 1] (edge replication)
    sample_idx = sample_coords.clamp(0, in_size - 1).long()
    # Distance from center (in input-pixel units), scaled if downsampling
    dist = (centers - sample_coords)
    if scale < 1.0:
        dist = dist * scale
    weights = _lanczos_weight(dist, a=a)  # (out_size, 2*support_int)
    # Normalize rows
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Gather contributing input pixels along `dim` and weight them
    # Move target dim to the end, gather, weight-sum, move back
    x_perm = x.movedim(dim, -1)  # (..., in_size)
    gathered = x_perm[..., sample_idx]  # (..., out_size, 2*support_int)
    w = weights.to(dtype)  # (out_size, 2*support_int)
    out = (gathered * w).sum(dim=-1)  # (..., out_size)
    return out.movedim(-1, dim)


@register("lanczos3")
def lanczos3(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Lanczos-3 upscale via separable 1-D resampling.

    PyTorch does not provide Lanczos in F.interpolate, so we implement it
    as two separable passes. This is slower than bicubic but closer to the
    ideal sinc-windowed reconstruction.
    """
    fh, fw = full_size
    # Resample width first (along dim=-1), then height (along dim=-2)
    resampled = _lanczos_resample_1d(delta_proxy, fw, dim=-1, a=3)
    resampled = _lanczos_resample_1d(resampled, fh, dim=-2, a=3)
    return resampled
```

Remove the placeholder `_lanczos_kernel` function you added above. The final `methods.py` has only the helpers + registered functions.

- [ ] **Step 3: Run the smoke test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v
```

Expected: 3 tests pass (`bilinear`, `bicubic`, `lanczos3`).

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py
git commit -m "$(cat <<'EOF'
feat(bench): add bicubic and lanczos3 upscale methods

bicubic is a 1-line F.interpolate swap. lanczos3 is a separable 1-D
Lanczos resampler (PyTorch has no Lanczos in F.interpolate) implemented
from scratch: compute taps per output row, gather+weighted-sum. Both
methods pass the parametrized smoke test.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Guided filter upscale method

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Add helpers + `guided_filter` function to methods.py**

Append to `methods.py`:

```python
from .oklab import rgb_to_oklab as _rgb_to_oklab


def _box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Separable box filter with radius r (window size 2r+1), reflection padding.

    Operates on (N, C, H, W). Used by the guided filter.
    """
    k = 2 * r + 1
    # Separable: horizontal then vertical average pool is a box filter
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.avg_pool2d(x, kernel_size=(1, k), stride=1)
    x = F.avg_pool2d(x, kernel_size=(k, 1), stride=1)
    return x


@register("guided_filter")
def guided_filter(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Joint bilateral-style upscale via the He/Sun/Tang 2010 guided filter.

    The guide is the full-resolution OKLab-L channel. The filter is run
    after a cheap initial upsample of the delta to full size.

    Hyperparameters: r=8 (full-res pixels), eps=1e-4.
    """
    r = 8
    eps = 1e-4
    fh, fw = full_size

    # 1. Compute the full-res luma guide from RGB
    guide = _rgb_to_oklab(orig_full)[:, 0:1]  # (1, 1, fh, fw) OKLab-L

    # 2. Cheap initial upsample of the delta to full size
    p = F.interpolate(delta_proxy, size=full_size, mode="bilinear", align_corners=False)

    # 3. Guided filter: process each of the 3 delta channels independently with the same guide
    out_channels = []
    for c in range(3):
        p_c = p[:, c:c + 1]  # (1, 1, fh, fw)
        mean_I = _box_filter(guide, r)
        mean_p = _box_filter(p_c, r)
        mean_Ip = _box_filter(guide * p_c, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = _box_filter(guide * guide, r)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = _box_filter(a, r)
        mean_b = _box_filter(b, r)

        q = mean_a * guide + mean_b
        out_channels.append(q)

    return torch.cat(out_channels, dim=1)
```

- [ ] **Step 2: Run the smoke test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v -k "bilinear or bicubic or lanczos3 or guided_filter"
```

Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py
git commit -m "$(cat <<'EOF'
feat(bench): add guided_filter upscale method

He/Sun/Tang 2010 guided filter using OKLab-L from orig_full as the guide,
radius=8 full-res pixels, eps=1e-4. Implemented as separable box filters
over the three delta channels independently, after a cheap bilinear
bootstrap upsample.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Joint bilateral upscale (Triton kernel) + asymmetric variant

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Add the Triton joint-bilateral kernel and register function**

Append to `methods.py`:

```python
import triton
import triton.language as tl


@triton.jit
def _joint_bilateral_kernel(
    # Delta at proxy: (3, ph, pw) fp32, channel-first flat
    delta_proxy_ptr,
    ph: tl.constexpr, pw: tl.constexpr,
    # Guide at full res: (fh, fw) fp32
    guide_full_ptr,
    # Output delta at full res: (3, fh, fw) fp32, channel-first flat
    out_ptr,
    fh: tl.constexpr, fw: tl.constexpr,
    # Hyperparameters
    sigma_spatial_L: tl.constexpr,
    sigma_spatial_ab: tl.constexpr,
    sigma_luma_sq_inv: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_pixels = fh * fw
    mask = offs < n_pixels

    y = offs // fw
    x = offs % fw

    # Map output pixel to proxy-grid coordinate (subpixel)
    scale_y = ph / fh
    scale_x = pw / fw
    py_f = (y.to(tl.float32) + 0.5) * scale_y - 0.5
    px_f = (x.to(tl.float32) + 0.5) * scale_x - 0.5

    py0 = tl.floor(py_f).to(tl.int32)
    px0 = tl.floor(px_f).to(tl.int32)
    # We gather a 2x2 neighborhood for simplicity (bilateral weights across 4 neighbors)
    # The spatial sigma controls how strongly the 4 contribute; guide similarity further modulates.

    # Clamp neighbor indices into [0, ph-1] / [0, pw-1]
    py0c = tl.minimum(tl.maximum(py0, 0), ph - 1)
    px0c = tl.minimum(tl.maximum(px0, 0), pw - 1)
    py1c = tl.minimum(py0 + 1, ph - 1)
    px1c = tl.minimum(px0 + 1, pw - 1)

    # Load the 4 delta-proxy neighbors for each of the 3 channels
    # channel stride = ph*pw
    ch_stride = ph * pw
    d00_L = tl.load(delta_proxy_ptr + 0 * ch_stride + py0c * pw + px0c, mask=mask)
    d01_L = tl.load(delta_proxy_ptr + 0 * ch_stride + py0c * pw + px1c, mask=mask)
    d10_L = tl.load(delta_proxy_ptr + 0 * ch_stride + py1c * pw + px0c, mask=mask)
    d11_L = tl.load(delta_proxy_ptr + 0 * ch_stride + py1c * pw + px1c, mask=mask)

    d00_a = tl.load(delta_proxy_ptr + 1 * ch_stride + py0c * pw + px0c, mask=mask)
    d01_a = tl.load(delta_proxy_ptr + 1 * ch_stride + py0c * pw + px1c, mask=mask)
    d10_a = tl.load(delta_proxy_ptr + 1 * ch_stride + py1c * pw + px0c, mask=mask)
    d11_a = tl.load(delta_proxy_ptr + 1 * ch_stride + py1c * pw + px1c, mask=mask)

    d00_b = tl.load(delta_proxy_ptr + 2 * ch_stride + py0c * pw + px0c, mask=mask)
    d01_b = tl.load(delta_proxy_ptr + 2 * ch_stride + py0c * pw + px1c, mask=mask)
    d10_b = tl.load(delta_proxy_ptr + 2 * ch_stride + py1c * pw + px0c, mask=mask)
    d11_b = tl.load(delta_proxy_ptr + 2 * ch_stride + py1c * pw + px1c, mask=mask)

    # Load the guide at the neighbor positions. The guide is full-res, so we
    # sample it at the corresponding full-res locations: for a proxy neighbor
    # (py, px), the full-res coord is (py + 0.5) / scale_y - 0.5 (rounded).
    ginv_y = fh / ph
    ginv_x = fw / pw

    def _guide_at(py_proxy, px_proxy):
        gy = ((py_proxy.to(tl.float32) + 0.5) * ginv_y - 0.5).to(tl.int32)
        gx = ((px_proxy.to(tl.float32) + 0.5) * ginv_x - 0.5).to(tl.int32)
        gy = tl.minimum(tl.maximum(gy, 0), fh - 1)
        gx = tl.minimum(tl.maximum(gx, 0), fw - 1)
        return tl.load(guide_full_ptr + gy * fw + gx, mask=mask)

    g_out = tl.load(guide_full_ptr + y * fw + x, mask=mask)
    g00 = _guide_at(py0c, px0c)
    g01 = _guide_at(py0c, px1c)
    g10 = _guide_at(py1c, px0c)
    g11 = _guide_at(py1c, px1c)

    # Spatial distances in proxy-pixel units
    dy0 = py_f - py0c.to(tl.float32)
    dy1 = py_f - py1c.to(tl.float32)
    dx0 = px_f - px0c.to(tl.float32)
    dx1 = px_f - px1c.to(tl.float32)

    def _w(dy, dx, dg, sigma_spatial):
        s2 = sigma_spatial * sigma_spatial
        spatial = tl.exp(-(dy * dy + dx * dx) / (2.0 * s2))
        luma = tl.exp(-(dg * dg) * sigma_luma_sq_inv * 0.5)
        return spatial * luma

    # L channel weights (sigma_spatial_L)
    wL00 = _w(dy0, dx0, g00 - g_out, sigma_spatial_L)
    wL01 = _w(dy0, dx1, g01 - g_out, sigma_spatial_L)
    wL10 = _w(dy1, dx0, g10 - g_out, sigma_spatial_L)
    wL11 = _w(dy1, dx1, g11 - g_out, sigma_spatial_L)
    wL_sum = wL00 + wL01 + wL10 + wL11
    out_L = (wL00 * d00_L + wL01 * d01_L + wL10 * d10_L + wL11 * d11_L) / (wL_sum + 1e-8)

    # a/b channels use sigma_spatial_ab
    wC00 = _w(dy0, dx0, g00 - g_out, sigma_spatial_ab)
    wC01 = _w(dy0, dx1, g01 - g_out, sigma_spatial_ab)
    wC10 = _w(dy1, dx0, g10 - g_out, sigma_spatial_ab)
    wC11 = _w(dy1, dx1, g11 - g_out, sigma_spatial_ab)
    wC_sum = wC00 + wC01 + wC10 + wC11
    out_a = (wC00 * d00_a + wC01 * d01_a + wC10 * d10_a + wC11 * d11_a) / (wC_sum + 1e-8)
    out_b = (wC00 * d00_b + wC01 * d01_b + wC10 * d10_b + wC11 * d11_b) / (wC_sum + 1e-8)

    tl.store(out_ptr + 0 * n_pixels + offs, out_L, mask=mask)
    tl.store(out_ptr + 1 * n_pixels + offs, out_a, mask=mask)
    tl.store(out_ptr + 2 * n_pixels + offs, out_b, mask=mask)


def _run_joint_bilateral_kernel(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
    sigma_spatial_L: float,
    sigma_spatial_ab: float,
    sigma_luma: float,
) -> torch.Tensor:
    """Shared driver for joint_bilateral and asymmetric_bilateral."""
    fh, fw = full_size
    assert delta_proxy.shape[0] == 1 and delta_proxy.shape[1] == 3
    _, _, ph, pw = delta_proxy.shape

    # Guide: full-resolution OKLab-L channel
    guide = _rgb_to_oklab(orig_full)[0, 0]  # (fh, fw)
    guide_flat = guide.contiguous()

    # Delta proxy: channel-first flat
    dp = delta_proxy[0].contiguous()  # (3, ph, pw)

    # Output: channel-first flat
    out = torch.empty(3, fh, fw, device="cuda", dtype=torch.float32)

    n_pixels = fh * fw
    BLOCK = 1024
    grid = ((n_pixels + BLOCK - 1) // BLOCK,)
    sigma_luma_sq_inv = 1.0 / (sigma_luma * sigma_luma)
    _joint_bilateral_kernel[grid](
        dp, ph, pw,
        guide_flat,
        out,
        fh, fw,
        sigma_spatial_L,
        sigma_spatial_ab,
        sigma_luma_sq_inv,
        BLOCK=BLOCK,
    )

    return out.unsqueeze(0)  # (1, 3, fh, fw)


@register("joint_bilateral")
def joint_bilateral(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Joint bilateral upsample with uniform spatial sigma across L/a/b."""
    return _run_joint_bilateral_kernel(
        delta_proxy, orig_full, full_size,
        sigma_spatial_L=1.5,
        sigma_spatial_ab=1.5,
        sigma_luma=0.1,
    )


@register("asymmetric_bilateral")
def asymmetric_bilateral(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Joint bilateral with larger spatial sigma for chroma (a/b) than luma (L)."""
    return _run_joint_bilateral_kernel(
        delta_proxy, orig_full, full_size,
        sigma_spatial_L=1.5,
        sigma_spatial_ab=3.0,
        sigma_luma=0.1,
    )
```

- [ ] **Step 2: Run the smoke test for the new methods**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v -k "joint_bilateral or asymmetric_bilateral"
```

Expected: 2 tests pass. Triton may emit a compile warning on first run — that's fine.

- [ ] **Step 3: Run the full smoke test**

```bash
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v
```

Expected: 6 tests pass (bilinear, bicubic, lanczos3, guided_filter, joint_bilateral, asymmetric_bilateral).

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py
git commit -m "$(cat <<'EOF'
feat(bench): add joint_bilateral + asymmetric_bilateral Triton kernels

Both methods share _joint_bilateral_kernel, a Triton kernel that gathers
4 neighboring proxy-grid samples per output pixel and weights them by
spatial distance × luma-guide similarity (using OKLab-L from orig_full).
The asymmetric variant uses a larger spatial sigma for chroma (a/b) than
for luma (L), on the theory that chroma bleeds are less visible.

Hyperparameters per spec:
- joint_bilateral:      σ_L = σ_ab = 1.5, σ_luma = 0.1
- asymmetric_bilateral: σ_L = 1.5, σ_ab = 3.0, σ_luma = 0.1

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Frame selection module

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/frame_select.py`

- [ ] **Step 1: Write `frame_select.py`**

```python
"""Frame selection: decode a subset of frames and score them on simple
heuristics to pick 3 diverse frames for the benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .oklab import rgb_to_oklab


def decode_frame(clip_path: Path, frame_idx: int) -> torch.Tensor:
    """Decode a single frame as (1, 3, H, W) fp32 CUDA tensor in [0, 1].

    Uses PyAV (already a dependency of raune_filter.py).
    """
    import av

    container = av.open(str(clip_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    target_pts = None
    # Seek to roughly the right place, then decode until we hit frame_idx
    # Simpler but slower: decode linearly until we reach the target index.
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            container.close()
            t = torch.from_numpy(arr).to("cuda", dtype=torch.float32) / 255.0
            return t.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
    container.close()
    raise IndexError(f"frame index {frame_idx} out of range for {clip_path}")


def _thumbnail(frame: torch.Tensor, target_long_edge: int = 540) -> torch.Tensor:
    """Downscale a (1, 3, H, W) frame to a thumbnail with long edge ``target_long_edge``."""
    _, _, h, w = frame.shape
    long_edge = max(h, w)
    if long_edge <= target_long_edge:
        return frame
    scale = target_long_edge / long_edge
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return F.interpolate(frame, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _score_edge_density(thumb: torch.Tensor) -> float:
    """Mean magnitude of Sobel gradients on OKLab-L."""
    L = rgb_to_oklab(thumb)[:, 0:1]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=L.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(L, sobel_x, padding=1)
    gy = F.conv2d(L, sobel_y, padding=1)
    mag = (gx * gx + gy * gy).sqrt()
    return mag.mean().item()


def _score_smoothness(thumb: torch.Tensor) -> float:
    """Negative of mean 32×32-block variance on OKLab-L (higher = flatter)."""
    L = rgb_to_oklab(thumb)[:, 0:1]
    b = 32
    _, _, h, w = L.shape
    h2 = (h // b) * b
    w2 = (w // b) * b
    if h2 == 0 or w2 == 0:
        return 0.0
    L = L[:, :, :h2, :w2]
    patches = L.unfold(2, b, b).unfold(3, b, b)  # (1, 1, Hb, Wb, 32, 32)
    mean_var = patches.reshape(1, 1, -1, b * b).var(dim=-1).mean().item()
    return -mean_var


def _score_chroma_magnitude(thumb: torch.Tensor) -> float:
    """Mean sqrt(a² + b²) in OKLab."""
    lab = rgb_to_oklab(thumb)
    a = lab[:, 1:2]
    b = lab[:, 2:3]
    chroma = (a * a + b * b).sqrt()
    return chroma.mean().item()


def auto_select(clip_path: Path, every_n: int = 30) -> List[int]:
    """Decode every Nth frame and pick 3 indices: one edge-heavy, one
    smooth, one high-chroma. Deterministic on the same clip."""
    import av

    container = av.open(str(clip_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    total = stream.frames or 0
    if total == 0:
        container.close()
        raise ValueError(f"cannot determine frame count of {clip_path}")
    candidates = list(range(0, total, every_n))

    thumbs: dict[int, torch.Tensor] = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in candidates:
            arr = frame.to_ndarray(format="rgb24")
            t = torch.from_numpy(arr).to("cuda", dtype=torch.float32) / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
            thumbs[i] = _thumbnail(t)
        if i >= candidates[-1]:
            break
    container.close()

    scores_edge: dict[int, float] = {}
    scores_smooth: dict[int, float] = {}
    scores_chroma: dict[int, float] = {}
    for idx, thumb in thumbs.items():
        scores_edge[idx] = _score_edge_density(thumb)
        scores_smooth[idx] = _score_smoothness(thumb)
        scores_chroma[idx] = _score_chroma_magnitude(thumb)

    best_edge = max(scores_edge, key=scores_edge.get)
    best_smooth = max(scores_smooth, key=scores_smooth.get)
    best_chroma = max(scores_chroma, key=scores_chroma.get)

    # Deduplicate while preserving order
    picked: List[int] = []
    for idx in (best_edge, best_smooth, best_chroma):
        if idx not in picked:
            picked.append(idx)

    # If dedup left us with fewer than 3 (rare: all criteria picked the same frame),
    # pad with the next best edge-density frames.
    if len(picked) < 3:
        ranked = sorted(scores_edge, key=scores_edge.get, reverse=True)
        for idx in ranked:
            if idx not in picked:
                picked.append(idx)
            if len(picked) == 3:
                break

    return picked[:3]
```

- [ ] **Step 2: Write a small sanity test for frame_select**

Append to `tests/test_methods.py` (or create `tests/test_frame_select.py`; create new to keep test files focused):

```python
# repos/dorea/benchmarks/upscale_bench/tests/test_frame_select.py
"""Sanity test for frame_select.auto_select on the real clip."""
from pathlib import Path

import pytest

from benchmarks.upscale_bench.frame_select import auto_select, decode_frame


CLIP = Path("/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4")


@pytest.mark.skipif(not CLIP.exists(), reason="test clip missing")
def test_auto_select_returns_three_distinct_indices():
    picks = auto_select(CLIP, every_n=60)
    assert len(picks) == 3
    assert len(set(picks)) == 3
    assert all(0 <= i for i in picks)


@pytest.mark.skipif(not CLIP.exists(), reason="test clip missing")
def test_decode_frame_shape():
    frame = decode_frame(CLIP, frame_idx=0)
    assert frame.shape[0] == 1
    assert frame.shape[1] == 3
    assert frame.shape[2] == 2160
    assert frame.shape[3] == 3840
    assert frame.dtype.is_floating_point
    assert frame.min().item() >= 0.0
    assert frame.max().item() <= 1.0
```

- [ ] **Step 3: Run the frame_select tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_frame_select.py -v
```

Expected: 2 tests pass (auto_select returns 3 distinct indices, decode_frame returns the right shape).

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/frame_select.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_frame_select.py
git commit -m "$(cat <<'EOF'
feat(bench): add frame_select with edge/smooth/chroma heuristics

auto_select decodes every Nth frame and picks 3 indices based on
(1) mean Sobel gradient on OKLab-L, (2) negative of 32x32 block variance,
and (3) mean chroma magnitude sqrt(a²+b²). Deterministic on the same clip.
decode_frame loads a single frame as (1, 3, H, W) fp32 CUDA tensor.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Gold-standard computation with tiled fallback + disk cache

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/gold.py`

- [ ] **Step 1: Write `gold.py`**

```python
"""Gold-standard computation: RAUNE at 4K (native or tiled) with disk cache."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from .oklab import rgb_to_oklab


@dataclass
class GoldResult:
    raune_4k: torch.Tensor      # (1, 3, 2160, 3840) fp32 in [0, 1]
    delta_4k: torch.Tensor      # (1, 3, 2160, 3840) OKLab delta fp32
    path_used: Literal["native_4k", "tiled_2x2_1984x1144_o128"]


def _sha1_file_prefix(path: Path, length: int = 8) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:length]


def _raune_forward(model, frame: torch.Tensor) -> torch.Tensor:
    """Run RAUNE on a full-size frame tensor. Handles [0,1] ↔ [-1,1] normalization."""
    x = frame * 2.0 - 1.0
    x = x.half()
    with torch.no_grad():
        y = model(x).float()
    return ((y + 1.0) / 2.0).clamp(0.0, 1.0)


def _load_raune_model(weights_path: Path, models_dir: Path) -> torch.nn.Module:
    """Load RauneNet, apply model.half() with InstanceNorm→fp32 fix from PR #67."""
    import torch.nn as nn

    # Add the models_dir/models/RAUNE-Net path so `from models.raune_net import RauneNet` works
    sys_path_entry = str(models_dir / "models" / "RAUNE-Net")
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)
    from models.raune_net import RauneNet  # type: ignore

    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model = model.cuda().eval()
    model.half()
    # InstanceNorm fp32 restoration (PR #67)
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.float()
    return model


def _tiled_4k_raune(model, frame_4k: torch.Tensor) -> torch.Tensor:
    """2×2 tiled RAUNE at 4K with 128-px overlap on inner seams, linear feathering."""
    assert frame_4k.shape == (1, 3, 2160, 3840)
    fh, fw = 2160, 3840
    overlap = 128
    tile_h = (fh + overlap) // 2  # 1144
    tile_w = (fw + overlap) // 2  # 1984

    # Tile positions (top-left):
    #   (0, 0)                  (0, fw - tile_w)
    #   (fh - tile_h, 0)        (fh - tile_h, fw - tile_w)
    positions = [
        (0, 0),
        (0, fw - tile_w),
        (fh - tile_h, 0),
        (fh - tile_h, fw - tile_w),
    ]

    out = torch.zeros_like(frame_4k)
    weight = torch.zeros(1, 1, fh, fw, device="cuda", dtype=torch.float32)

    # Per-tile feathering mask: linear ramp across the overlap band on the inner edges
    def _feather_mask(top: int, left: int) -> torch.Tensor:
        mask = torch.ones(1, 1, tile_h, tile_w, device="cuda", dtype=torch.float32)
        # Ramp on the inner edges only. If top > 0, the top edge is inner → ramp from 0→1 across first `overlap` rows.
        if top > 0:
            ramp = torch.linspace(0, 1, overlap, device="cuda", dtype=torch.float32).view(1, 1, -1, 1)
            mask[:, :, :overlap, :] *= ramp
        if top + tile_h < fh:
            ramp = torch.linspace(1, 0, overlap, device="cuda", dtype=torch.float32).view(1, 1, -1, 1)
            mask[:, :, -overlap:, :] *= ramp
        if left > 0:
            ramp = torch.linspace(0, 1, overlap, device="cuda", dtype=torch.float32).view(1, 1, 1, -1)
            mask[:, :, :, :overlap] *= ramp
        if left + tile_w < fw:
            ramp = torch.linspace(1, 0, overlap, device="cuda", dtype=torch.float32).view(1, 1, 1, -1)
            mask[:, :, :, -overlap:] *= ramp
        return mask

    for (top, left) in positions:
        tile = frame_4k[:, :, top:top + tile_h, left:left + tile_w].contiguous()
        enhanced = _raune_forward(model, tile)
        mask = _feather_mask(top, left)
        out[:, :, top:top + tile_h, left:left + tile_w] += enhanced * mask
        weight[:, :, top:top + tile_h, left:left + tile_w] += mask

    # Normalize by accumulated weights (no pixel should have weight=0)
    assert (weight > 0).all(), "tile stitching left gaps"
    out = out / weight
    return out.clamp(0.0, 1.0)


def compute_gold(
    frame_4k: torch.Tensor,
    raune_model: torch.nn.Module,
    *,
    force_path: Optional[Literal["native_4k", "tiled_2x2_1984x1144_o128"]] = None,
) -> GoldResult:
    """Run RAUNE at 4K, native-first with tiled fallback."""
    assert frame_4k.shape == (1, 3, 2160, 3840), f"unexpected shape {frame_4k.shape}"

    if force_path == "native_4k":
        raune_4k = _raune_forward(raune_model, frame_4k)
        path_used = "native_4k"
    elif force_path == "tiled_2x2_1984x1144_o128":
        raune_4k = _tiled_4k_raune(raune_model, frame_4k)
        path_used = "tiled_2x2_1984x1144_o128"
    else:
        try:
            raune_4k = _raune_forward(raune_model, frame_4k)
            path_used = "native_4k"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raune_4k = _tiled_4k_raune(raune_model, frame_4k)
            path_used = "tiled_2x2_1984x1144_o128"

    # Compute OKLab delta
    delta_4k = rgb_to_oklab(raune_4k) - rgb_to_oklab(frame_4k)
    return GoldResult(raune_4k=raune_4k, delta_4k=delta_4k, path_used=path_used)


def load_or_compute_gold(
    frame_4k: torch.Tensor,
    frame_idx: int,
    raune_model: torch.nn.Module,
    weights_path: Path,
    cache_dir: Path,
    regen: bool = False,
    force_path: Optional[str] = None,
) -> GoldResult:
    """Disk-cached wrapper around compute_gold."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sha_prefix = _sha1_file_prefix(weights_path)
    cache_file = cache_dir / f"{sha_prefix}_{frame_idx}.pt"

    if cache_file.exists() and not regen:
        try:
            blob = torch.load(cache_file, map_location="cuda", weights_only=False)
            return GoldResult(
                raune_4k=blob["raune_4k"],
                delta_4k=blob["delta_4k"],
                path_used=blob["path_used"],
            )
        except Exception as e:
            print(f"[gold] WARNING: corrupt cache {cache_file} ({e}); regenerating")

    result = compute_gold(frame_4k, raune_model, force_path=force_path)
    torch.save(
        {
            "raune_4k": result.raune_4k,
            "delta_4k": result.delta_4k,
            "path_used": result.path_used,
            "version": 1,
        },
        cache_file,
    )
    return result
```

- [ ] **Step 2: Write a smoke test for gold on a tiny synthetic 4K frame**

```python
# repos/dorea/benchmarks/upscale_bench/tests/test_gold.py
"""Smoke test for gold.compute_gold. Uses the real RAUNE model since the
benchmark depends on it anyway."""
from pathlib import Path

import pytest
import torch

from benchmarks.upscale_bench.gold import (
    compute_gold,
    load_or_compute_gold,
    _load_raune_model,
)


WEIGHTS = Path(
    "/workspaces/dorea-workspace/repos/dorea/models/raune_net/"
    "models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
)
MODELS_DIR = Path(
    "/workspaces/dorea-workspace/repos/dorea/models/raune_net"
)


@pytest.fixture(scope="module")
def raune_model():
    if not WEIGHTS.exists():
        pytest.skip("RAUNE weights missing; run scripts/download_raune_weights.sh")
    return _load_raune_model(WEIGHTS, MODELS_DIR)


def test_compute_gold_returns_expected_shape(raune_model):
    """On a random 4K frame, gold should return (1,3,2160,3840) raune_4k and delta_4k."""
    frame = torch.rand(1, 3, 2160, 3840, device="cuda", dtype=torch.float32) * 0.5
    result = compute_gold(frame, raune_model)
    assert result.raune_4k.shape == (1, 3, 2160, 3840)
    assert result.delta_4k.shape == (1, 3, 2160, 3840)
    assert result.path_used in ("native_4k", "tiled_2x2_1984x1144_o128")
    assert torch.isfinite(result.raune_4k).all()
    assert torch.isfinite(result.delta_4k).all()


def test_load_or_compute_gold_caches(raune_model, tmp_path):
    """Second call must hit cache and return same tensors."""
    frame = torch.rand(1, 3, 2160, 3840, device="cuda", dtype=torch.float32) * 0.5
    r1 = load_or_compute_gold(frame, 0, raune_model, WEIGHTS, tmp_path)
    r2 = load_or_compute_gold(frame, 0, raune_model, WEIGHTS, tmp_path)
    assert torch.allclose(r1.raune_4k, r2.raune_4k)
    assert torch.allclose(r1.delta_4k, r2.delta_4k)
    assert r1.path_used == r2.path_used
    # Confirm the cache file exists
    cache_files = list(tmp_path.glob("*.pt"))
    assert len(cache_files) == 1
```

- [ ] **Step 3: Run the gold test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_gold.py -v
```

Expected: 2 tests pass. The first call may take 5–30 seconds depending on whether native-4K fits; subsequent call is instant from the cache. If the test fails with OOM even in the tiled path, there's a real VRAM issue — investigate.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/gold.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_gold.py
git commit -m "$(cat <<'EOF'
feat(bench): add gold-standard computation with tiled fallback + cache

compute_gold runs RAUNE at native 4K first, catches OOM and falls back to
2x2 tiled (1984x1144 tiles with 128-px overlap, linear feathering).
load_or_compute_gold disk-caches results keyed by weights SHA prefix and
frame index, so re-runs are instant.

InstanceNorm fp32 restoration applied per PR #67 when loading the model.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `higher_proxy` control method (uses gold's tiled path)

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Add `higher_proxy` to methods.py**

The `higher_proxy` method is special — it doesn't take a proxy delta as input; it runs RAUNE at full resolution on `orig_full` directly. This means it doesn't fit the standard `(delta_proxy, orig_full, full_size) → delta_full` contract without help.

Resolution: the driver detects methods with a `__needs_model__` attribute and hands them the RAUNE model via a thread-local. Simpler: we add one attribute on the function and the driver checks for it. Append to `methods.py`:

```python
@register("higher_proxy")
def higher_proxy(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Control method: run RAUNE at full 4K on orig_full and compute the
    delta directly. Requires the RAUNE model — the driver injects it via
    a thread-local set before calling each method."""
    from .gold import compute_gold
    model = _get_injected_raune_model()
    gold = compute_gold(orig_full, model)
    return gold.delta_4k


higher_proxy.__needs_model__ = True  # type: ignore


# Thread-local injection point used by higher_proxy
_INJECTED_MODEL: dict[str, torch.nn.Module] = {}


def _get_injected_raune_model() -> torch.nn.Module:
    if "raune" not in _INJECTED_MODEL:
        raise RuntimeError(
            "higher_proxy called without an injected RAUNE model. "
            "The driver must call set_injected_raune_model() before running this method."
        )
    return _INJECTED_MODEL["raune"]


def set_injected_raune_model(model: torch.nn.Module) -> None:
    _INJECTED_MODEL["raune"] = model


def clear_injected_raune_model() -> None:
    _INJECTED_MODEL.pop("raune", None)
```

- [ ] **Step 2: Update the parametrized smoke test to skip methods that need the injected model**

The smoke test feeds synthetic tiny tensors. `higher_proxy` needs a real RAUNE model and a 4K-shaped frame. Skip it in the parametrized smoke and cover it separately.

Edit `tests/test_methods.py` — change the parametrize line:

```python
@pytest.mark.parametrize(
    "method_name",
    [name for name, fn in REGISTRY.items() if not getattr(fn, "__needs_model__", False)],
)
def test_method_smoke(method_name, synthetic_proxy_delta, synthetic_full_rgb):
    ...
```

- [ ] **Step 3: Add a specific test for `higher_proxy` in `tests/test_gold.py`**

Append:

```python
def test_higher_proxy_matches_gold(raune_model):
    """higher_proxy should produce the same delta as compute_gold on the same frame."""
    from benchmarks.upscale_bench.methods import (
        REGISTRY,
        set_injected_raune_model,
        clear_injected_raune_model,
    )
    from benchmarks.upscale_bench.gold import compute_gold

    frame = torch.rand(1, 3, 2160, 3840, device="cuda", dtype=torch.float32) * 0.5
    # Unused delta_proxy (higher_proxy ignores it)
    delta_proxy = torch.zeros(1, 3, 1080, 1920, device="cuda", dtype=torch.float32)

    set_injected_raune_model(raune_model)
    try:
        hp = REGISTRY["higher_proxy"](delta_proxy, frame, (2160, 3840))
    finally:
        clear_injected_raune_model()

    gold = compute_gold(frame, raune_model)
    assert torch.allclose(hp, gold.delta_4k, atol=1e-4)
```

- [ ] **Step 4: Run the tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/ -v
```

Expected: 6 smoke tests pass (bilinear, bicubic, lanczos3, guided_filter, joint_bilateral, asymmetric_bilateral — `higher_proxy` excluded by the parametrize filter), 2 frame_select tests pass, 3 gold tests pass (including the new `test_higher_proxy_matches_gold`).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_methods.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_gold.py
git commit -m "$(cat <<'EOF'
feat(bench): add higher_proxy control method

higher_proxy runs RAUNE at full 4K on the original frame and returns the
delta directly — this is the "information loss" control condition. Since
it needs the RAUNE model (not just the proxy delta), the driver injects
the model via a thread-local set_injected_raune_model() before calling.

The parametrized smoke test filters out methods with __needs_model__.
A specific test verifies higher_proxy's output matches compute_gold.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `sr_maxine` learned-SR method

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Add `sr_maxine` to methods.py**

Append:

```python
# Module-level lazy singletons for SR effects (avoid reloading per call)
_MAXINE_SR = None


def _get_maxine_sr():
    """Lazy-load the Maxine VideoSuperRes effect. Hard-errors if nvvfx is missing."""
    global _MAXINE_SR
    if _MAXINE_SR is not None:
        return _MAXINE_SR
    try:
        from nvvfx import VideoSuperRes
    except ImportError as e:
        raise RuntimeError(
            "sr_maxine requires nvidia-vfx (imports as nvvfx). "
            "Run scripts/setup_bench.sh or: pip install nvidia-vfx"
        ) from e

    vsr = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
    _MAXINE_SR = vsr
    return vsr


@register("sr_maxine")
def sr_maxine(
    delta_proxy: torch.Tensor,
    orig_full: torch.Tensor,
    full_size: Tuple[int, int],
) -> torch.Tensor:
    """Learned SR on the graded proxy (not the delta).

    Flow:
      1. Apply delta_proxy to orig_proxy (down-sampled orig_full) → graded_proxy
      2. Upscale graded_proxy 1080→2160 via Maxine VideoSuperRes
      3. Compute delta_method = lab(graded_4k) - lab(orig_full)
    """
    from .oklab import oklab_to_rgb, rgb_to_oklab

    fh, fw = full_size
    _, _, ph, pw = delta_proxy.shape

    # 1. Build graded_proxy from orig_proxy + delta
    orig_proxy = F.interpolate(orig_full, size=(ph, pw), mode="bilinear", align_corners=False)
    orig_proxy_lab = rgb_to_oklab(orig_proxy)
    graded_proxy_lab = orig_proxy_lab + delta_proxy
    graded_proxy = oklab_to_rgb(graded_proxy_lab)  # (1, 3, ph, pw) in [0, 1]

    # 2. Maxine VideoSuperRes expects (3, H, W) fp32 in [0, 1] on CUDA
    vsr = _get_maxine_sr()
    vsr.output_width = fw
    vsr.output_height = fh
    vsr.load()
    result = vsr.run(graded_proxy[0])  # (3, ph, pw) → (3, fh, fw)
    graded_4k = torch.from_dlpack(result.image).clone().unsqueeze(0)  # (1, 3, fh, fw)

    # 3. Delta vs orig_full in OKLab
    delta_4k = rgb_to_oklab(graded_4k) - rgb_to_oklab(orig_full)
    return delta_4k
```

- [ ] **Step 2: Add a specific test for `sr_maxine`**

Append to `tests/test_methods.py`:

```python
def test_sr_maxine_smoke(synthetic_proxy_delta, synthetic_full_rgb):
    """Maxine should run on tiny input and produce a 3-channel delta at full size."""
    pytest.importorskip("nvvfx")
    from benchmarks.upscale_bench.methods import REGISTRY
    fn = REGISTRY["sr_maxine"]
    full_size = (synthetic_full_rgb.shape[-2], synthetic_full_rgb.shape[-1])
    out = fn(synthetic_proxy_delta, synthetic_full_rgb, full_size=full_size)
    assert out.shape == (1, 3, *full_size)
    assert torch.isfinite(out).all()
```

Note: the parametrized smoke test already covers `sr_maxine` (no `__needs_model__` attr), so this specific test is redundant for pure smoke. Keep it anyway for documentation — it makes the Maxine dep explicit and provides a clear failure point if nvvfx regresses.

- [ ] **Step 3: Run the tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v
```

Expected: 7 parametrized smoke tests pass (bilinear, bicubic, lanczos3, guided_filter, joint_bilateral, asymmetric_bilateral, sr_maxine) + 1 explicit sr_maxine test passes. First Maxine call may take 1–2 s for SDK model load.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_methods.py
git commit -m "$(cat <<'EOF'
feat(bench): add sr_maxine learned-SR upscale method

Applies delta_proxy to orig_proxy to form a graded_proxy image, then
upscales that image (not the delta) via Maxine VideoSuperRes at HIGH
quality. The new graded_4k is used to derive a fresh delta against the
sharp full-res orig. Module-level lazy singleton avoids reloading the
Maxine model per call.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `sr_realesrgan` optional learned-SR method

**Files:**
- Modify: `repos/dorea/benchmarks/upscale_bench/methods.py`

- [ ] **Step 1: Verify Real-ESRGAN is installed**

```bash
/opt/dorea-venv/bin/python -c "from realesrgan import RealESRGANer; print('realesrgan OK')"
```

If it fails with `ModuleNotFoundError`, skip this task entirely — the soft-skip path in the driver will drop it at runtime. Log this as a follow-up.

If it succeeds, proceed.

- [ ] **Step 2: Add `sr_realesrgan` to methods.py (inside a try/except so import failure is soft-skipped at module-load)**

Append to `methods.py`:

```python
# Real-ESRGAN: optional, soft-skipped if not installed
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    _REALESRGAN_AVAILABLE = True
except ImportError:
    _REALESRGAN_AVAILABLE = False


_REALESRGAN_SR: Optional[object] = None


def _get_realesrgan_sr():
    """Lazy-load RealESRGANer with the RealESRGAN_x2plus model."""
    global _REALESRGAN_SR
    if _REALESRGAN_SR is not None:
        return _REALESRGAN_SR
    if not _REALESRGAN_AVAILABLE:
        raise RuntimeError("Real-ESRGAN not installed")

    # Model: RealESRGAN_x2plus
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    model_path = Path("/workspaces/dorea-workspace/repos/dorea/models/realesrgan/RealESRGAN_x2plus.pth")
    if not model_path.exists():
        # Fall back to the realesrgan auto-download URL; place into our models dir
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        print(f"[sr_realesrgan] downloading {url}")
        urllib.request.urlretrieve(url, model_path)

    upsampler = RealESRGANer(
        scale=2,
        model_path=str(model_path),
        model=model,
        tile=0,
        pre_pad=0,
        half=True,
    )
    _REALESRGAN_SR = upsampler
    return upsampler


if _REALESRGAN_AVAILABLE:
    @register("sr_realesrgan")
    def sr_realesrgan(
        delta_proxy: torch.Tensor,
        orig_full: torch.Tensor,
        full_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Real-ESRGAN x2 on the graded proxy image."""
        from .oklab import oklab_to_rgb, rgb_to_oklab

        fh, fw = full_size
        _, _, ph, pw = delta_proxy.shape

        orig_proxy = F.interpolate(orig_full, size=(ph, pw), mode="bilinear", align_corners=False)
        orig_proxy_lab = rgb_to_oklab(orig_proxy)
        graded_proxy_lab = orig_proxy_lab + delta_proxy
        graded_proxy = oklab_to_rgb(graded_proxy_lab)  # (1, 3, ph, pw) in [0,1]

        # RealESRGANer.enhance expects a (H, W, 3) uint8 BGR numpy array
        import numpy as np
        arr = (graded_proxy[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        arr_bgr = arr[:, :, ::-1]  # RGB → BGR

        upsampler = _get_realesrgan_sr()
        output_bgr, _ = upsampler.enhance(arr_bgr, outscale=fw / pw)
        # Output is (H*scale, W*scale, 3) uint8 BGR
        output_rgb = output_bgr[:, :, ::-1].copy()
        graded_4k = torch.from_numpy(output_rgb).to("cuda", dtype=torch.float32) / 255.0
        graded_4k = graded_4k.permute(2, 0, 1).unsqueeze(0)  # (1, 3, fh, fw)

        delta_4k = rgb_to_oklab(graded_4k) - rgb_to_oklab(orig_full)
        return delta_4k
```

Note the `Optional` import at the top of methods.py must already exist (`from typing import Callable, Tuple, Optional`) — if not, add `Optional` to the import line.

- [ ] **Step 3: Add `Path` import to `methods.py`**

Ensure `from pathlib import Path` is imported near the top of `methods.py`.

- [ ] **Step 4: Add LFS tracking for Real-ESRGAN weights**

Real-ESRGAN weights already covered by the `.gitattributes` glob `models/**/*.pth`. No additional config needed.

- [ ] **Step 5: Run the tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_methods.py -v
```

Expected: 8 parametrized smoke tests pass (all core + both SR). First Real-ESRGAN call downloads the model (~150 MB) and takes 1–3 minutes; subsequent calls use the cached model.

- [ ] **Step 6: Commit the methods.py change; the downloaded .pth gets committed via LFS**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/methods.py
# If the Real-ESRGAN model was downloaded during the test, add it via LFS
if [ -f repos/dorea/models/realesrgan/RealESRGAN_x2plus.pth ]; then
    git add repos/dorea/models/realesrgan/RealESRGAN_x2plus.pth
fi
git status --short
git lfs status
git commit -m "$(cat <<'EOF'
feat(bench): add optional sr_realesrgan method + LFS weights

RealESRGAN_x2plus as a learned-SR counterpoint to Maxine's more
conservative model. Auto-downloads the weights to
repos/dorea/models/realesrgan/ on first call. Registration is gated on
a try/except basicsr/realesrgan import so if the optional deps are
missing the method is soft-skipped at module load time.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Metrics module — delta errors, ΔE2000, SSIM, timing

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/metrics.py`
- Create: `repos/dorea/benchmarks/upscale_bench/tests/test_metrics.py`

- [ ] **Step 1: Write `metrics.py`**

```python
"""Per-(frame × method) metrics: delta-space error, final-image ΔE2000,
SSIM, and timing helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class MethodResult:
    frame_idx: int
    method: str
    delta_l1_L: float
    delta_l1_a: float
    delta_l1_b: float
    delta_l1: float
    delta_l2: float
    delta_max: float
    delta_p95: float
    final_delta_e: float
    final_delta_e_p95: float
    ssim: float
    wall_time_ms_upscale: float
    wall_time_ms_end_to_end: float
    status: str  # "ok", "oom", "skipped"

    def to_csv_row(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "method": self.method,
            "delta_l1_L": self.delta_l1_L,
            "delta_l1_a": self.delta_l1_a,
            "delta_l1_b": self.delta_l1_b,
            "delta_l1": self.delta_l1,
            "delta_l2": self.delta_l2,
            "delta_max": self.delta_max,
            "delta_p95": self.delta_p95,
            "final_delta_e": self.final_delta_e,
            "final_delta_e_p95": self.final_delta_e_p95,
            "ssim": self.ssim,
            "wall_time_ms_upscale": self.wall_time_ms_upscale,
            "wall_time_ms_end_to_end": self.wall_time_ms_end_to_end,
            "status": self.status,
        }


# ─── Delta-space metrics ────────────────────────────────────────────────────

def compute_delta_errors(
    delta_method: torch.Tensor,  # (1, 3, H, W)
    delta_gold: torch.Tensor,    # (1, 3, H, W)
) -> dict[str, float]:
    """Return a dict of delta-space error metrics."""
    diff = (delta_method - delta_gold).abs()
    per_pixel = diff.mean(dim=1)  # (1, H, W), averaged across L/a/b
    flat = per_pixel.flatten()
    return {
        "delta_l1_L": diff[:, 0].mean().item(),
        "delta_l1_a": diff[:, 1].mean().item(),
        "delta_l1_b": diff[:, 2].mean().item(),
        "delta_l1": diff.mean().item(),
        "delta_l2": (diff.pow(2).mean().sqrt()).item(),
        "delta_max": diff.max().item(),
        "delta_p95": torch.quantile(flat, 0.95).item(),
    }


# ─── Final-image metrics (ΔE2000 in CIELab) ─────────────────────────────────

def _srgb_to_xyz_d65(rgb: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) sRGB [0,1] → XYZ (D65)."""
    # Linearize
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([X, Y, Z], dim=1)


def _xyz_to_lab_d65(xyz: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) XYZ (D65) → CIELab."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883  # D65 white point
    x = xyz[:, 0:1] / Xn
    y = xyz[:, 1:2] / Yn
    z = xyz[:, 2:3] / Zn

    delta = 6 / 29
    def _f(t):
        return torch.where(t > delta ** 3, t.clamp_min(1e-12).pow(1.0 / 3.0), t / (3 * delta * delta) + 4 / 29)

    fx = _f(x)
    fy = _f(y)
    fz = _f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def delta_e_2000(
    rgb_a: torch.Tensor,
    rgb_b: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel ΔE2000 in CIELab between two sRGB [0,1] tensors.

    Returns (1, 1, H, W) of ΔE2000 values.
    """
    lab1 = _xyz_to_lab_d65(_srgb_to_xyz_d65(rgb_a))
    lab2 = _xyz_to_lab_d65(_srgb_to_xyz_d65(rgb_b))
    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3]

    C1 = (a1 * a1 + b1 * b1).sqrt()
    C2 = (a2 * a2 + b2 * b2).sqrt()
    Cm = (C1 + C2) / 2

    G = 0.5 * (1 - (Cm.pow(7) / (Cm.pow(7) + 25 ** 7)).sqrt())
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = (a1p * a1p + b1 * b1).sqrt()
    C2p = (a2p * a2p + b2 * b2).sqrt()

    def _atan2_deg(y, x):
        return torch.atan2(y, x) * (180.0 / torch.pi) % 360.0

    h1p = _atan2_deg(b1, a1p)
    h2p = _atan2_deg(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = torch.where(dhp > 180, dhp - 360, dhp)
    dhp = torch.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * (C1p * C2p).sqrt() * torch.sin(dhp * torch.pi / 360.0)

    Lp_mean = (L1 + L2) / 2
    Cp_mean = (C1p + C2p) / 2

    hp_sum = h1p + h2p
    hp_diff = (h1p - h2p).abs()
    hp_mean = torch.where(
        hp_diff <= 180,
        hp_sum / 2,
        torch.where(hp_sum < 360, (hp_sum + 360) / 2, (hp_sum - 360) / 2),
    )

    T = (1
         - 0.17 * torch.cos((hp_mean - 30) * torch.pi / 180)
         + 0.24 * torch.cos((2 * hp_mean) * torch.pi / 180)
         + 0.32 * torch.cos((3 * hp_mean + 6) * torch.pi / 180)
         - 0.20 * torch.cos((4 * hp_mean - 63) * torch.pi / 180))

    d_theta = 30 * torch.exp(-(((hp_mean - 275) / 25) ** 2))
    Rc = 2 * (Cp_mean.pow(7) / (Cp_mean.pow(7) + 25 ** 7)).sqrt()
    Sl = 1 + (0.015 * (Lp_mean - 50).pow(2)) / (20 + (Lp_mean - 50).pow(2)).sqrt()
    Sc = 1 + 0.045 * Cp_mean
    Sh = 1 + 0.015 * Cp_mean * T
    Rt = -torch.sin(2 * d_theta * torch.pi / 180) * Rc

    dE = ((dLp / Sl).pow(2)
          + (dCp / Sc).pow(2)
          + (dHp / Sh).pow(2)
          + Rt * (dCp / Sc) * (dHp / Sh)).sqrt()
    return dE  # (1, 1, H, W)


def compute_final_image_errors(
    final_method: torch.Tensor,
    final_gold: torch.Tensor,
) -> dict[str, float]:
    dE = delta_e_2000(final_method, final_gold)
    flat = dE.flatten()
    return {
        "final_delta_e": dE.mean().item(),
        "final_delta_e_p95": torch.quantile(flat, 0.95).item(),
    }


# ─── SSIM on BT.709 luma ────────────────────────────────────────────────────

def _rgb_to_luma_bt709(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def compute_ssim(final_method: torch.Tensor, final_gold: torch.Tensor) -> float:
    """SSIM on BT.709 luma. Uses torchmetrics if available, else inlined fallback."""
    y_a = _rgb_to_luma_bt709(final_method)
    y_b = _rgb_to_luma_bt709(final_gold)
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(y_a.device)
        return ssim_metric(y_a, y_b).item()
    except ImportError:
        # Inlined simple SSIM (no windowing, just luminance/contrast/structure)
        mu_a = y_a.mean()
        mu_b = y_b.mean()
        var_a = y_a.var()
        var_b = y_b.var()
        cov = ((y_a - mu_a) * (y_b - mu_b)).mean()
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
        den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
        return (num / den).item()


# ─── Timing helper ──────────────────────────────────────────────────────────

def timed_run(
    fn: Callable,
    warmup: int = 3,
    runs: int = 10,
) -> float:
    """Run ``fn`` ``warmup`` + ``runs`` times, return median wall-time in ms.

    Each run is bracketed by torch.cuda.synchronize() to capture real GPU
    completion time.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]
```

- [ ] **Step 2: Write tests for metrics.py**

```python
# repos/dorea/benchmarks/upscale_bench/tests/test_metrics.py
"""Tests for metrics helpers."""
import torch
import pytest

from benchmarks.upscale_bench.metrics import (
    compute_delta_errors,
    compute_final_image_errors,
    compute_ssim,
    delta_e_2000,
    timed_run,
)


def test_compute_delta_errors_identical_inputs_zero():
    d = torch.randn(1, 3, 32, 32, device="cuda")
    m = compute_delta_errors(d, d)
    assert m["delta_l1"] == 0.0
    assert m["delta_max"] == 0.0
    assert m["delta_p95"] == 0.0


def test_compute_delta_errors_shifted_inputs():
    d1 = torch.zeros(1, 3, 32, 32, device="cuda")
    d2 = torch.ones(1, 3, 32, 32, device="cuda")
    m = compute_delta_errors(d1, d2)
    assert abs(m["delta_l1"] - 1.0) < 1e-6
    assert abs(m["delta_max"] - 1.0) < 1e-6


def test_delta_e_2000_identical_inputs_zero():
    img = torch.rand(1, 3, 16, 16, device="cuda")
    dE = delta_e_2000(img, img)
    assert dE.max().item() < 1e-4


def test_delta_e_2000_shape():
    a = torch.rand(1, 3, 16, 16, device="cuda")
    b = torch.rand(1, 3, 16, 16, device="cuda")
    dE = delta_e_2000(a, b)
    assert dE.shape == (1, 1, 16, 16)
    assert torch.isfinite(dE).all()


def test_compute_ssim_identical_is_one():
    img = torch.rand(1, 3, 32, 32, device="cuda")
    s = compute_ssim(img, img)
    assert s > 0.999


def test_timed_run_returns_positive_ms():
    def fn():
        x = torch.randn(100, 100, device="cuda")
        y = x @ x
        return y

    ms = timed_run(fn, warmup=1, runs=3)
    assert ms > 0.0
```

- [ ] **Step 3: Run the tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_metrics.py -v
```

Expected: 6 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/metrics.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(bench): add metrics module — delta errors, ΔE2000, SSIM, timing

compute_delta_errors reports L1 per-channel and overall, L2, max, p95 on
the OKLab delta difference.
delta_e_2000 is an inlined CIELab-based ΔE2000 (not OKLab — industry
standard for perceptual distance).
compute_ssim uses torchmetrics if available, else a simple inlined
fallback.
timed_run is the warmup-then-median-of-N helper used by the driver.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Visualization module — summary grid, per-frame sheets, heatmaps

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/visualize.py`
- Create: `repos/dorea/benchmarks/upscale_bench/tests/test_visualize.py`

- [ ] **Step 1: Write `visualize.py`**

```python
"""Contact-sheet visualization: summary grid (A), per-frame sheets (B),
error heatmaps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ─── helpers ────────────────────────────────────────────────────────────────

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) or (3, H, W) fp32 [0,1] → PIL RGB Image."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize_pil(img: Image.Image, max_long_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_long_edge:
        return img
    scale = max_long_edge / max(w, h)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)


def _heatmap_pil(values: torch.Tensor, vmax: float) -> Image.Image:
    """(H, W) fp32 tensor → RGB PIL image via turbo colormap."""
    arr = values.detach().cpu().numpy()
    arr = np.clip(arr / max(vmax, 1e-8), 0, 1)
    # Turbo colormap (Google): encode as RGB lookup
    try:
        import matplotlib.pyplot as plt  # type: ignore
        cmap = plt.get_cmap("turbo")
        rgb = (cmap(arr)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        # Fallback: jet-ish gradient
        r = np.clip(1.5 - np.abs(4 * arr - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * arr - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * arr - 1), 0, 1)
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ─── Summary grid (contact sheet A) ─────────────────────────────────────────

@dataclass
class FrameCellData:
    frame_idx: int
    orig_proxy_final: torch.Tensor  # (1, 3, H, W) in [0, 1]
    gold_final: torch.Tensor
    method_finals: dict[str, torch.Tensor]  # name → (1, 3, H, W)
    method_metrics: dict[str, dict]          # name → metric dict


def generate_summary_grid(
    frames_data: Sequence[FrameCellData],
    method_order: Sequence[str],
    method_timings: dict[str, float],  # name → median wall_time_ms_end_to_end
    run_metadata: dict,                # e.g. {"date": ..., "weights_sha": ..., "clip": ..., "git_sha": ...}
    out_path: Path,
    cell_long_edge: int = 450,
) -> None:
    """Produce the summary grid PNG (contact sheet A)."""
    n_frames = len(frames_data)
    columns = ["orig_proxy", "gold"] + list(method_order)
    n_cols = len(columns)

    # Build cell images (downsample everything to cell_long_edge)
    cell_images: list[list[Image.Image]] = []
    for fd in frames_data:
        row = []
        row.append(_resize_pil(_tensor_to_pil(fd.orig_proxy_final), cell_long_edge))
        row.append(_resize_pil(_tensor_to_pil(fd.gold_final), cell_long_edge))
        for name in method_order:
            if name in fd.method_finals:
                row.append(_resize_pil(_tensor_to_pil(fd.method_finals[name]), cell_long_edge))
            else:
                # OOM/failed placeholder
                placeholder = Image.new("RGB", (cell_long_edge, cell_long_edge * 9 // 16), (64, 0, 0))
                ImageDraw.Draw(placeholder).text((10, 10), "OOM", fill=(255, 255, 255))
                row.append(placeholder)
        cell_images.append(row)

    # Determine canvas size
    cell_w = max(img.width for row in cell_images for img in row)
    cell_h = max(img.height for row in cell_images for img in row)
    header_h = 100
    footer_h = 180
    label_h = 40
    padding = 12
    col_label_h = 40

    canvas_w = n_cols * (cell_w + padding) + padding + 120  # 120px left gutter for row labels
    canvas_h = header_h + col_label_h + n_frames * (cell_h + padding + label_h) + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(20)
    font_small = _load_font(14)
    font_header = _load_font(22)

    # Header band
    header_text = (
        f"Delta Upscale Benchmark  |  {run_metadata.get('date', '?')}  |  "
        f"clip: {run_metadata.get('clip', '?')}  |  "
        f"weights: {run_metadata.get('weights_sha', '?')}  |  "
        f"git: {run_metadata.get('git_sha', '?')}"
    )
    draw.text((padding, padding), header_text, fill=(220, 220, 220), font=font_header)

    # Gold paths per frame (second header line)
    gold_paths_text = "  ".join(
        f"frame {fd.frame_idx}: {run_metadata.get('gold_paths', {}).get(fd.frame_idx, '?')}"
        for fd in frames_data
    )
    draw.text((padding, padding + 32), gold_paths_text, fill=(180, 180, 180), font=font_small)

    # Column labels (method name + timing)
    y_col = header_h
    for c, col_name in enumerate(columns):
        x = 120 + c * (cell_w + padding) + padding
        if col_name in ("orig_proxy", "gold"):
            label = col_name
        else:
            t = method_timings.get(col_name, 0.0)
            label = f"{col_name}  {t:.1f}ms"
        draw.text((x, y_col), label, fill=(220, 220, 220), font=font)

    # Grid body
    y = header_h + col_label_h
    for r, (fd, row) in enumerate(zip(frames_data, cell_images)):
        # Row label (frame index)
        draw.text((padding, y + cell_h // 2), f"frame\n{fd.frame_idx}", fill=(220, 220, 220), font=font)
        # Determine winner in this row (lowest final_delta_e)
        best_method = None
        best_val = float("inf")
        for name in method_order:
            val = fd.method_metrics.get(name, {}).get("final_delta_e", float("inf"))
            if val < best_val:
                best_val = val
                best_method = name
        for c, (col_name, img) in enumerate(zip(columns, row)):
            x = 120 + c * (cell_w + padding) + padding
            canvas.paste(img, (x, y))
            # Border
            if col_name == best_method:
                border_color = (0, 220, 0)  # green = winner
            elif col_name == "bilinear":
                border_color = (160, 160, 160)  # gray = baseline
            else:
                border_color = None
            if border_color is not None:
                draw.rectangle((x - 2, y - 2, x + img.width + 2, y + img.height + 2), outline=border_color, width=3)
        y += cell_h + padding + label_h

    # Footer: small aggregate metrics table
    y_footer = canvas_h - footer_h + padding
    draw.text((padding, y_footer), "Aggregate (mean across frames):", fill=(220, 220, 220), font=font)
    y_footer += 32
    header_cols = ["method", "delta_l1", "final_ΔE", "SSIM", "ms (end-to-end)"]
    col_widths = [160, 100, 100, 80, 160]
    x = padding
    for cname, cw in zip(header_cols, col_widths):
        draw.text((x, y_footer), cname, fill=(160, 160, 160), font=font_small)
        x += cw
    y_footer += 20
    for name in method_order:
        means = {}
        for key in ("delta_l1", "final_delta_e", "ssim"):
            vals = [fd.method_metrics.get(name, {}).get(key, float("nan")) for fd in frames_data]
            vals = [v for v in vals if v == v]
            means[key] = sum(vals) / max(len(vals), 1) if vals else float("nan")
        t = method_timings.get(name, 0.0)
        x = padding
        draw.text((x, y_footer), name, fill=(220, 220, 220), font=font_small)
        x += col_widths[0]
        draw.text((x, y_footer), f"{means['delta_l1']:.4f}", fill=(220, 220, 220), font=font_small)
        x += col_widths[1]
        draw.text((x, y_footer), f"{means['final_delta_e']:.3f}", fill=(220, 220, 220), font=font_small)
        x += col_widths[2]
        draw.text((x, y_footer), f"{means['ssim']:.4f}", fill=(220, 220, 220), font=font_small)
        x += col_widths[3]
        draw.text((x, y_footer), f"{t:.1f}", fill=(220, 220, 220), font=font_small)
        y_footer += 16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ─── Per-frame sheet ────────────────────────────────────────────────────────

def _find_disagreement_crop(
    method_deltas: dict[str, torch.Tensor],
    crop_size: int = 400,
) -> tuple[int, int]:
    """Find the top-left (y, x) of the crop_size × crop_size window with the
    highest across-method variance in delta_method."""
    stacked = torch.stack([d.squeeze(0) for d in method_deltas.values()], dim=0)  # (M, 3, H, W)
    var = stacked.var(dim=0).mean(dim=0)  # (H, W)
    H, W = var.shape
    # Coarse integral image to find max-sum window
    import torch.nn.functional as F
    kernel = torch.ones(1, 1, crop_size, crop_size, device=var.device)
    summed = F.conv2d(var.unsqueeze(0).unsqueeze(0), kernel, stride=crop_size // 4)
    idx = summed.flatten().argmax().item()
    sh, sw = summed.shape[-2], summed.shape[-1]
    r = idx // sw
    c = idx % sw
    y = r * (crop_size // 4)
    x = c * (crop_size // 4)
    y = min(max(y, 0), H - crop_size)
    x = min(max(x, 0), W - crop_size)
    return y, x


def generate_per_frame_sheet(
    frame_idx: int,
    orig_proxy_final: torch.Tensor,
    gold_final: torch.Tensor,
    method_finals: dict[str, torch.Tensor],
    method_deltas: dict[str, torch.Tensor],
    method_metrics: dict[str, dict],
    method_order: Sequence[str],
    out_path: Path,
    crop_size: int = 400,
) -> None:
    """Per-frame detail sheet with full gold, method thumbnails, and zoomed crops."""
    # Crop selection
    crop_y, crop_x = _find_disagreement_crop(method_deltas, crop_size=crop_size)

    # Gold full-width top
    gold_img = _resize_pil(_tensor_to_pil(gold_final), 1920)
    draw_scale_y = gold_img.height / gold_final.shape[-2]
    draw_scale_x = gold_img.width / gold_final.shape[-1]

    # Thumbnails (row of orig_proxy + gold + methods)
    thumb_long_edge = 300
    thumbs = [("orig_proxy", _resize_pil(_tensor_to_pil(orig_proxy_final), thumb_long_edge)),
              ("gold", _resize_pil(_tensor_to_pil(gold_final), thumb_long_edge))]
    for name in method_order:
        if name in method_finals:
            thumbs.append((name, _resize_pil(_tensor_to_pil(method_finals[name]), thumb_long_edge)))
        else:
            p = Image.new("RGB", (thumb_long_edge, thumb_long_edge * 9 // 16), (64, 0, 0))
            ImageDraw.Draw(p).text((10, 10), "OOM", fill=(255, 255, 255))
            thumbs.append((name, p))

    # Crops from same spatial location in every method (and orig/gold)
    def _crop_of(t: torch.Tensor) -> Image.Image:
        c = t[:, :, crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        return _tensor_to_pil(c)

    crops = [("orig_proxy", _crop_of(orig_proxy_final)),
             ("gold", _crop_of(gold_final))]
    for name in method_order:
        if name in method_finals:
            crops.append((name, _crop_of(method_finals[name])))
        else:
            p = Image.new("RGB", (crop_size, crop_size), (64, 0, 0))
            ImageDraw.Draw(p).text((10, 10), "OOM", fill=(255, 255, 255))
            crops.append((name, p))

    # Canvas layout
    padding = 12
    label_h = 24
    header_h = 40
    metrics_h = 220

    thumb_w = max(t.width for _, t in thumbs) + padding
    thumb_h = max(t.height for _, t in thumbs) + label_h
    crop_w = crop_size + padding
    crop_h = crop_size + label_h

    canvas_w = max(gold_img.width + padding * 2, len(thumbs) * thumb_w + padding)
    canvas_h = header_h + gold_img.height + padding + thumb_h + padding + crop_h + padding + metrics_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(22)
    font_small = _load_font(14)

    # Header
    draw.text((padding, padding), f"Frame {frame_idx}  —  crop @ ({crop_x}, {crop_y}) {crop_size}×{crop_size}",
              fill=(220, 220, 220), font=font)

    # Gold image + crop rectangle
    gy = header_h
    canvas.paste(gold_img, (padding, gy))
    rx0 = padding + int(crop_x * draw_scale_x)
    ry0 = gy + int(crop_y * draw_scale_y)
    rx1 = padding + int((crop_x + crop_size) * draw_scale_x)
    ry1 = gy + int((crop_y + crop_size) * draw_scale_y)
    draw.rectangle((rx0, ry0, rx1, ry1), outline=(255, 0, 0), width=4)

    # Thumbnails row
    ty = gy + gold_img.height + padding
    for i, (name, img) in enumerate(thumbs):
        tx = padding + i * thumb_w
        canvas.paste(img, (tx, ty + label_h))
        draw.text((tx, ty), name, fill=(220, 220, 220), font=font_small)

    # Crops row
    cy = ty + thumb_h + padding
    for i, (name, img) in enumerate(crops):
        cx = padding + i * crop_w
        canvas.paste(img, (cx, cy + label_h))
        draw.text((cx, cy), name, fill=(220, 220, 220), font=font_small)

    # Metrics strip
    my = cy + crop_h + padding
    draw.text((padding, my), "Per-method metrics for this frame:", fill=(220, 220, 220), font=font_small)
    my += 20
    header_cols = ["method", "delta_l1", "final_ΔE", "ΔE p95", "SSIM"]
    cw = [180, 100, 100, 100, 80]
    x = padding
    for hc, w in zip(header_cols, cw):
        draw.text((x, my), hc, fill=(160, 160, 160), font=font_small)
        x += w
    my += 18
    for name in method_order:
        m = method_metrics.get(name, {})
        x = padding
        draw.text((x, my), name, fill=(220, 220, 220), font=font_small)
        x += cw[0]
        draw.text((x, my), f"{m.get('delta_l1', float('nan')):.4f}", fill=(220, 220, 220), font=font_small)
        x += cw[1]
        draw.text((x, my), f"{m.get('final_delta_e', float('nan')):.3f}", fill=(220, 220, 220), font=font_small)
        x += cw[2]
        draw.text((x, my), f"{m.get('final_delta_e_p95', float('nan')):.3f}", fill=(220, 220, 220), font=font_small)
        x += cw[3]
        draw.text((x, my), f"{m.get('ssim', float('nan')):.4f}", fill=(220, 220, 220), font=font_small)
        my += 16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ─── Error heatmaps ─────────────────────────────────────────────────────────

def generate_heatmap(
    delta_method: torch.Tensor,
    delta_gold: torch.Tensor,
    frame_p95: float,
    out_path: Path,
    method_name: str,
) -> None:
    """Per-(frame, method) heatmap of |delta_method - delta_gold| averaged across channels."""
    diff = (delta_method - delta_gold).abs().mean(dim=1).squeeze(0)  # (H, W)
    img = _heatmap_pil(diff, vmax=frame_p95)
    # Add a small label on the image
    draw = ImageDraw.Draw(img)
    font = _load_font(24)
    draw.text((12, 12), method_name, fill=(255, 255, 255), font=font)
    draw.text((12, 40), f"max={diff.max().item():.4f}, p95={frame_p95:.4f}", fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
```

- [ ] **Step 2: Write a minimal test for visualize.py**

```python
# repos/dorea/benchmarks/upscale_bench/tests/test_visualize.py
"""Sanity test for visualize.py — generates output PNGs on synthetic data."""
from pathlib import Path

import torch
from PIL import Image

from benchmarks.upscale_bench.visualize import (
    FrameCellData,
    generate_summary_grid,
    generate_per_frame_sheet,
    generate_heatmap,
)


def _rand_final(h=128, w=200):
    return torch.rand(1, 3, h, w, device="cuda", dtype=torch.float32)


def _rand_delta(h=128, w=200):
    return torch.randn(1, 3, h, w, device="cuda", dtype=torch.float32) * 0.05


def test_generate_heatmap_writes_png(tmp_path):
    d_method = _rand_delta()
    d_gold = _rand_delta()
    out = tmp_path / "hm.png"
    generate_heatmap(d_method, d_gold, frame_p95=0.1, out_path=out, method_name="smoke_method")
    assert out.exists()
    Image.open(out).verify()


def test_generate_summary_grid_writes_png(tmp_path):
    frames = []
    method_order = ["bilinear", "bicubic"]
    method_timings = {"bilinear": 1.5, "bicubic": 2.1}
    for idx in (30, 180, 300):
        frames.append(FrameCellData(
            frame_idx=idx,
            orig_proxy_final=_rand_final(),
            gold_final=_rand_final(),
            method_finals={name: _rand_final() for name in method_order},
            method_metrics={name: {"delta_l1": 0.01, "final_delta_e": 0.5, "ssim": 0.99} for name in method_order},
        ))
    out = tmp_path / "grid.png"
    generate_summary_grid(frames, method_order, method_timings,
                          run_metadata={"date": "2026-04-10", "weights_sha": "deadbeef",
                                        "clip": "test.mp4", "git_sha": "abc123",
                                        "gold_paths": {30: "native_4k", 180: "native_4k", 300: "native_4k"}},
                          out_path=out)
    assert out.exists()
    Image.open(out).verify()


def test_generate_per_frame_sheet_writes_png(tmp_path):
    method_order = ["bilinear", "bicubic"]
    out = tmp_path / "frame.png"
    generate_per_frame_sheet(
        frame_idx=30,
        orig_proxy_final=_rand_final(),
        gold_final=_rand_final(),
        method_finals={name: _rand_final() for name in method_order},
        method_deltas={name: _rand_delta() for name in method_order},
        method_metrics={name: {"delta_l1": 0.01, "final_delta_e": 0.5, "final_delta_e_p95": 0.8, "ssim": 0.99}
                        for name in method_order},
        method_order=method_order,
        out_path=out,
        crop_size=64,
    )
    assert out.exists()
    Image.open(out).verify()
```

- [ ] **Step 3: Run the tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_visualize.py -v
```

Expected: 3 tests pass. If matplotlib is not installed, the heatmap test still passes using the inlined colormap fallback.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/visualize.py \
        repos/dorea/benchmarks/upscale_bench/tests/test_visualize.py
git commit -m "$(cat <<'EOF'
feat(bench): add visualization — summary grid, per-frame sheet, heatmaps

- generate_summary_grid: big N_frames × 11-columns contact sheet with
  header (date/clip/weights/git), column labels with method timings,
  green borders for row winners, gray for baseline, and a footer table
  of aggregate metrics.
- generate_per_frame_sheet: full gold at top with red crop rectangle,
  thumbnail row, crop row at auto-selected "highest method variance"
  location, and metrics strip.
- generate_heatmap: per-(frame, method) turbo-colormap heatmap of the
  |delta_method - delta_gold| magnitude, normalized to frame p95 so all
  methods on one frame share a scale.
- Uses PIL for all rendering; matplotlib is used only for the turbo
  colormap and falls back to a hand-rolled jet-ish gradient if missing.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: CLI driver — `run.py`

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/run.py`

- [ ] **Step 1: Write `run.py`**

```python
"""CLI entry point for the delta upscale benchmark.

Usage:
    python -m benchmarks.upscale_bench.run [OPTIONS]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .frame_select import auto_select, decode_frame
from .gold import GoldResult, _load_raune_model, load_or_compute_gold
from .methods import (
    REGISTRY,
    clear_injected_raune_model,
    set_injected_raune_model,
)
from .metrics import (
    MethodResult,
    compute_delta_errors,
    compute_final_image_errors,
    compute_ssim,
    timed_run,
)
from .oklab import oklab_to_rgb, rgb_to_oklab
from .visualize import (
    FrameCellData,
    generate_heatmap,
    generate_per_frame_sheet,
    generate_summary_grid,
)


DEFAULT_CLIP = Path(
    "/workspaces/dorea-workspace/footage/raw/2025-11-01/"
    "DJI_20251101111428_0055_D_3s.MP4"
)
DEFAULT_WEIGHTS = Path(
    "/workspaces/dorea-workspace/repos/dorea/models/raune_net/"
    "models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
)
DEFAULT_MODELS_DIR = Path(
    "/workspaces/dorea-workspace/repos/dorea/models/raune_net"
)
DEFAULT_OUT_DIR = Path("/workspaces/dorea-workspace/working/upscale_bench")


# ─── Driver helpers ─────────────────────────────────────────────────────────

def _file_sha1_prefix(path: Path, length: int = 16) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:length]


def _git_sha(cwd: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip())
        return sha[:10], dirty
    except subprocess.CalledProcessError:
        return "unknown", False


def _log_environment(args, weights_path: Path, clip_path: Path) -> dict:
    info = {
        "date": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "weights_sha": _file_sha1_prefix(weights_path),
        "clip_sha": _file_sha1_prefix(clip_path),
    }
    # git SHAs for workspace and dorea repo
    ws_sha, ws_dirty = _git_sha(Path("/workspaces/dorea-workspace"))
    dorea_sha, dorea_dirty = _git_sha(Path("/workspaces/dorea-workspace/repos/dorea"))
    info["workspace_git"] = f"{ws_sha}{'*' if ws_dirty else ''}"
    info["dorea_git"] = f"{dorea_sha}{'*' if dorea_dirty else ''}"
    # nvvfx version
    try:
        import nvvfx
        info["nvvfx_version"] = nvvfx.get_sdk_version()
    except Exception:
        info["nvvfx_version"] = "unavailable"
    try:
        import realesrgan
        info["realesrgan_version"] = getattr(realesrgan, "__version__", "present")
    except Exception:
        info["realesrgan_version"] = "unavailable"
    return info


def _apply_delta(
    delta_full: torch.Tensor,
    orig_full: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the final RGB image from a full-res OKLab delta + original."""
    lab = rgb_to_oklab(orig_full) + delta_full
    return oklab_to_rgb(lab)


# ─── Main orchestration ────────────────────────────────────────────────────

def run(args) -> None:
    # Set determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    clip_path = Path(args.clip)
    weights_path = Path(args.weights)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = out_dir / "gold"
    heatmap_dir = out_dir / "heatmaps"
    per_frame_dir = out_dir / "per_frame"

    if not clip_path.exists():
        raise FileNotFoundError(f"clip not found: {clip_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"RAUNE weights not found: {weights_path}. "
            f"Run scripts/download_raune_weights.sh"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this bench requires a GPU")

    env_info = _log_environment(args, weights_path, clip_path)
    print("=== upscale_bench ===")
    for k, v in env_info.items():
        print(f" {k}: {v}")

    # Select methods
    if args.methods == "all":
        method_names = list(REGISTRY.keys())
    else:
        method_names = [m.strip() for m in args.methods.split(",")]
        unknown = [m for m in method_names if m not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}. Available: {list(REGISTRY.keys())}")

    # sr_maxine hard-required if it's in the list
    if "sr_maxine" in method_names:
        try:
            import nvvfx  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "sr_maxine requested but nvvfx is not installed. "
                "Run scripts/setup_bench.sh or pip install nvidia-vfx"
            ) from e

    # Select frames
    if args.frames == "auto":
        frame_indices = auto_select(clip_path, every_n=30)
    else:
        frame_indices = [int(s) for s in args.frames.split(",")]
    print(f" frames: {frame_indices}")

    # Load model once
    print(" loading RAUNE model...")
    raune_model = _load_raune_model(weights_path, models_dir)
    set_injected_raune_model(raune_model)

    # Collect results
    all_results: list[MethodResult] = []
    frames_data: list[FrameCellData] = []
    gold_paths: dict[int, str] = {}
    oom_events: list[str] = []

    try:
        for frame_idx in frame_indices:
            print(f"\n─── frame {frame_idx} ───")
            frame_4k = decode_frame(clip_path, frame_idx)
            gold = load_or_compute_gold(
                frame_4k, frame_idx, raune_model, weights_path, gold_dir,
                regen=args.regen_gold, force_path=None,
            )
            gold_paths[frame_idx] = gold.path_used
            print(f"  gold: {gold.path_used}")

            # Optional sanity check: run both paths, report disagreement
            if args.gold_sanity_check:
                other_path = "tiled_2x2_1984x1144_o128" if gold.path_used == "native_4k" else "native_4k"
                try:
                    alt = load_or_compute_gold(
                        frame_4k, frame_idx, raune_model, weights_path, gold_dir,
                        regen=True, force_path=other_path,
                    )
                    diff = (alt.delta_4k - gold.delta_4k).abs().max().item()
                    print(f"  gold-sanity: {other_path} vs {gold.path_used} delta_max={diff:.5f}")
                except Exception as e:
                    print(f"  gold-sanity: failed to run alt path: {e}")

            # Proxy delta from the standard pipeline
            ph = 1080 if args.proxy_size == 1080 else args.proxy_size
            pw = int(round(ph * 3840 / 2160))
            orig_proxy = F.interpolate(frame_4k, size=(ph, pw), mode="bilinear", align_corners=False)
            # Run RAUNE at proxy
            x = (orig_proxy * 2 - 1).half()
            with torch.no_grad():
                y = raune_model(x).float()
            raune_proxy = ((y + 1) / 2).clamp(0, 1)
            delta_proxy = rgb_to_oklab(raune_proxy) - rgb_to_oklab(orig_proxy)

            # Compute orig_proxy final (what the reference column shows)
            orig_proxy_final = orig_proxy  # just the original without grading
            gold_final = gold.raune_4k

            # Run each method
            method_finals: dict[str, torch.Tensor] = {}
            method_deltas: dict[str, torch.Tensor] = {}
            method_metrics: dict[str, dict] = {}
            method_timings: dict[str, float] = {}

            frame_p95 = None  # placeholder; set after first method for heatmap normalization

            for name in method_names:
                print(f"  method: {name}")
                fn = REGISTRY[name]
                try:
                    # Warmup + timed upscale-only
                    def _upscale_only():
                        return fn(delta_proxy, frame_4k, (2160, 3840))
                    t_upscale = timed_run(_upscale_only, warmup=args.timing_warmup, runs=args.timing_runs)
                    # Warmup + timed end-to-end (upscale + apply delta)
                    def _end_to_end():
                        d = fn(delta_proxy, frame_4k, (2160, 3840))
                        return _apply_delta(d, frame_4k)
                    t_e2e = timed_run(_end_to_end, warmup=args.timing_warmup, runs=args.timing_runs)

                    # One more call to capture output
                    delta_method = fn(delta_proxy, frame_4k, (2160, 3840))
                    final_method = _apply_delta(delta_method, frame_4k)

                    derrs = compute_delta_errors(delta_method, gold.delta_4k)
                    ferrs = compute_final_image_errors(final_method, gold_final)
                    ssim = compute_ssim(final_method, gold_final)

                    method_finals[name] = final_method
                    method_deltas[name] = delta_method
                    method_timings[name] = t_e2e
                    metrics_dict = {**derrs, **ferrs, "ssim": ssim,
                                    "wall_time_ms_upscale": t_upscale,
                                    "wall_time_ms_end_to_end": t_e2e}
                    method_metrics[name] = metrics_dict

                    all_results.append(MethodResult(
                        frame_idx=frame_idx, method=name,
                        delta_l1_L=derrs["delta_l1_L"], delta_l1_a=derrs["delta_l1_a"],
                        delta_l1_b=derrs["delta_l1_b"], delta_l1=derrs["delta_l1"],
                        delta_l2=derrs["delta_l2"], delta_max=derrs["delta_max"],
                        delta_p95=derrs["delta_p95"],
                        final_delta_e=ferrs["final_delta_e"],
                        final_delta_e_p95=ferrs["final_delta_e_p95"],
                        ssim=ssim,
                        wall_time_ms_upscale=t_upscale,
                        wall_time_ms_end_to_end=t_e2e,
                        status="ok",
                    ))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    oom_events.append(f"{frame_idx}/{name}")
                    print(f"    OOM: {name}")
                    all_results.append(MethodResult(
                        frame_idx=frame_idx, method=name,
                        delta_l1_L=float("nan"), delta_l1_a=float("nan"),
                        delta_l1_b=float("nan"), delta_l1=float("nan"),
                        delta_l2=float("nan"), delta_max=float("nan"),
                        delta_p95=float("nan"),
                        final_delta_e=float("nan"), final_delta_e_p95=float("nan"),
                        ssim=float("nan"),
                        wall_time_ms_upscale=float("nan"),
                        wall_time_ms_end_to_end=float("nan"),
                        status="oom",
                    ))

            # Heatmaps for this frame
            if method_deltas:
                # Frame p95 = max over methods of the p95 of |delta_method - delta_gold|
                frame_p95 = max(
                    torch.quantile((d - gold.delta_4k).abs().mean(dim=1).flatten(), 0.95).item()
                    for d in method_deltas.values()
                )
                for name, d in method_deltas.items():
                    heatmap_path = heatmap_dir / f"frame{frame_idx}_{name}.png"
                    generate_heatmap(d, gold.delta_4k, frame_p95, heatmap_path, name)

            # Per-frame sheet
            per_frame_path = per_frame_dir / f"frame{frame_idx}_sheet.png"
            generate_per_frame_sheet(
                frame_idx=frame_idx,
                orig_proxy_final=F.interpolate(orig_proxy_final, size=(2160, 3840), mode="bilinear", align_corners=False),
                gold_final=gold_final,
                method_finals=method_finals,
                method_deltas=method_deltas,
                method_metrics=method_metrics,
                method_order=method_names,
                out_path=per_frame_path,
            )

            # Store for summary grid
            frames_data.append(FrameCellData(
                frame_idx=frame_idx,
                orig_proxy_final=F.interpolate(orig_proxy_final, size=(2160, 3840), mode="bilinear", align_corners=False),
                gold_final=gold_final,
                method_finals=method_finals,
                method_metrics=method_metrics,
            ))

            # Free memory between frames
            del frame_4k, gold, orig_proxy, raune_proxy, delta_proxy, method_finals, method_deltas
            torch.cuda.empty_cache()
    finally:
        clear_injected_raune_model()

    # Aggregate timings (median across frames) for the summary grid
    aggregate_timings: dict[str, float] = {}
    for name in method_names:
        vals = [r.wall_time_ms_end_to_end for r in all_results
                if r.method == name and r.status == "ok"]
        aggregate_timings[name] = sum(vals) / len(vals) if vals else float("nan")

    # Summary grid
    grid_path = out_dir / "summary_grid.png"
    generate_summary_grid(
        frames_data=frames_data,
        method_order=method_names,
        method_timings=aggregate_timings,
        run_metadata={
            **env_info,
            "clip": clip_path.name,
            "gold_paths": gold_paths,
        },
        out_path=grid_path,
    )
    print(f"\nsummary grid: {grid_path}")

    # CSV
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        fields = list(all_results[0].to_csv_row().keys()) if all_results else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.to_csv_row())
        # Aggregate rows
        for name in method_names:
            ok_rows = [r for r in all_results if r.method == name and r.status == "ok"]
            if not ok_rows:
                continue
            mean_row = {"frame_idx": "ALL_MEAN", "method": name, "status": "ok"}
            keys = [k for k in fields if k not in ("frame_idx", "method", "status")]
            for k in keys:
                vals = [getattr(r, k) for r in ok_rows]
                mean_row[k] = sum(vals) / len(vals)
            writer.writerow(mean_row)
    print(f"metrics csv:  {csv_path}")

    # Run report
    report_path = out_dir / "run_report.md"
    _write_run_report(
        report_path, env_info, frame_indices, gold_paths, oom_events,
        all_results, method_names, out_dir,
    )
    print(f"run report:   {report_path}")

    print("\n=== done ===")


def _write_run_report(
    report_path: Path,
    env_info: dict,
    frame_indices: list[int],
    gold_paths: dict[int, str],
    oom_events: list[str],
    all_results: list,
    method_names: list[str],
    out_dir: Path,
) -> None:
    lines = []
    lines.append(f"# Upscale Bench Run Report — {env_info['date']}\n")
    lines.append("## Environment\n")
    for k, v in env_info.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("\n## Frames\n")
    for idx in frame_indices:
        lines.append(f"- frame {idx}: gold path = `{gold_paths.get(idx, 'unknown')}`")
    if oom_events:
        lines.append("\n## OOM events\n")
        for ev in oom_events:
            lines.append(f"- {ev}")
    lines.append("\n## Winners\n")
    ok_results = [r for r in all_results if r.status == "ok"]
    if ok_results:
        # Per-metric winner averaged across frames
        def best(metric_getter, lower_better=True):
            per_method = {}
            for r in ok_results:
                per_method.setdefault(r.method, []).append(metric_getter(r))
            means = {m: sum(v) / len(v) for m, v in per_method.items()}
            return sorted(means.items(), key=lambda kv: kv[1] if lower_better else -kv[1])[0]
        bw_de, v_de = best(lambda r: r.final_delta_e)
        bw_dl1, v_dl1 = best(lambda r: r.delta_l1)
        bw_t_u, v_t_u = best(lambda r: r.wall_time_ms_upscale)
        bw_t_e, v_t_e = best(lambda r: r.wall_time_ms_end_to_end)
        lines.append(f"- Best `final_delta_e`: **{bw_de}** ({v_de:.4f})")
        lines.append(f"- Best `delta_l1`: **{bw_dl1}** ({v_dl1:.4f})")
        lines.append(f"- Best `wall_time_ms_upscale`: **{bw_t_u}** ({v_t_u:.2f} ms)")
        lines.append(f"- Best `wall_time_ms_end_to_end`: **{bw_t_e}** ({v_t_e:.2f} ms)")
    lines.append("\n## Per-method aggregates\n")
    lines.append("| method | delta_l1 | final_ΔE | SSIM | ms (e2e) |")
    lines.append("|---|---|---|---|---|")
    for name in method_names:
        ok_rows = [r for r in all_results if r.method == name and r.status == "ok"]
        if not ok_rows:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        d = sum(r.delta_l1 for r in ok_rows) / len(ok_rows)
        de = sum(r.final_delta_e for r in ok_rows) / len(ok_rows)
        s = sum(r.ssim for r in ok_rows) / len(ok_rows)
        t = sum(r.wall_time_ms_end_to_end for r in ok_rows) / len(ok_rows)
        lines.append(f"| {name} | {d:.4f} | {de:.3f} | {s:.4f} | {t:.1f} |")
    lines.append("\n## Per-frame sheets\n")
    for idx in frame_indices:
        rel = f"per_frame/frame{idx}_sheet.png"
        lines.append(f"![frame {idx}]({rel})")
    lines.append("\n## Notes\n")
    lines.append("- RAUNE-at-4K is out-of-distribution for a model likely trained on small patches; gold is _what RAUNE produces at 4K_, not ground truth.")
    lines.append(f"- Open `{out_dir.name}/summary_grid.png` for the full visual comparison.")

    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta upscale benchmark.")
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--frames", default="auto", help='"auto" or comma-separated indices')
    parser.add_argument("--methods", default="all", help='"all" or comma-separated method names')
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--proxy-size", type=int, default=1080)
    parser.add_argument("--regen-gold", action="store_true")
    parser.add_argument("--gold-sanity-check", action="store_true")
    parser.add_argument("--timing-runs", type=int, default=10)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `run.py` imports cleanly**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -c "from benchmarks.upscale_bench import run; print('run.py imports OK')"
```

Expected: `run.py imports OK` (ignore any DeprecationWarning from dependencies).

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/run.py
git commit -m "$(cat <<'EOF'
feat(bench): add run.py CLI driver

Orchestrates: env logging (git SHAs, weights SHA, clip SHA, GPU, versions),
frame selection (auto or explicit), RAUNE model load (once), per-frame
gold computation via load_or_compute_gold, per-method timing + quality
measurement, visualization (summary grid, per-frame sheets, heatmaps),
metrics.csv, and run_report.md.

sr_maxine is hard-required (raises if nvvfx is missing). Other methods
OOM-tolerant — a failed method gets a NaN row and the run continues.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: End-to-end smoke test (subprocess)

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/tests/test_e2e.py`

- [ ] **Step 1: Write `tests/test_e2e.py`**

```python
"""End-to-end smoke test: run.py as a subprocess on a tiny synthetic clip."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path("/workspaces/dorea-workspace/repos/dorea")
VENV_PY = Path("/opt/dorea-venv/bin/python")
REAL_CLIP = Path(
    "/workspaces/dorea-workspace/footage/raw/2025-11-01/"
    "DJI_20251101111428_0055_D_3s.MP4"
)


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="test clip missing")
def test_e2e_bilinear_bicubic_on_real_clip(tmp_path):
    """Runs the driver on a single frame with only fast methods.

    This is an integration sanity check — not a performance test. Uses
    bilinear + bicubic (both run in a few ms) and regen-gold so we don't
    pollute the real cache directory.
    """
    out_dir = tmp_path / "bench_out"
    result = subprocess.run(
        [
            str(VENV_PY), "-m", "benchmarks.upscale_bench.run",
            "--clip", str(REAL_CLIP),
            "--frames", "30",
            "--methods", "bilinear,bicubic",
            "--out-dir", str(out_dir),
            "--timing-runs", "2",
            "--timing-warmup", "1",
        ],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, f"run.py exited with {result.returncode}"
    assert (out_dir / "summary_grid.png").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "run_report.md").exists()
    # Verify the CSV has the expected rows (2 methods × 1 frame = 2, + 2 aggregate rows)
    with (out_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    ok_rows = [r for r in rows if r["status"] == "ok" and r["frame_idx"] != "ALL_MEAN"]
    assert len(ok_rows) == 2
```

- [ ] **Step 2: Run the e2e test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m pytest benchmarks/upscale_bench/tests/test_e2e.py -v -s
```

Expected: 1 test passes. First run triggers a real RAUNE-at-4K gold computation on frame 30 (slow, ~5–30 s); subsequent runs from the cache directory structure inside `tmp_path` will each be fresh, so expect ~30 s per run of this test. That's fine for an integration test run manually.

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/tests/test_e2e.py
git commit -m "$(cat <<'EOF'
test(bench): add end-to-end CLI subprocess smoke test

Runs the driver on frame 30 of the real test clip with only bilinear and
bicubic (fast methods) and regen-gold into a tmp dir. Verifies
summary_grid.png, metrics.csv, and run_report.md are produced, and that
the CSV has the expected number of rows.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: README + first real run + follow-up capture

**Files:**
- Create: `repos/dorea/benchmarks/upscale_bench/README.md`

- [ ] **Step 1: Write the README**

```markdown
# Upscale Bench

Re-runnable benchmark comparing 9 delta-upscale methods for the RAUNE
OKLab delta path in `raune_filter.py::_process_batch`, against a
RAUNE-at-4K gold standard.

See `docs/decisions/2026-04-10-delta-upscale-bench-design.md` for the
design rationale.

## Setup (one-time)

```bash
# 1. Install dependencies + git-lfs
bash repos/dorea/scripts/setup_bench.sh

# 2. Download RAUNE weights (idempotent; uses gdown from Google Drive)
bash repos/dorea/scripts/download_raune_weights.sh
```

## Run

```bash
# From repos/dorea/
/opt/dorea-venv/bin/python -m benchmarks.upscale_bench.run
```

This produces:
- `working/upscale_bench/summary_grid.png` — the big contact sheet
- `working/upscale_bench/per_frame/frame{idx}_sheet.png` — per-frame detail
- `working/upscale_bench/heatmaps/frame{idx}_{method}.png` — error heatmaps
- `working/upscale_bench/metrics.csv` — machine-readable metrics
- `working/upscale_bench/run_report.md` — human-readable summary

First run: ~1–3 minutes (gold computation + Real-ESRGAN model download).
Subsequent runs: ~30 seconds (gold cache hits).

## Useful flags

```bash
# Override which frames are used
--frames 30,180,300

# Run a subset of methods
--methods bilinear,bicubic,joint_bilateral

# Force gold recomputation (ignore cache)
--regen-gold

# Sanity-check the gold by running both native and tiled paths
--gold-sanity-check

# See everything
--verbose
```

## Adding a new method

One function, one decorator, zero other files touched:

```python
# benchmarks/upscale_bench/methods.py
@register("my_new_method")
def my_new_method(
    delta_proxy: torch.Tensor,       # (1, 3, ph, pw) fp32 OKLab delta
    orig_full:   torch.Tensor,       # (1, 3, fh, fw) fp32 RGB [0,1]
    full_size:   tuple[int, int],
) -> torch.Tensor:                   # (1, 3, fh, fw) fp32 OKLab delta
    return my_implementation(delta_proxy, orig_full, full_size)
```

The parametrized smoke test picks up the new method automatically, and
the driver runs it next time the bench is invoked.

If your method needs the RAUNE model (e.g. for re-running it at full
resolution), set `my_method.__needs_model__ = True` and use
`_get_injected_raune_model()` inside the function — see `higher_proxy`.

## Methods

| name | category | notes |
|---|---|---|
| `bilinear` | baseline | current production behavior |
| `bicubic` | classical | F.interpolate mode=bicubic |
| `lanczos3` | classical | separable 1-D Lanczos taps |
| `joint_bilateral` | edge-aware | Triton kernel, luma-guided |
| `guided_filter` | edge-aware | He/Sun/Tang 2010, box filters |
| `asymmetric_bilateral` | edge-aware | joint bilateral with larger σ for a/b |
| `higher_proxy` | control | RAUNE at 4K, shares tiled fallback with gold |
| `sr_maxine` | learned SR | NVIDIA Maxine VideoSuperRes, conservative |
| `sr_realesrgan` | learned SR | Real-ESRGAN x2plus, aggressive (optional) |

## Troubleshooting

- **`nvvfx` import fails:** run `pip install nvidia-vfx` in `/opt/dorea-venv`. This is hard-required.
- **RAUNE weights missing:** run `scripts/download_raune_weights.sh`. The script is idempotent.
- **OOM on gold tiled path:** free some VRAM (close other tools) and try again. The fallback already tiles to 1984×1144 per tile.
- **Real-ESRGAN not installed:** soft-skipped, the run continues with 8 methods instead of 9.
```

- [ ] **Step 2: Commit the README**

```bash
cd /workspaces/dorea-workspace
git add repos/dorea/benchmarks/upscale_bench/README.md
git commit -m "$(cat <<'EOF'
docs(bench): add upscale_bench README with setup/run/extend instructions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Run the full benchmark on the real clip**

```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -m benchmarks.upscale_bench.run
```

Expected: completes in 1–3 minutes. Output at `working/upscale_bench/`. Inspect:

```bash
ls -la /workspaces/dorea-workspace/working/upscale_bench/
cat /workspaces/dorea-workspace/working/upscale_bench/run_report.md
```

- [ ] **Step 4: Eyeball the summary grid and per-frame sheets**

Open `working/upscale_bench/summary_grid.png` and the three `working/upscale_bench/per_frame/frame*_sheet.png` files. Look for:

- The gold column should look clearly better than `bilinear`.
- `higher_proxy` should be close to (or equal to) the gold — if not, something's wrong with the tiled path.
- Crops should zoom into visible "argument areas" where methods disagree.
- Heatmaps should light up mostly along edges for classical methods.

- [ ] **Step 5: Record the first run's findings into corvia**

Use `corvia_write` to record:

- Winning method on `final_delta_e`
- Winning method on `wall_time_ms_end_to_end`
- Any unexpected findings (e.g. if `higher_proxy` doesn't win, that's a scoop about information loss)
- Which gold path was used per frame

Example (adjust values from the actual run):

```bash
/opt/dorea-venv/bin/python - <<'PYEOF'
# Pseudo-code — actual corvia_write is via MCP. Use the MCP tool in the agent.
# This is a placeholder note for the human runner: write the finding to corvia
# with content_role="finding" and source_origin="repo:dorea".
print("Record the winner + any surprises in corvia before moving on.")
PYEOF
```

- [ ] **Step 6: Final commit — any fixes from the dry run**

If the first real run surfaced bugs, fix them and commit separately. If the run was clean:

```bash
cd /workspaces/dorea-workspace
git status --short
# If clean, nothing to commit. If changes, create a follow-up fix commit.
```

---

## Self-review

### Spec coverage

| Spec section | Covered by |
|---|---|
| Package layout | Tasks 1, 4 |
| Method contract | Task 4 |
| Driver flow | Task 15 |
| `bilinear` | Task 4 |
| `bicubic` | Task 5 |
| `lanczos3` | Task 5 |
| `joint_bilateral` | Task 7 |
| `guided_filter` | Task 6 |
| `asymmetric_bilateral` | Task 7 |
| `higher_proxy` | Task 10 |
| `sr_maxine` | Task 11 |
| `sr_realesrgan` | Task 12 |
| Frame selection (auto + explicit) | Task 8 |
| Gold native path | Task 9 |
| Gold tiled fallback (1984×1144 / 128 overlap) | Task 9 |
| Gold caching | Task 9 |
| Gold sanity check | Task 15 (CLI flag) |
| Delta L1/L2/max/p95 per channel + overall | Task 13 |
| ΔE2000 (CIELab D65) | Task 13 |
| SSIM on BT.709 luma | Task 13 |
| Wall time (upscale-only + end-to-end) | Tasks 13, 15 |
| Timing methodology (10 runs, 3 warmup, median) | Task 13 |
| Summary grid | Task 14 |
| Per-frame sheets (auto-crop) | Task 14 |
| Error heatmaps (turbo, frame-p95 normalized) | Task 14 |
| metrics.csv (per-row + aggregates) | Task 15 |
| run_report.md | Task 15 |
| Setup scripts (setup_bench.sh + download_raune_weights.sh) | Tasks 1, 2 |
| Git-LFS for weights | Tasks 1, 2, 12 |
| dorea.toml path update | Task 3 |
| `.gitattributes` for LFS | Task 1 |
| CLI flags (--clip, --frames, --methods, --out-dir, --proxy-size, --regen-gold, --gold-sanity-check, --timing-runs, --timing-warmup, --verbose) | Task 15 |
| Error handling: RAUNE weights missing | Task 15 |
| Error handling: CUDA unavailable | Task 15 |
| Error handling: nvvfx hard-required | Task 15 |
| Error handling: realesrgan soft-skipped | Task 12 |
| Error handling: OOM per-method | Task 15 |
| Error handling: corrupt gold cache | Task 9 |
| Environment logging | Task 15 |
| Determinism (cudnn.deterministic, seed) | Task 15 |
| Per-method smoke tests | Tasks 4, 7, 10, 11 |
| End-to-end smoke test | Task 16 |
| README + how-to-add-a-method | Task 17 |

All spec sections covered.

### Placeholder scan

- No TBD/TODO placeholders.
- No "similar to task N" — each task's code is self-contained.
- `EXPECTED_SHA256=""` in `download_raune_weights.sh` is explicitly a runtime placeholder that the operator fills in during Task 2 Step 4; the plan calls this out.
- The Step 1 of Task 5 has a `_lanczos_kernel` stub that raises NotImplementedError — the plan explicitly instructs the engineer to remove it and replace with the real implementation. Flagged as intentional.
- The `urllib.request.urlretrieve` for Real-ESRGAN weights downloads at first use. Not a placeholder.

### Type / name consistency

- `REGISTRY`, `@register`, `MethodResult`, `GoldResult`, `FrameCellData`, `compute_gold`, `load_or_compute_gold`, `timed_run`, `generate_summary_grid`, `generate_per_frame_sheet`, `generate_heatmap`, `_load_raune_model`, `set_injected_raune_model`, `_get_injected_raune_model`, `clear_injected_raune_model` — all names consistent across tasks.
- Method signature `(delta_proxy, orig_full, full_size) → delta_full` — all methods conform.
- Paths: `repos/dorea/models/raune_net/models/RAUNE-Net/...` is used consistently in Tasks 2, 3, 9, 15.
- `EXPECTED_SHA256` variable name consistent between download script and verify step.

### Decomposition check

17 tasks. Each task:
- Writes at most 1–3 files
- Ends in a commit
- Has running tests that pass before the commit (except Task 17 Step 3 which runs the real bench — that's intentional integration work, not a unit test)

The plan is appropriately scoped for one implementation plan.

---

## Execution handoff

**Plan complete and saved to `docs/plans/2026-04-10-delta-upscale-bench-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
