# UIE Model Comparison POC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python benchmark script under `repos/dorea/scripts/poc/` that compares fp16 RAUNE-Net against FA+Net, Shallow-UWnet, and Color-Accurate UIE, producing a full-resolution contact sheet PNG and a speed/quality markdown table.

**Architecture:** Standalone throwaway research code. A `uie_bench.py` entrypoint orchestrates decode → per-model load → warm benchmark → contact-sheet inference → render. Model wrappers implement a common `UIEModel` ABC with `load(variant)` and `infer(batch)` methods. No production pipeline imports; no coupling to `dorea_inference/` or `dorea-cli`.

**Tech Stack:** Python 3 (existing `/opt/dorea-venv`), PyTorch with CUDA, PyAV (HEVC decode), NumPy, PIL, scikit-image, pytest. All already in the venv.

**Spec:** `docs/decisions/2026-04-10-uie-comparison-poc-design.md`

**Important up-front context for the implementer:**

- **Three of four model checkouts are already vendored** under `working/sea_thru_poc/models/`:
  - RAUNE-Net at `working/sea_thru_poc/models/RAUNE-Net/` (weights at `pretrained/RAUNENet/test/weights_95.pth`)
  - FA+Net at `working/sea_thru_poc/models/FiveAPlus-Network/` (weights at `model/FAPlusNet-alpha-0.4.pth`)
  - Shallow-UWnet at `working/sea_thru_poc/models/Shallow-UWnet/` (weights at `snapshots/model.ckpt`, a full pickled model)
- **Color-Accurate UIE weights are NOT present.** Task 10 wires the module correctly but assumes weights may be missing at runtime — the spec's per-cell graceful-failure path handles this. The contact sheet will show a red cell for that column if weights are absent; this is the intended behavior and is NOT a test failure.
- Reference (non-POC) scripts that show the loading recipe for each already-vendored model live at `working/sea_thru_poc/run_raune_net.py`, `run_five_aplus.py`, `run_shallow_uwnet.py`. Use them as the source of truth for model class names, constructor args, and preprocessing quirks — do not re-derive from upstream READMEs.
- **Work happens in `repos/dorea/` and `working/poc_weights/` — never touch `python/dorea_inference/` or any `crates/` directory.**
- `repos/dorea/` is a nested git repo separate from the workspace root. All commits in this plan are to `repos/dorea/` — run `cd repos/dorea` before each `git` command.

---

### Task 0: Scaffold directories and stub README

**Files:**
- Create: `repos/dorea/scripts/poc/README.md`
- Create: `repos/dorea/scripts/poc/__init__.py` (empty)
- Create: `repos/dorea/scripts/poc/models/__init__.py` (empty)
- Create: `repos/dorea/scripts/poc/tests/__init__.py` (empty)

- [ ] **Step 1: Create directory tree**

```bash
cd /workspaces/dorea-workspace/repos/dorea
mkdir -p scripts/poc/models scripts/poc/tests
touch scripts/poc/__init__.py scripts/poc/models/__init__.py scripts/poc/tests/__init__.py
```

- [ ] **Step 2: Write README**

Create `repos/dorea/scripts/poc/README.md`:

```markdown
# UIE Model Comparison POC

Throwaway benchmark comparing fp16 RAUNE-Net against three lightweight
alternatives on a dive clip. Produces a contact sheet PNG and a speed/quality
markdown table.

**This is research code. It is not used by the production pipeline.**

## Models compared

| Model              | Source                                                       | Weights file                                                             |
|--------------------|--------------------------------------------------------------|--------------------------------------------------------------------------|
| RAUNE-Net          | `working/sea_thru_poc/models/RAUNE-Net/`                     | `working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth` |
| FA+Net             | `working/sea_thru_poc/models/FiveAPlus-Network/`             | `working/sea_thru_poc/models/FiveAPlus-Network/model/FAPlusNet-alpha-0.4.pth`   |
| Shallow-UWnet      | `working/sea_thru_poc/models/Shallow-UWnet/`                 | `working/sea_thru_poc/models/Shallow-UWnet/snapshots/model.ckpt`         |
| Color-Accurate UIE | Upstream: https://arxiv.org/abs/2603.16363 (code not yet vendored) | `working/poc_weights/color_accurate/model.pth` (must be hand-placed)      |

If Color-Accurate UIE weights are missing, that cell fails gracefully and the
rest of the bench runs. Drop weights in place and rerun to include it.

## Running

```bash
cd /workspaces/dorea-workspace
/opt/dorea-venv/bin/python repos/dorea/scripts/poc/uie_bench.py \
    --input footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4
```

Output lands in `working/poc_out/<timestamp>/`:

- `contact_sheet.png` — 8 rows × 5 cols, BEST_EFFORT variant only
- `bench.json` — machine-readable results
- `bench.md` — human-readable table, sorted by FPS desc
- `logs/` — per-cell stderr captures

## Running tests

```bash
/opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/ -v
```
```

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/README.md scripts/poc/__init__.py scripts/poc/models/__init__.py scripts/poc/tests/__init__.py
git commit -m "chore(poc): scaffold UIE comparison POC directory + README"
```

---

### Task 1: Base ABC and shared fp16 helpers

**Files:**
- Create: `repos/dorea/scripts/poc/models/base.py`
- Create: `repos/dorea/scripts/poc/tests/test_base_helpers.py`

- [ ] **Step 1: Write failing test for `force_norm_fp32`**

Create `repos/dorea/scripts/poc/tests/test_base_helpers.py`:

```python
"""Tests for the shared fp16 helpers in models.base."""
import torch
import torch.nn as nn
import pytest

from scripts.poc.models.base import force_norm_fp32, Variant


class TinyNetWithNorms(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.inorm = nn.InstanceNorm2d(8, affine=True)
        self.bnorm = nn.BatchNorm2d(8)
        self.gnorm = nn.GroupNorm(2, 8)
        self.out = nn.Conv2d(8, 3, 1)

    def forward(self, x):
        return self.out(self.gnorm(self.bnorm(self.inorm(self.conv(x)))))


def test_force_norm_fp32_preserves_norm_precision():
    m = TinyNetWithNorms().half()
    force_norm_fp32(m)
    assert m.conv.weight.dtype == torch.float16
    assert m.out.weight.dtype == torch.float16
    # Norm layers kept in fp32 (their parameters exist for affine=True)
    assert m.inorm.weight.dtype == torch.float32
    assert m.bnorm.weight.dtype == torch.float32
    assert m.gnorm.weight.dtype == torch.float32


def test_variant_enum_has_two_members():
    assert Variant.STRICT_PARITY.value == "strict_parity"
    assert Variant.BEST_EFFORT.value == "best_effort"
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_base_helpers.py -v
```

Expected: ImportError — `scripts.poc.models.base` doesn't exist yet.

- [ ] **Step 3: Implement `models/base.py`**

Create `repos/dorea/scripts/poc/models/base.py`:

```python
"""Base interface and shared helpers for the UIE POC model wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn


class Variant(Enum):
    """fp16 quantization recipe variant."""
    STRICT_PARITY = "strict_parity"
    BEST_EFFORT = "best_effort"


# Tuple of norm classes whose parameters must stay in fp32 after a global
# .half() conversion to avoid per-channel variance underflow.
_NORM_TYPES = (
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.BatchNorm1d,    nn.BatchNorm2d,    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
)


def force_norm_fp32(model: nn.Module) -> int:
    """Force all norm layers in ``model`` back to fp32 after .half().

    Generalizes the RAUNE-Net direct-mode rule (raune_filter.py:680-685).
    Returns the count of norm modules touched.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, _NORM_TYPES):
            m.float()
            count += 1
    return count


class UIEModel(ABC):
    """Common interface for every POC model wrapper.

    Lifecycle:
        1. construct() — cheap, no tensors allocated
        2. load(variant, device) — allocates weights + applies fp16 recipe
        3. infer(batch) — called repeatedly with a pre-shaped CUDA fp16 tensor
        4. unload() — releases tensors (implicit via del + empty_cache)
    """
    name: str  # override in subclass

    @abstractmethod
    def load(self, variant: Variant, device: torch.device) -> None:
        """Load weights and apply the fp16 recipe for this variant.

        Called exactly once per wrapper instance.
        """

    @abstractmethod
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Input:  ``(N, 3, H, W)`` float16 tensor on CUDA, values in ``[0, 1]``.
        Output: ``(N, 3, H, W)`` float16 tensor on CUDA, values in ``[0, 1]``.
        """

    @abstractmethod
    def preferred_proxy_size(self) -> int:
        """Long-edge pixel count this model wants for its proxy input."""
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_base_helpers.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/models/base.py scripts/poc/tests/test_base_helpers.py
git commit -m "feat(poc): add UIEModel ABC and force_norm_fp32 helper"
```

---

### Task 2: Metrics module — timing, VRAM, UIQM, SSIM

**Files:**
- Create: `repos/dorea/scripts/poc/metrics.py`
- Create: `repos/dorea/scripts/poc/tests/test_metrics.py`

- [ ] **Step 1: Write failing test for metrics**

Create `repos/dorea/scripts/poc/tests/test_metrics.py`:

```python
"""Tests for scripts.poc.metrics."""
import numpy as np
import pytest

from scripts.poc.metrics import (
    BenchResult,
    compute_ssim,
    compute_uiqm,
)


def test_ssim_identity_is_one():
    rng = np.random.default_rng(42)
    img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-5)


def test_ssim_symmetric():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    assert compute_ssim(a, b) == pytest.approx(compute_ssim(b, a), abs=1e-6)


def test_uiqm_returns_finite_positive_on_real_image():
    rng = np.random.default_rng(7)
    # UIQM is defined on natural images, not pure noise — use a smooth gradient
    x = np.linspace(0, 255, 128, dtype=np.float32)
    img = np.stack([
        np.tile(x, (128, 1)),
        np.tile(x, (128, 1)).T,
        np.full((128, 128), 96, dtype=np.float32),
    ], axis=-1).astype(np.uint8)
    u = compute_uiqm(img)
    assert np.isfinite(u)
    assert u > 0.0


def test_bench_result_serialisable():
    r = BenchResult(
        model="raune",
        variant="strict_parity",
        proxy_size=(608, 1080),
        batch_size=4,
        time_to_first_frame_s=9.4,
        latency_mean_ms=62.0,
        latency_p50_ms=61.5,
        latency_p95_ms=78.0,
        throughput_fps=64.0,
        vram_peak_mib=1842.0,
        uiqm_mean=3.24,
        ssim_vs_raune_mean=1.0,
        error=None,
    )
    from dataclasses import asdict
    d = asdict(r)
    assert d["model"] == "raune"
    assert d["ssim_vs_raune_mean"] == 1.0
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_metrics.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `metrics.py`**

Create `repos/dorea/scripts/poc/metrics.py`:

```python
"""Metrics and timing helpers for the UIE comparison POC.

UIQM reference: Panetta, Gao, Agaian, "Human-Visual-System-Inspired
Underwater Image Quality Measures", IEEE J. Oceanic Eng. 41(3) 541-551 (2016).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as _skimage_ssim


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    """One row in the output benchmark table."""
    model: str
    variant: str
    proxy_size: tuple[int, int]       # (H, W) actually used
    batch_size: int
    time_to_first_frame_s: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_fps: float
    vram_peak_mib: float
    uiqm_mean: Optional[float]
    ssim_vs_raune_mean: Optional[float]
    error: Optional[str]              # set iff the cell failed


# ─────────────────────────────────────────────────────────────────────────────
# Timing and VRAM
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def cuda_timer():
    """Measure wall-time around a CUDA region, handling async kernels.

    Usage:
        with cuda_timer() as t:
            model(batch)
        elapsed_ms = t.elapsed_ms
    """
    class _T:
        elapsed_ms: float = 0.0

    t = _T()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t.elapsed_ms = (time.perf_counter() - t0) * 1000.0


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def vram_peak_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# SSIM — wraps skimage
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM across channels. Inputs must be uint8 HxWx3 of the same shape."""
    assert a.shape == b.shape, f"SSIM shape mismatch: {a.shape} vs {b.shape}"
    assert a.dtype == np.uint8 and b.dtype == np.uint8
    # skimage wants channel_axis for multichannel
    return float(_skimage_ssim(a, b, channel_axis=-1, data_range=255))


# ─────────────────────────────────────────────────────────────────────────────
# UIQM — vendored numpy implementation
# ─────────────────────────────────────────────────────────────────────────────

def _uicm(img: np.ndarray) -> float:
    """Underwater image colourfulness measure (chrominance)."""
    r = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    b = img[:, :, 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b

    def _asymmetric_alpha_trimmed(x: np.ndarray, a_l: float = 0.1, a_r: float = 0.1) -> tuple[float, float]:
        flat = np.sort(x.flatten())
        n = flat.size
        lo = int(np.floor(a_l * n))
        hi = int(np.ceil((1 - a_r) * n))
        if hi <= lo:
            return float(flat.mean()), float(flat.var())
        trimmed = flat[lo:hi]
        return float(trimmed.mean()), float(trimmed.var())

    mu_rg, var_rg = _asymmetric_alpha_trimmed(rg)
    mu_yb, var_yb = _asymmetric_alpha_trimmed(yb)

    lhs = np.sqrt(mu_rg ** 2 + mu_yb ** 2)
    rhs = np.sqrt(var_rg + var_yb)
    # Coefficients from Panetta 2016 eq. 16
    return float(-0.0268 * lhs + 0.1586 * rhs)


def _sobel_magnitude(ch: np.ndarray) -> np.ndarray:
    """Simple Sobel gradient magnitude using numpy (avoids scipy dep churn)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    # Use FFT-based convolution via np.pad + manual slide (small image ok)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(ch.astype(np.float64), 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    gx = (windows * kx).sum(axis=(-1, -2))
    gy = (windows * ky).sum(axis=(-1, -2))
    return np.sqrt(gx * gx + gy * gy)


def _uism(img: np.ndarray) -> float:
    """Underwater image sharpness measure via EME of Sobel edges."""
    # Compute per-channel sharpness, weighted by luminance contributions
    weights = (0.299, 0.587, 0.114)
    total = 0.0
    for c in range(3):
        edges = _sobel_magnitude(img[:, :, c])
        # Scale edges by intensity mask (EME-like)
        scaled = edges * img[:, :, c].astype(np.float64)
        total += weights[c] * _eme(scaled)
    return float(total)


def _eme(block: np.ndarray, n_blocks: int = 8) -> float:
    """Enhancement Measure Estimation — mean of log(max/min) across sub-blocks."""
    h, w = block.shape
    bh = max(h // n_blocks, 1)
    bw = max(w // n_blocks, 1)
    vals: list[float] = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            sub = block[i:i + bh, j:j + bw]
            if sub.size == 0:
                continue
            mx = float(sub.max())
            mn = float(sub.min())
            if mn <= 1e-6 or mx <= 1e-6:
                continue
            vals.append(np.log(mx / mn))
    if not vals:
        return 0.0
    return (2.0 / (n_blocks * n_blocks)) * float(np.sum(vals))


def _uiconm(img: np.ndarray) -> float:
    """Contrast measure — EME of the luminance channel."""
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    return _eme(lum.astype(np.float64))


def compute_uiqm(img: np.ndarray) -> float:
    """UIQM = c1·UICM + c2·UISM + c3·UIConM, coefficients from Panetta 2016."""
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img)
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_metrics.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/metrics.py scripts/poc/tests/test_metrics.py
git commit -m "feat(poc): add metrics module — UIQM, SSIM, cuda_timer, BenchResult"
```

---

### Task 3: Render module — contact sheet composition

**Files:**
- Create: `repos/dorea/scripts/poc/render.py`
- Create: `repos/dorea/scripts/poc/tests/test_render.py`

- [ ] **Step 1: Write failing test for render**

Create `repos/dorea/scripts/poc/tests/test_render.py`:

```python
"""Tests for scripts.poc.render."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from scripts.poc.metrics import BenchResult
from scripts.poc.render import build_contact_sheet


def _dummy_frame(h: int, w: int, fill: int) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _dummy_result(model: str, error: str | None = None) -> BenchResult:
    return BenchResult(
        model=model,
        variant="best_effort",
        proxy_size=(64, 64),
        batch_size=4,
        time_to_first_frame_s=1.0,
        latency_mean_ms=10.0,
        latency_p50_ms=10.0,
        latency_p95_ms=12.0,
        throughput_fps=400.0,
        vram_peak_mib=100.0,
        uiqm_mean=3.0 if error is None else None,
        ssim_vs_raune_mean=1.0 if error is None else None,
        error=error,
    )


def test_build_contact_sheet_writes_png(tmp_path: Path):
    originals = [_dummy_frame(64, 64, 128) for _ in range(3)]
    model_outputs = {
        "raune":          [_dummy_frame(64, 64, 200) for _ in range(3)],
        "fa_net":         [_dummy_frame(64, 64, 150) for _ in range(3)],
        "shallow_uwnet":  [_dummy_frame(64, 64, 100) for _ in range(3)],
        "color_accurate": [_dummy_frame(64, 64, 50)  for _ in range(3)],
    }
    results = {m: _dummy_result(m) for m in model_outputs}
    out = tmp_path / "sheet.png"
    build_contact_sheet(originals, model_outputs, results, tile_size="native", out_path=out)
    assert out.exists()
    assert out.stat().st_size > 10_000  # non-trivial PNG
    img = Image.open(out)
    assert img.mode in ("RGB", "RGBA")


def test_build_contact_sheet_renders_error_cell_red(tmp_path: Path):
    originals = [_dummy_frame(64, 64, 128)]
    model_outputs = {
        "raune":          [_dummy_frame(64, 64, 200)],
        "fa_net":         [_dummy_frame(64, 64, 0)],    # placeholder — should be overwritten
        "shallow_uwnet":  [_dummy_frame(64, 64, 100)],
        "color_accurate": [_dummy_frame(64, 64, 50)],
    }
    results = {
        "raune":          _dummy_result("raune"),
        "fa_net":         _dummy_result("fa_net", error="weights not found"),
        "shallow_uwnet":  _dummy_result("shallow_uwnet"),
        "color_accurate": _dummy_result("color_accurate"),
    }
    out = tmp_path / "sheet.png"
    build_contact_sheet(originals, model_outputs, results, tile_size="native", out_path=out)
    arr = np.array(Image.open(out).convert("RGB"))
    # The failed fa_net column should contain red pixels somewhere
    # (dark red fill is (128, 0, 0) — dominant R channel)
    reds = (arr[:, :, 0] > 100) & (arr[:, :, 1] < 40) & (arr[:, :, 2] < 40)
    assert reds.any(), "Expected red error fill in fa_net column"
```

- [ ] **Step 2: Run test, confirm failure**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_render.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `render.py`**

Create `repos/dorea/scripts/poc/render.py`:

```python
"""Contact sheet composition for the UIE comparison POC."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.poc.metrics import BenchResult


# Fixed column order — must match uie_bench.py's MODEL_ORDER
_COLUMN_ORDER = ("original", "raune", "fa_net", "shallow_uwnet", "color_accurate")

_BORDER_PX = 4
_BORDER_RGB = (32, 32, 32)
_ROW_LABEL_W = 120
_COL_HEADER_H = 80
_ERROR_FILL = (128, 0, 0)
_TEXT_RGB = (240, 240, 240)


def _get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=22)  # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int = 24) -> str:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _resize_to_tile(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a uint8 HxWx3 array to (target_h, target_w) via Lanczos."""
    if arr.shape[:2] == (target_h, target_w):
        return arr
    img = Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(img)


def _pad_to_tile(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad a smaller uint8 HxWx3 array to (target_h, target_w) with dark fill."""
    h, w = arr.shape[:2]
    if (h, w) == (target_h, target_w):
        return arr
    out = np.full((target_h, target_w, 3), _BORDER_RGB, dtype=np.uint8)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def _error_tile(h: int, w: int, message: str, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.new("RGB", (w, h), _ERROR_FILL)
    draw = ImageDraw.Draw(img)
    wrapped = _wrap_text(f"FAILED: {message}")
    draw.multiline_text((20, 20), wrapped, fill=_TEXT_RGB, font=font, spacing=6)
    return np.array(img)


def _overlay_metrics(arr: np.ndarray, result: BenchResult, font: ImageFont.ImageFont) -> np.ndarray:
    """Paint a small metrics box in the bottom-right corner."""
    if result.error is not None:
        return arr
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    p50 = f"{result.latency_p50_ms:.0f}ms" if result.latency_p50_ms else "—"
    uiqm = f"{result.uiqm_mean:.2f}" if result.uiqm_mean is not None else "—"
    text = f"p50={p50}\nUIQM={uiqm}"
    # Simple black background box
    box_w, box_h = 180, 54
    h, w = arr.shape[:2]
    draw.rectangle([w - box_w - 10, h - box_h - 10, w - 10, h - 10], fill=(0, 0, 0))
    draw.multiline_text((w - box_w, h - box_h - 4), text, fill=_TEXT_RGB, font=font, spacing=2)
    return np.array(img)


def build_contact_sheet(
    original_frames: list[np.ndarray],
    model_outputs: dict[str, list[np.ndarray]],
    bench_results: dict[str, BenchResult],
    tile_size: Literal["native", "4k"],
    out_path: Path,
) -> None:
    """Render an N-row × 5-col contact sheet and save it as PNG.

    Columns (fixed): original | raune | fa_net | shallow_uwnet | color_accurate.
    Rows: one per element of ``original_frames``. Every model in
    ``model_outputs`` must provide that many frames.

    For failed cells (``bench_results[m].error is not None``), the model's
    output frames are replaced with dark-red error tiles.
    """
    n_rows = len(original_frames)
    assert n_rows > 0
    for m in ("raune", "fa_net", "shallow_uwnet", "color_accurate"):
        assert m in model_outputs, f"missing model output: {m}"
        assert len(model_outputs[m]) == n_rows, f"{m}: {len(model_outputs[m])} != {n_rows}"

    # Determine tile size
    if tile_size == "4k":
        tile_h, tile_w = 2160, 3840
    else:
        # Native: max dims across all tiles including originals
        all_h: list[int] = []
        all_w: list[int] = []
        for row_frames in [original_frames] + [model_outputs[m] for m in ("raune", "fa_net", "shallow_uwnet", "color_accurate")]:
            for f in row_frames:
                all_h.append(f.shape[0])
                all_w.append(f.shape[1])
        tile_h = max(all_h)
        tile_w = max(all_w)

    # Build per-cell arrays (post-resize / post-error)
    n_cols = len(_COLUMN_ORDER)

    def _cell(col: str, row_idx: int) -> np.ndarray:
        if col == "original":
            src = original_frames[row_idx]
        else:
            res = bench_results[col]
            if res.error is not None:
                return _error_tile(tile_h, tile_w, res.error, _get_font())
            src = model_outputs[col][row_idx]
        if tile_size == "4k":
            return _resize_to_tile(src, tile_h, tile_w)
        # native: pad smaller tiles, downsample the 4K original to max tile size
        if src.shape[0] > tile_h or src.shape[1] > tile_w:
            return _resize_to_tile(src, tile_h, tile_w)
        return _pad_to_tile(src, tile_h, tile_w)

    font = _get_font()

    # Compose the canvas
    sheet_w = _ROW_LABEL_W + n_cols * tile_w + (n_cols + 1) * _BORDER_PX
    sheet_h = _COL_HEADER_H + n_rows * tile_h + (n_rows + 1) * _BORDER_PX
    canvas = Image.new("RGB", (sheet_w, sheet_h), _BORDER_RGB)

    # Column headers
    draw = ImageDraw.Draw(canvas)
    for ci, col in enumerate(_COLUMN_ORDER):
        x0 = _ROW_LABEL_W + ci * (tile_w + _BORDER_PX) + _BORDER_PX
        label = col if col == "original" else f"{col} / best_effort"
        draw.text((x0 + 10, 20), label, fill=_TEXT_RGB, font=font)

    # Rows
    for ri in range(n_rows):
        y0 = _COL_HEADER_H + ri * (tile_h + _BORDER_PX) + _BORDER_PX
        draw.text((10, y0 + tile_h // 2 - 10), f"f={ri:03d}", fill=_TEXT_RGB, font=font)
        for ci, col in enumerate(_COLUMN_ORDER):
            x0 = _ROW_LABEL_W + ci * (tile_w + _BORDER_PX) + _BORDER_PX
            tile = _cell(col, ri)
            if col != "original":
                tile = _overlay_metrics(tile, bench_results[col], font)
            canvas.paste(Image.fromarray(tile), (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=False)
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_render.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/render.py scripts/poc/tests/test_render.py
git commit -m "feat(poc): add contact sheet render with error cell fallback"
```

---

### Task 4: RAUNE-Net wrapper

**Files:**
- Create: `repos/dorea/scripts/poc/models/raune.py`
- Create: `repos/dorea/scripts/poc/tests/test_raune_wrapper.py`

**Reference:** `working/sea_thru_poc/models/RAUNE-Net/models/raune_net.py` defines the `RauneNet` class. Direct mode constructs it with `input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64` and normalizes inputs with mean/std `(0.5, 0.5, 0.5)`, so model input is in `[-1, 1]` and output is in `[-1, 1]`. Our `UIEModel.infer` contract uses `[0, 1]`, so the wrapper must convert on both sides.

- [ ] **Step 1: Write failing smoke test**

Create `repos/dorea/scripts/poc/tests/test_raune_wrapper.py`:

```python
"""Smoke test that the RAUNE wrapper respects the UIEModel contract."""
from pathlib import Path
import torch
import pytest

from scripts.poc.models.base import Variant
from scripts.poc.models.raune import RauneWrapper


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RAUNE wrapper requires CUDA",
)

WEIGHTS = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth")
MODELS_DIR = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not present")
def test_raune_wrapper_round_trips_shape_and_dtype():
    w = RauneWrapper(weights_path=WEIGHTS, models_dir=MODELS_DIR)
    w.load(Variant.BEST_EFFORT, torch.device("cuda"))
    batch = torch.rand((2, 3, 256, 256), dtype=torch.float16, device="cuda")
    out = w.infer(batch)
    assert out.shape == batch.shape
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -0.01 and float(out.max()) <= 1.01


def test_preferred_proxy_size_is_1080():
    w = RauneWrapper(weights_path=WEIGHTS, models_dir=MODELS_DIR)
    assert w.preferred_proxy_size() == 1080
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_raune_wrapper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `models/raune.py`**

Create `repos/dorea/scripts/poc/models/raune.py`:

```python
"""RAUNE-Net wrapper — fp16 with InstanceNorm kept in fp32 + torch.compile."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from scripts.poc.models.base import UIEModel, Variant, force_norm_fp32


# Normalization constants from direct mode: maps [0,1] → [-1,1] on input,
# and [-1,1] → [0,1] on output.
_NORM_MEAN = 0.5
_NORM_STD = 0.5


class RauneWrapper(UIEModel):
    name = "raune"

    def __init__(self, weights_path: Path, models_dir: Path) -> None:
        self._weights_path = Path(weights_path)
        self._models_dir = Path(models_dir)
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def preferred_proxy_size(self) -> int:
        return 1080  # matches direct mode's default raune_proxy_size

    def load(self, variant: Variant, device: torch.device) -> None:
        # Make the vendored RAUNE-Net models/ dir importable
        models_pkg = self._models_dir
        if not (models_pkg / "models" / "raune_net.py").exists():
            raise FileNotFoundError(
                f"RAUNE-Net models dir invalid: {models_pkg} "
                f"(expected models/raune_net.py inside)"
            )
        if str(models_pkg) not in sys.path:
            sys.path.insert(0, str(models_pkg))

        from models.raune_net import RauneNet  # type: ignore

        if not self._weights_path.exists():
            raise FileNotFoundError(f"RAUNE-Net weights not found at {self._weights_path}")

        model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).to(device)
        state = torch.load(self._weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # fp16 conversion — same recipe for both variants (RAUNE is the baseline)
        model = model.half()
        force_norm_fp32(model)

        # torch.compile for reduce-overhead CUDA graph path
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        self._model = model
        self._device = device

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "load() must be called first"
        assert batch.dtype == torch.float16, f"expected fp16, got {batch.dtype}"
        # [0,1] → [-1,1]
        x = (batch - _NORM_MEAN) / _NORM_STD
        with torch.no_grad():
            y = self._model(x)
        # [-1,1] → [0,1]
        y = ((y + 1.0) * 0.5).clamp(0.0, 1.0)
        return y.to(torch.float16)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_raune_wrapper.py -v
```

Expected: 2 passed (or 1 passed + 1 skipped if no GPU / weights absent).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/models/raune.py scripts/poc/tests/test_raune_wrapper.py
git commit -m "feat(poc): add RAUNE-Net wrapper (fp16 + compile reduce-overhead)"
```

---

### Task 5: FA+Net wrapper

**Files:**
- Create: `repos/dorea/scripts/poc/models/fa_net.py`
- Create: `repos/dorea/scripts/poc/tests/test_fa_net_wrapper.py`

**Reference:** `working/sea_thru_poc/run_five_aplus.py` and `models/FiveAPlus-Network/archs/FIVE_APLUS.py`. Key quirks:
- Class name is `FIVE_APLUSNet`, imported from `archs.FIVE_APLUS`
- Weights: `model/FAPlusNet-alpha-0.4.pth`, loaded via `load_state_dict(torch.load(..., weights_only=True))`
- Input is `[0, 1]` via `ToTensor()` (no mean/std normalization)
- **Forward returns a tuple `(output_tensor, stage2_head)`** — we use the first element only
- Output is in `[0, 1]` (tanh/sigmoid internally)
- **Input H and W must be divisible by 128** (Enhance module uses `F.avg_pool2d` with kernels up to 128)

- [ ] **Step 1: Write failing smoke test**

Create `repos/dorea/scripts/poc/tests/test_fa_net_wrapper.py`:

```python
"""Smoke test that the FA+Net wrapper respects the UIEModel contract."""
from pathlib import Path
import torch
import pytest

from scripts.poc.models.base import Variant
from scripts.poc.models.fa_net import FaNetWrapper


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FA+Net wrapper requires CUDA",
)

WEIGHTS = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/FiveAPlus-Network/model/FAPlusNet-alpha-0.4.pth")
ARCH_DIR = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/FiveAPlus-Network")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="FA+Net weights not present")
def test_fa_net_wrapper_round_trips_shape_and_dtype():
    w = FaNetWrapper(weights_path=WEIGHTS, arch_dir=ARCH_DIR)
    w.load(Variant.BEST_EFFORT, torch.device("cuda"))
    # Must be divisible by 128
    batch = torch.rand((2, 3, 256, 256), dtype=torch.float16, device="cuda")
    out = w.infer(batch)
    assert out.shape == batch.shape
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_fa_net_wrapper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `models/fa_net.py`**

Create `repos/dorea/scripts/poc/models/fa_net.py`:

```python
"""FA+Net (Five A+ Network) wrapper — fp16, input dims must be divisible by 128."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from scripts.poc.models.base import UIEModel, Variant, force_norm_fp32


class FaNetWrapper(UIEModel):
    name = "fa_net"

    def __init__(self, weights_path: Path, arch_dir: Path) -> None:
        self._weights_path = Path(weights_path)
        self._arch_dir = Path(arch_dir)
        self._model: torch.nn.Module | None = None
        self._variant: Variant | None = None

    def preferred_proxy_size(self) -> int:
        return 1024  # divisible by 128, same as reference script

    def load(self, variant: Variant, device: torch.device) -> None:
        if not (self._arch_dir / "archs" / "FIVE_APLUS.py").exists():
            raise FileNotFoundError(
                f"FA+Net arch dir invalid: {self._arch_dir} "
                f"(expected archs/FIVE_APLUS.py inside)"
            )
        if str(self._arch_dir) not in sys.path:
            sys.path.insert(0, str(self._arch_dir))

        from archs.FIVE_APLUS import FIVE_APLUSNet  # type: ignore

        if not self._weights_path.exists():
            raise FileNotFoundError(f"FA+Net weights not found at {self._weights_path}")

        model = FIVE_APLUSNet().to(device)
        state = torch.load(self._weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        model = model.half()
        force_norm_fp32(model)

        if variant is Variant.STRICT_PARITY:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        self._model = model
        self._variant = variant

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "load() must be called first"
        assert batch.dtype == torch.float16
        _, _, h, w = batch.shape
        assert h % 128 == 0 and w % 128 == 0, (
            f"FA+Net requires H,W divisible by 128, got ({h},{w})"
        )
        with torch.no_grad():
            out = self._model(batch)
        # FA+Net returns (output, stage2_head) — take output only
        if isinstance(out, tuple):
            out = out[0]
        return out.clamp(0.0, 1.0).to(torch.float16)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_fa_net_wrapper.py -v
```

Expected: 1 passed (or skipped if no GPU).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/models/fa_net.py scripts/poc/tests/test_fa_net_wrapper.py
git commit -m "feat(poc): add FA+Net wrapper"
```

---

### Task 6: Shallow-UWnet wrapper

**Files:**
- Create: `repos/dorea/scripts/poc/models/shallow_uwnet.py`
- Create: `repos/dorea/scripts/poc/tests/test_shallow_uwnet_wrapper.py`

**Reference:** `working/sea_thru_poc/run_shallow_uwnet.py`. Key quirks:
- Checkpoint is a **full pickled model**, not a state_dict — must `torch.load(weights_only=False)`
- The unpickler needs `working/sea_thru_poc/models/Shallow-UWnet/` on `sys.path` first
- Input is `[0, 1]` via `ToTensor()`; output clamped to `[0, 1]`
- Original reference runs on CPU; we run on GPU with `.half()`
- No dimension constraint (standard CNN, any size)

- [ ] **Step 1: Write failing smoke test**

Create `repos/dorea/scripts/poc/tests/test_shallow_uwnet_wrapper.py`:

```python
from pathlib import Path
import torch
import pytest

from scripts.poc.models.base import Variant
from scripts.poc.models.shallow_uwnet import ShallowUwnetWrapper


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Shallow-UWnet wrapper requires CUDA",
)

WEIGHTS = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/Shallow-UWnet/snapshots/model.ckpt")
CODE_DIR = Path("/workspaces/dorea-workspace/working/sea_thru_poc/models/Shallow-UWnet")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="Shallow-UWnet weights not present")
def test_shallow_uwnet_wrapper_round_trips():
    w = ShallowUwnetWrapper(weights_path=WEIGHTS, code_dir=CODE_DIR)
    w.load(Variant.BEST_EFFORT, torch.device("cuda"))
    batch = torch.rand((2, 3, 256, 256), dtype=torch.float16, device="cuda")
    out = w.infer(batch)
    assert out.shape == batch.shape
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -0.01 and float(out.max()) <= 1.01
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_shallow_uwnet_wrapper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `models/shallow_uwnet.py`**

Create `repos/dorea/scripts/poc/models/shallow_uwnet.py`:

```python
"""Shallow-UWnet wrapper — fp16, loads a full pickled model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from scripts.poc.models.base import UIEModel, Variant, force_norm_fp32


class ShallowUwnetWrapper(UIEModel):
    name = "shallow_uwnet"

    def __init__(self, weights_path: Path, code_dir: Path) -> None:
        self._weights_path = Path(weights_path)
        self._code_dir = Path(code_dir)
        self._model: torch.nn.Module | None = None

    def preferred_proxy_size(self) -> int:
        return 1024

    def load(self, variant: Variant, device: torch.device) -> None:
        if not (self._code_dir / "model.py").exists() and not (self._code_dir / "training.py").exists():
            raise FileNotFoundError(
                f"Shallow-UWnet code dir invalid: {self._code_dir}"
            )
        if str(self._code_dir) not in sys.path:
            sys.path.insert(0, str(self._code_dir))

        if not self._weights_path.exists():
            raise FileNotFoundError(f"Shallow-UWnet weights not found at {self._weights_path}")

        # Full pickled model — must use weights_only=False
        model = torch.load(self._weights_path, map_location=device, weights_only=False)
        model.eval()
        model = model.to(device)

        model = model.half()
        force_norm_fp32(model)

        if variant is Variant.STRICT_PARITY:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        self._model = model

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "load() must be called first"
        assert batch.dtype == torch.float16
        with torch.no_grad():
            out = self._model(batch)
        return out.clamp(0.0, 1.0).to(torch.float16)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_shallow_uwnet_wrapper.py -v
```

Expected: 1 passed (or skipped).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/models/shallow_uwnet.py scripts/poc/tests/test_shallow_uwnet_wrapper.py
git commit -m "feat(poc): add Shallow-UWnet wrapper (full pickled model)"
```

---

### Task 7: Color-Accurate UIE wrapper (graceful-fail path)

**Files:**
- Create: `repos/dorea/scripts/poc/models/color_accurate.py`
- Create: `repos/dorea/scripts/poc/tests/test_color_accurate_wrapper.py`

**Note:** Color-Accurate UIE (arXiv:2603.16363) code is not yet vendored. This task implements the wrapper *structure* and asserts that `load()` raises `FileNotFoundError` with a clear message when weights are absent — which matches the spec's per-cell graceful-failure path. When upstream code becomes available, Step 3 can be revised to actually load the model; no other file changes will be needed.

- [ ] **Step 1: Write failing test**

Create `repos/dorea/scripts/poc/tests/test_color_accurate_wrapper.py`:

```python
from pathlib import Path
import torch
import pytest

from scripts.poc.models.base import Variant
from scripts.poc.models.color_accurate import ColorAccurateWrapper


def test_missing_weights_raises_clear_error(tmp_path):
    w = ColorAccurateWrapper(weights_path=tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="Color-Accurate UIE weights not found"):
        w.load(Variant.BEST_EFFORT, torch.device("cpu"))


def test_preferred_proxy_size_is_reasonable():
    w = ColorAccurateWrapper(weights_path=Path("/tmp/missing"))
    size = w.preferred_proxy_size()
    assert 128 <= size <= 2048
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_color_accurate_wrapper.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `models/color_accurate.py`**

Create `repos/dorea/scripts/poc/models/color_accurate.py`:

```python
"""Color-Accurate UIE wrapper — paper arXiv:2603.16363.

Upstream code/weights are not yet vendored. When available, drop them under
``working/poc_weights/color_accurate/`` and fill in the architecture load in
``load()``. Until then, ``load()`` fails with a clear message and the POC
bench loop gracefully fails this cell per the spec.
"""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.poc.models.base import UIEModel, Variant, force_norm_fp32


class ColorAccurateWrapper(UIEModel):
    name = "color_accurate"

    def __init__(self, weights_path: Path) -> None:
        self._weights_path = Path(weights_path)
        self._model: torch.nn.Module | None = None

    def preferred_proxy_size(self) -> int:
        # Paper reports 640×480 on Jetson; use 1024 long-edge for 3060 comparison.
        return 1024

    def load(self, variant: Variant, device: torch.device) -> None:
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Color-Accurate UIE weights not found at {self._weights_path}. "
                f"See repos/dorea/scripts/poc/README.md for upstream paper URL. "
                f"This cell will be rendered as FAILED in the contact sheet — "
                f"the rest of the bench is unaffected."
            )
        # When weights become available, construct the model here, load state,
        # .half(), force_norm_fp32, and optionally torch.compile.
        raise NotImplementedError(
            "Color-Accurate UIE architecture not yet vendored. "
            "Fill in this branch once upstream code is available."
        )

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None, "load() must be called first"
        with torch.no_grad():
            out = self._model(batch)
        return out.clamp(0.0, 1.0).to(torch.float16)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_color_accurate_wrapper.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/models/color_accurate.py scripts/poc/tests/test_color_accurate_wrapper.py
git commit -m "feat(poc): add Color-Accurate UIE wrapper stub (graceful-fail path)"
```

---

### Task 8: Shape-contract test across all wrappers

**Files:**
- Create: `repos/dorea/scripts/poc/tests/test_shape_contract.py`

This is an integration test that runs every wrapper through its contract and is the canonical "did I wire this right" check. It should be run every time any wrapper is touched.

- [ ] **Step 1: Write the integration test**

Create `repos/dorea/scripts/poc/tests/test_shape_contract.py`:

```python
"""Shape-contract test — every wrapper round-trips fp16 CUDA tensors correctly.

Uses 256×256 inputs (divisible by 128 for FA+Net). Wrappers whose weights are
absent are skipped. Color-Accurate UIE is expected to be skipped until its
upstream code is vendored.
"""
from pathlib import Path
import torch
import pytest

from scripts.poc.models.base import Variant, UIEModel
from scripts.poc.models.raune import RauneWrapper
from scripts.poc.models.fa_net import FaNetWrapper
from scripts.poc.models.shallow_uwnet import ShallowUwnetWrapper
from scripts.poc.models.color_accurate import ColorAccurateWrapper


WORKING = Path("/workspaces/dorea-workspace/working/sea_thru_poc")
POC_WEIGHTS = Path("/workspaces/dorea-workspace/working/poc_weights")


def _all_wrappers() -> list[tuple[str, UIEModel, Path]]:
    return [
        (
            "raune",
            RauneWrapper(
                weights_path=WORKING / "models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth",
                models_dir=WORKING / "models/RAUNE-Net",
            ),
            WORKING / "models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth",
        ),
        (
            "fa_net",
            FaNetWrapper(
                weights_path=WORKING / "models/FiveAPlus-Network/model/FAPlusNet-alpha-0.4.pth",
                arch_dir=WORKING / "models/FiveAPlus-Network",
            ),
            WORKING / "models/FiveAPlus-Network/model/FAPlusNet-alpha-0.4.pth",
        ),
        (
            "shallow_uwnet",
            ShallowUwnetWrapper(
                weights_path=WORKING / "models/Shallow-UWnet/snapshots/model.ckpt",
                code_dir=WORKING / "models/Shallow-UWnet",
            ),
            WORKING / "models/Shallow-UWnet/snapshots/model.ckpt",
        ),
        (
            "color_accurate",
            ColorAccurateWrapper(
                weights_path=POC_WEIGHTS / "color_accurate/model.pth",
            ),
            POC_WEIGHTS / "color_accurate/model.pth",
        ),
    ]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="shape-contract test requires CUDA",
)


@pytest.mark.parametrize("name,wrapper,weights", _all_wrappers(), ids=lambda x: x[0] if isinstance(x, tuple) else str(x))
@pytest.mark.parametrize("variant", [Variant.STRICT_PARITY, Variant.BEST_EFFORT])
def test_wrapper_shape_contract(name: str, wrapper: UIEModel, weights: Path, variant: Variant):
    if not weights.exists():
        pytest.skip(f"{name} weights not present at {weights}")
    wrapper.load(variant, torch.device("cuda"))
    batch = torch.rand((2, 3, 256, 256), dtype=torch.float16, device="cuda")
    out = wrapper.infer(batch)
    assert out.shape == batch.shape
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    # Output should land in [0, 1] with slack for fp16 rounding
    assert float(out.min()) >= -0.02 and float(out.max()) <= 1.02
```

- [ ] **Step 2: Run the contract test**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/test_shape_contract.py -v
```

Expected: Mix of passed and skipped. `color_accurate/*` should be skipped; the other three should each pass for both variants (8 passed, 2 skipped total if all weights present).

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/tests/test_shape_contract.py
git commit -m "test(poc): add cross-wrapper fp16 shape contract test"
```

---

### Task 9: The bench loop — `uie_bench.py`

**Files:**
- Create: `repos/dorea/scripts/poc/uie_bench.py`

No dedicated pytest for this — it's the integration harness. Task 10 runs it end-to-end against the real clip as the final verification.

- [ ] **Step 1: Implement `uie_bench.py`**

Create `repos/dorea/scripts/poc/uie_bench.py`:

```python
#!/usr/bin/env python3
"""UIE model comparison POC — main entrypoint.

Decodes a dive clip, runs RAUNE-Net + FA+Net + Shallow-UWnet + Color-Accurate
UIE at fp16, records speed/quality metrics, and renders a contact sheet PNG.

Usage:
    python repos/dorea/scripts/poc/uie_bench.py \\
        --input footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4

See docs/decisions/2026-04-10-uie-comparison-poc-design.md for the full spec.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# ensure we can import scripts.poc.* regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]  # repos/dorea
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.poc.metrics import (
    BenchResult,
    compute_ssim,
    compute_uiqm,
    cuda_timer,
    reset_vram_peak,
    vram_peak_mib,
)
from scripts.poc.models.base import UIEModel, Variant
from scripts.poc.models.raune import RauneWrapper
from scripts.poc.models.fa_net import FaNetWrapper
from scripts.poc.models.shallow_uwnet import ShallowUwnetWrapper
from scripts.poc.models.color_accurate import ColorAccurateWrapper
from scripts.poc.render import build_contact_sheet


logger = logging.getLogger("uie_bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

# Fixed order matters for the contact sheet columns.
MODEL_ORDER = ("raune", "fa_net", "shallow_uwnet", "color_accurate")

WORKING = Path("/workspaces/dorea-workspace/working/sea_thru_poc")
POC_WEIGHTS = Path("/workspaces/dorea-workspace/working/poc_weights")


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_wrapper(name: str) -> UIEModel:
    if name == "raune":
        return RauneWrapper(
            weights_path=WORKING / "models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth",
            models_dir=WORKING / "models/RAUNE-Net",
        )
    if name == "fa_net":
        return FaNetWrapper(
            weights_path=WORKING / "models/FiveAPlus-Network/model/FAPlusNet-alpha-0.4.pth",
            arch_dir=WORKING / "models/FiveAPlus-Network",
        )
    if name == "shallow_uwnet":
        return ShallowUwnetWrapper(
            weights_path=WORKING / "models/Shallow-UWnet/snapshots/model.ckpt",
            code_dir=WORKING / "models/Shallow-UWnet",
        )
    if name == "color_accurate":
        return ColorAccurateWrapper(
            weights_path=POC_WEIGHTS / "color_accurate/model.pth",
        )
    raise ValueError(f"unknown model: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Input decode
# ─────────────────────────────────────────────────────────────────────────────

def decode_all_frames(clip_path: Path) -> np.ndarray:
    """Decode every frame from the clip as uint8 HWC RGB via PyAV."""
    import av
    container = av.open(str(clip_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
    container.close()
    if not frames:
        raise RuntimeError(f"Decoded zero frames from {clip_path}")
    logger.info("Decoded %d frames at %dx%d", len(frames), frames[0].shape[1], frames[0].shape[0])
    return np.stack(frames, axis=0)


def sample_contact_sheet_indices(n_total: int, n_rows: int = 8) -> list[int]:
    if n_total < n_rows:
        return list(range(n_total))
    step = n_total // n_rows
    return [i * step for i in range(n_rows)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-model input resize + normalize
# ─────────────────────────────────────────────────────────────────────────────

def resize_frame_for_model(
    frame_hwc_u8: np.ndarray,
    long_edge: int,
    divisor: int,
) -> np.ndarray:
    """Resize so long edge == long_edge, rounded down to a multiple of ``divisor``."""
    from PIL import Image
    h, w = frame_hwc_u8.shape[:2]
    if w >= h:
        new_w = long_edge
        new_h = int(round(h * long_edge / w))
    else:
        new_h = long_edge
        new_w = int(round(w * long_edge / h))
    # Round to multiples of divisor (FA+Net needs 128)
    new_w = max(divisor, (new_w // divisor) * divisor)
    new_h = max(divisor, (new_h // divisor) * divisor)
    return np.array(
        Image.fromarray(frame_hwc_u8).resize((new_w, new_h), Image.Resampling.BILINEAR)
    )


def stack_to_cuda_fp16(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """uint8 HWC → fp16 NCHW [0,1] on ``device``."""
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (N,H,W,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
    return t.to(device=device, dtype=torch.float16, non_blocking=True)


def fp16_tensor_to_uint8_hwc(tensor: torch.Tensor) -> list[np.ndarray]:
    """fp16 NCHW [0,1] CUDA → list of uint8 HWC arrays on CPU."""
    arr = (tensor.clamp(0, 1).float() * 255.0).to(torch.uint8).cpu().numpy()
    return [arr[i].transpose(1, 2, 0) for i in range(arr.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# Bench cell runner
# ─────────────────────────────────────────────────────────────────────────────

def _divisor_for(name: str) -> int:
    return 128 if name == "fa_net" else 8


def _run_warm_bench(
    wrapper: UIEModel,
    device: torch.device,
    long_edge: int,
    divisor: int,
    sample_frame: np.ndarray,
    batch_size: int,
    n_measured_batches: int,
    n_warmup_batches: int,
) -> tuple[float, float, float, float]:
    """Returns (mean_ms, p50_ms, p95_ms, total_warm_time_s)."""
    resized = resize_frame_for_model(sample_frame, long_edge, divisor)
    batch_list = [resized] * batch_size
    batch = stack_to_cuda_fp16(batch_list, device)

    # Warmup
    for _ in range(n_warmup_batches):
        _ = wrapper.infer(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measured
    latencies_ms: list[float] = []
    t0 = None
    for i in range(n_measured_batches):
        with cuda_timer() as t:
            _ = wrapper.infer(batch)
        latencies_ms.append(t.elapsed_ms)
        if i == 0:
            import time as _time
            t0 = _time.perf_counter()
    import time as _time
    total_warm_time_s = (_time.perf_counter() - (t0 or _time.perf_counter())) + latencies_ms[0] / 1000.0

    lat = np.array(latencies_ms)
    return float(lat.mean()), float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), float(total_warm_time_s)


def _run_cell(
    model_name: str,
    variant: Variant,
    all_frames_u8: np.ndarray,
    sheet_indices: list[int],
    raune_baseline_outputs: dict[int, np.ndarray] | None,
    batch_size: int,
    device: torch.device,
) -> tuple[BenchResult, list[np.ndarray] | None]:
    """Run one (model, variant) cell. Returns (result, contact_sheet_outputs)."""
    import time as _time

    # Skip RAUNE best_effort — it's identical to strict_parity
    if model_name == "raune" and variant is Variant.BEST_EFFORT:
        logger.info("[skip] raune / best_effort (same as strict_parity)")
        return (
            BenchResult(
                model=model_name,
                variant=variant.value,
                proxy_size=(0, 0),
                batch_size=batch_size,
                time_to_first_frame_s=0.0,
                latency_mean_ms=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                throughput_fps=0.0,
                vram_peak_mib=0.0,
                uiqm_mean=None,
                ssim_vs_raune_mean=None,
                error="SKIP (same as raune/strict_parity)",
            ),
            None,
        )

    reset_vram_peak()
    wrapper: UIEModel | None = None
    try:
        wrapper = _make_wrapper(model_name)
        long_edge = wrapper.preferred_proxy_size()
        divisor = _divisor_for(model_name)

        # Time-to-first-frame = load + first forward
        tff_start = _time.perf_counter()
        wrapper.load(variant, device)
        sample = all_frames_u8[0]
        resized = resize_frame_for_model(sample, long_edge, divisor)
        first_batch = stack_to_cuda_fp16([resized], device)
        _ = wrapper.infer(first_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_to_first_frame_s = _time.perf_counter() - tff_start

        # Warm bench — 3 warmup batches + 87 measured
        mean_ms, p50_ms, p95_ms, total_warm_s = _run_warm_bench(
            wrapper=wrapper,
            device=device,
            long_edge=long_edge,
            divisor=divisor,
            sample_frame=sample,
            batch_size=batch_size,
            n_measured_batches=87,
            n_warmup_batches=3,
        )
        throughput_fps = (87 * batch_size) / max(total_warm_s, 1e-9)

        # Contact sheet inference (real frames, not repeated)
        sheet_inputs = [
            resize_frame_for_model(all_frames_u8[i], long_edge, divisor)
            for i in sheet_indices
        ]
        sheet_batch = stack_to_cuda_fp16(sheet_inputs, device)
        sheet_out_tensor = wrapper.infer(sheet_batch)
        sheet_outputs_u8 = fp16_tensor_to_uint8_hwc(sheet_out_tensor)

        # Metrics
        uiqm_vals = [compute_uiqm(o) for o in sheet_outputs_u8]
        if raune_baseline_outputs is not None:
            ssim_vals: list[float] = []
            for idx_in_sheet, frame_idx in enumerate(sheet_indices):
                if frame_idx not in raune_baseline_outputs:
                    continue
                ref = raune_baseline_outputs[frame_idx]
                # Resize candidate to match ref shape for SSIM
                from PIL import Image as _PI
                cand = sheet_outputs_u8[idx_in_sheet]
                if cand.shape != ref.shape:
                    cand = np.array(
                        _PI.fromarray(cand).resize((ref.shape[1], ref.shape[0]), _PI.Resampling.LANCZOS)
                    )
                ssim_vals.append(compute_ssim(cand, ref))
            ssim_mean = float(np.mean(ssim_vals)) if ssim_vals else None
        else:
            ssim_mean = 1.0 if model_name == "raune" else None

        vram = vram_peak_mib()

        result = BenchResult(
            model=model_name,
            variant=variant.value,
            proxy_size=(sheet_inputs[0].shape[0], sheet_inputs[0].shape[1]),
            batch_size=batch_size,
            time_to_first_frame_s=time_to_first_frame_s,
            latency_mean_ms=mean_ms,
            latency_p50_ms=p50_ms,
            latency_p95_ms=p95_ms,
            throughput_fps=throughput_fps,
            vram_peak_mib=vram,
            uiqm_mean=float(np.mean(uiqm_vals)),
            ssim_vs_raune_mean=ssim_mean,
            error=None,
        )
        logger.info(
            "[ok] %s / %s  p50=%.1fms  FPS=%.0f  VRAM=%.0f MiB  UIQM=%.2f  SSIM=%.2f",
            model_name, variant.value, p50_ms, throughput_fps, vram,
            result.uiqm_mean or 0.0, result.ssim_vs_raune_mean or 0.0,
        )
        return result, sheet_outputs_u8

    except Exception as e:
        tb = traceback.format_exc()
        logger.warning("[fail] %s / %s: %s", model_name, variant.value, e)
        return (
            BenchResult(
                model=model_name,
                variant=variant.value,
                proxy_size=(0, 0),
                batch_size=batch_size,
                time_to_first_frame_s=0.0,
                latency_mean_ms=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                throughput_fps=0.0,
                vram_peak_mib=vram_peak_mib(),
                uiqm_mean=None,
                ssim_vs_raune_mean=None,
                error=f"{type(e).__name__}: {e}",
            ),
            None,
        )

    finally:
        if wrapper is not None:
            del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# bench.md formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_bench_md(results: list[BenchResult]) -> str:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    ok.sort(key=lambda r: -r.throughput_fps)

    lines = ["# UIE POC Benchmark", "", "## Successful cells (sorted by FPS desc)", ""]
    lines.append("| Model | Variant | TTFF (s) | p50 (ms) | p95 (ms) | FPS | VRAM (MiB) | UIQM | SSIM→RAUNE |")
    lines.append("|-------|---------|---------:|---------:|---------:|----:|-----------:|-----:|-----------:|")
    for r in ok:
        lines.append(
            f"| {r.model} | {r.variant} | {r.time_to_first_frame_s:.1f} | "
            f"{r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} | "
            f"{r.throughput_fps:.0f} | {r.vram_peak_mib:.0f} | "
            f"{r.uiqm_mean:.2f} | "
            f"{r.ssim_vs_raune_mean:.2f}" if r.ssim_vs_raune_mean is not None else "—"
        )
    if failed:
        lines += ["", "## Failed / skipped cells", ""]
        for r in failed:
            lines.append(f"- **{r.model} / {r.variant}** — {r.error}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="UIE model comparison POC")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", default=Path("/workspaces/dorea-workspace/working/poc_out"), type=Path)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--variants", default="strict_parity,best_effort")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tile-size", default="native", choices=["native", "4k"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — POC is GPU-only")
        return 1
    if not args.input.exists():
        logger.error("input clip does not exist: %s", args.input)
        return 1

    device = torch.device("cuda")

    # Decode clip once
    all_frames = decode_all_frames(args.input)
    sheet_indices = sample_contact_sheet_indices(all_frames.shape[0], n_rows=8)
    logger.info("Contact sheet indices: %s", sheet_indices)

    # Build model × variant matrix
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    requested_variants = [Variant(v.strip()) for v in args.variants.split(",") if v.strip()]
    cells: list[tuple[str, Variant]] = [(m, v) for m in requested_models for v in requested_variants]

    results: list[BenchResult] = []
    contact_sheet_outputs: dict[str, list[np.ndarray] | None] = {m: None for m in MODEL_ORDER}
    raune_baseline: dict[int, np.ndarray] | None = None

    for i, (m, v) in enumerate(cells, start=1):
        logger.info("[%d/%d] %s / %s starting…", i, len(cells), m, v.value)
        result, sheet_outputs = _run_cell(
            model_name=m,
            variant=v,
            all_frames_u8=all_frames,
            sheet_indices=sheet_indices,
            raune_baseline_outputs=raune_baseline,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(result)

        # Capture RAUNE baseline outputs for SSIM comparison later
        if m == "raune" and v is Variant.STRICT_PARITY and sheet_outputs is not None:
            raune_baseline = {
                sheet_indices[k]: sheet_outputs[k] for k in range(len(sheet_indices))
            }
            contact_sheet_outputs["raune"] = sheet_outputs

        # For best_effort cells (other than RAUNE), that's what goes on the contact sheet
        if v is Variant.BEST_EFFORT and m != "raune" and sheet_outputs is not None:
            contact_sheet_outputs[m] = sheet_outputs

    # Output dir
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # bench.json
    (out_dir / "bench.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2) + "\n"
    )

    # bench.md
    md = format_bench_md(results)
    (out_dir / "bench.md").write_text(md)
    logger.info("\n%s", md)

    # Contact sheet — assemble the best-effort (or RAUNE strict) outputs per model
    # For any missing model, build an error placeholder by copying the relevant result
    per_model_results: dict[str, BenchResult] = {}
    for m in MODEL_ORDER:
        chosen: BenchResult | None = None
        for r in results:
            if r.model != m:
                continue
            if m == "raune" and r.variant == "strict_parity":
                chosen = r
                break
            if m != "raune" and r.variant == "best_effort":
                chosen = r
                break
        if chosen is None:
            chosen = BenchResult(
                model=m, variant="best_effort",
                proxy_size=(0, 0), batch_size=args.batch_size,
                time_to_first_frame_s=0.0, latency_mean_ms=0.0,
                latency_p50_ms=0.0, latency_p95_ms=0.0,
                throughput_fps=0.0, vram_peak_mib=0.0,
                uiqm_mean=None, ssim_vs_raune_mean=None,
                error="cell not run",
            )
        per_model_results[m] = chosen

    # For missing sheet outputs, build dummy frames matching original count
    originals = [all_frames[i] for i in sheet_indices]
    model_outputs_for_render: dict[str, list[np.ndarray]] = {}
    for m in MODEL_ORDER:
        if contact_sheet_outputs.get(m) is not None:
            model_outputs_for_render[m] = contact_sheet_outputs[m]  # type: ignore[assignment]
        else:
            # Error path — build_contact_sheet will overwrite these with red tiles
            model_outputs_for_render[m] = [np.zeros((64, 64, 3), dtype=np.uint8)] * len(originals)

    build_contact_sheet(
        original_frames=originals,
        model_outputs=model_outputs_for_render,
        bench_results=per_model_results,
        tile_size=args.tile_size,
        out_path=out_dir / "contact_sheet.png",
    )

    logger.info("Contact sheet: %s", out_dir / "contact_sheet.png")
    logger.info("bench.md:      %s", out_dir / "bench.md")
    logger.info("bench.json:    %s", out_dir / "bench.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax sanity check**

```bash
cd /workspaces/dorea-workspace
/opt/dorea-venv/bin/python -c "import ast; ast.parse(open('repos/dorea/scripts/poc/uie_bench.py').read()); print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add scripts/poc/uie_bench.py
git commit -m "feat(poc): add uie_bench.py orchestrator"
```

---

### Task 10: End-to-end smoke run

This is the final verification: run the whole pipeline against the canonical dive clip and visually inspect the output. Any failure surfaced here either produces a real answer (a model is worse/slower) or exposes a POC bug worth fixing.

- [ ] **Step 1: Run full bench**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python repos/dorea/scripts/poc/uie_bench.py \
    --input footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 2>&1 | tee /tmp/uie_bench_run.log
```

Expected outcomes:
- Decode reports ~360 frames at 3840×2160
- 8 cells attempted (4 models × 2 variants)
- RAUNE strict_parity: completes, reports FPS in the 40-100 range, VRAM ~1.5-2.5 GiB
- RAUNE best_effort: SKIP (by design)
- FA+Net strict_parity + best_effort: both complete, FPS in the 200-600 range, VRAM < 1 GiB
- Shallow-UWnet strict_parity + best_effort: both complete, FPS in the 100-400 range, VRAM < 1 GiB
- Color-Accurate UIE strict_parity + best_effort: both FAIL with "Color-Accurate UIE weights not found at …" — this is expected
- Final markdown table printed to stdout
- Contact sheet PNG path printed

If any **expected** cell fails (e.g. FA+Net strict_parity errors out), inspect the log under the run's `logs/` directory and fix the root cause — do not apply the graceful-failure path as a shortcut. The graceful-failure path is a safety net for genuinely missing assets, not a way to paper over bugs.

- [ ] **Step 2: Inspect contact sheet**

```bash
ls -lh /workspaces/dorea-workspace/working/poc_out/*/contact_sheet.png | tail
```

Expected:
- Exactly one new directory under `working/poc_out/` with a recent timestamp
- `contact_sheet.png` present and non-empty (30–100 MB range for native tile size)
- `bench.md` present
- `bench.json` present

Open the PNG in a viewer and confirm:
- 8 rows × 5 columns (original + 4 models)
- Original column shows real dive footage, not garbage
- RAUNE column shows plausible colour-corrected output
- FA+Net and Shallow-UWnet columns show different-looking but plausible enhancements
- Color-Accurate UIE column is solid dark red with "FAILED: weights not found" text (expected)
- Each enhanced tile has a small metrics overlay in the bottom-right
- Column headers readable, row labels readable

- [ ] **Step 3: Run the full test suite once**

```bash
cd /workspaces/dorea-workspace
PYTHONPATH=repos/dorea /opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/ -v
```

Expected: all non-skipped tests pass. Color-Accurate wrapper tests pass (they test the failure path). Shape contract for color_accurate is skipped.

- [ ] **Step 4: Commit run artifacts reference (not the artifacts themselves — they go under `working/` which is gitignored)**

This is a documentation-only commit noting the run happened. Nothing to stage in the main tree — `working/poc_out/` is not tracked. Write a short note under `docs/decisions/` referencing the first run:

```bash
cd /workspaces/dorea-workspace
cat > docs/decisions/2026-04-10-uie-poc-first-run.md <<'EOF'
# UIE POC first run

**Date:** 2026-04-10

Initial end-to-end smoke run of `repos/dorea/scripts/poc/uie_bench.py` against
`footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4`.

Per-model results are in `working/poc_out/<timestamp>/bench.md` and
`contact_sheet.png`. See design spec
`docs/decisions/2026-04-10-uie-comparison-poc-design.md` and impl plan
`docs/plans/2026-04-10-uie-comparison-poc-impl.md`.

The Color-Accurate UIE cell failed gracefully (weights not yet vendored); the
other three models populated successfully. Next step: visual review and
decision on whether any candidate is worth pursuing for multi-instance
parallelism.
EOF
git add docs/decisions/2026-04-10-uie-poc-first-run.md
git commit -m "docs: note initial UIE POC smoke run"
```

- [ ] **Step 5: Record the decision in corvia**

Use `corvia_write` with `source_origin="workspace"` to save the comparison summary and link to the run artifacts. This lets future sessions find the numbers without re-running the bench.

```bash
# This step is a prompt to the executor, not a literal shell command.
# Invoke the corvia_write MCP tool with:
#   scope_id="dorea"
#   source_origin="workspace"
#   content_role="decision"
#   title="UIE POC — RAUNE vs FA+Net vs Shallow-UWnet vs Color-Accurate"
#   body=<markdown summary of bench.md + contact sheet observations + recommendation>
```

---

## Self-review

**Spec coverage check:**
- Section 1 (architecture & file layout) → Task 0 (scaffold), Task 9 (uie_bench.py location) ✓
- Section 2 (model interface + variants) → Task 1 (base.py), Tasks 4–7 (per-model wrappers) ✓
- Section 3 (bench loop + metrics) → Task 2 (metrics.py), Task 9 (bench loop) ✓
- Section 4 (contact sheet render) → Task 3 (render.py) ✓
- Section 5 (error handling, testing, run UX) → cell-level try/except in Task 9, Tasks 2/3/8 tests, Task 0 README ✓
- bench.md format → Task 9's `format_bench_md` ✓
- fp16 + force_norm_fp32 + torch.compile(reduce-overhead) → Task 1 helper + Tasks 4–7 wrappers ✓
- Per-cell graceful failure → Task 9 `_run_cell` try/except + Task 7 Color-Accurate stub ✓
- Shape contract test → Task 8 ✓

**Placeholder scan:** No "TBD"/"TODO"/"fill in later" in any task. Task 7 (Color-Accurate) has an *explicit* `NotImplementedError` that is tested for — that's the design, not a placeholder.

**Type / name consistency:**
- `BenchResult` fields used consistently across Task 2 (definition), Task 3 (render consumes `.error`, `.latency_p50_ms`, `.uiqm_mean`), Task 9 (produces BenchResult). All field names match.
- `UIEModel` methods: `load(variant, device)`, `infer(batch)`, `preferred_proxy_size()`. Used consistently in all four wrappers and in the bench loop.
- `Variant` enum: `STRICT_PARITY`, `BEST_EFFORT`. Used consistently.
- `force_norm_fp32` defined in Task 1 and imported in Tasks 4, 5, 6, 7.
- `MODEL_ORDER` in Task 9 matches `_COLUMN_ORDER` in Task 3 (`("original", "raune", "fa_net", "shallow_uwnet", "color_accurate")`).
- Wrapper names (`RauneWrapper`, `FaNetWrapper`, `ShallowUwnetWrapper`, `ColorAccurateWrapper`) consistent across tests and bench loop.

**Scope check:** Plan targets one deliverable (the POC script + one smoke run). Not decomposable further without sacrificing cohesion.

**Arithmetic check:** 3 warmup + 87 measured = 90 batches × 4 = 360 frames. ✓

No issues found. Plan ready for execution.
