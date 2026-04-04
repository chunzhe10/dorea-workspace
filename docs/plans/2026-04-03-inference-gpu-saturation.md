# GPU Inference Saturation — Fused RAUNE+Depth Batch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate GPU idle valleys during `auto_calibrate` by (1) fusing RAUNE and depth inference into a single IPC call so the enhanced tensor never leaves the GPU between models, and (2) enabling depth estimation on RAUNE-enhanced frames for better zone-boundary quality.

**Architecture:** One new fused IPC message `raune_depth_batch`. Python receives original frames, runs RAUNE → keeps enhanced tensors on GPU → GPU-resize + re-normalise → Depth Anything → dtoh both results together. Rust calls this once per chunk instead of two separate batch calls. Replaces the two-phase (all RAUNE, then all depth) approach in the previous plan with a single fused phase.

**Batching clarification:** "Batch" here means N frames stacked into one `(N, 3, H, W)` tensor passed through a single forward pass — GPU tensor-core parallelism, not multiple processes or threads. `N=32` keeps the GPU saturated; `N=1` leaves ~80% of CUDA cores idle.

**Tech Stack:** Python (torch, torch.nn.functional, numpy, PIL, pytest), Rust (serde_json, base64), existing JSON-lines IPC protocol.

---

## Data flow: before vs. after

**Before (two separate batch calls, depth on original):**
```
Rust ──original_px──► IPC ──► RAUNE ──► IPC ──► Rust (enhanced_px stored)
Rust ──original_px──► IPC ──► Depth ──► IPC ──► Rust (depths stored)
         ^── 4 IPC data crossings, enhanced_px goes GPU→CPU→pipe→CPU→GPU for nothing
```

**After (fused call, depth on RAUNE output):**
```
Rust ──original_px──► IPC ──► RAUNE ──resize+renorm(GPU)──► Depth ──► IPC ──► Rust
         ^── 2 IPC data crossings, enhanced tensor stays on GPU the whole time
```

Saved per 8-frame chunk: one IPC round-trip (~30–50 ms), one dtoh of enhanced frames, one htod of enhanced frames. Better depth maps as a bonus.

---

## Resize and renormalise between models

RAUNE outputs at `max_size=1024` (no patch constraint). Depth Anything V2 requires dimensions ≤ 518 that are **multiples of 14** (ViT patch size). Between the two models, on GPU:

```python
import torch.nn.functional as F

# RAUNE output: enhanced (N, 3, H_r, W_r) in [0, 1]
scale = 518 / max(H_r, W_r)
H_d = max(14, int(H_r * scale) // 14 * 14)
W_d = max(14, int(W_r * scale) // 14 * 14)

resized = F.interpolate(enhanced, size=(H_d, W_d), mode='bilinear', align_corners=False)

# Re-normalise [0,1] → ImageNet stats that Depth Anything expects
MEAN = torch.tensor([0.485, 0.456, 0.406], device=enhanced.device).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=enhanced.device).view(1, 3, 1, 1)
depth_input = (resized - MEAN) / STD   # (N, 3, H_d, W_d)
```

---

## File Map

| File | Change |
|------|--------|
| `repos/dorea/python/dorea_inference/raune_net.py` | Add `infer_batch` (building block) |
| `repos/dorea/python/dorea_inference/depth_anything.py` | Add `infer_batch_from_tensors`; pinned memory in `infer_batch` |
| `repos/dorea/python/dorea_inference/protocol.py` | Add `RauneDepthBatchResult` |
| `repos/dorea/python/dorea_inference/server.py` | Add `raune_depth_batch` handler |
| `repos/dorea/crates/dorea-video/src/inference_subprocess.rs` | Add `RauneDepthBatchItem`, `run_raune_depth_batch` |
| `repos/dorea/crates/dorea-cli/src/grade.rs` | Refactor `auto_calibrate` to single fused phase |
| `repos/dorea/python/tests/test_infer_batch.py` | New — pytest unit tests (CPU, no weights needed) |

---

## Constants

- `FUSED_BATCH_SIZE = 8` — calibration images ~1024×577; 8 frames × ~57 MB GPU input, safe under 6 GB with both models loaded.
- `DEPTH_BATCH_SIZE = 32` — unchanged in `grade.rs`, used by the main grading pass.

---

## Task 1: Add `infer_batch` to `RauneNetInference`

This is the building block. The fused handler in Task 3 calls a GPU-tensor variant, but `infer_batch` (returns numpy) is still useful standalone and makes the fused path testable in pieces.

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_net.py`
- Create: `repos/dorea/python/tests/__init__.py` (empty)
- Create: `repos/dorea/python/tests/test_infer_batch.py`

- [ ] **Step 1.1: Write failing tests**

Create `repos/dorea/python/tests/__init__.py` — empty file.

Create `repos/dorea/python/tests/test_infer_batch.py`:

```python
"""Tests for infer_batch methods — CPU mode, no GPU or real weights required."""
import numpy as np
import pytest
import torch


def _make_rgb(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _fake_raune_model():
    """Identity RauneNetInference on CPU, no weights file needed."""
    import types, sys
    fake_mod = types.ModuleType("models")
    fake_mod.raune_net = types.ModuleType("models.raune_net")

    class _IdentityNet(torch.nn.Module):
        def forward(self, x):
            return x  # identity; output in same range as input ([-1,1] after normalize)

    fake_mod.raune_net.RauneNet = lambda **_: _IdentityNet()
    sys.modules.setdefault("models", fake_mod)
    sys.modules.setdefault("models.raune_net", fake_mod.raune_net)

    from dorea_inference.raune_net import RauneNetInference
    m = RauneNetInference.__new__(RauneNetInference)
    m.device = torch.device("cpu")
    m.model = _IdentityNet()
    m.model.eval()
    return m


class TestRauneInferBatch:
    def test_returns_same_count(self):
        model = _fake_raune_model()
        imgs = [_make_rgb(60, 80) for _ in range(5)]
        results = model.infer_batch(imgs, max_size=80)
        assert len(results) == 5

    def test_output_dtype_and_channels(self):
        model = _fake_raune_model()
        results = model.infer_batch([_make_rgb(60, 80)], max_size=80)
        assert results[0].dtype == np.uint8
        assert results[0].ndim == 3
        assert results[0].shape[2] == 3

    def test_empty_returns_empty(self):
        model = _fake_raune_model()
        assert model.infer_batch([], max_size=80) == []

    def test_mixed_dims_falls_back_to_sequential(self):
        model = _fake_raune_model()
        imgs = [_make_rgb(60, 80), _make_rgb(40, 60)]  # different sizes
        results = model.infer_batch(imgs, max_size=160)
        assert len(results) == 2
```

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py::TestRauneInferBatch -v 2>&1 | head -20
```
Expected: `FAILED` — `AttributeError: 'RauneNetInference' object has no attribute 'infer_batch'`

- [ ] **Step 1.2: Add `infer_batch` to `raune_net.py`**

Append after `infer_gpu` (after line 135 of `repos/dorea/python/dorea_inference/raune_net.py`):

```python
    def infer_batch(self, imgs: "list[np.ndarray]", max_size: int = 1024) -> "list[np.ndarray]":
        """Run RAUNE-Net on a batch of uint8 HxWx3 RGB images.

        All images must resize to the same dims (guaranteed for frames from one video).
        Falls back to sequential infer() if post-resize dims differ.
        Returns list of uint8 HxWx3 arrays.
        """
        if not imgs:
            return []

        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensors = []
        wh_set = set()
        for img in imgs:
            r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
            wh_set.add((tw, th))
            tensors.append(normalize(transforms.ToTensor()(r)))

        if len(wh_set) > 1:
            import sys
            print(
                f"[dorea_inference] WARNING: infer_batch(raune): mixed post-resize dims "
                f"{wh_set}; falling back to sequential",
                file=sys.stderr,
            )
            return [self.infer(img, max_size) for img in imgs]

        batch = torch.stack(tensors)  # (N, 3, H, W)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)  # (N, 3, H, W) in [-1, 1]

        out = ((out + 1.0) / 2.0).clamp(0.0, 1.0)
        out_np = out.cpu().numpy()  # (N, 3, H, W) float32
        return [(out_np[i].transpose(1, 2, 0) * 255).astype(np.uint8) for i in range(len(imgs))]

    def infer_batch_gpu(self, imgs: "list[np.ndarray]", max_size: int = 1024) -> "tuple[torch.Tensor, int, int]":
        """Run RAUNE-Net batch, returning enhanced tensors on device (no dtoh).

        Returns (batch_tensor, out_w, out_h) where batch_tensor is (N, 3, H, W)
        float32 in [0, 1], still on self.device. Caller must not let it be GC'd.
        Falls back to stacking infer() results if dims differ.
        """
        if not imgs:
            import torch
            return torch.zeros(0, 3, 1, 1, device=self.device), 0, 0

        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensors = []
        wh_set = set()
        out_w, out_h = 0, 0
        for img in imgs:
            r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
            wh_set.add((tw, th))
            out_w, out_h = tw, th
            tensors.append(normalize(transforms.ToTensor()(r)))

        if len(wh_set) > 1:
            # Different sizes: run sequentially, collect results as GPU tensors
            results = []
            for img in imgs:
                r, tw, th = _resize_maintain_aspect(_Image.fromarray(img), max_size)
                t = normalize(transforms.ToTensor()(r)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(t)
                out = ((out.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
                results.append(out)
            stacked = torch.stack(results)
            _, _, H, W = stacked.shape
            return stacked, W, H

        batch = torch.stack(tensors)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)  # (N, 3, H, W) in [-1, 1]

        out = ((out + 1.0) / 2.0).clamp(0.0, 1.0)  # [0, 1], still on device
        return out, out_w, out_h
```

- [ ] **Step 1.3: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py::TestRauneInferBatch -v
```
Expected: 4 tests PASS.

- [ ] **Step 1.4: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/raune_net.py python/tests/__init__.py python/tests/test_infer_batch.py
git commit -m "feat(inference): add RauneNetInference.infer_batch + infer_batch_gpu"
```

---

## Task 2: Add `infer_batch_from_tensors` to `DepthAnythingInference` + pinned memory

The fused server handler needs depth to accept GPU tensors directly (skipping PIL → numpy → htod). Also add pinned memory to the existing `infer_batch`.

**Files:**
- Modify: `repos/dorea/python/dorea_inference/depth_anything.py`
- Modify: `repos/dorea/python/tests/test_infer_batch.py`

- [ ] **Step 2.1: Write failing tests**

Append to `repos/dorea/python/tests/test_infer_batch.py`:

```python
class TestDepthInferBatchFromTensors:
    def test_accepts_gpu_tensor_shape(self):
        """infer_batch_from_tensors returns one depth map per input tensor."""
        from dorea_inference.depth_anything import DepthAnythingInference

        class _FakeDepthModel(torch.nn.Module):
            class _Out:
                def __init__(self, t): self.predicted_depth = t
            def forward(self, pixel_values=None):
                N, _, H, W = pixel_values.shape
                return self._Out(pixel_values[:, 0, :, :])  # (N, H, W) dummy

        model = DepthAnythingInference.__new__(DepthAnythingInference)
        model.device = torch.device("cpu")
        model.model = _FakeDepthModel()

        # Simulate RAUNE output: (3, 3, H, W) float32 in [0, 1]
        fake_enhanced = torch.rand(3, 3, 56, 84)  # 3 frames, not yet patch-aligned
        depths = model.infer_batch_from_tensors(fake_enhanced, depth_max_size=56)
        assert len(depths) == 3
        assert all(isinstance(d, np.ndarray) for d in depths)
        assert all(d.dtype == np.float32 for d in depths)

    def test_output_dims_are_patch_aligned(self):
        """Output depth map dimensions are multiples of 14."""
        from dorea_inference.depth_anything import DepthAnythingInference

        class _FakeDepthModel(torch.nn.Module):
            class _Out:
                def __init__(self, t): self.predicted_depth = t
            def forward(self, pixel_values=None):
                N, _, H, W = pixel_values.shape
                return self._Out(pixel_values[:, 0, :, :])

        model = DepthAnythingInference.__new__(DepthAnythingInference)
        model.device = torch.device("cpu")
        model.model = _FakeDepthModel()

        fake_enhanced = torch.rand(1, 3, 100, 180)
        depths = model.infer_batch_from_tensors(fake_enhanced, depth_max_size=98)
        H, W = depths[0].shape
        assert H % 14 == 0, f"height {H} not multiple of 14"
        assert W % 14 == 0, f"width {W} not multiple of 14"
```

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py::TestDepthInferBatchFromTensors -v 2>&1 | head -20
```
Expected: `FAILED` — `AttributeError: 'DepthAnythingInference' object has no attribute 'infer_batch_from_tensors'`

- [ ] **Step 2.2: Add `infer_batch_from_tensors` and update pinned memory in `depth_anything.py`**

**First**, update the `infer_batch` htod line (line 146 of `repos/dorea/python/dorea_inference/depth_anything.py`):

```python
        # Before:
        batch = torch.stack(arrays).to(self.device)  # (N, 3, H, W)

        # After:
        batch = torch.stack(arrays)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)
```

**Then**, append after `infer_gpu` (end of the class):

```python
    def infer_batch_from_tensors(
        self,
        enhanced: "torch.Tensor",
        depth_max_size: int = 518,
    ) -> "list[np.ndarray]":
        """Run depth estimation on a batch of RAUNE-output GPU tensors.

        Args:
            enhanced: (N, 3, H_r, W_r) float32 in [0, 1], on self.device.
                      This is the direct output of RauneNetInference.infer_batch_gpu().
            depth_max_size: max long-edge for depth model (default 518).

        Returns list of (H_d, W_d) float32 depth maps normalized to [0, 1].
        The resize and re-normalisation happen on-device — no dtoh between models.
        """
        import torch
        import torch.nn.functional as F

        N, _, H_r, W_r = enhanced.shape

        # GPU resize: keep aspect, snap to multiples of _PATCH_SIZE
        scale = depth_max_size / max(H_r, W_r)
        H_d = max(_PATCH_SIZE, int(H_r * scale) // _PATCH_SIZE * _PATCH_SIZE)
        W_d = max(_PATCH_SIZE, int(W_r * scale) // _PATCH_SIZE * _PATCH_SIZE)

        resized = F.interpolate(
            enhanced, size=(H_d, W_d), mode="bilinear", align_corners=False
        )  # (N, 3, H_d, W_d) in [0, 1]

        # Re-normalise [0,1] → ImageNet stats Depth Anything expects
        mean = torch.tensor(self._MEAN, dtype=torch.float32, device=enhanced.device).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD,  dtype=torch.float32, device=enhanced.device).view(1, 3, 1, 1)
        depth_input = (resized - mean) / std  # (N, 3, H_d, W_d)

        with torch.no_grad():
            outputs = self.model(pixel_values=depth_input)

        depths_raw = outputs.predicted_depth.cpu().numpy()  # (N, H_d, W_d)

        result = []
        for i in range(N):
            d = depths_raw[i]
            d_min, d_max = float(d.min()), float(d.max())
            if d_max - d_min < 1e-6:
                result.append(np.zeros_like(d, dtype=np.float32))
            else:
                result.append(((d - d_min) / (d_max - d_min)).astype(np.float32))
        return result
```

- [ ] **Step 2.3: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py -v
```
Expected: all tests PASS.

- [ ] **Step 2.4: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/depth_anything.py python/tests/test_infer_batch.py
git commit -m "feat(inference): add infer_batch_from_tensors for GPU-tensor depth; pinned memory in infer_batch"
```

---

## Task 3: Add `RauneDepthBatchResult` to protocol + fused server handler

**Files:**
- Modify: `repos/dorea/python/dorea_inference/protocol.py`
- Modify: `repos/dorea/python/dorea_inference/server.py`
- Modify: `repos/dorea/python/tests/test_infer_batch.py`

- [ ] **Step 3.1: Write failing test**

Append to `repos/dorea/python/tests/test_infer_batch.py`:

```python
class TestRauneDepthBatchProtocol:
    def test_raune_depth_batch_result_roundtrip(self):
        """RauneDepthBatchResult serialises/deserialises correctly."""
        from dorea_inference.protocol import RauneDepthBatchResult, DepthResult, RauneResult, encode_png
        import json

        dummy_img = np.zeros((14, 14, 3), dtype=np.uint8)
        dummy_depth = np.zeros((14, 14), dtype=np.float32)

        item = {
            "id": "kf0000",
            "image_b64": encode_png(dummy_img),
            "enhanced_width": 14,
            "enhanced_height": 14,
            **DepthResult.from_array("kf0000", dummy_depth).to_dict(),
        }
        resp = RauneDepthBatchResult(results=[item])
        d = resp.to_dict()
        assert d["type"] == "raune_depth_batch_result"
        assert len(d["results"]) == 1
        assert d["results"][0]["id"] == "kf0000"
        # Round-trip through JSON
        assert json.loads(json.dumps(d))["type"] == "raune_depth_batch_result"
```

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py::TestRauneDepthBatchProtocol -v 2>&1 | head -10
```
Expected: `FAILED` — `ImportError: cannot import name 'RauneDepthBatchResult'`

- [ ] **Step 3.2: Add `RauneDepthBatchResult` to `protocol.py`**

In `repos/dorea/python/dorea_inference/protocol.py`, add after `DepthBatchResult` (after line 122):

```python
@dataclass
class RauneDepthBatchResult:
    """Fused RAUNE+depth batch response.

    Each result dict contains:
      id, image_b64 (PNG, enhanced frame), enhanced_width, enhanced_height,
      depth_f32_b64, depth_width, depth_height, type="depth_result"
    """
    results: list
    type: str = "raune_depth_batch_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "results": self.results}
```

- [ ] **Step 3.3: Add `raune_depth_batch` handler to `server.py`**

Update the import line in `server.py` (line 28):

```python
from .protocol import (
    PongResponse,
    RauneResult,
    DepthResult,
    DepthBatchResult,
    RauneDepthBatchResult,
    ErrorResponse,
    OkResponse,
    decode_png,
    decode_raw_rgb,
    encode_png,
)
```

Add the handler after `depth_batch` and before `shutdown` in `server.py`:

```python
            elif req_type == "raune_depth_batch":
                if raune_model is None:
                    raise RuntimeError("RAUNE-Net not loaded — pass --raune-weights")
                if depth_model is None:
                    raise RuntimeError("Depth Anything not loaded — pass --depth-model")

                items = req.get("items", [])
                imgs = []
                for item in items:
                    fmt = item.get("format", "png")
                    if fmt == "raw_rgb":
                        imgs.append(decode_raw_rgb(item["image_b64"], int(item["width"]), int(item["height"])))
                    else:
                        imgs.append(decode_png(item["image_b64"]))

                raune_max = int(items[0].get("raune_max_size", 1024)) if items else 1024
                depth_max = int(items[0].get("depth_max_size", 518)) if items else 518

                # RAUNE → enhanced tensors stay on GPU
                enhanced_batch, enh_w, enh_h = raune_model.infer_batch_gpu(imgs, max_size=raune_max)

                # Depth on enhanced tensors — no dtoh between models
                depth_maps = depth_model.infer_batch_from_tensors(enhanced_batch, depth_max_size=depth_max)

                # dtoh enhanced frames for output
                enhanced_np = (
                    enhanced_batch.permute(0, 2, 3, 1).cpu().numpy() * 255
                ).astype("uint8")  # (N, H, W, 3)

                results = []
                for i, (item, depth) in enumerate(zip(items, depth_maps)):
                    depth_result = DepthResult.from_array(item.get("id"), depth)
                    results.append({
                        "id": item.get("id"),
                        "image_b64": encode_png(enhanced_np[i]),
                        "enhanced_width": int(enh_w),
                        "enhanced_height": int(enh_h),
                        **{k: v for k, v in depth_result.to_dict().items()
                           if k in ("depth_f32_b64", "depth_width", "depth_height", "type")},
                    })
                resp = RauneDepthBatchResult(results=results)
```

- [ ] **Step 3.4: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea/python
/opt/dorea-venv/bin/pytest tests/test_infer_batch.py -v
```
Expected: all tests PASS.

- [ ] **Step 3.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/protocol.py python/dorea_inference/server.py python/tests/test_infer_batch.py
git commit -m "feat(inference): add raune_depth_batch fused IPC handler — enhanced tensor stays on GPU"
```

---

## Task 4: Add `run_raune_depth_batch` to Rust IPC client

**Files:**
- Modify: `repos/dorea/crates/dorea-video/src/inference_subprocess.rs`

- [ ] **Step 4.1: Write failing compile test**

In the `#[cfg(test)]` module of `inference_subprocess.rs`, add:

```rust
#[test]
fn raune_depth_batch_item_fields_compile() {
    let item = RauneDepthBatchItem {
        id: "kf0000".to_string(),
        pixels: vec![0u8; 60 * 80 * 3],
        width: 80,
        height: 60,
        raune_max_size: 80,
        depth_max_size: 56,
    };
    assert_eq!(item.id, "kf0000");
}
```

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video raune_depth_batch_item_fields_compile 2>&1 | tail -5
```
Expected: `error[E0412]: cannot find type 'RauneDepthBatchItem'`

- [ ] **Step 4.2: Add `RauneDepthBatchItem` and `run_raune_depth_batch`**

After `DepthBatchItem` (around line 74 of `inference_subprocess.rs`):

```rust
/// A single image for the fused raune_depth_batch IPC call.
pub struct RauneDepthBatchItem {
    pub id: String,
    /// Raw RGB24 pixels, row-major.
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Max long-edge for RAUNE resize (pixels).
    pub raune_max_size: usize,
    /// Max long-edge for depth resize (pixels, must be ≤518 for Depth Anything).
    pub depth_max_size: usize,
}
```

After `run_depth_batch`:

```rust
/// Run RAUNE then Depth Anything on a batch, with the enhanced tensor staying on GPU
/// between the two models. Returns enhanced RGB and depth map for each item.
///
/// Returns `Vec<(id, enhanced_rgb_u8, enh_w, enh_h, depth_f32, depth_w, depth_h)>`.
#[allow(clippy::type_complexity)]
pub fn run_raune_depth_batch(
    &mut self,
    items: &[RauneDepthBatchItem],
) -> Result<Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>, InferenceError> {
    if items.is_empty() {
        return Ok(vec![]);
    }

    let json_items: Vec<serde_json::Value> = items.iter().map(|item| {
        serde_json::json!({
            "id": item.id,
            "image_b64": B64.encode(&item.pixels),
            "format": "raw_rgb",
            "width": item.width,
            "height": item.height,
            "raune_max_size": item.raune_max_size,
            "depth_max_size": item.depth_max_size,
        })
    }).collect();

    let req = serde_json::json!({ "type": "raune_depth_batch", "items": json_items });
    self.send_line(&req.to_string())?;

    let resp = self.recv_line()?;
    let v: serde_json::Value = serde_json::from_str(&resp)
        .map_err(|e| InferenceError::Ipc(format!("raune_depth_batch parse: {e}")))?;

    if v["type"].as_str() == Some("error") {
        return Err(InferenceError::ServerError(
            v["message"].as_str().unwrap_or("unknown").to_string(),
        ));
    }
    if v["type"].as_str() != Some("raune_depth_batch_result") {
        return Err(InferenceError::Ipc(format!("unexpected type: {resp}")));
    }

    let results = v["results"].as_array()
        .ok_or_else(|| InferenceError::Ipc("missing results array".to_string()))?;

    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let id = r["id"].as_str().unwrap_or("").to_string();

        // Enhanced image (PNG-encoded)
        let enh_w = r["enhanced_width"].as_u64().unwrap_or(0) as usize;
        let enh_h = r["enhanced_height"].as_u64().unwrap_or(0) as usize;
        let enh_b64 = r["image_b64"].as_str()
            .ok_or_else(|| InferenceError::Ipc(format!("missing image_b64 for {id}")))?;
        let enh_png = B64.decode(enh_b64)
            .map_err(|e| InferenceError::Ipc(format!("base64 enh for {id}: {e}")))?;
        let enh_rgb = decode_png_bytes(&enh_png)?;

        // Depth map (raw f32 LE)
        let depth_w = r["depth_width"].as_u64().unwrap_or(0) as usize;
        let depth_h = r["depth_height"].as_u64().unwrap_or(0) as usize;
        let depth_b64 = r["depth_f32_b64"].as_str()
            .ok_or_else(|| InferenceError::Ipc(format!("missing depth_f32_b64 for {id}")))?;
        let raw = B64.decode(depth_b64)
            .map_err(|e| InferenceError::Ipc(format!("base64 depth for {id}: {e}")))?;
        if raw.len() != depth_w * depth_h * 4 {
            return Err(InferenceError::Ipc(format!(
                "depth size mismatch for {id}: got {} want {}",
                raw.len(), depth_w * depth_h * 4
            )));
        }
        let depth: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        out.push((id, enh_rgb, enh_w, enh_h, depth, depth_w, depth_h));
    }
    Ok(out)
}
```

- [ ] **Step 4.3: Run compile test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video raune_depth_batch_item_fields_compile 2>&1 | tail -5
```
Expected: `test raune_depth_batch_item_fields_compile ... ok`

- [ ] **Step 4.4: Full check**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-video 2>&1 | tail -5
```
Expected: `Finished` with no errors.

- [ ] **Step 4.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference_subprocess.rs
git commit -m "feat(inference): add RauneDepthBatchItem + run_raune_depth_batch fused IPC method"
```

---

## Task 5: Refactor `auto_calibrate` to use fused batch call

**Files:**
- Modify: `repos/dorea/crates/dorea-cli/src/grade.rs`

- [ ] **Step 5.1: Update import and add constant**

In `grade.rs`, update the inference import line:

```rust
use dorea_video::inference::{DepthBatchItem, RauneDepthBatchItem, InferenceConfig, InferenceServer};
```

After `DEPTH_BATCH_SIZE` (line 116), add:

```rust
/// Maximum frames per fused RAUNE+depth inference batch.
/// Calibration images are ~1024×577; 8 frames ≈ 57 MB RAUNE input on GPU.
const FUSED_BATCH_SIZE: usize = 8;
```

- [ ] **Step 5.2: Replace the `auto_calibrate` inference loop**

Find the loop starting at line 549 (`for i in 0..n_kf {`) through `let _ = inf_server.shutdown();` and replace with:

```rust
    // --- Phase 1: collect pixel buffers (N individual ffmpeg seeks — unchanged) ---
    log::info!("Auto-calibration: extracting {n_kf} keyframes...");
    let mut kf_pixels: Vec<Vec<u8>> = Vec::with_capacity(n_kf);
    for i in 0..n_kf {
        let ts = (i as f64 + 0.5) * safe_duration / n_kf as f64;
        let pixels = ffmpeg::extract_frame_at(&args.input, ts, kf_w, kf_h)
            .with_context(|| format!("failed to extract keyframe at {ts:.2}s"))?;
        kf_pixels.push(pixels);
    }
    log::info!("Collected {} keyframe pixels", kf_pixels.len());

    // --- Phase 2: fused RAUNE+depth batch inference ---
    // RAUNE output stays on GPU between models — no IPC round-trip for enhanced frames.
    // Depth runs on RAUNE-enhanced frames for better zone boundaries.
    log::info!(
        "Fused RAUNE+depth inference ({} frames, batch_size={FUSED_BATCH_SIZE})...",
        kf_pixels.len()
    );

    let fused_items: Vec<RauneDepthBatchItem> = kf_pixels.iter().enumerate().map(|(i, px)| {
        RauneDepthBatchItem {
            id: format!("kf{i:04}"),
            pixels: px.clone(),
            width: kf_w,
            height: kf_h,
            raune_max_size: kf_w,
            depth_max_size: 518,
        }
    }).collect();

    // (id, enhanced_rgb, enh_w, enh_h, depth_f32, depth_w, depth_h)
    let mut kf_results: Vec<(Vec<u8>, Vec<f32>, usize, usize)> = Vec::with_capacity(n_kf);

    for (chunk_items, chunk_pixels) in fused_items
        .chunks(FUSED_BATCH_SIZE)
        .zip(kf_pixels.chunks(FUSED_BATCH_SIZE))
    {
        let mut results = inf_server.run_raune_depth_batch(chunk_items)
            .unwrap_or_else(|e| {
                log::warn!("Fused RAUNE+depth batch failed: {e} — using originals + uniform depth");
                chunk_items.iter().zip(chunk_pixels.iter()).map(|(item, px)| {
                    (item.id.clone(), px.clone(), kf_w, kf_h,
                     vec![0.5f32; kf_w * kf_h], kf_w, kf_h)
                }).collect()
            });

        if results.len() < chunk_items.len() {
            log::warn!(
                "Fused batch returned {} results for {} items — padding with originals",
                results.len(), chunk_items.len()
            );
            let have = results.len();
            for (item, px) in chunk_items[have..].iter().zip(chunk_pixels[have..].iter()) {
                results.push((item.id.clone(), px.clone(), kf_w, kf_h,
                               vec![0.5f32; kf_w * kf_h], kf_w, kf_h));
            }
        }

        for (_, enhanced, _, _, depth_raw, dw, dh) in results {
            kf_results.push((enhanced, depth_raw, dw, dh));
        }
    }
    log::info!("Fused inference complete: {} frames", kf_results.len());

    let _ = inf_server.shutdown();

    // --- Phase 3: push to store ---
    for (i, (pixels, (enhanced, depth_proxy, dw, dh))) in
        kf_pixels.iter().zip(kf_results.iter()).enumerate()
    {
        let depth = InferenceServer::upscale_depth(depth_proxy, *dw, *dh, kf_w, kf_h);
        store.push(pixels, enhanced, &depth, kf_w, kf_h)
            .with_context(|| format!("failed to page frame {i} to store"))?;
    }
```

- [ ] **Step 5.3: Cargo check**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-cli 2>&1 | tail -10
```
Expected: `Finished` with no errors. Fix any type mismatches before continuing.

- [ ] **Step 5.4: Cargo test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -10
```
Expected: no failures.

- [ ] **Step 5.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): fused RAUNE+depth in auto_calibrate — depth runs on enhanced frames, single IPC phase"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Depth runs on RAUNE-enhanced frames | Task 3 handler, Task 5 caller |
| Enhanced tensor stays on GPU between models | Task 2 `infer_batch_from_tensors`, Task 3 handler |
| Single IPC round-trip for both models | Task 3 fused handler |
| Pinned memory for initial htod | Tasks 1 + 2 |
| Fallback to originals + uniform depth on failure | Task 5 |
| `infer_batch` standalone (for non-fused uses) | Task 1 |
| `pyo3_backend.rs` parity | NOT in scope — note below |

**Out of scope:** `pyo3_backend.rs` (the `python` Cargo feature path) is not updated. If that feature is ever enabled, it needs matching `RauneDepthBatchItem` + `run_raune_depth_batch` additions.

**Placeholder scan:** None. All code blocks are complete with exact types.

**Type consistency:**
- `run_raune_depth_batch` returns `Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>`
- Destructured in Task 5 as `(_, enhanced, _, _, depth_raw, dw, dh)` — matches
- `kf_results` stores `(Vec<u8>, Vec<f32>, usize, usize)` — consumed correctly in Phase 3

---

## Expected impact vs. prior plan

| Metric | Sequential (original) | Separate batches (prior plan) | Fused batch (this plan) |
|---|---|---|---|
| IPC round-trips (55 frames) | 110 | 7 + 2 = 9 | 7 |
| Enhanced frame IPC crossings | N/A | 2 × 97 MB | 0 |
| Depth input quality | original | original | RAUNE-enhanced |
| GPU idle between RAUNE→depth | yes (IPC gap) | yes (separate phases) | no (same forward call) |
