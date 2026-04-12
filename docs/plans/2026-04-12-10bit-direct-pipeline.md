# 10-Bit Direct Pipeline + GPU Frame Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve 10-bit precision through the RAUNE + OKLab delta grading pipeline and eliminate a redundant per-frame GPU upload.

**Architecture:** All changes in `repos/dorea/python/dorea_inference/raune_filter.py` plus a new test file. Decode as rgb48le (uint16), cache frames on GPU as int32 (PyTorch lacks uint16), process in fp32, download as uint16, encode as rgb48le. Triton wrapper updated to pass fp32 input (was unnecessarily casting to fp16). Pipe mode deleted in a separate commit.

**Tech Stack:** Python 3.13, PyTorch (CUDA), Triton, PyAV, numpy, pytest

**Spec:** `docs/decisions/2026-04-12-10bit-direct-pipeline.md`

---

### Task 1: Add synthetic round-trip test for `_process_batch`

**Files:**
- Create: `repos/dorea/python/tests/test_raune_filter.py`

This test validates the current 8-bit behavior before we change anything, then will be extended in Task 5 for 16-bit.

- [ ] **Step 1: Write the test file with an identity-model mock and 8-bit round-trip test**

```python
"""Tests for raune_filter _process_batch and related functions."""

import numpy as np
import pytest
import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    """Mock RAUNE that returns input unchanged (identity function).

    Expects input in [-1, 1] (RAUNE normalization range).
    Returns input unchanged, so the OKLab delta will be zero
    and the output should match the input.
    """

    def forward(self, x):
        return x


def _make_transfer_fn():
    """Return the PyTorch OKLab transfer function (no Triton dependency)."""
    from dorea_inference.raune_filter import pytorch_oklab_transfer
    return pytorch_oklab_transfer


@pytest.fixture
def identity_model():
    """Identity model on GPU in fp16 (matching real RAUNE dtype)."""
    model = IdentityModel().cuda().half()
    model.eval()
    return model


@pytest.fixture
def model_dtype(identity_model):
    return next(identity_model.parameters(), torch.tensor(0, dtype=torch.float16)).dtype


class TestProcessBatch8Bit:
    """Baseline: current 8-bit _process_batch behavior."""

    def test_output_shape_and_dtype(self, identity_model, model_dtype):
        from dorea_inference.raune_filter import _process_batch

        # 16x16 uint8 frame with a gradient
        frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 1
        assert results[0].shape == (16, 16, 3)
        # Before 10-bit change: uint8. After: uint16.
        # This test will be updated in Task 5.
        assert results[0].dtype == np.uint8

    def test_identity_model_preserves_values(self, identity_model, model_dtype):
        """With identity RAUNE (zero delta), output ≈ input within quantization."""
        from dorea_inference.raune_filter import _process_batch

        # Uniform mid-gray frame — avoids edge effects from OKLab nonlinearity
        frame = np.full((16, 16, 3), 128, dtype=np.uint8)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # Identity model → zero delta → output should be close to input.
        # Allow ±2 for OKLab round-trip quantization at 8-bit.
        diff = np.abs(results[0].astype(np.int16) - frame.astype(np.int16))
        assert diff.max() <= 2, f"max diff {diff.max()}, expected ≤2"
```

- [ ] **Step 2: Run the test to verify it passes (baseline)**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: 2 tests PASS (if GPU available) or SKIP (if no CUDA).

- [ ] **Step 3: Commit baseline test**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/tests/test_raune_filter.py && \
  git commit -m "test: add baseline _process_batch 8-bit round-trip tests"
```

---

### Task 2: Update `_process_batch` — int32 GPU cache + 16-bit math

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:475-541`

This is the core change. Replace the double-upload uint8 path with single-upload int32 cache and 16-bit output.

- [ ] **Step 1: Update the `_process_batch` function**

Replace lines 475-541 (the entire `_process_batch` function) with:

```python
def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype):
    """Process a batch of frames on GPU. Returns list of uint16 HWC numpy arrays."""
    n = len(batch_frames_np)
    results = []

    with torch.no_grad():
        # Upload once as int32, keep on GPU for reuse in apply loop.
        # int32 because PyTorch has no uint16 — torch.from_numpy(uint16)
        # reinterprets as int16 (signed), corrupting values > 32767.
        full_gpu_cache = []
        proxy_tensors = []
        for rgb_np in batch_frames_np:
            t_i32 = torch.from_numpy(rgb_np.astype(np.int32)).cuda()  # int32 on GPU
            full_gpu_cache.append(t_i32)
            full_f32 = t_i32.float() / 65535.0                   # (H,W,3) fp32 [0,1]
            full_f32 = full_f32.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
            # Downscale to proxy
            proxy_t = F.interpolate(full_f32, size=(ph, pw), mode="bilinear", align_corners=False)
            proxy_norm = (proxy_t - 0.5) / 0.5  # Normalize for RAUNE: [0,1] → [-1,1]
            proxy_tensors.append(proxy_norm.squeeze(0))
            del full_f32                                          # free fp32, keep int32

        # RAUNE inference + OKLab delta — unchanged from 8-bit path.
        # The proxy normalization produces [-1, 1] range regardless of whether
        # the source was 8-bit or 10-bit, so this section is unaffected.
        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        # Cast to model's dtype (passed in from main(), set once at model load)
        if proxy_batch.dtype != model_dtype:
            proxy_batch = proxy_batch.to(model_dtype)

        # RAUNE inference (may run in fp16)
        raune_out = model(proxy_batch)
        # Cast back to fp32 for downstream OKLab math
        raune_out = raune_out.float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)

        # Handle U-Net padding
        rh, rw = raune_out.shape[2], raune_out.shape[3]
        if rh != ph or rw != pw:
            raune_out = F.interpolate(raune_out, size=(ph, pw), mode="bilinear", align_corners=False)

        # Original proxy (un-normalized) — cast back to fp32 for OKLab math
        orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch

        # OKLab deltas at proxy resolution
        raune_lab = rgb_to_lab(raune_out)
        orig_lab = rgb_to_lab(orig_proxy)
        delta_lab = raune_lab - orig_lab
        del raune_out, orig_proxy, raune_lab, orig_lab

        # Upscale deltas to full resolution
        delta_full = F.interpolate(delta_lab, size=(fh, fw), mode="bilinear", align_corners=False)
        del delta_lab

        # Apply transfer per frame — reuse GPU cache, no re-upload
        for i in range(n):
            full_t = full_gpu_cache[i].float() / 65535.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

            result = transfer_fn(full_t, delta_full[i:i+1])
            del full_t

            # GPU → CPU → uint16 (round-half-up to avoid systematic -0.5 LSB bias)
            result_u16 = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 65535.0 + 0.5
                         ).to(torch.int32).cpu().numpy().astype(np.uint16)
            results.append(result_u16)
            del result

        del full_gpu_cache, delta_full

    return results
```

- [ ] **Step 2: Run the baseline test — it should now fail (dtype changed)**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: `test_output_shape_and_dtype` FAILS because output is now uint16 not uint8.

- [ ] **Step 3: Commit the `_process_batch` change**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "feat: 16-bit _process_batch with int32 GPU cache"
```

---

### Task 3: Update decoder thread — rgb48le decode

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:310-320`

- [ ] **Step 1: Change decode format and resize fallback**

Replace lines 312-317:

```python
                    t0 = time.perf_counter()
                    rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                    if rgb.shape[1] != fw or rgb.shape[0] != fh:
                        rgb = np.array(
                            frame.to_image().resize((fw, fh)),
                            dtype=np.uint8,
                        )
```

With:

```python
                    t0 = time.perf_counter()
                    rgb = frame.to_ndarray(format="rgb48le")  # (H, W, 3) uint16
                    if rgb.shape[1] != fw or rgb.shape[0] != fh:
                        # PyAV reformat preserves 16-bit (PIL .to_image() is 8-bit)
                        frame_resized = frame.reformat(width=fw, height=fh, format="rgb48le")
                        rgb = frame_resized.to_ndarray(format="rgb48le")
```

- [ ] **Step 2: Update the queue memory comment**

Find the line containing `q_decoded = queue.Queue(maxsize=2)` in `run_single_process` and replace:

```python
    q_decoded = queue.Queue(maxsize=2)   # holds: list[np.ndarray] (one batch)
```

With:

```python
    q_decoded = queue.Queue(maxsize=2)   # holds: list[np.ndarray] uint16 (~794 MB at 4K, batch=8)
```

- [ ] **Step 3: Commit decoder change**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "feat: decode as rgb48le for 10-bit preservation"
```

---

### Task 4: Update encoder thread — rgb48le encode

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:379`

- [ ] **Step 1: Change encode format**

Replace line 379:

```python
                    out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb24")
```

With:

```python
                    out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb48le")
```

- [ ] **Step 2: Commit encoder change**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "feat: encode via rgb48le for 10-bit output"
```

---

### Task 5: Update tests for 16-bit output

**Files:**
- Modify: `repos/dorea/python/tests/test_raune_filter.py`

- [ ] **Step 1: Update existing tests and add 16-bit specific tests**

Replace the entire test file with:

```python
"""Tests for raune_filter _process_batch and related functions."""

import numpy as np
import pytest
import torch
import torch.nn as nn


class IdentityModel(nn.Module):
    """Mock RAUNE that returns input unchanged (identity function).

    Expects input in [-1, 1] (RAUNE normalization range).
    Returns input unchanged, so the OKLab delta will be zero
    and the output should match the input.
    """

    def forward(self, x):
        return x


def _make_transfer_fn():
    """Return the PyTorch OKLab transfer function (no Triton dependency)."""
    from dorea_inference.raune_filter import pytorch_oklab_transfer
    return pytorch_oklab_transfer


@pytest.fixture
def identity_model():
    """Identity model on GPU in fp16 (matching real RAUNE dtype)."""
    model = IdentityModel().cuda().half()
    model.eval()
    return model


@pytest.fixture
def model_dtype(identity_model):
    return next(identity_model.parameters(), torch.tensor(0, dtype=torch.float16)).dtype


class TestProcessBatch16Bit:
    """16-bit _process_batch with int32 GPU cache."""

    def test_output_shape_and_dtype(self, identity_model, model_dtype):
        """Output must be uint16 HWC numpy arrays."""
        from dorea_inference.raune_filter import _process_batch

        frame = np.full((16, 16, 3), 32768, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 1
        assert results[0].shape == (16, 16, 3)
        assert results[0].dtype == np.uint16

    def test_identity_preserves_midgray(self, identity_model, model_dtype):
        """With identity RAUNE (zero delta), mid-gray output ≈ input."""
        from dorea_inference.raune_filter import _process_batch

        # Mid-gray in 16-bit
        val = 32768
        frame = np.full((16, 16, 3), val, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        diff = np.abs(results[0].astype(np.int32) - frame.astype(np.int32))
        # Allow ±128 (≈ 2 LSBs at 16-bit, ~0.2%) for OKLab round-trip
        assert diff.max() <= 128, f"max diff {diff.max()}, expected ≤128"

    def test_values_above_32767_not_corrupted(self, identity_model, model_dtype):
        """Values > 32767 must not become negative (int16 sign corruption check)."""
        from dorea_inference.raune_filter import _process_batch

        # Frame with value 60000 — would be negative if treated as int16
        frame = np.full((16, 16, 3), 60000, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # If int16 corruption occurred, values would be near 0 or wrapped
        assert results[0].min() > 50000, (
            f"min={results[0].min()}, expected >50000 — likely int16 sign corruption"
        )

    def test_8bit_input_zero_extended(self, identity_model, model_dtype):
        """8-bit source zero-extended to uint16 should produce valid output."""
        from dorea_inference.raune_filter import _process_batch

        # Simulate rgb48le zero-extension of 8-bit source: val * 257
        val_8bit = 128
        val_16bit = val_8bit * 257  # = 32896
        frame = np.full((16, 16, 3), val_16bit, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            [frame], identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert results[0].dtype == np.uint16
        diff = np.abs(results[0].astype(np.int32) - frame.astype(np.int32))
        assert diff.max() <= 128, f"max diff {diff.max()}, expected ≤128"

    def test_rounding_edge_values(self, identity_model, model_dtype):
        """Verify +0.5 rounding works for edge values (0, max)."""
        from dorea_inference.raune_filter import _process_batch

        # Black frame — should stay near 0
        black = np.zeros((8, 8, 3), dtype=np.uint16)
        # Near-white frame — should stay near 65535
        white = np.full((8, 8, 3), 65535, dtype=np.uint16)
        transfer_fn = _make_transfer_fn()

        results_black = _process_batch(
            [black], identity_model, None,
            fw=8, fh=8, pw=4, ph=4,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )
        results_white = _process_batch(
            [white], identity_model, None,
            fw=8, fh=8, pw=4, ph=4,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        # Black should be very close to 0
        assert results_black[0].max() <= 128, f"black max={results_black[0].max()}"
        # White should be very close to 65535
        assert results_white[0].min() >= 65000, f"white min={results_white[0].min()}"

    def test_batch_of_multiple_frames(self, identity_model, model_dtype):
        """Multiple frames in a batch should all be processed."""
        from dorea_inference.raune_filter import _process_batch

        frames = [
            np.full((16, 16, 3), v, dtype=np.uint16)
            for v in [16384, 32768, 49152]
        ]
        transfer_fn = _make_transfer_fn()

        results = _process_batch(
            frames, identity_model, None,
            fw=16, fh=16, pw=8, ph=8,
            transfer_fn=transfer_fn,
            model_dtype=model_dtype,
        )

        assert len(results) == 3
        for r in results:
            assert r.shape == (16, 16, 3)
            assert r.dtype == np.uint16
```

- [ ] **Step 2: Run the updated tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 3: Commit updated tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/tests/test_raune_filter.py && \
  git commit -m "test: update _process_batch tests for 16-bit output"
```

---

### Task 6: Update Triton wrapper — fp32 input

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:173-188`

- [ ] **Step 1: Update `triton_oklab_transfer` to pass fp32 to kernel**

Replace lines 173-188:

```python
def triton_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """Fused Triton kernel: entire OKLab transfer in one kernel launch."""
    _, C, H, W = frame_nchw_f32.shape
    frame_flat = frame_nchw_f32.squeeze(0).half().reshape(3, -1).contiguous()
    delta_flat = delta_nchw_f32.squeeze(0).half().reshape(3, -1).contiguous()
    n_pixels = H * W
    out_flat = torch.empty_like(frame_flat)

    BLOCK_SIZE = 1024
    grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _oklab_transfer_kernel[grid](
        frame_flat, delta_flat, out_flat,
        n_pixels=n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.reshape(1, 3, H, W).float()
```

With:

```python
def triton_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """Fused Triton kernel: entire OKLab transfer in one kernel launch."""
    _, C, H, W = frame_nchw_f32.shape
    # Pass fp32 to kernel — avoid precision loss before nonlinear transforms.
    # Kernel output stays fp16 (sufficient for 10-bit).
    frame_flat = frame_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    delta_flat = delta_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    n_pixels = H * W
    out_flat = torch.empty(3, n_pixels, dtype=torch.float16, device=frame_flat.device)

    BLOCK_SIZE = 1024
    grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _oklab_transfer_kernel[grid](
        frame_flat, delta_flat, out_flat,
        n_pixels=n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.reshape(1, 3, H, W).float()
```

- [ ] **Step 2: Run tests to verify nothing broke**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit Triton wrapper change**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "perf: pass fp32 input to Triton kernel (avoid fp16 round-trip)"
```

---

### Task 7: Update module docstring and queue comment

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:1-17`

- [ ] **Step 1: Update module docstring**

Replace lines 1-17:

```python
"""RAUNE-Net OKLab chroma transfer — single-process, zero-pipe.

Decodes input video via PyAV, processes on GPU, encodes output via PyAV.
No stdin/stdout pipe I/O — all frame data stays in-process.

Pipeline per batch:
  1. PyAV decode → numpy → GPU upload
  2. Downscale to proxy on GPU (torch.interpolate)
  3. Batch RAUNE inference on proxy
  4. OKLab delta computation at proxy on GPU
  5. Upscale deltas to full-res on GPU (Triton kernel)
  6. Full-res OKLab transfer via fused Triton kernel
  7. GPU download → PyAV encode

Supports both single-process mode (--input/--output) and legacy pipe mode
(reads rgb48le from stdin, writes to stdout) for backward compatibility.
"""
```

With:

```python
"""RAUNE-Net OKLab chroma transfer — single-process, 10-bit.

Decodes input video via PyAV (rgb48le for 10-bit preservation), processes
on GPU with int32 frame cache to avoid redundant PCIe transfers, encodes
output via PyAV (rgb48le → yuv422p10le for ProRes).

Pipeline per batch:
  1. PyAV decode (rgb48le uint16) → numpy → int32 GPU upload (cached)
  2. Downscale to proxy on GPU (torch.interpolate)
  3. Batch RAUNE inference on proxy (fp16)
  4. OKLab delta computation at proxy on GPU (fp32)
  5. Upscale deltas to full-res on GPU
  6. Full-res OKLab transfer via fused Triton kernel (fp32 in, fp16 out)
  7. GPU download (uint16) → PyAV encode (rgb48le)
"""
```

- [ ] **Step 2: Commit docstring update**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "docs: update raune_filter docstring for 10-bit pipeline"
```

---

### Task 8: Delete pipe mode (separate commit)

**Files:**
- Modify: `repos/dorea/python/dorea_inference/raune_filter.py:544-635, 696-703`

- [ ] **Step 1: Delete `run_pipe_mode` function**

Delete the entire block from line 544 (`# Legacy pipe mode`) through line 635 (`return frame_count`). This includes the section comment and the function body.

- [ ] **Step 2: Update `main()` dispatch to require --input/--output**

Replace lines 696-703:

```python
    # Dispatch to single-process or pipe mode
    if args.input:
        if not args.output:
            print("error: --output required with --input", file=sys.stderr)
            sys.exit(1)
        run_single_process(args, model, normalize, model_dtype)
    else:
        run_pipe_mode(args, model, normalize, model_dtype)
```

With:

```python
    # Single-process mode only (pipe mode removed)
    if not args.input or not args.output:
        print("error: --input and --output are required", file=sys.stderr)
        sys.exit(1)
    run_single_process(args, model, normalize, model_dtype)
```

- [ ] **Step 3: Run tests to verify nothing broke**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Verify no pipe-mode callers exist in Rust CLI**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  grep -rn "raune_filter" crates/dorea-cli/src/
```

Expected: All matches show `--input` and `--output` being passed. No pipe-mode invocation.

- [ ] **Step 5: Commit pipe mode deletion**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  git add python/dorea_inference/raune_filter.py && \
  git commit -m "chore: delete legacy pipe mode from raune_filter"
```

---

### Task 9: Smoke test on real hardware

This task requires the RTX 3060 workstation and actual RAUNE weights. It cannot be run in CI.

**Files:** None (manual verification)

- [ ] **Step 1: Run automated tests one final time**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_raune_filter.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run dorea on a 10-bit test clip**

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  ./target/release/dorea footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4
```

Expected: Completes without OOM. Check stderr for stage timing output.

- [ ] **Step 3: Verify output is real 10-bit via ffprobe**

```bash
ffprobe -v quiet -select_streams v:0 -show_entries stream=bits_per_raw_sample,pix_fmt,codec_name \
  footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s_graded.mov
```

Expected: `bits_per_raw_sample=10`, `pix_fmt=yuv422p10le`, `codec_name=prores`.

- [ ] **Step 4: Pixel-level verification that lower bits are non-zero**

```bash
/opt/dorea-venv/bin/python -c "
import subprocess, numpy as np
cmd = ['ffmpeg', '-i', 'footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s_graded.mov',
       '-frames:v', '1', '-pix_fmt', 'rgb48le', '-f', 'rawvideo', '-']
raw = subprocess.run(cmd, capture_output=True).stdout
arr = np.frombuffer(raw, dtype=np.uint16)
lower_6_bits = arr & 0x3F
nonzero_pct = (lower_6_bits != 0).sum() / len(lower_6_bits) * 100
print(f'Lower 6 bits non-zero: {nonzero_pct:.1f}% of pixels')
assert nonzero_pct > 10, f'Only {nonzero_pct:.1f}% — likely 8-bit padded'
print('PASS: output has real 10-bit content')
"
```

Expected: `PASS: output has real 10-bit content` with >10% of pixels having non-zero lower 6 bits.

- [ ] **Step 5: Test 8-bit input backward compatibility**

If an 8-bit H264 clip is available, run:

```bash
cd /workspaces/dorea-workspace/repos/dorea && \
  ./target/release/dorea path/to/8bit-clip.mp4
```

Expected: Completes successfully. Output file is valid and playable.

- [ ] **Step 6: Compare stage timing against baseline**

Check the stderr output from step 2 for `[raune-filter] stage timing` line. Compare `decode`, `gpu`, and `encode` busy times against pre-change baseline. No specific threshold — log for reference.
