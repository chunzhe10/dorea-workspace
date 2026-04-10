# Batched _process_batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `_process_batch()` to do single batched H2D upload, batched downscale, batched RAUNE+OKLab, single batched D2H. Update Triton wrapper to accept batched input.

**Architecture:** Two changes in `raune_filter.py`. (1) `_process_batch()` rewritten to use np.stack + single .cuda() + batched ops throughout. (2) `triton_oklab_transfer()` wrapper accepts (N, 3, H, W) by permuting to channel-first and flattening N into the pixel dimension. The Triton kernel itself is unchanged.

**Tech Stack:** Python (PyTorch, numpy, Triton).

**Spec:** `docs/decisions/2026-04-10-batched-process-batch.md`

**Issue:** chunzhe10/dorea#68

---

## File Map

| File | Change |
|---|---|
| `python/dorea_inference/raune_filter.py` | `_process_batch()` rewritten; `triton_oklab_transfer()` wrapper supports batched input |

The Triton kernel `_oklab_transfer_kernel` is NOT modified. The PyTorch fallback `pytorch_oklab_transfer` is NOT modified (already supports batches). `run_pipe_mode` is NOT modified.

---

## Task 1: Update Triton wrapper to accept batched input

**File:** `python/dorea_inference/raune_filter.py`

- [ ] **Step 1: Replace `triton_oklab_transfer()` to handle batched (N, 3, H, W) input**

Current code (around lines 173-188):
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

Replace with:
```python
def triton_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """Fused Triton kernel: entire OKLab transfer in one kernel launch.

    Supports batched input (N, 3, H, W) by flattening N into the pixel dimension.
    The kernel processes pixels independently — no spatial dependencies — so
    flattening across the batch is mathematically equivalent to per-frame calls
    but uses a single kernel launch.
    """
    N, C, H, W = frame_nchw_f32.shape
    # Permute (N, 3, H, W) → (3, N, H, W) so each channel is contiguous across
    # all batches. The Triton kernel expects layout (3, n_pixels) where each
    # channel's pixels are stored contiguously. After permute we have:
    # channel 0 = [b0_p0, b0_p1, ..., b0_pHW, b1_p0, ..., bN_pHW]
    frame_chnw = frame_nchw_f32.half().permute(1, 0, 2, 3).contiguous()
    delta_chnw = delta_nchw_f32.half().permute(1, 0, 2, 3).contiguous()
    frame_flat = frame_chnw.reshape(3, -1)  # (3, N*H*W)
    delta_flat = delta_chnw.reshape(3, -1)
    n_pixels = N * H * W
    out_flat = torch.empty_like(frame_flat)

    BLOCK_SIZE = 1024
    grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _oklab_transfer_kernel[grid](
        frame_flat, delta_flat, out_flat,
        n_pixels=n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reverse the permutation: (3, N*H*W) → (3, N, H, W) → (N, 3, H, W)
    return out_flat.reshape(3, N, H, W).permute(1, 0, 2, 3).contiguous().float()
```

- [ ] **Step 2: Smoke-test the wrapper with the existing single-frame benchmark**

```bash
cd /workspaces/dorea-workspace
python3 working/sea_thru_poc/bench_oklab_transfer.py --iters 5 --warmup 2 2>&1 | tail -25
```

The benchmark uses single-frame input (shape (1,3,H,W)). The wrapper change must still work for N=1. Expected: similar timing (~3 ms/frame for the Triton path) and identical visual output to the previous implementation.

If the benchmark fails or produces wildly different numbers, the wrapper change broke something — investigate before proceeding.

---

## Task 2: Rewrite _process_batch with batched operations

**File:** `python/dorea_inference/raune_filter.py`

- [ ] **Step 1: Replace the entire `_process_batch()` function**

Current code (lines 467-532, approximately):
```python
def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype):
    """Process a batch of frames on GPU. Returns list of uint8 HWC numpy arrays."""
    n = len(batch_frames_np)
    results = []

    with torch.no_grad():
        # Build proxy batch for RAUNE
        proxy_tensors = []
        for rgb_np in batch_frames_np:
            full_t = torch.from_numpy(rgb_np).cuda().float() / 255.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)
            proxy_t = F.interpolate(full_t, size=(ph, pw), mode="bilinear", align_corners=False)
            proxy_norm = (proxy_t - 0.5) / 0.5
            proxy_tensors.append(proxy_norm.squeeze(0))
            del full_t

        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        if proxy_batch.dtype != model_dtype:
            proxy_batch = proxy_batch.to(model_dtype)

        raune_out = model(proxy_batch)
        raune_out = raune_out.float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)

        rh, rw = raune_out.shape[2], raune_out.shape[3]
        if rh != ph or rw != pw:
            raune_out = F.interpolate(raune_out, size=(ph, pw), mode="bilinear", align_corners=False)

        orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch

        raune_lab = rgb_to_lab(raune_out)
        orig_lab = rgb_to_lab(orig_proxy)
        delta_lab = raune_lab - orig_lab
        del raune_out, orig_proxy, raune_lab, orig_lab

        delta_full = F.interpolate(delta_lab, size=(fh, fw), mode="bilinear", align_corners=False)
        del delta_lab

        for i in range(n):
            full_t = torch.from_numpy(batch_frames_np[i]).cuda().float() / 255.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)
            result = transfer_fn(full_t, delta_full[i:i+1])
            del full_t
            result_u8 = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0
                         ).to(torch.uint8).cpu().numpy()
            results.append(result_u8)
            del result

        del delta_full

    return results
```

Replace with:

```python
def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype):
    """Process a batch of frames on GPU. Returns list of uint8 HWC numpy arrays.

    Uses batched ops throughout: single H2D upload, batched downscale, batched
    RAUNE inference, batched OKLab transfer, single D2H download. The full-res
    frame tensor is uploaded ONCE and reused for both the proxy downscale and
    the full-res transfer step.
    """
    n = len(batch_frames_np)
    if n == 0:
        return []

    with torch.no_grad():
        # ── Step 1: Single batched H2D upload ──────────────────────────────
        # Stack frames CPU-side, then one .cuda() call.
        rgb_np_stack = np.stack(batch_frames_np)  # (N, H, W, 3) uint8
        full_batch = torch.from_numpy(rgb_np_stack).cuda().float() / 255.0
        # (N, H, W, 3) → (N, 3, H, W)
        full_batch = full_batch.permute(0, 3, 1, 2).contiguous()

        # ── Step 2: Batched downscale to proxy ─────────────────────────────
        proxy_batch = F.interpolate(
            full_batch, size=(ph, pw), mode="bilinear", align_corners=False
        )
        # Normalize for RAUNE: [0,1] → [-1,1]
        proxy_norm = (proxy_batch - 0.5) / 0.5

        # ── Step 3: Cast to model dtype and run RAUNE ──────────────────────
        if proxy_norm.dtype != model_dtype:
            proxy_norm = proxy_norm.to(model_dtype)
        raune_out = model(proxy_norm).float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)

        # Handle U-Net padding
        rh, rw = raune_out.shape[2], raune_out.shape[3]
        if rh != ph or rw != pw:
            raune_out = F.interpolate(
                raune_out, size=(ph, pw), mode="bilinear", align_corners=False
            )

        # ── Step 4: OKLab deltas at proxy resolution ───────────────────────
        # proxy_batch is already in [0,1] fp32 (no normalize/de-normalize cycle).
        raune_lab = rgb_to_lab(raune_out)
        orig_lab = rgb_to_lab(proxy_batch)
        delta_lab = raune_lab - orig_lab
        del raune_out, raune_lab, orig_lab, proxy_norm

        # ── Step 5: Upscale deltas to full resolution ──────────────────────
        delta_full = F.interpolate(
            delta_lab, size=(fh, fw), mode="bilinear", align_corners=False
        )
        del delta_lab

        # ── Step 6: Batched OKLab transfer (REUSES full_batch from step 1) ─
        # Both transfer_fn variants (Triton and PyTorch) accept batched input.
        result_batch = transfer_fn(full_batch, delta_full)
        del full_batch, delta_full

        # ── Step 7: Single batched D2H ─────────────────────────────────────
        # (N, 3, H, W) → (N, H, W, 3) uint8 numpy
        result_u8_batch = (
            result_batch.permute(0, 2, 3, 1).clamp(0, 1) * 255.0
        ).to(torch.uint8).cpu().numpy()
        del result_batch

    # Slice into per-frame numpy arrays for the encoder
    return [result_u8_batch[i] for i in range(n)]
```

Note: `proxy_batch` is kept in fp32 (we cast `proxy_norm` to model_dtype but keep `proxy_batch` for the OKLab orig_lab computation). This is correct because OKLab math runs in fp32.

- [ ] **Step 2: Verify Python syntax**

```bash
cd /workspaces/dorea-workspace/repos/dorea
python3 -c "import ast; ast.parse(open('python/dorea_inference/raune_filter.py').read()); print('OK')"
```

- [ ] **Step 3: Run the existing benchmark to verify the wrapper still works for single-frame**

```bash
cd /workspaces/dorea-workspace
python3 working/sea_thru_poc/bench_oklab_transfer.py --iters 5 --warmup 2 2>&1 | tail -25
```

Expected: still completes successfully, similar timing to before.

- [ ] **Step 4: End-to-end smoke test with current default batch=4**

```bash
cd /workspaces/dorea-workspace/repos/dorea
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_batched_b4.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -10
```

Expected:
- 360 frames complete
- Throughput should be roughly equal to the previous fp16 result (~4.25 fps) or slightly better
- gpu busy time should be slightly lower than the previous 230 ms (some Python loop overhead removed)

Report the actual numbers.

- [ ] **Step 5: Frame parity vs previous fp16 output**

```bash
ffmpeg -v quiet -ss 1.5 -i "/workspaces/dorea-workspace/working/oklab_batched_b4.mov" \
  -frames:v 1 "/workspaces/dorea-workspace/working/oklab_batched_b4_frame.png" -y

ffmpeg -v error \
  -i /workspaces/dorea-workspace/working/oklab_batched_b4_frame.png \
  -i /workspaces/dorea-workspace/working/oklab_fp16_e2e_frame.png \
  -filter_complex "format=rgb24[a];[1:v]format=rgb24[b];[a][b]blend=all_mode=difference,signalstats" \
  -f null - 2>&1 | grep -E "YAVG|YMAX"
```

Expected: YAVG=0, YMAX=0 (the batched ops should produce bit-identical output to the per-frame loop). If diff is non-zero, investigate.

- [ ] **Step 6: Throughput test at batch=8**

```bash
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_batched_b8.mov" \
  --output-codec prores \
  --direct \
  --direct-batch-size 8 \
  --verbose \
  2>&1 | tail -10
```

Expected: throughput should now be MEANINGFULLY higher than batch=4 (target: ~5 fps). Report numbers.

- [ ] **Step 7: Throughput test at batch=16**

```bash
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_batched_b16.mov" \
  --output-codec prores \
  --direct \
  --direct-batch-size 16 \
  --verbose \
  2>&1 | tail -10
```

Expected: throughput should NOT regress vs batch=8 (target: ~5-6 fps). If it still regresses, the per-frame loop wasn't the only bottleneck and we need to investigate further.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/raune_filter.py
git commit -m "$(cat <<'EOF'
perf(direct): batched _process_batch — single H2D upload, batched transfer

Refactor _process_batch() to eliminate the per-frame upload loop and the
double upload of full-res frames. Now does:
- Single batched H2D via np.stack + .cuda()
- Batched F.interpolate for downscale and delta upscale
- Batched OKLab transfer (full_batch reused from step 1, no re-upload)
- Single batched D2H

Triton wrapper updated to accept (N, 3, H, W) by permuting to channel-first
and flattening N into the pixel dimension. The kernel itself is unchanged
because it processes pixels independently with no spatial dependencies.

The PyTorch fallback transfer was already batch-aware (NCHW math).
run_pipe_mode and the Triton kernel definition are unchanged.

Performance reviewers in PR #65 and #67 flagged this as the next bottleneck
after fp16 landed. Eliminates ~N × 25MB redundant PCIe traffic per batch.

Closes #68

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the Implementer

### The proxy_batch / proxy_norm distinction

In the new code:
- `proxy_batch` is the un-normalized (in [0,1]) downscaled batch, fp32
- `proxy_norm` is the RAUNE-normalized version (in [-1,1]), cast to model_dtype

After RAUNE inference, we use `proxy_batch` directly for the OKLab orig_lab computation — NO need to "denormalize" because we kept the un-normalized version. This is cleaner than the old code which computed `orig_proxy = (proxy_batch * 0.5 + 0.5)` from the normalized tensor.

### Why permute+contiguous in the Triton wrapper

The kernel addressing is `frame_ptr + channel_idx * n_pixels + pixel_offset`. This requires channel-major layout: all channel-0 pixels stored contiguously, then all channel-1, then all channel-2.

For input `(N, 3, H, W)` contiguous, the natural memory layout is batch-major: `[b0_c0_p, b0_c1_p, b0_c2_p, b1_c0_p, ...]`. We need to transpose so it becomes `[c0_b0_p, c0_b1_p, ..., c0_bN_p, c1_b0_p, ...]`.

`permute(1, 0, 2, 3)` swaps batch and channel dims. `.contiguous()` materializes the new memory layout. Cost is one GPU memcpy of ~400MB at batch=8 4K fp16, which is ~0.6ms on RTX 3060.

### What if batch=16 still regresses

If the batch=16 test in Step 7 still shows regression, the per-frame loop wasn't the only issue. Possible additional culprits:
- VRAM pressure (peak memory exceeds 6GB at batch=16)
- L2 cache thrashing
- delta_full at fp32 4K is large (4 × 4K_pixels × 4 bytes per channel = ~100MB per frame, batch=16 = 1.6GB)

In that case: report it as a finding, don't try to fix in this PR. The plan succeeded in eliminating the per-frame loop overhead — further optimization is a separate issue.

### What NOT to do

- DO NOT modify the Triton kernel `_oklab_transfer_kernel` itself
- DO NOT modify `pytorch_oklab_transfer` (already batch-aware)
- DO NOT modify `run_pipe_mode` (separate code path)
- DO NOT modify `rgb_to_lab` / `lab_to_rgb` (already batch-aware via NCHW)
- DO NOT add pinned memory (separate optimization for a future PR)
- DO NOT add unit tests (separate issue)
