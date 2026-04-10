# Batched _process_batch — Eliminate Per-Frame Upload Loop

## Problem

After fp16 landed (chunzhe10/dorea#67), direct mode is at 4.25 fps. Performance reviewers
flagged that `_process_batch()` does N serial CPU→GPU uploads per batch, then re-uploads
the same numpy buffers a second time for the OKLab transfer. Larger batch sizes show no
benefit (batch=8 ≈ batch=4) and batch=16 actively regresses, because the per-frame upload
loop scales linearly with batch_size and dominates as RAUNE inference shrinks.

## Decision

Refactor `_process_batch()` to do a single batched H2D upload, batched downscale,
batched transfer, and single batched D2H. Update the Triton wrapper to accept batched
input by flattening N into the pixel dimension.

## Architecture

### Before
```
for rgb_np in batch:
    upload(rgb_np)  → discard         # N uploads, ~25MB each
proxy_batch = stack(proxy_tensors)
RAUNE(proxy_batch)
for i in range(n):
    upload(batch_frames_np[i])         # N more uploads (same data!)
    transfer_fn(full_t[i:i+1], delta_full[i:i+1])  # N kernel launches
    download(result)
```

### After
```
full_batch = upload(np.stack(batch))   # 1 upload, contains all frames
proxy_batch = downscale_batched(full_batch)
RAUNE(proxy_batch)
result_batch = transfer_fn(full_batch, delta_full)  # 1 batched call
results = download_batched(result_batch)             # 1 download
```

## Triton Wrapper Change

The kernel itself stays unchanged — pixels are processed independently with no spatial
dependencies, so flattening N into `n_pixels` is mathematically valid.

The wrapper permutes `(N,3,H,W)` → `(3,N,H,W)` (channel-first) and flattens to
`(3, N*H*W)`. After the kernel, it reverses the permutation. Permute+contiguous costs
~0.6ms at batch=8 4K fp16, negligible vs the ~N × ~25ms PCIe uploads it eliminates.

## Expected Outcome

- `batch=4` should match current 4.25 fps (no regression)
- `batch=8` should now actually deliver ~10-20% (target ~5 fps)
- `batch=16` should no longer regress (~5-6 fps)
- Cumulative: 2.47 → ~5-6 fps (additional 18-40% over PR #67)

## Scope

- `python/dorea_inference/raune_filter.py`:
  - `_process_batch()` rewritten to use batched ops
  - `triton_oklab_transfer()` wrapper accepts (N, 3, H, W) input
  - `_oklab_transfer_kernel` itself UNCHANGED (still single-frame conceptually,
    just processes a flattened pixel range)
- `run_pipe_mode()` unchanged — pipe mode is rarely used and the per-frame structure
  is required by the rgb48le streaming protocol

## Backward Compatibility

- Single-frame call (`N=1`) still works — permute is a no-op for that case
- `pytorch_oklab_transfer` already handles batches via NCHW math
- Output must remain bit-identical to current per-frame loop

## Risks

### Risk: Permute layout error causes frame ordering corruption
**Mitigation**: Frame parity test must show YAVG=0, YMAX=0 vs PR #67 baseline.

### Risk: VRAM peak grows
Keeping `full_batch` on GPU through the whole batch instead of freeing per-frame
adds ~N × 25MB peak VRAM. At batch=8 that's ~200MB extra. At batch=16 it's ~400MB.
Within the 6GB budget but worth measuring with `torch.cuda.max_memory_allocated()`.

### Risk: Triton wrapper change breaks single-frame callers
**Mitigation**: The benchmark script `bench_oklab_transfer.py` tests single-frame
input. Verify it still works after the wrapper change.

## Issue
chunzhe10/dorea#68

## Plan
docs/plans/2026-04-10-batched-process-batch.md
