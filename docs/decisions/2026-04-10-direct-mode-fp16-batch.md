# Direct Mode GPU Optimization: fp16 RAUNE + Larger Batch

## Problem

After the 3-thread pipeline landed (chunzhe10/dorea#65), direct mode is GPU-bound at
~398 ms/frame. RAUNE inference dominates. Per the user constraint, optimizations must
preserve quality (no proxy reduction, no frame skipping).

This decision covers the first PR in a multi-PR sweep targeting the GPU stage.

## Decision

Two stacked, low-risk optimizations:

1. **fp16 RAUNE inference**: call `model.half()` after loading, convert input tensors
   to half precision before forward pass. Output is upcast to fp32 for the OKLab
   transfer (which already runs in mixed precision via the Triton kernel).

2. **Larger batch size**: raise the default from 4 to 8, expose `--direct-batch-size`
   CLI flag for tuning. fp16 halves activation memory, so batch=8 has the same VRAM
   footprint as batch=4 in fp32 — no risk of OOM on the 6GB RTX 3060.

## Why these two together

- fp16 alone gives ~2× on Ampere fp16 throughput
- Larger batch alone gives ~10-20% from amortizing per-call overhead and improving SM
  occupancy
- They stack cleanly: fp16 frees the VRAM that larger batch needs
- Both are bit-preserving from a perceptual standpoint:
  - fp16 has 11 bits of mantissa precision (10 explicit + 1 implicit), more than enough
    for 8-bit (or even 10-bit) output
  - Batch size doesn't change the mathematical operation, only its scheduling

## Expected outcome

- Throughput: 2.47 fps → ~5 fps (combination of 2× fp16 and 1.1-1.2× larger batch)
- Output: visually equivalent to baseline (fp16 introduces sub-1/255 noise)
- VRAM: comparable to current usage (fp16 halves activations, batch=8 doubles them)

## Scope

- `python/dorea_inference/raune_filter.py` — `main()` calls `model.half()` after
  loading state dict, `_process_batch()` converts proxy tensors to half before model
  call and casts output back to float32
- `crates/dorea-cli/src/grade.rs` — add `--direct-batch-size` CLI flag, default 8
- `crates/dorea-cli/src/pipeline/grading.rs` — `DirectModeConfig.batch_size` already
  exists; just thread the new CLI value through
- `crates/dorea-cli/src/config.rs` — add `direct_batch_size` to `[grade]` config
  section

### Not in scope (deferred to follow-up PRs)

- `torch.compile(model)` — separate PR after we verify fp16 baseline works
- NVDEC hardware decode — separate PR (more complex API surface)
- `_process_batch()` PCIe inefficiencies (double upload, per-frame transfer loop) —
  separate issue

## Risks and mitigations

### Risk: RAUNE-Net layers don't support fp16
**Mitigation**: RAUNE-Net is a standard U-Net with conv/batchnorm/relu/instance-norm.
All these layers support fp16 in PyTorch. If a layer fails, the error will be obvious
at first call. Fallback: keep the model in fp32 if `model.half()` fails (try/except
in main).

### Risk: fp16 numerical differences cause visible quality regression
**Mitigation**: Frame-by-frame diff against fp32 baseline must be ≤ 2/255 mean,
≤ 5/255 max. If exceeded, escalate.

### Risk: batch=8 OOMs on 6GB VRAM
**Mitigation**: fp16 halves activation memory, so net VRAM should be similar to
batch=4 fp32. If OOM occurs, the error is immediate and the CLI flag allows rollback.

### Risk: CLI flag conflicts with existing behavior
**Mitigation**: Default of 8 changes the previous hardcoded default of 4. The flag
is opt-out (always present), so users can set 4 to restore old behavior.

## Issue
chunzhe10/dorea#66

## Plan
docs/plans/2026-04-10-direct-mode-fp16-batch.md
