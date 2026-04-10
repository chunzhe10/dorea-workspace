# torch.compile RAUNE-Net for Inductor JIT Fusion

## Problem

After fp16 landed (chunzhe10/dorea#67), direct mode is at 4.25 fps with RAUNE forward
pass at 230 ms/frame. Issue #68 proved that PCIe upload batching is NOT the bottleneck —
CUDA streams already overlap transfers with compute. The critical path stage is RAUNE
itself.

## Decision

Wrap the fp16 RAUNE model with `torch.compile(model, mode="reduce-overhead", dynamic=False)`
after the `model.half()` + InstanceNorm fp32 restoration. This enables Inductor JIT
fusion of the 30 residual blocks.

## Design Decisions

### `mode="reduce-overhead"`
Optimized for repeated inference on same-shape inputs. Uses CUDA graphs where possible
to eliminate kernel launch overhead. Better than default `mode=None` for our use case.

### `dynamic=False`
Batch size is fixed per run (from `--direct-batch-size`). Static shapes let Inductor
specialize and use CUDA graphs more aggressively. Dynamic shapes would disable CUDA
graph optimization.

### `try/except` around compile (unlike model.half())
torch.compile is genuinely optional — if Inductor fails on an exotic layer, the model
still works in eager mode. The try/except here is NOT the same fail-fast violation as
the original model.half() wrapper, because:
- fp16 conversion is a *required* optimization — failure means broken assumptions
- torch.compile is a *pure performance* optimization — failure means no speedup

Falls back to eager mode with a clear stderr warning. The pipeline still works.

### Persistent Inductor cache
Set `TORCHINDUCTOR_CACHE_DIR` to a workspace-persistent path so subsequent runs skip
the ~10-30s compile. First run pays the cost, all subsequent runs reuse the compiled
kernels.

### Warmup forward pass
After compile, do one dummy forward pass with the correct input shape to trigger
Inductor compilation during startup. This moves the compile delay from "first frame"
to "startup" which is more predictable.

## Expected Outcome

- ~1.2-1.5× speedup on RAUNE forward pass (standard Inductor gain on Ampere vision models)
- 4.25 fps → ~5-6 fps target
- First run: ~10-30s additional startup time (one-time compile)
- Subsequent runs: negligible startup overhead (cache hit)
- Output numerically equivalent to eager mode within fp16 rounding

## Risks

### Risk: torch.compile + mixed precision (fp16 conv + fp32 InstanceNorm) fails
**Mitigation**: try/except fallback to eager mode. Test on the POC benchmark script
first, before running on real footage. If compile fails, log the error clearly.

### Risk: Compilation produces numerically different output
**Mitigation**: Frame parity check vs PR #67 baseline (target YAVG < 1.0, YMAX < 5).

### Risk: CUDA graphs with `mode="reduce-overhead"` are incompatible with our control flow
**Mitigation**: If CUDA graphs fail, `mode="default"` is a fallback that still gets
Inductor fusion without the graph overhead.

### Risk: First-run cold start blocks the user
**Mitigation**: Log clearly that compilation is happening. Persistent cache means this
is a one-time cost.

## Scope

Only `python/dorea_inference/raune_filter.py::main()`. No other files change.
No new CLI flags. No new config options. Inductor behavior is controlled via env var.

## What stays the same

- `_process_batch()` — unchanged
- `triton_oklab_transfer` and OKLab math — unchanged
- `run_pipe_mode` — unchanged (still uses eager model — pipe mode is rarely used)
- CLI flags and config — unchanged
- 3-thread pipeline — unchanged
- fp16 + InstanceNorm fp32 pattern — preserved

## Issue
chunzhe10/dorea#69

## Plan
docs/plans/2026-04-10-torch-compile-raune.md
