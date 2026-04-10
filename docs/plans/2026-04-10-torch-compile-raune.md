# torch.compile RAUNE-Net — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Wrap the fp16 RAUNE model with `torch.compile()` for Inductor JIT fusion. No other changes.

**Architecture:** Single-function change in `main()` of `raune_filter.py`. After `model.half()` + InstanceNorm restoration, add a `torch.compile` wrapper with try/except fallback and a warmup forward pass.

**Tech Stack:** PyTorch 2.6 + Inductor + CUDA 12.4.

**Spec:** `docs/decisions/2026-04-10-torch-compile-raune.md`

**Issue:** chunzhe10/dorea#69

---

## File Map

| File | Change |
|---|---|
| `python/dorea_inference/raune_filter.py` | Add `torch.compile` wrapping after fp16 conversion in `main()`; add warmup forward pass |

---

## Task 1: Wrap model with torch.compile

**File:** `python/dorea_inference/raune_filter.py`

- [ ] **Step 1: Set up persistent Inductor cache directory at module level**

Near the top of `raune_filter.py`, after the existing imports, add:

```python
# Persistent Inductor compile cache so subsequent runs skip the ~10-30s JIT.
# The workspace /workspaces mount is persistent across devcontainer restarts.
_INDUCTOR_CACHE = "/workspaces/dorea-workspace/working/.torch_inductor_cache"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _INDUCTOR_CACHE)
```

You'll also need to import `os` if not already imported at the top of the file. Check the imports and add `import os` if needed.

- [ ] **Step 2: Add torch.compile wrapper in main() after fp16 conversion**

Find the fp16 conversion block in `main()` (currently around line 667-682):

```python
    # Convert RAUNE to fp16 for ~2× throughput on Ampere.
    # ...
    import torch.nn as nn
    model = model.half()
    instance_norm_count = 0
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.float()
            instance_norm_count += 1
    model_dtype = next(model.parameters()).dtype
    print(f"[raune-filter] RAUNE converted to fp16 "
          f"({instance_norm_count} InstanceNorm layers kept in fp32, "
          f"model_dtype={model_dtype})",
          file=sys.stderr, flush=True)
```

Immediately after this block, add:

```python
    # Wrap with torch.compile for Inductor JIT fusion.
    # torch.compile is a pure performance optimization — if it fails on any
    # layer we fall back to eager mode, which still works (just slower).
    # This is DIFFERENT from the fp16 try/except which we removed in PR #67:
    # fp16 is required for correctness; torch.compile is optional for speed.
    try:
        import time as _compile_time_start_mod
        t_compile_start = _compile_time_start_mod.time()
        print(f"[raune-filter] compiling RAUNE with torch.compile "
              f"(cache: {os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'default')})",
              file=sys.stderr, flush=True)
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        # Warmup forward pass to trigger compilation during startup so the first
        # real frame doesn't pay the cold-start cost.
        with torch.no_grad():
            dummy_input = torch.zeros(
                args.batch_size, 3, args.proxy_height, args.proxy_width,
                device="cuda", dtype=model_dtype,
            )
            _ = model(dummy_input)
            torch.cuda.synchronize()
            del dummy_input

        compile_duration = _compile_time_start_mod.time() - t_compile_start
        print(f"[raune-filter] torch.compile ready ({compile_duration:.1f}s, "
              f"Inductor reduce-overhead mode, static shape "
              f"[{args.batch_size}, 3, {args.proxy_height}, {args.proxy_width}])",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[raune-filter] WARNING: torch.compile failed ({e}); "
              f"using eager mode (no Inductor fusion)",
              file=sys.stderr, flush=True)
```

Note: we use `_compile_time_start_mod` instead of shadowing the already-imported `time` module, because the existing file already has `import time` and we don't want to confuse the reader.

Actually, simpler: just use the existing `time` module:

```python
    try:
        t_compile_start = time.time()
        print(f"[raune-filter] compiling RAUNE with torch.compile "
              f"(cache: {os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'default')})",
              file=sys.stderr, flush=True)
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        # Warmup forward pass to trigger compilation during startup
        with torch.no_grad():
            dummy_input = torch.zeros(
                args.batch_size, 3, args.proxy_height, args.proxy_width,
                device="cuda", dtype=model_dtype,
            )
            _ = model(dummy_input)
            torch.cuda.synchronize()
            del dummy_input

        compile_duration = time.time() - t_compile_start
        print(f"[raune-filter] torch.compile ready ({compile_duration:.1f}s, "
              f"Inductor reduce-overhead mode, static shape "
              f"[{args.batch_size}, 3, {args.proxy_height}, {args.proxy_width}])",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[raune-filter] WARNING: torch.compile failed ({e}); "
              f"using eager mode (no Inductor fusion)",
              file=sys.stderr, flush=True)
```

Use this simpler version.

- [ ] **Step 3: Verify Python syntax**

```bash
cd /workspaces/dorea-workspace/repos/dorea
python3 -c "import ast; ast.parse(open('python/dorea_inference/raune_filter.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: First-run smoke test (expect compile time)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_compile_run1.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -20
```

Expected on the FIRST run:
- `[raune-filter] compiling RAUNE with torch.compile ...`
- `[raune-filter] torch.compile ready (N.Ns, ...)` where N is 10-60 seconds
- 360 frames complete
- Per-stage timing at the end should show `gpu` busy time LOWER than the previous 230 ms (target: ~160-190 ms)
- Wall throughput at least slightly better than 4.25 fps (target: ~5-6 fps)

Report the compile duration and the per-stage timings.

- [ ] **Step 5: Second-run smoke test (expect cached compile)**

```bash
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_compile_run2.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -20
```

Expected on the SECOND run:
- Same log lines but `torch.compile ready (X.Xs ...)` should be MUCH faster (< 5s, ideally < 2s)
- Same throughput as run 1 (steady state)
- Cache hit confirmed

Report the compile duration on run 2.

- [ ] **Step 6: Visual quality check vs previous fp16 output**

```bash
ffmpeg -v quiet -ss 1.5 -i "/workspaces/dorea-workspace/working/oklab_compile_run2.mov" \
  -frames:v 1 "/workspaces/dorea-workspace/working/oklab_compile_frame.png" -y

ffmpeg -v error \
  -i /workspaces/dorea-workspace/working/oklab_compile_frame.png \
  -i /workspaces/dorea-workspace/working/oklab_fp16_e2e_frame.png \
  -filter_complex "format=rgb24[a];[1:v]format=rgb24[b];[a][b]blend=all_mode=difference,signalstats" \
  -f null - 2>&1 | grep -E "YAVG|YMAX"
```

Expected: YAVG < 1.0, YMAX < 5 (Inductor-fused kernels may introduce sub-1/255 numerical noise but visual output should be identical).

If YMAX > 5, investigate — Inductor may have introduced a reordering that changes fp16 rounding.

- [ ] **Step 7: If torch.compile fails (fallback path works)**

If `torch.compile` raises an exception on first call, the fallback should print a warning and the pipeline should still complete successfully at eager-mode speed (~4.25 fps). Check that the `WARNING: torch.compile failed` line appears in stderr and the run completes.

If compile works, skip this step.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/raune_filter.py
git commit -m "$(cat <<'EOF'
perf(direct): torch.compile RAUNE for Inductor JIT fusion

Wrap fp16 RAUNE model with torch.compile(mode="reduce-overhead", dynamic=False)
after the InstanceNorm fp32 restoration. Inductor JIT-fuses the 30 residual
blocks and uses CUDA graphs where possible to eliminate kernel launch overhead.

Persistent compile cache via TORCHINDUCTOR_CACHE_DIR so subsequent runs skip
the ~10-30s cold start. Warmup forward pass triggers compilation during
startup rather than on the first real frame.

try/except fallback to eager mode if Inductor fails — torch.compile is a
pure performance optimization, not a correctness requirement. Unlike the
model.half() try/except we removed in PR #67, falling back to eager just
means losing speedup, not broken output.

Closes #69
Refs #66 (GPU optimization sweep PR 2/3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the Implementer

### Why `try/except` is OK here but wasn't for `model.half()`

The 5-persona review of PR #67 flagged the `model.half()` try/except as a fail-fast violation. That was correct because fp16 is a correctness-sensitive choice — silently falling back to fp32 means the user thinks they're getting fp16 behavior but they're not.

torch.compile is different. It's purely a performance optimization. If compile fails:
- The model still runs correctly (in eager mode)
- The output is still numerically valid (same as without compile)
- The only thing lost is some speedup

Silently falling back doesn't mask a correctness issue. The stderr warning is sufficient observability.

### Why `mode="reduce-overhead"` instead of default

PyTorch's `torch.compile` has three main modes:
- `mode=None` (default): basic Inductor fusion, no CUDA graphs
- `mode="reduce-overhead"`: CUDA graphs + Inductor fusion, best for small batches
- `mode="max-autotune"`: most aggressive, longest compile time, best for large batches

Our workload is 4K at batch=4-16, which is "small batch" in ML terms. `reduce-overhead` is the right choice.

### Why `dynamic=False`

Batch size is fixed at startup from `--direct-batch-size`. Input resolution is fixed at `proxy_width × proxy_height`. All shapes are static. Telling Inductor to specialize for static shapes enables CUDA graphs and more aggressive fusion.

If the user ran multiple videos with different resolutions in one process, `dynamic=True` would be needed — but we run one video per process.

### Why warmup with zeros instead of real data

Inductor compiles the graph the first time a specific input shape is seen. Using `torch.zeros` with the exact shape `(batch_size, 3, proxy_height, proxy_width)` triggers compilation for exactly the shape the real frames will use. The actual values don't matter for compilation — only the shape and dtype.

### What NOT to do

- DO NOT remove the try/except fallback
- DO NOT compile `_process_batch` itself (it has data-dependent control flow)
- DO NOT compile `rgb_to_lab` / `lab_to_rgb` (too simple to benefit; already handled by Triton kernel path)
- DO NOT modify the Triton kernel or OKLab math
- DO NOT change any CLI flags or config options
- DO NOT add unit tests (separate issue)

### What if torch.compile makes it SLOWER

Inductor compilation can occasionally produce slower output for certain layer topologies, especially if the graph is dominated by small ops. If the measured gpu_busy time goes UP after compile:
- Report as DONE_WITH_CONCERNS
- Don't delete the compile path — it may help on different hardware
- Add a note to the corvia record
