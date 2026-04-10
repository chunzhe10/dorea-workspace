# Direct Mode fp16 + Larger Batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert RAUNE inference to fp16 and raise default direct-mode batch size from 4 to 8, with a new `--direct-batch-size` CLI flag for tuning.

**Architecture:** Two stacked changes in `raune_filter.py` (fp16 conversion in `main()` and `_process_batch()`) plus a new CLI flag in Rust that threads through `DirectModeConfig.batch_size` to the existing `--batch-size` filter argument.

**Tech Stack:** Python (PyTorch fp16), Rust (clap CLI flags).

**Spec:** `docs/decisions/2026-04-10-direct-mode-fp16-batch.md`

**Issue:** chunzhe10/dorea#66 (PR 1 of 3 in the GPU sweep)

---

## File Map

| File | Change |
|------|--------|
| `python/dorea_inference/raune_filter.py` | `main()` calls `model.half()` after loading state dict; `_process_batch()` casts proxy tensors to half before model call and result back to float32 |
| `crates/dorea-cli/src/grade.rs` | Add `--direct-batch-size` CLI flag, pass to `DirectModeConfig` |
| `crates/dorea-cli/src/config.rs` | Add `direct_batch_size: Option<usize>` to `[grade]` config section |

`run_pipe_mode()` is NOT modified — it already takes `batch_size` from args, so it picks up the new default for free, and fp16 is applied at model load (so it benefits too).

---

## Task 1: Convert RAUNE to fp16 in raune_filter.py

**Files:**
- Modify: `python/dorea_inference/raune_filter.py`

- [ ] **Step 1: Convert model to fp16 after loading state dict**

In `main()`, find the model load block (currently around line 651):

```python
    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).cuda()
    state = torch.load(args.weights, map_location="cuda", weights_only=True)
    model.load_state_dict(state)
    model.eval()
```

Replace with:

```python
    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).cuda()
    state = torch.load(args.weights, map_location="cuda", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Convert to fp16 for ~2× throughput on Ampere. fp16 has 11 bits of mantissa
    # precision, more than enough for 8-bit (or 10-bit) output. The forward pass
    # outputs fp16; we cast back to fp32 in _process_batch before the OKLab transfer.
    try:
        model = model.half()
        print("[raune-filter] RAUNE model converted to fp16", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[raune-filter] WARNING: model.half() failed ({e}); falling back to fp32",
              file=sys.stderr, flush=True)
```

This is wrapped in try/except as a safety net — if RAUNE has any layer that doesn't support fp16, we fall back gracefully instead of crashing.

- [ ] **Step 2: Cast proxy tensor to fp16 before RAUNE forward in _process_batch**

In `_process_batch()` (around lines 470-503), find the section that builds and runs RAUNE:

```python
        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        # RAUNE inference
        raune_out = model(proxy_batch)
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

Replace with:

```python
        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        # Cast to model's dtype (fp16 if model.half() was called, fp32 otherwise).
        # Match input dtype to model dtype to avoid PyTorch type mismatch errors.
        model_dtype = next(model.parameters()).dtype
        if proxy_batch.dtype != model_dtype:
            proxy_batch = proxy_batch.to(model_dtype)

        # RAUNE inference (may run in fp16)
        raune_out = model(proxy_batch)
        # Cast back to fp32 for downstream OKLab math
        raune_out = raune_out.float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

The key changes:
1. Match input dtype to model dtype (`proxy_batch.to(model_dtype)`) — avoids type mismatch errors
2. Cast `raune_out.float()` immediately after the forward pass — downstream OKLab math runs in fp32 (the Triton kernel already handles fp16 internally for memory but expects fp32 inputs, and the PyTorch fallback uses fp32)

Note: `orig_proxy = (proxy_batch * 0.5 + 0.5)` is computed below this — it will use the fp16 `proxy_batch`. We need to fix that too.

- [ ] **Step 3: Cast orig_proxy to fp32 in _process_batch**

A few lines below the model call, find:

```python
        # Original proxy (un-normalized)
        orig_proxy = (proxy_batch * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch
```

Replace with:

```python
        # Original proxy (un-normalized) — cast back to fp32 for OKLab math
        orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch
```

- [ ] **Step 4: Apply same dtype-matching to run_pipe_mode**

In `run_pipe_mode()` (around lines 562-590), find the equivalent block:

```python
            proxy_batch = torch.stack(proxy_tensors).cuda()
            del proxy_tensors

            raune_out = model(proxy_batch)
            raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

Replace with:

```python
            proxy_batch = torch.stack(proxy_tensors).cuda()
            del proxy_tensors

            # Match input dtype to model dtype (fp16 if model.half() was called)
            model_dtype = next(model.parameters()).dtype
            if proxy_batch.dtype != model_dtype:
                proxy_batch = proxy_batch.to(model_dtype)

            raune_out = model(proxy_batch).float()
            raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

And a few lines down, find:

```python
            orig_proxy = (proxy_batch * 0.5 + 0.5).clamp(0.0, 1.0)
            del proxy_batch
```

Replace with:

```python
            orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            del proxy_batch
```

- [ ] **Step 5: Verify syntax**

```bash
cd /workspaces/dorea-workspace/repos/dorea
python3 -c "import ast; ast.parse(open('python/dorea_inference/raune_filter.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Smoke test fp16 path with current default batch=4**

```bash
cd /workspaces/dorea-workspace/repos/dorea
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_fp16.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -10
```

Expected:
- `[raune-filter] RAUNE model converted to fp16` in stderr
- 360 frames complete
- gpu busy time significantly lower than baseline ~398ms (target: ~200ms)
- wall fps ~4-5 (up from 2.47)
- No type errors or NaN warnings

If gpu busy time is NOT lower, fp16 conversion didn't take effect — investigate before proceeding.

- [ ] **Step 7: Visual quality check vs baseline**

```bash
ffmpeg -v quiet -ss 1.5 -i "/workspaces/dorea-workspace/working/oklab_fp16.mov" \
  -frames:v 1 "/workspaces/dorea-workspace/working/oklab_fp16_frame.png" -y

ffmpeg -v error \
  -i /workspaces/dorea-workspace/working/oklab_fp16_frame.png \
  -i /workspaces/dorea-workspace/working/oklab_3thread_fixed_frame.png \
  -filter_complex "blend=all_mode=difference,signalstats" \
  -f null - 2>&1 | grep -E "YAVG|YMAX"
```

Expected: YAVG < 1.0, YMAX < 5.0 (fp16 introduces tiny rounding noise but should be visually identical).

If YMAX > 5, the fp16 numerical drift is larger than expected — investigate.

- [ ] **Step 8: Commit fp16 changes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/raune_filter.py
git commit -m "$(cat <<'EOF'
perf(direct): fp16 RAUNE inference

Convert RAUNE model to half precision via model.half() after loading.
Forward pass runs in fp16; output cast back to fp32 for OKLab transfer
which expects fp32 inputs.

fp16 has 11 bits of mantissa precision, more than enough for 8/10-bit
output. Expected ~2× speedup on Ampere fp16 throughput.

Wrapped in try/except as a safety net — if any RAUNE layer doesn't
support fp16, falls back to fp32 with a stderr warning.

Both single-process and pipe modes affected; both pick up the new dtype
via runtime check (next(model.parameters()).dtype).

Refs #66

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add --direct-batch-size CLI flag

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`
- Modify: `crates/dorea-cli/src/config.rs`

- [ ] **Step 1: Add config field**

In `crates/dorea-cli/src/config.rs`, find the `[grade]` section struct (around line 57 where `raune_proxy_size` is defined). Add a new field after it:

```rust
    pub raune_proxy_size: Option<usize>,
    /// Batch size for direct-mode RAUNE inference (number of frames per forward pass)
    pub direct_batch_size: Option<usize>,
```

Place it adjacent to `raune_proxy_size` so direct-mode config fields are grouped.

- [ ] **Step 2: Add CLI flag**

In `crates/dorea-cli/src/grade.rs`, find the `GradeArgs` struct around line 122 where `raune_proxy_size` is defined. Add a new field after it:

```rust
    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1920)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,

    /// Frames per batch in direct mode (default: 8). Larger = better GPU
    /// utilization, more VRAM. fp16 inference allows larger batches than fp32.
    #[arg(long)]
    pub direct_batch_size: Option<usize>,
}
```

- [ ] **Step 3: Resolve the value and pass it through**

In `crates/dorea-cli/src/grade.rs`, find the direct-mode block (around line 200). Locate where `direct_cfg` is constructed (around line 232):

```rust
        let direct_cfg = pipeline::grading::DirectModeConfig {
            python: python.clone(),
            raune_weights: rw.clone(),
            raune_models_dir: rmd.clone(),
            raune_proxy_size,
            batch_size: 4,
            output: output.clone(),
        };
```

Replace `batch_size: 4,` with the resolved value. Add a resolution line before the struct construction:

```rust
        let direct_batch_size = args.direct_batch_size
            .or(cfg.grade.direct_batch_size)
            .unwrap_or(8);

        log::info!(
            "Direct mode: single-process OKLab transfer, RAUNE proxy {}x{} (max {raune_proxy_size}), batch={direct_batch_size}, output {}x{}",
            proxy_w, proxy_h, info.width, info.height,
        );

        let direct_cfg = pipeline::grading::DirectModeConfig {
            python: python.clone(),
            raune_weights: rw.clone(),
            raune_models_dir: rmd.clone(),
            raune_proxy_size,
            batch_size: direct_batch_size,
            output: output.clone(),
        };
```

Also remove the existing `log::info!` line above that didn't include batch — it's now replaced by the version above.

- [ ] **Step 4: Build**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release -p dorea-cli 2>&1 | tail -5
```

Expected: clean build, no new warnings.

- [ ] **Step 5: Smoke test with new default batch=8**

```bash
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_fp16_batch8.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -15
```

Expected:
- Log line shows `batch=8`
- 360 frames complete
- Throughput should be slightly better than batch=4 fp16 result (target: ~5 fps)
- No OOM errors

- [ ] **Step 6: Test CLI override**

```bash
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_fp16_batch16.mov" \
  --output-codec prores \
  --direct \
  --direct-batch-size 16 \
  --verbose \
  2>&1 | tail -10
```

Expected: log line shows `batch=16`, completes without OOM. Throughput may or may not improve over batch=8 (depends on diminishing returns).

- [ ] **Step 7: Commit Rust changes**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/config.rs
git commit -m "$(cat <<'EOF'
perf(direct): add --direct-batch-size CLI flag, default 8

Raise default direct-mode batch from 4 to 8. Larger batches improve
GPU SM occupancy and amortize per-call Python overhead. fp16 inference
halves activation memory, so batch=8 has comparable VRAM usage to the
old batch=4 fp32.

Configurable via:
  - CLI: --direct-batch-size N
  - Config: [grade].direct_batch_size = N
  - Default: 8

Refs #66

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the Implementer

### Why `next(model.parameters()).dtype` instead of a flag

Using a runtime dtype check makes the code work for both fp16 and fp32 model loads with no extra plumbing. If `model.half()` fails and falls back to fp32, the dtype check picks up the fp32 dtype and skips the cast. Robust to both paths.

### Why output is cast to float() immediately after model call

The OKLab transfer code (Triton kernel and PyTorch fallback) was written assuming fp32 inputs. Casting to float right after the model call keeps the change minimal — we don't need to audit every downstream operation for fp16 compatibility.

### Why try/except around model.half()

RAUNE-Net is a standard U-Net with conv/batchnorm/relu — all standard layers support fp16 in PyTorch. But there's a small risk that a specific batchnorm or instance norm layer has a quirk. The try/except ensures any failure produces a clear stderr warning and falls back gracefully rather than crashing the pipeline. We expect this fallback to never trigger in practice.

### Don't change `_process_batch()` PCIe inefficiencies

The performance reviewer flagged the double full-res upload and per-frame transfer loop in `_process_batch()`. Those are out of scope for this PR (separate issue). Stay focused on the dtype conversion only.

### Don't add torch.compile

torch.compile is in scope for issue #66 overall, but it's the next PR. Stay focused on fp16 + batch only.

### Frame parity tolerance

fp16 introduces sub-1/255 numerical noise. The visual quality check is:
- YAVG < 1.0 (mean diff)
- YMAX < 5.0 (max diff)

If exceeded, escalate — fp16 should not visibly change output.
