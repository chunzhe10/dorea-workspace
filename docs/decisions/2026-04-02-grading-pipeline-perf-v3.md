# Grading Pipeline Performance v3 — Design Spec

**Date:** 2026-04-02
**Status:** Proposed (reviewed by 5-persona panel)
**Scope:** `dorea grade` command — depth inference, color grading, IPC protocol

---

## Problem

After the box blur (sliding window) and proxy resize fixes landed, three compounding
bottlenecks remain in the per-frame grading pipeline, plus a silent CPU fallback that
can cause a 50x performance degradation without warning:

1. **Redundant depth inference at high FPS** — Depth Anything V2 runs on every frame,
   even at 120fps slow-mo where consecutive frames differ by sub-pixel amounts.
2. **Triple LAB roundtrip** — `depth_aware_ambiance`, clarity, and warmth each do
   independent RGB-to-LAB-to-RGB conversions (~7 conversion calls per pixel at 8.3M pixels).
3. **PNG encode/decode in IPC** — Frames are PNG-compressed, base64-encoded, piped to
   Python, then PNG-decompressed — all unnecessary at proxy resolution (453KB raw).
4. **HuggingFace processor overhead + silent CPU fallback** — The HF `AutoImageProcessor`
   adds ~30ms of redundant resize/normalize per frame. Separately, if CUDA fails at
   runtime, grading silently falls back to CPU with no user indication.

## Decision

Implement four targeted optimizations:

### Optimization 1: Motion-Adaptive Depth Interpolation

Skip depth inference for frames that are visually similar to the previous keyframe.
Use linear interpolation of bracketing keyframe depth maps for skipped frames.

**Motion detection:** Compute normalized MSE between the current proxy-resolution frame
and the last keyframe's proxy frame, in Rust (`grade.rs`), before any IPC call. This is
~0.1ms on a 518x292 frame and avoids the entire Python round-trip when skipped.

```
MSE = sum((curr[i] - keyframe[i])^2) / (n_pixels * 255^2)
```

**Keyframe selection rules:**
- Frame 0 is always a keyframe (no prior reference)
- A frame is a keyframe if `MSE > threshold` OR `frames_since_last_keyframe >= max_interval`
- **Scene cuts**: If `MSE > 10 * threshold`, treat as a scene cut — force keyframe and
  flush the interpolation buffer. Never interpolate across scene boundaries (produces
  visible ghosting due to semantic depth discontinuity).
- The max interval caps drift for static scenes

**Interpolation:** For non-keyframe frame `i` between keyframes `k_before` and `k_after`:
```
t = clamp((i - k_before) / max(k_after - k_before, 1), 0.0, 1.0)
depth[i] = depth[k_before] * (1 - t) + depth[k_after] * t
```
Guard: if `k_before == k_after`, use `k_before`'s depth directly (no division by zero).

**Lookahead buffer:** Accumulate frames until the next keyframe is identified, run depth
inference on it, then interpolate and grade the buffered frames. The buffer is bounded
by `max_interval`. **Safety valve:** if buffer exceeds `1.5 * max_interval` (e.g., due
to Python server backpressure), force a keyframe and flush to prevent unbounded growth.

**Edge cases:**
- **Frame 0**: Always a keyframe, no interpolation.
- **Last frames of video**: If no `k_after` exists, use `k_before`'s depth for all
  trailing frames (no interpolation, just reuse).
- **Single-frame input**: Frame 0 is a keyframe, done.
- **Very short clips (< max_interval frames)**: Frame 0 is keyframe, all others use
  frame 0's depth unless MSE triggers additional keyframes.

**CLI arguments:**
- `--depth-skip-threshold <f32>` — MSE threshold for keyframe decision (default: 0.005)
- `--depth-max-interval <usize>` — Maximum frames between keyframes (default: 12)
- `--no-depth-interp` — Disable interpolation, run depth on every frame

**Quality justification:** At 120fps, frames are 8.3ms apart. Underwater camera sway is
1-3 cm/s = 0.08-0.25mm between frames (sub-pixel in 518px depth map space). The depth
map feeds `depth_aware_ambiance` which uses depth as a smooth continuous parameter —
sub-1% depth errors are invisible in the 8-bit RGB output. Validate with ΔE comparison
on test footage during implementation.

**Expected savings:** 4-12x reduction in depth inference calls depending on content.

### Optimization 2: Fused LAB Pass (Ambiance + Warmth)

Merge the per-pixel work from `depth_aware_ambiance` and `finish_grade`'s warmth scaling
into a single RGB-to-LAB-to-RGB roundtrip, parallelized with rayon.

**Current flow (3 LAB roundtrips for ambiance+warmth):**
```
1. depth_aware_ambiance:  RGB → LAB → [shadow, s-curve, compress, warmth, vibrance] → LAB → RGB
2. apply_cpu_clarity:     RGB → LAB (L only) → blur → detail → LAB → RGB   [separate pass, unchanged]
3. warmth in finish_grade: RGB → LAB → scale a*/b* → LAB → RGB
```

**New flow (1 LAB roundtrip for ambiance+warmth, rayon-parallelized):**
```
1. fused_ambiance_warmth: RGB → LAB → [shadow, s-curve, compress, warmth push, vibrance, user warmth scale] → LAB → RGB
   (parallelized over pixel chunks via rayon par_chunks_exact_mut)
2. apply_cpu_clarity:     RGB → LAB (L only) → blur → detail → LAB → RGB   [unchanged]
```

**What changes:**
- New function `fused_ambiance_warmth(rgb, depth, width, height, contrast_scale, warmth_factor)`
  in `cpu.rs` that does all per-pixel LAB work in one pass using `par_chunks_exact_mut`
- `finish_grade` calls `fused_ambiance_warmth` instead of `depth_aware_ambiance` + separate warmth loop
- The clarity pass remains separate because (a) it requires a full-image box blur between
  per-pixel operations, and (b) the CUDA path already handles clarity via `clarity.cu`

**Color science note:** Fusing the LAB operations changes the order of operations compared
to the three-pass pipeline. Because sRGB↔LAB conversions are nonlinear, the output will
NOT be bit-exact with the old pipeline. This is an accepted tradeoff — the old output was
from a POC, not a golden reference. Validate that ΔE < 2 on test footage (imperceptible
to human vision). Document the change.

**Savings:** Eliminates ~1.45 billion redundant FLOPs per 4K frame (~0.25s on single core).
With rayon parallelism across available CPU cores, expect 4-6x additional speedup on the
fused pass itself.

### Optimization 3: Raw RGB IPC Protocol

Replace PNG encode/decode with raw RGB bytes in the Rust-to-Python inference protocol
for both depth and RAUNE requests.

**Current IPC per request:**
```
Rust:   proxy_pixels → encode_png_bytes (filter rows + zlib store + Adler-32) → base64 → JSON pipe
Python: JSON parse → base64 decode → PIL.Image.open(BytesIO).convert("RGB") → np.array
```

**New IPC:**
```
Rust:   proxy_pixels → base64 encode (raw bytes) → JSON pipe (with width, height, format fields)
Python: JSON parse → base64 decode → np.frombuffer(dtype=uint8).reshape(h, w, 3)
```

**Protocol change:** Add `"format": "raw_rgb"`, `"width"`, `"height"` fields to both
depth and RAUNE request types. Python server dispatches on `format` field: if absent or
`"png"`, use existing `decode_png()` (backward compatibility). If `"raw_rgb"`, use new
`decode_raw_rgb()`. Both `run_depth` and `run_raune` in `inference.rs` switch to raw mode.

**Byte format specification:**
- Encoding: raw interleaved RGB, uint8, row-major, no padding
- Byte order: not applicable (uint8, single-byte values)
- Layout: `[R0, G0, B0, R1, G1, B1, ...]`, length = `width * height * 3`

At proxy resolution (518x292), raw RGB is ~453KB vs ~460KB for uncompressed PNG.
The size difference is negligible; the savings come from skipping PNG filter/inflate
and PIL construction.

**Savings:** ~25ms/frame (PNG encode ~10ms, PNG decode + PIL ~15ms).

### Optimization 4: Bypass HuggingFace Processor + GPU-Required Runtime

**4a. Direct tensor construction:**

Replace `AutoImageProcessor(images=..., return_tensors="pt")` with direct numpy/torch
operations using known Depth Anything V2 normalization constants:

```python
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

arr = img_rgb.astype(np.float32) / 255.0
arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
```

The ViT patch-alignment resize (`_resize_for_depth`) is already handled. The processor's
only remaining job is normalize + tensorize, which the above replaces.
Remove `AutoImageProcessor` import and `self.processor` attribute entirely.

**Savings:** ~5-10ms/frame (revised estimate after perf review — original 20ms was high).

**4b. GPU-required runtime:**

Make CUDA a hard requirement for `dorea grade` at runtime:
- `grade_frame()` in `lib.rs`: if `cuda` feature is not compiled, return an error
  immediately (not a silent fallback)
- If `grade_frame_cuda()` returns a CUDA error, propagate it to the caller as a hard
  error — no CPU fallback. This includes clarity kernel failures (currently swallowed
  with a `log::warn` in `cuda/mod.rs:164-167` — change to hard error).
- CPU grading code (`grade_frame_cpu`, `finish_grade`, `apply_cpu_clarity`) remains
  available for `cargo test` — tests don't require a GPU
- `depth_aware_ambiance` and clarity CPU functions remain as reference implementations
  and for test use
- Python `DepthAnythingInference` should also hard-error if `device="cuda"` but
  `torch.cuda.is_available()` is false (currently silently falls back to CPU at
  `depth_anything.py:51-52`)

**Affected files:**
- `crates/dorea-gpu/src/lib.rs` — remove CPU fallback in `grade_frame()`
- `crates/dorea-gpu/src/cuda/mod.rs` — clarity failure becomes hard error
- `python/dorea_inference/depth_anything.py` — hard error on CUDA unavailable

### Optimization 5: Temporal Grade Interpolation

Extend the motion-adaptive keyframe concept from depth-only (Opt 1) to the **entire
grading pipeline**. For non-keyframe frames, skip depth inference AND grading entirely —
interpolate the final graded RGB output between bracketing keyframe graded outputs.

**How it works:** The depth interpolation lookahead buffer (Opt 1) already identifies
keyframes and buffers intermediate frames. Instead of running `grade_frame()` on every
buffered frame with interpolated depth, we:

1. Grade keyframe frames fully (depth inference + LUT + HSL + ambiance + clarity)
2. Cache the graded u8 output for each keyframe
3. For buffered (non-keyframe) frames: `graded[i] = lerp(graded[k_before], graded[k_after], t)`
4. The lerp operates on the final u8 RGB pixels — just a weighted average per channel

**Per non-keyframe frame cost:**
```
ffmpeg decode:    ~5ms   (still needed — source pixels required for MSE check)
Proxy resize:     ~1ms   (needed for MSE)
MSE computation:  ~0.1ms
Graded frame lerp: ~1ms  (4K u8: 8.3M × 3 channels × 1 multiply+add = ~25M ops)
ffmpeg encode:    ~5ms
Total:            ~12ms
```

Compare to **full pipeline per frame**: ~80-180ms (depth + grade).

**This subsumes Opt 1:** Depth interpolation (Opt 1) becomes unnecessary for non-keyframe
frames because we skip grading entirely. Depth is only needed for keyframes. The MSE-based
keyframe detection, scene-cut handling, buffer overflow safety, and edge cases from Opt 1
all still apply — they just gate the full pipeline vs. a simple lerp instead of gating
depth inference vs. depth interpolation.

**Lerp implementation:**
```rust
fn lerp_graded(a: &[u8], b: &[u8], t: f32) -> Vec<u8> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| {
            let v = va as f32 + (vb as f32 - va as f32) * t;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}
```

**Quality justification:** Same as Opt 1 — at 120fps, frames are 8.3ms apart with
sub-pixel motion. Linearly interpolating the graded output is equivalent to the camera
smoothly transitioning between two very similar graded frames. At the 8-bit output
quantization level, the interpolation error is invisible.

**CLI interaction:** Same args as Opt 1 (`--depth-skip-threshold`, `--depth-max-interval`,
`--no-depth-interp`). When `--no-depth-interp` is set, every frame runs full pipeline.

**Expected savings:** For N=6 keyframe interval, 5 of 6 frames cost ~12ms instead of
~80-180ms. At 1671 frames with ~1393 non-keyframe frames:
- Old: 1393 × ~130ms = ~3 min (depth interp + grade)
- New: 1393 × ~12ms = ~17s (lerp only)
- **Additional savings: ~2.5 min on top of Opt 1-4**

---

## File Map

| File | Optimization | Change |
|------|-------------|--------|
| `crates/dorea-cli/src/grade.rs` | Opt 1+5 | Frame buffer, MSE, keyframe logic, graded frame interpolation, scene-cut flush |
| `crates/dorea-cli/src/grade.rs` | Opt 1+5 | New CLI args: `--depth-skip-threshold`, `--depth-max-interval`, `--no-depth-interp` |
| `crates/dorea-gpu/src/cpu.rs` | Opt 2 | New `fused_ambiance_warmth()` with rayon, update `finish_grade` to call it |
| `crates/dorea-gpu/src/lib.rs` | Opt 4b | Remove CPU fallback, hard error on CUDA failure |
| `crates/dorea-gpu/src/cuda/mod.rs` | Opt 4b | Clarity kernel failure becomes hard error |
| `crates/dorea-video/src/inference.rs` | Opt 3 | Raw RGB encode path in `run_depth` and `run_raune` |
| `python/dorea_inference/protocol.py` | Opt 3 | `decode_raw_rgb()` helper, format-aware dispatch |
| `python/dorea_inference/server.py` | Opt 3 | Dispatch on `format` field in request |
| `python/dorea_inference/depth_anything.py` | Opt 4a, 4b | Replace processor with direct tensor; hard error on no CUDA |

## Instrumentation

Add timing spans (`Instant::now()` in Rust, `time.perf_counter()` in Python) around
each optimization's hot path to validate savings independently:

| Optimization | Metric | Location |
|-------------|--------|----------|
| Opt 1+5 | MSE computation time, keyframe %, frames graded vs interpolated | `grade.rs` |
| Opt 2 | `fused_ambiance_warmth()` wall time vs old `depth_aware_ambiance()` | `cpu.rs` |
| Opt 3 | Encode time (raw vs PNG), decode time | `inference.rs`, `server.py` |
| Opt 4a | Tensor construction time vs processor time | `depth_anything.py` |

Log per-frame timing at `--verbose` level. Aggregate stats (p50, p90, p99) at end of run.

## Expected Impact

| Optimization | Savings/frame (4K) | For 1671 frames |
|-------------|-------------------|-----------------|
| Temporal grade interpolation (Opt 5, avg N=6) | ~120-170ms on non-KF frames | ~2.5-4 min |
| Fused LAB pass + rayon (Opt 2, keyframes only) | ~250ms on KF frames | ~1.2 min (278 KFs) |
| Raw RGB IPC (Opt 3, keyframes only) | ~25ms on KF frames | ~7s (278 KFs) |
| Bypass HF processor (Opt 4a, keyframes only) | ~5-10ms on KF frames | ~1-3s (278 KFs) |
| GPU-required (Opt 4b) | risk elimination | N/A |
| **Combined** | | **~4-5.5 min** |

Note: With Opt 5 (temporal grade interpolation), Opts 2-4 only apply to keyframe
frames (~278 of 1671 at N=6). The dominant savings come from skipping the full pipeline
on ~83% of frames. Total per-frame cost: keyframes ~50-80ms, non-keyframes ~12ms.
Average: ~18-23ms/frame → **~30-38s total** for 1671 frames (vs hours before all fixes).

## Non-Goals

- CUDA kernel for fused ambiance+warmth (future Phase 3 work)
- Optical flow-guided depth warping (overkill for current use case)
- Batch depth inference (VRAM constraint: 6GB RTX 3060)
- Changes to the calibration pipeline (`dorea calibrate`, `dorea preview`)
- Bit-exact color reproduction vs the old three-pass LAB pipeline (accepted tradeoff)

## Review Panel Findings (2026-04-02)

Reviewed by 5 personas: SWE, QA, PM, Video Pipeline Specialist, Performance Engineer.

**Resolved blockers:**
- Scene-cut ghosting → force keyframe + buffer flush on MSE > 10x threshold
- LAB fusion non-bit-exactness → accepted as color-science tradeoff, validate ΔE < 2
- GPU clarity fallback swallowed → changed to hard error (consistent with GPU-required)

**Key additions from review:**
- Rayon parallelism for fused LAB pass (Perf Eng)
- Buffer overflow safety valve at 1.5x max_interval (Video Specialist)
- Division-by-zero guard on interpolation t (SWE)
- Raw RGB for both depth AND RAUNE, not just depth (SWE)
- Backward-compatible format field in protocol (QA)
- Timing instrumentation for per-optimization validation (Perf Eng)
- Edge case handling: single frame, last frame, short clips (QA)
- HF processor savings revised from 20ms to 5-10ms (Perf Eng)
