# 3D LUT Temporal Interpolation — Design Decision

**Date**: 2026-04-03
**Status**: Approved
**Supersedes**: Temporal grade interpolation (perf v3 opt 5, commit bc8934a — reverted in #23)

## Context

The perf v3 temporal interpolation skipped depth inference for non-keyframe frames
by interpolating final graded RGB pixels between keyframes. This caused visible
ghosting on moving subjects (pixel-wise blend of two spatial positions).

The fix (#23, commit c4710cf) interpolates depth maps instead and runs the full
grading pipeline on every frame. This eliminates ghosting but is slower: 11m37s
vs 3m33s (buggy) on the 1671-frame 4K 120fps test clip.

## Decision

Replace `grade_frame` on non-keyframe frames with a precomputed per-depth-band
3D color LUT. The LUT captures the entire per-pixel grading pipeline
(LUT apply + HSL correct + fused ambiance + warmth + blend) as a function of
(R, G, B, depth). Clarity is excluded — it's spatial and will be replaced by
NVIDIA Maxine.

## Design

### LUT Structure

- **Type**: Array of N standard 3D color LUTs, one per depth band
- **Grid resolution**: 33x33x33 (industry standard, matches DaVinci Resolve)
- **Depth bands**: 8 evenly spaced values in [0.0, 1.0]
- **Per entry**: 3x f32 (R', G', B') graded output
- **Total memory**: 33^3 × 8 × 3 × 4 bytes = 3.3MB
- **Lifetime**: Built once per clip (calibration + params are constant)

### Build Phase

After calibration is loaded, before the grading loop begins:

1. For each of the 8 depth bands (d = 0.0, 0.143, ..., 1.0):
2. For each (R, G, B) on the 33x33x33 grid:
   - Construct a synthetic 1-pixel input at (R/32, G/32, B/32) with depth d
   - Run the per-pixel grading pipeline: LUT apply → HSL correct →
     fused ambiance+warmth → blend with original
   - Store the output (R', G', B') in the LUT
3. Parallelized with rayon across grid points

Total: 33^3 × 8 = 287,496 single-pixel evaluations. Estimated build time: <500ms.

Requires a new `grade_pixel()` function that evaluates the per-pixel pipeline
without spatial operations (clarity). This is `grade_frame` for a 1x1 frame
with clarity disabled.

### Apply Phase (Non-Keyframe Frames)

For each pixel with input (R, G, B) and interpolated depth d:

1. Find bracketing depth bands: lo, hi, t_depth
2. Trilinear interpolate in `luts[lo]` → graded_lo
3. Trilinear interpolate in `luts[hi]` → graded_hi
4. Linear blend: `output = lerp(graded_lo, graded_hi, t_depth)`

Per-pixel cost: 48 LUT reads + 45 lerps. Parallelized with rayon.

### Pipeline Integration

```
Keyframe path (unchanged):
  depth_inference → grade_frame (full pipeline incl. clarity) → encode

Non-keyframe path (changed):
  Before: lerp_depth → grade_frame (~40ms)
  After:  lerp_depth → apply_grade_lut (~15ms)
```

The `grade.rs` main loop changes:
- Build `GradeLut` once after calibration, before the frame loop
- In the non-keyframe branch: call `grade_lut.apply(pixels, depth)` instead of `grade_frame`
- Keyframe branch: unchanged (full pipeline, stores depth for interpolation)

### Code Location

| File | Change |
|------|--------|
| `dorea-gpu/src/lut3d.rs` | New module: `GradeLut`, `Lut3D`, `build_grade_lut()`, `apply_grade_lut()` |
| `dorea-gpu/src/cpu.rs` | New: `grade_pixel()` — single-pixel pipeline without clarity |
| `dorea-gpu/src/lib.rs` | Export `lut3d` module |
| `dorea-cli/src/grade.rs` | Build LUT after calibration; use in non-keyframe path |

### Clarity Handling

Clarity (box blur detail extraction) is **skipped on non-keyframes**:
- It's the only spatial operation — cannot be captured in a per-pixel LUT
- At 120fps, frame-to-frame clarity difference is imperceptible
- Clarity will be replaced by NVIDIA Maxine post-grading enhancement
  (see `2026-04-02-maxine-post-grading-enhancement.md`)
- Keyframes still run full clarity via `grade_frame`

### Quality Characteristics

- **Trilinear quantization**: 33-point grid with trilinear interpolation is standard
  in film color grading. Max error ~0.5/255 for smooth transfer functions.
- **Depth band interpolation**: 8 bands with linear blending. The depth-dependent
  effects (shadow lift, S-curve contrast, warmth, vibrance) all vary linearly or
  near-linearly with depth — 8 bands is more than sufficient.
- **No ghosting**: Source pixels come from the actual frame, not blended keyframe outputs.
- **Keyframe quality**: Unchanged — full pipeline including clarity.

### Performance Estimates

| Frame type | Current (depth interp) | With 3D LUT |
|-----------|------------------------|-------------|
| Keyframe (~17%) | ~1.5s | ~1.5s (unchanged) |
| Non-keyframe (~83%) | ~40ms | ~15-17ms (rayon) |
| **1671 frames, 4K 120fps** | **11m 37s** | **~6-7m** |

Future optimization: CUDA 3D texture kernel for LUT apply → non-keyframe ~5ms → total ~5m.

### Future: CUDA Texture Acceleration

The 3D LUT lookup maps directly to GPU texture hardware:
- Upload each depth-band LUT as a CUDA 3D texture with trilinear filtering
- Single texture fetch per pixel per depth band (hardware-interpolated)
- Per-pixel cost drops from ~45 ALU ops to ~2 texture fetches + 1 lerp
- Expected: ~2ms per 4K frame

This is a follow-up optimization — the CPU/rayon path is the initial implementation.

## Alternatives Considered

### 4D Tetrahedral LUT
Single monolithic 4D array with tetrahedral interpolation. Same memory, slightly
fewer lookups, but 4D tetrahedral is exotic — harder to implement, debug, and
optimize. No CUDA texture hardware support for 4D. Rejected.

### Depth-Parameterized Polynomial Fit
Fit low-order polynomials to the grading function per grid point. Fewer memory
reads but the nonlinear LAB transforms (tanh, exp) risk polynomial
ringing/overshoot. More complex build. Rejected.

### Interpolate graded RGB (status quo ante)
The original perf v3 approach. Fast (~12ms) but produces ghosting on moving
subjects. Not acceptable for production output. Rejected and reverted in #23.
