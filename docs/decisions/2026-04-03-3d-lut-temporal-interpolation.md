# 3D LUT Temporal Interpolation — Design Decision

**Date**: 2026-04-03
**Status**: Approved (post 5-persona review)
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

- **Type**: Array of N standard 3D color LUTs, one per depth sample point
- **Grid resolution**: 65x65x65 (validated — see Numerical Validation below)
- **Depth samples**: Zone-aligned — `[0.0] + calibration.zone_centers + [1.0]`
  (7 points for 5 calibration zones). Uses the actual adaptive zone centers from
  calibration, not evenly-spaced bands. This guarantees sampling exactly where each
  zone's LUT dominates, with smooth soft-weight transitions between samples.
- **Per entry**: 3x f32 (R', G', B') graded output
- **Total memory**: 65^3 × 7 × 3 × 4 bytes = **23MB**
- **Lifetime**: Built once per clip (calibration + params are constant)

### Why 65x65x65 (not 33)

Numerical validation (100K random samples, fused_ambiance_warmth + blend pipeline):

| Grid | p99 error | Max error | Pixels > 2/255 |
|------|-----------|-----------|-----------------|
| 33x33x33 | 1.71/255 | 5.23/255 | 0.69% |
| **65x65x65** | **0.81/255** | **2.61/255** | **0.03%** |

The 33-grid fails at `tanh` highlight knees (L=0.88, output=0.92) where second
derivative is high. The 65-grid keeps 99.97% of pixels within 2/255.

**10-bit note**: At 10-bit (1023 levels), 65-grid max error maps to ~10.5/1023 —
too high. A 129x129x129 grid (~180MB) would be needed. Grid size is parameterized
for future upgrade. Locked at 65 for the current 8-bit pipeline.

### Build Phase

After calibration is loaded, before the grading loop begins:

1. For each depth sample d in `[0.0] + calibration.zone_centers + [1.0]`:
2. For each (R, G, B) on the 65x65x65 grid:
   - Call `grade_frame_inner` on a synthetic 1x1 frame at (R/64, G/64, B/64)
     with depth d and `skip_clarity=true`
   - Store the output (R', G', B') in the LUT
3. Parallelized with rayon across grid points

Total: 65^3 × 7 = ~1.9M single-pixel evaluations. Estimated build time: ~2-3s
in Rust with rayon.

**`grade_pixel` is NOT a separate function** — it calls the real pipeline
(`grade_frame_inner` with skip_clarity) on a 1x1 frame. Any future pipeline
changes automatically propagate to the LUT build. Zero divergence risk.

This requires extracting the per-pixel pipeline into `grade_frame_inner` with a
`skip_clarity` parameter. The existing `finish_grade` already accepts
`skip_clarity`, so this is mostly wiring.

### Apply Phase (Non-Keyframe Frames)

For each pixel with input (R, G, B) and interpolated depth d:

1. Find bracketing depth samples: lo, hi, t_depth
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
  After:  lerp_depth → apply_grade_lut (~15-25ms est, benchmark on real footage)
```

The `grade.rs` main loop changes:
- Build `GradeLut` once after calibration, before the frame loop
- In the non-keyframe branch: call `grade_lut.apply(pixels, depth)` instead of `grade_frame`
- Keyframe branch: unchanged (full pipeline, stores depth for interpolation)

### CLI Flags

- `--no-grade-lut`: Disable 3D LUT, force full `grade_frame` on every frame.
  For debugging and A/B quality comparison. Mirrors existing `--no-depth-interp`.

### Code Location

| File | Change |
|------|--------|
| `dorea-gpu/src/lut3d.rs` | New module: `GradeLut`, `Lut3D`, `build_grade_lut()`, `apply_grade_lut()` |
| `dorea-gpu/src/lib.rs` | Extract `grade_frame_inner(skip_clarity)` from `grade_frame`; export `lut3d` |
| `dorea-cli/src/grade.rs` | Build LUT after calibration; use in non-keyframe path; add `--no-grade-lut` |

### Clarity Handling

Clarity (box blur detail extraction) is **skipped on non-keyframes**:
- It's the only spatial operation — cannot be captured in a per-pixel LUT
- At 120fps, frame-to-frame clarity difference is imperceptible
- Clarity will be replaced by NVIDIA Maxine post-grading enhancement
  (see `2026-04-02-maxine-post-grading-enhancement.md`)
- Keyframes still run full clarity via `grade_frame`
- **Low frame rate note**: For content below 60fps, recommend reducing
  `--depth-max-interval` to 4-6 to increase keyframe density and reduce
  the quality asymmetry between keyframes (with clarity) and non-keyframes.

### Maxine Interaction

The Maxine decision doc (2026-04-02) described non-keyframes as lerping enhanced
RGB between keyframes — this is the approach reverted in PR #23 for ghosting.
The 3D LUT replaces that strategy. Combined pipeline:

| Configuration | Keyframe path | Non-keyframe path |
|---|---|---|
| LUT only | grade_frame + clarity | apply_grade_lut |
| LUT + Maxine | grade_frame + clarity + enhance | apply_grade_lut + enhance |
| Neither (current) | grade_frame + clarity | grade_frame + clarity |

Maxine runs on ALL frames post-grade (not just keyframes). The Maxine decision
doc must be updated to reflect this when Maxine is implemented.

### Invariant Enforcement

The LUT is built from calibration + GradeParams, which must be constant per clip.
`GradeLut` stores a hash of its build inputs. Before each `apply` call, assert
that the current params match the build hash:

```rust
impl GradeLut {
    fn is_valid_for(&self, cal: &Calibration, params: &GradeParams) -> bool;
}
```

This turns a silent correctness bug into a loud failure if params ever change
mid-clip.

### Quality Characteristics

**Numerical validation** (100K random samples at 65x65x65 grid, 7 zone-aligned
depth bands, fused_ambiance_warmth + blend pipeline):
- **Mean error**: 0.072/255
- **p50**: 0.035/255
- **p95**: 0.253/255
- **p99**: 0.812/255
- **Max**: 2.607/255 (0.03% of pixels, extreme highlights at tanh knee)

**Error budget breakdown**:
- Trilinear RGB quantization: dominant source, worst at highlight knees (L=0.88,
  output=0.92) where `tanh` second derivative is high
- Depth band interpolation: minimal — zone-aligned sampling captures transitions
- HSV hue shift: known property of all 3D LUTs in RGB space. Negligible for
  underwater footage (blues/greens, moderate saturation). Validated for saturated
  primaries in boundary condition tests.

**Other characteristics**:
- **No ghosting**: Source pixels come from the actual frame
- **Keyframe quality**: Unchanged — full pipeline including clarity

### Test Plan

**Unit tests** (in `dorea-gpu/src/lut3d.rs`):
1. `trilinear_accuracy`: 10K random (R,G,B,d) samples. Assert max error < 3/255,
   p99 < 1/255 at grid=65.
2. `boundary_conditions`: Black (0,0,0), white (1,1,1), midgray (0.5,0.5,0.5),
   saturated primaries (R,G,B,C,M,Y) at each depth band. Assert exact match at
   grid nodes (error = 0). Verify hue shift < 2 degrees for saturated primaries
   at off-grid positions.
3. `lut_validity_check`: Assert `is_valid_for` returns false when params change.

**Integration test**:
4. Grade a short clip with `--no-grade-lut` (full pipeline) and with LUT enabled.
   Compare frame-by-frame: assert max per-pixel error < 3/255 for non-keyframes.

### Performance Estimates

| Frame type | Current (depth interp) | With 3D LUT |
|-----------|------------------------|-------------|
| Keyframe (~17%) | ~1.5s | ~1.5s (unchanged) |
| Non-keyframe (~83%) | ~40ms | ~15-25ms (rayon, pending real benchmark) |
| LUT build (once) | — | ~2-3s |
| **1671 frames, 4K 120fps** | **11m 37s** | **~6-8m** (est) |

Timing breakdown of current ~40ms non-keyframe cost:
- GPU LUT apply + HSL correct: ~5ms
- GPU Clarity: ~5ms
- CPU fused_ambiance_warmth (rayon, LAB roundtrip): ~25ms ← eliminated by 3D LUT
- CPU blend + f32→u8: ~5ms ← eliminated by 3D LUT

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

### 33x33x33 Grid
Industry standard for film grading, but insufficient for this pipeline's
nonlinear operations. Max error 5.2/255 at highlight knees. Rejected after
numerical validation.

### Interpolate graded RGB (status quo ante)
The original perf v3 approach. Fast (~12ms) but produces ghosting on moving
subjects. Not acceptable for production output. Rejected and reverted in #23.

## Review Panel Findings Addressed

5-persona review (Senior Rust/GPU Engineer, Color Science Specialist, QA Engineer,
Performance Engineer, Video Pipeline Architect):

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Adaptive zone boundaries vs even spacing | Zone-aligned depth sampling using calibration.zone_centers |
| 2 | Trilinear error exceeds 0.5/255 | Upgraded to 65x65x65 grid; validated at p99=0.81/255 |
| 3 | No acceptance test plan | Unit tests + integration test defined (see Test Plan) |
| 4 | Maxine interaction undefined | Maxine runs on ALL frames post-grade; doc update required |
| 5 | grade_pixel divergence risk | Uses grade_frame_inner (real pipeline), not separate function |
| 6 | 8 depth bands may cause banding | Zone-aligned sampling eliminates banding at zone transitions |
| 7 | HSV hue shift in RGB trilinear | Known LUT property; validated for saturated primaries in tests |
| 8 | Maxine doc stale | Will update when Maxine is implemented |
| 9 | No --no-grade-lut flag | Added to CLI flags |
| 10 | 15ms estimate optimistic | Marked as estimate, benchmark on real footage required |
| 11 | Quality asymmetry at low fps | Added recommendation: reduce keyframe interval below 60fps |
| 12 | Performance table lacks breakdown | Added timing breakdown per stage |
| 13 | GradeParams not enforced constant | Added hash-based validity check on GradeLut |
