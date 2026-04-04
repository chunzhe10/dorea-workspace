# Per-Keyframe Adaptive Depth Zones — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Supersedes:** Global depth zone computation (single `adaptive_zone_boundaries` call per clip)

## Problem

The current pipeline computes one set of depth zone boundaries from a reservoir sample of ALL keyframe depths across the entire clip. This produces a single global depth distribution that feeds both LUT calibration and per-frame grading. When the camera moves significantly (reef close-up → open water) or subjects shift depth (diver ascending), the global zones are a compromise that serves no scene well:

1. **LUT calibration quality degrades.** Zone boundaries averaged across disparate scenes place correction samples at depths that don't match any individual scene. A close-up reef (depths 0.05–0.35) and open water (depths 0.40–0.95) produce zones centered around 0.45 — relevant to neither.

2. **Per-frame depth-aware grading is inaccurate.** The combined GPU texture evaluates `grade_pixel_device` at zone center depths. With global zones, center depths don't align with the actual depth distribution of the frame being graded. Soft zone blending interpolates between irrelevant depth samples.

Both problems compound: the LUT is trained on wrong depth partitions, AND it's applied at wrong depth sample points.

## Solution: Two-Tier Zone Architecture

Split depth zones into two tiers with different lifetimes and granularity:

### Tier 1 — Base LUT (32 fine zones, per-segment)

- Built from all keyframes within a scene segment
- 32 zones provide fine depth resolution across the full [0.0–1.0] range
- Many keyframes contribute → good color cell coverage in the 33³ grid
- Rebuilt only at scene segment boundaries
- Encodes the **color science**: "at depth X, what RGB correction do we apply?"

### Tier 2 — Runtime Zones (8 adaptive zones, per-keyframe)

- Computed from each keyframe's own depth map via `adaptive_zone_boundaries(kf_depth, 8)`
- Places zone centers where the depth values actually ARE in this frame
- Combined GPU texture rebuilt per-keyframe at these 8 local zone centers, reading from the 32-zone base LUT
- Encodes the **depth context**: "which depths matter for this frame?"

### Why this works

The 32-zone base LUT has enough depth resolution that `grade_pixel_device` can accurately evaluate at any depth value via trilinear interpolation across fine zones. The 8 runtime zones choose which depths to sample — and they choose locally relevant ones. Color science (stable per-segment) is decoupled from depth context (adapts per-keyframe).

## Data Flow

```
Pass 0 (existing, unchanged):
  Proxy decode → keyframe detection → fused RAUNE+depth inference
  Output: all keyframe depth maps + RAUNE targets in memory

Pre-compute phase (new, ~500ms total):
  ┌─────────────────────────────────┐
  │ 1. Per-KF zone boundaries (raw) │  ~5ms
  │    adaptive_zone_boundaries      │  (all keyframes, trivially parallel)
  │    per keyframe depth map        │
  │    Output: Vec<[f32; 9]>         │
  ├─────────────────────────────────┤
  │ 2. Segment detection             │  ~2ms
  │    Wasserstein-1 distance on     │  (consecutive KF depth histograms)
  │    depth histograms              │
  │    Output: Vec<SegmentRange>     │
  ├─────────────────────────────────┤
  │ 3. Zone boundary smoothing       │  ~microseconds
  │    Weighted avg over 3 KFs       │  (0.6 center + 0.2 each neighbor)
  │    Respects segment boundaries   │  (segments must be known first)
  ├─────────────────────────────────┤
  │ 4. Per-segment base LUT build    │  ~100ms × N_segments
  │    32-zone StreamingLutBuilder   │
  │    + HSL corrections             │
  │    Output: Vec<(DepthLuts,       │
  │            HslCorrections)>      │
  └─────────────────────────────────┘

Grading pass (restructured):
  CUDA Stream A: grade frames using current texture pair + temporal blend
  CUDA Stream B: build next KF's combined texture (pipelined, hidden)
  At KF boundary: swap textures, upload new zone boundaries
  At segment boundary: also upload new base LUT to device
```

## Scene Segmentation

### Detection metric

Wasserstein-1 distance (earth mover's distance) between consecutive keyframes' depth histograms. Measures how much "work" it takes to reshape one depth distribution into the other. Computed from sorted CDF difference on quantized depth bins.

```rust
fn depth_distribution_distance(depth_a: &[f32], depth_b: &[f32], n_bins: usize) -> f32 {
    // Quantize into n_bins histogram bins (e.g., 64 bins)
    // Compute CDFs for both
    // Wasserstein-1 = sum of |CDF_a[i] - CDF_b[i]| / n_bins
}
```

This is distinct from the MSE-based `ChangeDetector` used for keyframe selection. MSE detects visual change; Wasserstein-1 detects depth distribution shift.

### Segment rules

1. Scene boundary triggered when `depth_distribution_distance` > `scene_threshold` between consecutive keyframes
2. Minimum segment length: 5 keyframes. Segments shorter than this are merged with the previous segment (too few frames for good LUT cell coverage)
3. Scene cuts reset the `ChangeDetector` reference (existing behavior)

### What is rebuilt at segment boundaries

- 32-zone base StreamingLUT (re-accumulate from new segment's keyframes)
- HSL corrections (re-derive from new segment's LUT-vs-RAUNE residuals)
- Cost: ~100ms per segment boundary

## Zone Boundary Smoothing

Weighted moving average over 3 consecutive keyframes:

```
smoothed[n] = 0.6 × raw[n] + 0.2 × raw[n-1] + 0.2 × raw[n+1]
```

- Center keyframe gets dominant weight (0.6) — stays adaptive to local scene
- Adjacent keyframes dampen single-frame outliers (e.g., fish darting in front of camera)
- At segment boundaries: don't smooth across segments. Use only same-segment neighbors with re-normalized weights
- First/last keyframe in segment: use available neighbors only, re-normalize

At 120fps with ~6–12 frame keyframe intervals, 3 keyframes spans ~150–300ms — enough to filter outliers without lagging real scene evolution.

## Pipelined Combined Texture Build

Combined GPU textures (8 zones × 97³ × float4 = ~91MB each) cannot all be pre-stored in VRAM. Instead, use double-buffered pipelining:

### Double-buffer mechanism

Two texture sets (A and B) are permanently allocated in VRAM (~182MB total). Two CUDA streams operate concurrently:

```
Stream A (grading):   grade frames 0-5 ──────→ grade frames 6-11 ──────→ ...
                      using texture set A       using sets A+B (blend)

Stream B (build):     build KF_2 into set B ──→ build KF_3 into set A ──→ ...
                      (50ms, hidden behind       (50ms, hidden behind
                       120-240ms of grading)      grading work)
```

### Timing

- Frames between keyframes: ~6–12 frames × ~20ms grading = 120–240ms
- Combined texture build: ~50ms (kernel <1ms + cuMemcpy3D ~20ms + overhead)
- Build is fully hidden behind grading work — zero visible stall

### At keyframe boundaries

1. Swap active texture designation (pointer swap, instant)
2. Upload new zone boundaries to device (9 floats, instant)
3. If entering new segment: upload new 32-zone base LUT to device (~14MB, ~2ms)
4. Stream B begins building next keyframe's combined texture

## Per-Frame Grading Kernel

### Dual-texture temporal blending

Each non-keyframe frame sits between two keyframes with potentially different zone boundaries and textures. The grading kernel samples both and blends by temporal position:

```
For frame at position t (0.0 = KF_before, 1.0 = KF_after):
  For each pixel:
    color_a = sample_combined_texture(texture_A, zones_A, pixel_rgb, pixel_depth)
    color_b = sample_combined_texture(texture_B, zones_B, pixel_rgb, pixel_depth)
    output = color_a * (1 - t) + color_b * t
```

Each `sample_combined_texture` call does the existing zone-blending logic: soft triangular weights based on depth proximity to zone centers, hardware trilinear tex3D sampling per zone.

### Performance

- Current kernel: ~10–20ms at 1080p, ~80ms at 4K (single texture set)
- New kernel: ~2× texture lookups → ~20–35ms at 1080p, ~140ms at 4K
- On keyframes (t = 0.0): kernel can skip texture B sampling entirely — same speed as current

## VRAM Budget

| Component | Size | Notes |
|-----------|------|-------|
| 32-zone base LUT (device) | ~14MB | Uploaded per-segment |
| Combined texture set A | ~91MB | Permanent allocation |
| Combined texture set B | ~91MB | Permanent allocation |
| Zone boundaries (2 sets) | ~72 bytes | Trivial |
| Frame I/O buffers | ~50MB | Existing (pixels + depth + output) |
| **Total grading VRAM** | **~246MB** | |
| RTX 3060 available | 6GB | Models unloaded before grading |

Comfortable fit. Depth model (~1.5GB) and RAUNE (~3GB) are unloaded after Pass 0 inference completes, before grading begins.

## Configuration

### `dorea.toml` fields under `[grade]`

```toml
[grade]
depth_zones = 8              # runtime zones per keyframe (existing, unchanged meaning)
base_lut_zones = 32          # fine zones for segment-level base LUT (new)
scene_threshold = 0.15       # Wasserstein-1 distance for segment boundary (new)
min_segment_keyframes = 5    # minimum KFs per segment before merge (new)
zone_smoothing_window = 3    # boundary smoothing kernel width, 1 = no smoothing (new)
```

### CLI flags on `dorea grade`

```
--base-lut-zones <N>         Override base LUT zone count (default: config or 32)
--scene-threshold <F>        Override scene boundary threshold (default: config or 0.15)
```

## Files Changed

| File | Change |
|------|--------|
| `crates/dorea-lut/src/build.rs` | `StreamingLutBuilder` accepts `base_zones` separately from runtime zones. New `depth_distribution_distance()` function. |
| `crates/dorea-cli/src/grade.rs` | Pre-compute depth timeline (per-KF zones, segments). Pipelined double-buffer texture build. Dual-stream grading pass. Remove global reservoir sampling. |
| `crates/dorea-cli/src/change_detect.rs` | Add `depth_distribution_distance()` for scene segment detection. |
| `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu` | Build 8 runtime zones reading from 32-zone base LUT (different zone counts for input vs output). |
| `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu` | Dual-texture sampling with blend factor `t`. Two zone boundary arrays. Skip texture B when `t = 0.0`. |
| `crates/dorea-gpu/src/combined_lut.rs` | Double-buffer texture management. Async build on stream B. Swap API. |
| `crates/dorea-cli/src/config.rs` | New fields: `base_lut_zones`, `scene_threshold`, `min_segment_keyframes`, `zone_smoothing_window`. |
| `dorea.toml` | New defaults under `[grade]`. |

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Pre-compute phase | N/A | +~500ms one-time |
| Per-frame grading (non-KF) | ~20ms @ 1080p | ~35ms @ 1080p (2× texture sampling) |
| Per-frame grading (KF, t=0) | ~20ms @ 1080p | ~20ms @ 1080p (unchanged) |
| Combined texture rebuild | Once at startup | Per-keyframe, pipelined (hidden) |
| Segment LUT rebuild | N/A | ~100ms per boundary (hidden in pre-compute) |
| **Net grading pass impact** | Baseline | **~50–70% slower per non-KF frame** |

The throughput cost comes from dual-texture sampling — every non-keyframe pixel reads from two texture sets instead of one. This is the price of accurate depth-adaptive grading.

## Supersedes

- Global reservoir-sampled `adaptive_zone_boundaries()` call in `grade.rs`
- Single-pass `StreamingLutBuilder` with unified zone count
- Single combined texture set in `CombinedLut`
- `--depth-zones` as the only zone configuration knob

## Depends On

- Unified keyframe sampling (2026-04-04) — provides all keyframe depths before grading
- GPU-space IPC / combined LUT architecture — provides the `build_combined_lut` kernel and texture infrastructure
- `ChangeDetector` trait — provides per-keyframe scores for segment boundary detection
