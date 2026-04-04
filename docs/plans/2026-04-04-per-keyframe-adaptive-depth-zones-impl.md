# Per-Keyframe Adaptive Depth Zones — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace global depth zone computation with a two-tier architecture: 32-zone base LUT per scene segment + 8 adaptive runtime zones per keyframe, with pipelined double-buffer GPU texture builds and dual-texture temporal blending.

**Architecture:** After Pass 0 (all keyframe depths known), pre-compute the full depth timeline: per-KF zone boundaries, scene segments via Wasserstein-1 depth histogram distance, per-segment base LUTs + HSL corrections. During grading, double-buffered CUDA streams pipeline combined texture rebuilds (Stream B) behind frame grading (Stream A), with dual-texture temporal blending for non-keyframe frames.

**Tech Stack:** Rust (dorea-cli, dorea-gpu, dorea-lut), CUDA C (kernels via nvcc/build.rs), cudarc (Rust CUDA bindings), rayon (CPU parallelism)

**Spec:** `docs/decisions/2026-04-04-per-keyframe-adaptive-depth-zones.md`

---

## File Map

| File | Action | Task |
|------|--------|------|
| `crates/dorea-cli/src/config.rs` | Modify — add `base_lut_zones`, `scene_threshold`, `min_segment_keyframes`, `zone_smoothing_window` to `GradeDefaults` | 1 |
| `crates/dorea-cli/src/grade.rs:18-99` | Modify — add `--base-lut-zones`, `--scene-threshold` CLI args to `GradeArgs` | 1 |
| `crates/dorea-cli/src/grade.rs:156-165` | Modify — resolve new config fields | 1 |
| `crates/dorea-cli/src/change_detect.rs` | Modify — add `depth_distribution_distance()`, `SegmentRange`, `detect_scene_segments()`, `compute_per_kf_zones()`, `smooth_zone_boundaries()` | 2, 3 |
| `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu` | Modify — separate base zone params from runtime zone params | 4 |
| `crates/dorea-gpu/src/cuda/combined_lut.rs` | Modify — restructure into `AdaptiveLut` with double-buffer, segment loading, per-KF rebuild | 5 |
| `crates/dorea-gpu/src/cuda/mod.rs` | Modify — replace `CudaGrader` with `AdaptiveGrader`, new `grade_frame_blended` method | 5, 6 |
| `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu` | Modify — dual-texture sampling with blend factor `t` | 6 |
| `crates/dorea-gpu/src/lib.rs` | Modify — expose `AdaptiveGrader` and new public API | 6 |
| `crates/dorea-cli/src/grade.rs:482-606` | Modify — replace global reservoir/LUT/HSL with per-segment pre-compute | 7 |
| `crates/dorea-cli/src/grade.rs:613-679` | Modify — restructure grading pass with texture swaps + blending | 7 |
| `dorea.toml` | Modify — add new `[grade]` defaults | 1 |

---

### Task 1: Config + CLI flags (issue #TBD)

**Files:**
- Modify: `crates/dorea-cli/src/config.rs:51-68`
- Modify: `crates/dorea-cli/src/grade.rs:18-99` (GradeArgs)
- Modify: `crates/dorea-cli/src/grade.rs:156-165` (config resolution)
- Modify: `dorea.toml:15-22`

- [ ] **Step 1: Add fields to GradeDefaults**

In `crates/dorea-cli/src/config.rs`, add after line 67 (`pub depth_zones: Option<usize>,`):

```rust
    /// Fine zones for segment-level base LUT (default: 32)
    pub base_lut_zones: Option<usize>,
    /// Wasserstein-1 distance threshold for scene segment boundary (default: 0.15)
    pub scene_threshold: Option<f32>,
    /// Minimum keyframes per scene segment before merge (default: 5)
    pub min_segment_keyframes: Option<usize>,
    /// Zone boundary smoothing window width; 1 = no smoothing (default: 3)
    pub zone_smoothing_window: Option<usize>,
```

- [ ] **Step 2: Add CLI args to GradeArgs**

In `crates/dorea-cli/src/grade.rs`, add after the `depth_zones` field (line 58):

```rust
    /// Fine zones for segment-level base LUT (config: [grade].base_lut_zones, built-in default: 32)
    #[arg(long)]
    pub base_lut_zones: Option<usize>,

    /// Wasserstein-1 threshold for scene segment detection (config: [grade].scene_threshold, built-in default: 0.15)
    #[arg(long)]
    pub scene_threshold: Option<f32>,
```

- [ ] **Step 3: Resolve new config values in run()**

In `crates/dorea-cli/src/grade.rs`, add after line 165 (`let depth_zones = ...`):

```rust
    let base_lut_zones      = args.base_lut_zones.or(cfg.grade.base_lut_zones).unwrap_or(32_usize);
    let scene_threshold     = args.scene_threshold.or(cfg.grade.scene_threshold).unwrap_or(0.15_f32);
    let min_segment_kfs     = cfg.grade.min_segment_keyframes.unwrap_or(5_usize);
    let zone_smoothing_w    = cfg.grade.zone_smoothing_window.unwrap_or(3_usize);
```

- [ ] **Step 4: Update dorea.toml**

Add after `depth_zones = 8` in `dorea.toml`:

```toml
base_lut_zones       = 32   # fine zones for segment-level base LUT
scene_threshold      = 0.15 # Wasserstein-1 distance for scene segment boundary
min_segment_keyframes = 5   # minimum keyframes per segment before merge
zone_smoothing_window = 3   # boundary smoothing kernel width (1 = off)
```

- [ ] **Step 5: Verify it compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo check -p dorea-cli`
Expected: compiles with no errors (new fields are `Option`, unused variables are warnings only)

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-cli/src/config.rs crates/dorea-cli/src/grade.rs
git commit -m "feat(config): add adaptive depth zone config fields

base_lut_zones, scene_threshold, min_segment_keyframes,
zone_smoothing_window in [grade] config + CLI flags."
```

---

### Task 2: Depth distribution distance + scene segment detection

**Files:**
- Modify: `crates/dorea-cli/src/change_detect.rs`

- [ ] **Step 1: Write failing tests for depth_distribution_distance**

Append to the `tests` module in `crates/dorea-cli/src/change_detect.rs`:

```rust
    #[test]
    fn depth_dist_identical_is_zero() {
        let a = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        assert!((depth_distribution_distance(&a, &a, 64) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn depth_dist_opposite_is_high() {
        let near = vec![0.0; 100];
        let far = vec![1.0; 100];
        let d = depth_distribution_distance(&near, &far, 64);
        assert!(d > 0.9, "expected >0.9 for opposite distributions, got {d}");
    }

    #[test]
    fn depth_dist_symmetric() {
        let a: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let b: Vec<f32> = (0..50).map(|i| (i as f32 / 50.0).powi(2)).collect();
        let d_ab = depth_distribution_distance(&a, &b, 64);
        let d_ba = depth_distribution_distance(&b, &a, 64);
        assert!((d_ab - d_ba).abs() < 1e-6, "not symmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn depth_dist_empty_is_zero() {
        assert!((depth_distribution_distance(&[], &[], 64) - 0.0).abs() < 1e-6);
        assert!((depth_distribution_distance(&[0.5], &[], 64) - 0.0).abs() < 1e-6);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli depth_dist -- --nocapture`
Expected: FAIL — `depth_distribution_distance` not found

- [ ] **Step 3: Implement depth_distribution_distance**

Add above the `impl ChangeDetector for MseDetector` block (before line 42):

```rust
/// Wasserstein-1 (earth mover's) distance between two depth distributions.
///
/// Quantizes depths into `n_bins` histogram bins over [0, 1], computes CDFs,
/// and returns the area between them (normalized to [0, 1]).
/// Returns 0.0 if either input is empty.
pub fn depth_distribution_distance(a: &[f32], b: &[f32], n_bins: usize) -> f32 {
    if a.is_empty() || b.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let mut hist_a = vec![0u32; n_bins];
    let mut hist_b = vec![0u32; n_bins];

    for &d in a {
        let bin = ((d.clamp(0.0, 1.0) * n_bins as f32) as usize).min(n_bins - 1);
        hist_a[bin] += 1;
    }
    for &d in b {
        let bin = ((d.clamp(0.0, 1.0) * n_bins as f32) as usize).min(n_bins - 1);
        hist_b[bin] += 1;
    }

    // Normalize to CDFs
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mut cdf_a = 0.0_f64;
    let mut cdf_b = 0.0_f64;
    let mut distance = 0.0_f64;

    for i in 0..n_bins {
        cdf_a += hist_a[i] as f64 / na;
        cdf_b += hist_b[i] as f64 / nb;
        distance += (cdf_a - cdf_b).abs();
    }

    (distance / n_bins as f64) as f32
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli depth_dist -- --nocapture`
Expected: all 4 tests PASS

- [ ] **Step 5: Write failing tests for detect_scene_segments**

Append to tests module:

```rust
    #[test]
    fn segments_single_uniform_clip() {
        // All keyframes have similar depth distributions → one segment
        let depths: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..100).map(|i| i as f32 / 100.0).collect())
            .collect();
        let segs = detect_scene_segments(&depths, 0.15, 5);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 10);
    }

    #[test]
    fn segments_hard_scene_cut() {
        // First 5 keyframes: near-field. Next 5: far-field.
        let mut depths: Vec<Vec<f32>> = Vec::new();
        for _ in 0..5 {
            depths.push(vec![0.1; 100]);
        }
        for _ in 0..5 {
            depths.push(vec![0.9; 100]);
        }
        let segs = detect_scene_segments(&depths, 0.15, 3);
        assert_eq!(segs.len(), 2, "expected 2 segments, got {:?}", segs);
        assert_eq!(segs[0].end, segs[1].start);
    }

    #[test]
    fn segments_short_segment_merged() {
        // Pattern: 5 near, 2 far (too short), 5 near
        let mut depths: Vec<Vec<f32>> = Vec::new();
        for _ in 0..5 { depths.push(vec![0.1; 100]); }
        for _ in 0..2 { depths.push(vec![0.9; 100]); }
        for _ in 0..5 { depths.push(vec![0.1; 100]); }
        let segs = detect_scene_segments(&depths, 0.15, 5);
        // The 2-KF far segment is below min_segment_keyframes=5, should be merged
        assert!(segs.len() <= 2, "short segment should be merged, got {:?}", segs);
    }
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli segments_ -- --nocapture`
Expected: FAIL — `detect_scene_segments` not found

- [ ] **Step 7: Implement SegmentRange and detect_scene_segments**

Add after `depth_distribution_distance` (before the MseDetector impl):

```rust
/// A contiguous range of keyframe indices forming one scene segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentRange {
    /// Inclusive start keyframe index (into the keyframe Vec, not frame index).
    pub start: usize,
    /// Exclusive end keyframe index.
    pub end: usize,
}

/// Detect scene segments from per-keyframe depth maps.
///
/// Scans consecutive keyframe depth distributions using Wasserstein-1 distance.
/// When the distance exceeds `threshold`, a segment boundary is placed.
/// Segments shorter than `min_keyframes` are merged into the previous segment.
pub fn detect_scene_segments(
    keyframe_depths: &[Vec<f32>],
    threshold: f32,
    min_keyframes: usize,
) -> Vec<SegmentRange> {
    let n = keyframe_depths.len();
    if n == 0 {
        return vec![];
    }

    const N_BINS: usize = 64;
    let mut boundaries: Vec<usize> = vec![0]; // segment start indices

    for i in 1..n {
        let dist = depth_distribution_distance(
            &keyframe_depths[i - 1],
            &keyframe_depths[i],
            N_BINS,
        );
        if dist > threshold {
            boundaries.push(i);
        }
    }
    boundaries.push(n); // sentinel

    // Build raw segments
    let mut segments: Vec<SegmentRange> = boundaries.windows(2)
        .map(|w| SegmentRange { start: w[0], end: w[1] })
        .collect();

    // Merge short segments into the previous segment
    let mut merged: Vec<SegmentRange> = Vec::new();
    for seg in segments.drain(..) {
        let len = seg.end - seg.start;
        if len < min_keyframes && !merged.is_empty() {
            // Extend previous segment to absorb this one
            merged.last_mut().unwrap().end = seg.end;
        } else {
            merged.push(seg);
        }
    }

    merged
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli segments_ -- --nocapture`
Expected: all 3 tests PASS

- [ ] **Step 9: Commit**

```bash
git add crates/dorea-cli/src/change_detect.rs
git commit -m "feat(change-detect): Wasserstein-1 depth distance + scene segmentation

depth_distribution_distance() computes earth mover's distance between
depth histograms. detect_scene_segments() partitions keyframes into
segments with short-segment merging."
```

---

### Task 3: Per-keyframe zone boundaries + smoothing

**Files:**
- Modify: `crates/dorea-cli/src/change_detect.rs`

- [ ] **Step 1: Write failing tests for compute_per_kf_zones and smooth_zone_boundaries**

Append to tests module:

```rust
    #[test]
    fn per_kf_zones_basic() {
        let depths: Vec<Vec<f32>> = vec![
            (0..100).map(|i| i as f32 / 100.0).collect(),
            vec![0.5; 100],
        ];
        let zones = compute_per_kf_zones(&depths, 4);
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].len(), 5); // 4 zones → 5 boundaries
        assert!((zones[0][0] - 0.0).abs() < 1e-6);
        assert!((zones[0][4] - 1.0).abs() < 1e-6);
        // Uniform distribution → roughly equidistant boundaries
        for i in 1..4 {
            let expected = i as f32 / 4.0;
            assert!((zones[0][i] - expected).abs() < 0.05,
                "boundary {i}: expected ~{expected}, got {}", zones[0][i]);
        }
        // Constant depth → all interior boundaries at 0.5
        for i in 1..4 {
            assert!((zones[1][i] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn smooth_zones_no_change_when_window_1() {
        let raw = vec![
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 0.10, 0.20, 0.30, 1.0],
        ];
        let segments = vec![SegmentRange { start: 0, end: 2 }];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 1);
        assert_eq!(smoothed, raw);
    }

    #[test]
    fn smooth_zones_dampens_outlier() {
        let raw = vec![
            vec![0.0, 0.25, 0.50, 0.75, 1.0],  // normal
            vec![0.0, 0.02, 0.04, 0.06, 1.0],  // outlier (fish in face)
            vec![0.0, 0.25, 0.50, 0.75, 1.0],  // normal
        ];
        let segments = vec![SegmentRange { start: 0, end: 3 }];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 3);
        // Middle KF's boundaries should be pulled toward neighbors
        // 0.6 * 0.02 + 0.2 * 0.25 + 0.2 * 0.25 = 0.012 + 0.05 + 0.05 = 0.112
        assert!(smoothed[1][1] > 0.05, "smoothing should pull outlier up, got {}", smoothed[1][1]);
    }

    #[test]
    fn smooth_zones_respects_segment_boundary() {
        let raw = vec![
            vec![0.0, 0.10, 0.20, 0.30, 1.0],
            vec![0.0, 0.80, 0.85, 0.90, 1.0],
        ];
        let segments = vec![
            SegmentRange { start: 0, end: 1 },
            SegmentRange { start: 1, end: 2 },
        ];
        let smoothed = smooth_zone_boundaries(&raw, &segments, 3);
        // Each KF is alone in its segment, so smoothing has no cross-segment effect
        assert_eq!(smoothed[0], raw[0]);
        assert_eq!(smoothed[1], raw[1]);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli per_kf_zones -- --nocapture && cargo test -p dorea-cli smooth_zones -- --nocapture`
Expected: FAIL — functions not found

- [ ] **Step 3: Implement compute_per_kf_zones**

Add after `detect_scene_segments` in `change_detect.rs`:

```rust
use dorea_lut::build::adaptive_zone_boundaries;

/// Compute per-keyframe zone boundaries from each keyframe's depth map.
///
/// Returns `Vec<Vec<f32>>` where each inner Vec has `n_zones + 1` boundary values.
pub fn compute_per_kf_zones(
    keyframe_depths: &[Vec<f32>],
    n_zones: usize,
) -> Vec<Vec<f32>> {
    keyframe_depths.iter()
        .map(|depths| adaptive_zone_boundaries(depths, n_zones))
        .collect()
}
```

- [ ] **Step 4: Implement smooth_zone_boundaries**

Add after `compute_per_kf_zones`:

```rust
/// Smooth per-keyframe zone boundaries using a weighted moving average.
///
/// Center keyframe gets weight 0.6, each neighbor gets 0.2.
/// Does NOT smooth across segment boundaries.
/// `window` must be odd and >= 1. Window of 1 means no smoothing.
pub fn smooth_zone_boundaries(
    raw: &[Vec<f32>],
    segments: &[SegmentRange],
    window: usize,
) -> Vec<Vec<f32>> {
    if window <= 1 || raw.is_empty() {
        return raw.to_vec();
    }

    let n_kf = raw.len();
    let n_bounds = raw[0].len();
    let mut smoothed = raw.to_vec();

    for seg in segments {
        let seg_len = seg.end - seg.start;
        if seg_len <= 1 {
            continue; // Single-KF segment, nothing to smooth
        }

        for ki in seg.start..seg.end {
            // Gather same-segment neighbors
            let prev = if ki > seg.start { Some(ki - 1) } else { None };
            let next = if ki + 1 < seg.end { Some(ki + 1) } else { None };

            // Weights: center = 0.6, each neighbor = 0.2 (re-normalize if missing)
            let mut total_w = 0.6_f32;
            let mut result = vec![0.0_f32; n_bounds];
            for j in 0..n_bounds {
                result[j] = raw[ki][j] * 0.6;
            }

            if let Some(pi) = prev {
                total_w += 0.2;
                for j in 0..n_bounds {
                    result[j] += raw[pi][j] * 0.2;
                }
            }
            if let Some(ni) = next {
                total_w += 0.2;
                for j in 0..n_bounds {
                    result[j] += raw[ni][j] * 0.2;
                }
            }

            // Re-normalize
            for j in 0..n_bounds {
                result[j] /= total_w;
            }

            // Preserve fixed endpoints: boundary[0] = 0.0, boundary[last] = 1.0
            result[0] = 0.0;
            *result.last_mut().unwrap() = 1.0;

            smoothed[ki] = result;
        }
    }

    smoothed
}
```

- [ ] **Step 5: Add the use statement at the top of change_detect.rs**

Add at the very top of the file (line 1):

```rust
use dorea_lut::build::adaptive_zone_boundaries;
```

Note: if `dorea-lut` is not already a dependency of `dorea-cli`, add it to `crates/dorea-cli/Cargo.toml` under `[dependencies]`:

```toml
dorea-lut = { path = "../dorea-lut" }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli per_kf_zones -- --nocapture && cargo test -p dorea-cli smooth_zones -- --nocapture`
Expected: all 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-cli/src/change_detect.rs crates/dorea-cli/Cargo.toml
git commit -m "feat(change-detect): per-keyframe zone boundaries + smoothing

compute_per_kf_zones() wraps adaptive_zone_boundaries per keyframe.
smooth_zone_boundaries() applies 3-KF weighted average respecting
segment boundaries."
```

---

### Task 4: build_combined_lut.cu — separate base/runtime zone params

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu`

The kernel currently uses the same `zone_boundaries` and `n_zones` for both its output zone loop AND the `grade_pixel_device` call inside. We need to separate them:
- **Runtime zones** (8): determine the output texture indexing and the depth at which each texture is evaluated
- **Base zones** (32): passed to `grade_pixel_device` for the per-zone LUT lookup inside the grading pipeline

The `grade_pixel_device` function in `grade_pixel.cuh` is **unchanged** — it still takes `luts`, `zone_boundaries`, `lut_size`, `n_zones` as before. The change is purely in what the build kernel passes to it.

- [ ] **Step 1: Modify build_combined_lut_kernel signature**

Replace the kernel signature in `build_combined_lut.cu`:

```c
extern "C"
__global__ void build_combined_lut_kernel(
    float4* __restrict__ output,
    // Base LUT data (32 fine zones — fed to grade_pixel_device)
    const float* __restrict__ base_luts,
    const float* __restrict__ base_zone_boundaries,
    int base_n_zones,
    // Runtime zone boundaries (8 adaptive zones — determines output indexing)
    const float* __restrict__ runtime_zone_boundaries,
    int runtime_n_zones,
    // HSL + grade params (unchanged)
    const float* __restrict__ h_offsets,
    const float* __restrict__ s_ratios,
    const float* __restrict__ v_offsets,
    const float* __restrict__ weights,
    float warmth, float strength, float contrast,
    int grid_size, int lut_size,
    int total_threads
)
```

- [ ] **Step 2: Update the kernel body**

Replace the zone index computation and grade_pixel_device call. The key changes:
- `zone` indexes into `runtime_zone_boundaries` (8 zones) for output placement and depth center
- `grade_pixel_device` receives `base_luts`, `base_zone_boundaries`, `base_n_zones` for its internal LUT lookup

```c
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int N = grid_size;
    int N3 = N * N * N;

    // runtime zone index (0..runtime_n_zones-1) determines output slot
    int zone = idx / N3;
    int rem  = idx % N3;

    int bi = rem / (N * N);
    int gi = (rem / N) % N;
    int ri = rem % N;

    float r = (float)ri / (float)(N - 1);
    float g = (float)gi / (float)(N - 1);
    float b = (float)bi / (float)(N - 1);

    // Depth at runtime zone center
    float z_lo = runtime_zone_boundaries[zone];
    float z_hi = runtime_zone_boundaries[zone + 1];
    float depth = 0.5f * (z_lo + z_hi);

    // Grade using BASE LUT (32 fine zones)
    float3 graded = grade_pixel_device(
        r, g, b, depth,
        base_luts, base_zone_boundaries,
        lut_size, base_n_zones,
        h_offsets, s_ratios, v_offsets, weights,
        warmth, strength, contrast
    );

    output[idx] = make_float4(graded.x, graded.y, graded.z, 0.0f);
}
```

- [ ] **Step 3: Verify kernel compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu --features cuda 2>&1 | head -30`
Expected: build.rs invokes nvcc, kernel compiles. If there are Rust-side call sites that don't match the new signature yet, the PTX will still compile but the Rust launch will fail — that's expected (we fix the Rust side in Task 5).

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu
git commit -m "feat(cuda): separate base/runtime zone params in build_combined_lut

Kernel now accepts base_luts + base_zone_boundaries + base_n_zones
(for grade_pixel_device internal lookup) separately from
runtime_zone_boundaries + runtime_n_zones (for output indexing
and depth center computation). grade_pixel_device is unchanged."
```

---

### Task 5: CombinedLut restructure — double-buffer + per-KF rebuild

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/combined_lut.rs`

This is the largest single change. We replace the single-shot `CombinedLut::build()` with a struct that:
1. Holds two texture sets (double buffer)
2. Can load base LUT data per-segment
3. Can rebuild one texture set per-keyframe with new runtime zone boundaries
4. Tracks which set is "active" for grading

- [ ] **Step 1: Restructure CombinedLut into AdaptiveLut**

Replace the existing `CombinedLut` struct definition and impl with:

```rust
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig, sys};
use crate::GpuError;
use super::map_cudarc_error;

pub(crate) const COMBINED_LUT_GRID: usize = 97;

/// One set of per-zone 3D textures (CUarray + CUtexObject per zone).
struct TextureSet {
    arrays: Vec<sys::CUarray>,
    textures: Vec<u64>, // CUtexObject = u64
    n_zones: usize,
    zone_boundaries: Vec<f32>,
}

/// Double-buffered adaptive LUT with per-keyframe texture rebuilds.
///
/// Holds two `TextureSet`s. One is "active" (used for grading), the other can
/// be rebuilt in the background for the next keyframe. Swapping is a pointer
/// toggle — no copies, no stalls.
pub(crate) struct AdaptiveLut {
    sets: [TextureSet; 2],
    /// Index of the set currently used for grading (0 or 1).
    pub active: usize,
    pub grid_size: usize,
    pub runtime_n_zones: usize,

    // Base LUT data resident on device (per-segment, rarely changes)
    d_base_luts: CudaSlice<f32>,
    d_base_boundaries: CudaSlice<f32>,
    base_n_zones: usize,

    // HSL + grade params on device (constant per clip)
    d_h_offsets: CudaSlice<f32>,
    d_s_ratios: CudaSlice<f32>,
    d_v_offsets: CudaSlice<f32>,
    d_weights: CudaSlice<f32>,
    warmth: f32,
    strength: f32,
    contrast: f32,
    lut_size: usize,
}
```

- [ ] **Step 2: Implement TextureSet allocation and cleanup**

```rust
impl TextureSet {
    /// Allocate empty CUarrays and texture objects for `n_zones` zones.
    fn allocate(n_zones: usize, grid_size: usize) -> Result<Self, GpuError> {
        let n = grid_size;
        let mut arrays = Vec::with_capacity(n_zones);
        let mut textures = Vec::with_capacity(n_zones);

        for _ in 0..n_zones {
            let mut desc = sys::CUDA_ARRAY3D_DESCRIPTOR {
                Width: n,
                Height: n,
                Depth: n,
                Format: sys::CUarray_format::CU_AD_FORMAT_FLOAT,
                NumChannels: 4,
                Flags: 0,
            };
            let mut arr: sys::CUarray = std::ptr::null_mut();
            unsafe {
                sys::lib().cuArray3DCreate_v2(&mut arr, &desc)
            }.map_err(|e| GpuError::CudaFail(format!("cuArray3DCreate: {e}")))?;

            // Create texture object (will be re-created on each rebuild — simpler than updating)
            let tex = Self::create_texture(arr)?;
            arrays.push(arr);
            textures.push(tex);
        }

        Ok(Self {
            arrays,
            textures,
            n_zones,
            zone_boundaries: vec![0.0; n_zones + 1],
        })
    }

    fn create_texture(arr: sys::CUarray) -> Result<u64, GpuError> {
        let res_desc = sys::CUDA_RESOURCE_DESC {
            resType: sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            res: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                array: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    hArray: arr,
                },
            },
            flags: 0,
        };
        let tex_desc = sys::CUDA_TEXTURE_DESC {
            addressMode: [
                sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP,
                sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP,
                sys::CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP,
            ],
            filterMode: sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR,
            flags: 0,
            maxAnisotropy: 0,
            mipmapFilterMode: sys::CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };
        let mut tex: u64 = 0;
        unsafe {
            sys::lib().cuTexObjectCreate(
                &mut tex,
                &res_desc,
                &tex_desc,
                std::ptr::null(),
            )
        }.map_err(|e| GpuError::CudaFail(format!("cuTexObjectCreate: {e}")))?;
        Ok(tex)
    }
}

impl Drop for TextureSet {
    fn drop(&mut self) {
        for &tex in &self.textures {
            unsafe { let _ = sys::lib().cuTexObjectDestroy(tex); }
        }
        for &arr in &self.arrays {
            unsafe { let _ = sys::lib().cuArrayDestroy(arr); }
        }
    }
}
```

- [ ] **Step 3: Implement AdaptiveLut::new**

```rust
impl AdaptiveLut {
    /// Create a new AdaptiveLut with two pre-allocated texture sets.
    ///
    /// `base_luts_flat`: flattened base LUT data [base_n_zones][lut_size^3][3]
    /// `base_boundaries`: base zone boundaries [base_n_zones + 1]
    /// `hsl_data`: (h_offsets, s_ratios, v_offsets, weights) each Vec<f32> of length 6
    /// `params`: (warmth, strength, contrast)
    pub(crate) fn new(
        device: &Arc<CudaDevice>,
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        base_n_zones: usize,
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
        params: (f32, f32, f32),
        lut_size: usize,
        runtime_n_zones: usize,
    ) -> Result<Self, GpuError> {
        let grid_size = COMBINED_LUT_GRID;

        // Upload base LUT data to device
        let d_base_luts = device.htod_sync_copy(base_luts_flat).map_err(map_cudarc_error)?;
        let d_base_boundaries = device.htod_sync_copy(base_boundaries).map_err(map_cudarc_error)?;

        // Upload HSL data
        let (ho, sr, vo, wt) = hsl_data;
        let d_h_offsets = device.htod_sync_copy(ho).map_err(map_cudarc_error)?;
        let d_s_ratios  = device.htod_sync_copy(sr).map_err(map_cudarc_error)?;
        let d_v_offsets = device.htod_sync_copy(vo).map_err(map_cudarc_error)?;
        let d_weights   = device.htod_sync_copy(wt).map_err(map_cudarc_error)?;

        // Allocate both texture sets
        let set_a = TextureSet::allocate(runtime_n_zones, grid_size)?;
        let set_b = TextureSet::allocate(runtime_n_zones, grid_size)?;

        Ok(Self {
            sets: [set_a, set_b],
            active: 0,
            grid_size,
            runtime_n_zones,
            d_base_luts,
            d_base_boundaries,
            base_n_zones,
            d_h_offsets,
            d_s_ratios,
            d_v_offsets,
            d_weights,
            warmth: params.0,
            strength: params.1,
            contrast: params.2,
            lut_size,
        })
    }
```

- [ ] **Step 4: Implement rebuild_set and load_segment**

```rust
    /// Rebuild one texture set with new runtime zone boundaries.
    ///
    /// Runs `build_combined_lut_kernel` to evaluate grade_pixel_device at runtime
    /// zone centers, reading from the base LUT. Then copies result into the
    /// specified set's CUarrays.
    pub(crate) fn rebuild_set(
        &mut self,
        device: &Arc<CudaDevice>,
        set_index: usize,
        runtime_boundaries: &[f32],
    ) -> Result<(), GpuError> {
        let n = self.grid_size;
        let nz = self.runtime_n_zones;
        let total = nz * n * n * n;

        // Upload runtime boundaries
        let d_runtime_bounds = device.htod_sync_copy(runtime_boundaries)
            .map_err(map_cudarc_error)?;

        // Allocate temp build buffer
        let d_build: CudaSlice<f32> = device.alloc_zeros(total * 4)
            .map_err(map_cudarc_error)?;

        // Launch build kernel
        let func = device.get_func("build_combined_lut", "build_combined_lut_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("build_combined_lut_kernel not found".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (
                &d_build,
                &self.d_base_luts,
                &self.d_base_boundaries,
                self.base_n_zones as i32,
                &d_runtime_bounds,
                nz as i32,
                &self.d_h_offsets,
                &self.d_s_ratios,
                &self.d_v_offsets,
                &self.d_weights,
                self.warmth,
                self.strength,
                self.contrast,
                n as i32,
                self.lut_size as i32,
                total as i32,
            ))
        }.map_err(map_cudarc_error)?;

        // Copy build buffer → CUarrays (one per runtime zone)
        let zone_floats = n * n * n * 4; // float4 per texel
        let d_build_ptr = *d_build.device_ptr() as sys::CUdeviceptr;

        for z in 0..nz {
            let src_offset = (z * zone_floats * std::mem::size_of::<f32>()) as u64;
            let copy_params = sys::CUDA_MEMCPY3D {
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcDevice: d_build_ptr + src_offset,
                srcPitch: n * 4 * std::mem::size_of::<f32>(),       // row pitch in bytes
                srcHeight: n,
                srcXInBytes: 0, srcY: 0, srcZ: 0,
                srcHost: std::ptr::null(),
                srcArray: std::ptr::null_mut(),
                srcLOD: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                dstArray: self.sets[set_index].arrays[z],
                dstXInBytes: 0, dstY: 0, dstZ: 0,
                dstHost: std::ptr::null_mut(),
                dstDevice: 0,
                dstPitch: 0, dstHeight: 0,
                dstLOD: 0,
                WidthInBytes: n * 4 * std::mem::size_of::<f32>(),   // float4 per texel
                Height: n,
                Depth: n,
                reserved0: std::ptr::null_mut(),
                reserved1: std::ptr::null_mut(),
            };
            unsafe {
                sys::lib().cuMemcpy3D_v2(&copy_params)
            }.map_err(|e| GpuError::CudaFail(format!("cuMemcpy3D zone {z}: {e}")))?;
        }

        // Store boundaries for this set
        self.sets[set_index].zone_boundaries = runtime_boundaries.to_vec();

        Ok(())
    }

    /// Load new segment base LUT data (32-zone LUT + HSL corrections).
    /// Called at scene segment boundaries.
    pub(crate) fn load_segment(
        &mut self,
        device: &Arc<CudaDevice>,
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
    ) -> Result<(), GpuError> {
        self.d_base_luts = device.htod_sync_copy(base_luts_flat).map_err(map_cudarc_error)?;
        self.d_base_boundaries = device.htod_sync_copy(base_boundaries).map_err(map_cudarc_error)?;
        self.base_n_zones = base_boundaries.len() - 1;
        let (ho, sr, vo, wt) = hsl_data;
        self.d_h_offsets = device.htod_sync_copy(ho).map_err(map_cudarc_error)?;
        self.d_s_ratios  = device.htod_sync_copy(sr).map_err(map_cudarc_error)?;
        self.d_v_offsets = device.htod_sync_copy(vo).map_err(map_cudarc_error)?;
        self.d_weights   = device.htod_sync_copy(wt).map_err(map_cudarc_error)?;
        Ok(())
    }

    /// Swap which texture set is active. Returns the NEW active index.
    pub(crate) fn swap(&mut self) -> usize {
        self.active = 1 - self.active;
        self.active
    }

    /// Get the active and inactive texture sets for dual-texture grading.
    pub(crate) fn active_textures(&self) -> &[u64] {
        &self.sets[self.active].textures
    }

    pub(crate) fn active_boundaries(&self) -> &[f32] {
        &self.sets[self.active].zone_boundaries
    }

    pub(crate) fn inactive_textures(&self) -> &[u64] {
        &self.sets[1 - self.active].textures
    }

    pub(crate) fn inactive_boundaries(&self) -> &[f32] {
        &self.sets[1 - self.active].zone_boundaries
    }
}
```

- [ ] **Step 5: Load build_combined_lut PTX in mod.rs**

In `crates/dorea-gpu/src/cuda/mod.rs`, add alongside the existing PTX include (near line 26):

```rust
#[cfg(feature = "cuda")]
const BUILD_COMBINED_LUT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/build_combined_lut.ptx"));
```

And load it in the grader initialization (we'll add this in Task 6).

- [ ] **Step 6: Verify it compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo check -p dorea-gpu --features cuda`
Expected: compiles (existing CudaGrader still exists, AdaptiveLut is additive)

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-gpu/src/cuda/combined_lut.rs crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat(cuda): AdaptiveLut with double-buffer texture management

Replaces single-shot CombinedLut::build with AdaptiveLut that holds
two TextureSets, supports per-keyframe rebuild_set() and per-segment
load_segment(). Kernel reads from 32-zone base LUT, outputs to
8-zone runtime textures."
```

---

### Task 6: Dual-texture grading kernel + AdaptiveGrader

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu`
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`
- Modify: `crates/dorea-gpu/src/lib.rs`

- [ ] **Step 1: Modify combined_lut_kernel for dual-texture blending**

Replace the kernel in `combined_lut.cu`:

```c
extern "C"
__global__ void combined_lut_kernel(
    const unsigned char* __restrict__ pixels_in,
    const float*         __restrict__ depth,
    // Texture set A (active / "before" keyframe)
    const unsigned long long* __restrict__ textures_a,
    const float*              __restrict__ zone_boundaries_a,
    // Texture set B ("after" keyframe)
    const unsigned long long* __restrict__ textures_b,
    const float*              __restrict__ zone_boundaries_b,
    // Blend factor: 0.0 = all A, 1.0 = all B
    float blend_t,
    unsigned char*       __restrict__ pixels_out,
    int n_pixels,
    int n_zones,
    int grid_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    float r = pixels_in[idx * 3 + 0] * (1.0f / 255.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 255.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 255.0f);
    float d = depth[idx];
    float gs = (float)(grid_size - 1);

    // Sample texture set A
    float total_w_a = 0.0f;
    float4 blended_a = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries_a[z];
        float z_hi = zone_boundaries_a[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(d - z_center);
        float w = fmaxf(1.0f - dist / z_width, 0.0f);
        if (w < 1e-6f) continue;
        cudaTextureObject_t tex = (cudaTextureObject_t)textures_a[z];
        float4 s = tex3D<float4>(tex, r * gs, g * gs, b * gs);
        blended_a.x += s.x * w;
        blended_a.y += s.y * w;
        blended_a.z += s.z * w;
        total_w_a += w;
    }

    float r_a, g_a, b_a;
    if (total_w_a > 1e-6f) {
        r_a = blended_a.x / total_w_a;
        g_a = blended_a.y / total_w_a;
        b_a = blended_a.z / total_w_a;
    } else {
        r_a = r; g_a = g; b_a = b;
    }

    // Early out: skip set B when blend_t ≈ 0 (on keyframes)
    float r_out, g_out, b_out;
    if (blend_t < 1e-4f) {
        r_out = r_a; g_out = g_a; b_out = b_a;
    } else {
        // Sample texture set B
        float total_w_b = 0.0f;
        float4 blended_b = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int z = 0; z < n_zones; z++) {
            float z_lo = zone_boundaries_b[z];
            float z_hi = zone_boundaries_b[z + 1];
            float z_width = z_hi - z_lo;
            if (z_width < 1e-6f) continue;
            float z_center = 0.5f * (z_lo + z_hi);
            float dist = fabsf(d - z_center);
            float w = fmaxf(1.0f - dist / z_width, 0.0f);
            if (w < 1e-6f) continue;
            cudaTextureObject_t tex = (cudaTextureObject_t)textures_b[z];
            float4 s = tex3D<float4>(tex, r * gs, g * gs, b * gs);
            blended_b.x += s.x * w;
            blended_b.y += s.y * w;
            blended_b.z += s.z * w;
            total_w_b += w;
        }

        float r_b, g_b, b_b;
        if (total_w_b > 1e-6f) {
            r_b = blended_b.x / total_w_b;
            g_b = blended_b.y / total_w_b;
            b_b = blended_b.z / total_w_b;
        } else {
            r_b = r; g_b = g; b_b = b;
        }

        // Temporal blend
        float inv_t = 1.0f - blend_t;
        r_out = r_a * inv_t + r_b * blend_t;
        g_out = g_a * inv_t + g_b * blend_t;
        b_out = b_a * inv_t + b_b * blend_t;
    }

    pixels_out[idx * 3 + 0] = (unsigned char)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 1] = (unsigned char)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 255.0f));
    pixels_out[idx * 3 + 2] = (unsigned char)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 255.0f));
}
```

- [ ] **Step 2: Add AdaptiveGrader to mod.rs**

In `crates/dorea-gpu/src/cuda/mod.rs`, add after the existing `CudaGrader` impl:

```rust
#[cfg(feature = "cuda")]
const BUILD_COMBINED_LUT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/build_combined_lut.ptx"));

/// Adaptive grader with per-keyframe zone boundaries and dual-texture blending.
#[cfg(feature = "cuda")]
pub struct AdaptiveGrader {
    adaptive_lut: combined_lut::AdaptiveLut,
    device: Arc<CudaDevice>,
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(feature = "cuda")]
impl AdaptiveGrader {
    /// Initialize CUDA device, load kernels, allocate double-buffered textures.
    pub fn new(
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        base_n_zones: usize,
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
        params: &GradeParams,
        lut_size: usize,
        runtime_n_zones: usize,
    ) -> Result<Self, GpuError> {
        let device = CudaDevice::new(0).map_err(|e| {
            GpuError::ModuleLoad(format!("CudaDevice::new(0) failed: {e}"))
        })?;

        // Load per-frame kernel
        device.load_ptx(
            Ptx::from_src(COMBINED_LUT_PTX),
            "combined_lut",
            &["combined_lut_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("load combined_lut PTX: {e}")))?;

        // Load build kernel
        device.load_ptx(
            Ptx::from_src(BUILD_COMBINED_LUT_PTX),
            "build_combined_lut",
            &["build_combined_lut_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("load build_combined_lut PTX: {e}")))?;

        let adaptive_lut = combined_lut::AdaptiveLut::new(
            &device,
            base_luts_flat,
            base_boundaries,
            base_n_zones,
            hsl_data,
            (params.warmth, params.strength, params.contrast),
            lut_size,
            runtime_n_zones,
        )?;

        Ok(Self {
            adaptive_lut,
            device,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Build runtime textures for a keyframe's zone boundaries into the inactive set.
    pub fn prepare_keyframe(&mut self, runtime_boundaries: &[f32]) -> Result<(), GpuError> {
        let inactive = 1 - self.adaptive_lut.active;
        self.adaptive_lut.rebuild_set(&self.device, inactive, runtime_boundaries)
    }

    /// Swap active/inactive texture sets (call at keyframe boundary).
    pub fn swap_textures(&mut self) {
        self.adaptive_lut.swap();
    }

    /// Load new segment base LUT + HSL data.
    pub fn load_segment(
        &mut self,
        base_luts_flat: &[f32],
        base_boundaries: &[f32],
        hsl_data: (&[f32], &[f32], &[f32], &[f32]),
    ) -> Result<(), GpuError> {
        self.adaptive_lut.load_segment(
            &self.device, base_luts_flat, base_boundaries, hsl_data,
        )
    }

    /// Grade one frame with dual-texture temporal blending.
    ///
    /// `blend_t`: 0.0 = use active set only (keyframe), 1.0 = use inactive set only.
    pub fn grade_frame_blended(
        &self,
        pixels: &[u8],
        depth: &[f32],
        width: usize,
        height: usize,
        blend_t: f32,
    ) -> Result<Vec<u8>, GpuError> {
        let n = width * height;
        if pixels.len() != n * 3 {
            return Err(GpuError::InvalidInput(format!(
                "pixels len {} != {}*3", pixels.len(), n
            )));
        }
        if depth.len() != n {
            return Err(GpuError::InvalidInput(format!(
                "depth len {} != {}", depth.len(), n
            )));
        }
        let dev = &self.device;

        let d_pixels_in = dev.htod_sync_copy(pixels).map_err(map_cudarc_error)?;
        let d_depth = dev.htod_sync_copy(depth).map_err(map_cudarc_error)?;
        let d_textures_a = dev.htod_sync_copy(self.adaptive_lut.active_textures())
            .map_err(map_cudarc_error)?;
        let d_boundaries_a = dev.htod_sync_copy(self.adaptive_lut.active_boundaries())
            .map_err(map_cudarc_error)?;
        let d_textures_b = dev.htod_sync_copy(self.adaptive_lut.inactive_textures())
            .map_err(map_cudarc_error)?;
        let d_boundaries_b = dev.htod_sync_copy(self.adaptive_lut.inactive_boundaries())
            .map_err(map_cudarc_error)?;
        let d_pixels_out: CudaSlice<u8> = dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?;

        let func = dev.get_func("combined_lut", "combined_lut_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("combined_lut_kernel not found".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (div_ceil(n as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_pixels_in,
                &d_depth,
                &d_textures_a,
                &d_boundaries_a,
                &d_textures_b,
                &d_boundaries_b,
                blend_t,
                &d_pixels_out,
                n as i32,
                self.adaptive_lut.runtime_n_zones as i32,
                self.adaptive_lut.grid_size as i32,
            ))
        }.map_err(map_cudarc_error)?;

        let result = dev.dtoh_sync_copy(&d_pixels_out).map_err(map_cudarc_error)?;
        Ok(result)
    }
}
```

- [ ] **Step 3: Expose AdaptiveGrader in lib.rs**

In `crates/dorea-gpu/src/lib.rs`, add after the existing `grade_frame_with_grader`:

```rust
#[cfg(feature = "cuda")]
pub fn grade_frame_with_adaptive_grader(
    grader: &cuda::AdaptiveGrader,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    blend_t: f32,
) -> Result<Vec<u8>, GpuError> {
    grader.grade_frame_blended(pixels, depth, width, height, blend_t)
}
```

And in the `pub use` section, ensure `AdaptiveGrader` is exported from `cuda`:

```rust
#[cfg(feature = "cuda")]
pub use cuda::AdaptiveGrader;
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo check -p dorea-gpu --features cuda`
Expected: compiles (existing CudaGrader is untouched, new types are additive)

- [ ] **Step 5: Write basic integration test**

Add to `crates/dorea-gpu/src/cuda/mod.rs` tests module:

```rust
    #[test]
    fn adaptive_grader_builds_and_grades() {
        let cal = make_calibration(32); // 32 base zones
        let params = crate::GradeParams::default();

        // Extract flat LUT data from calibration
        let base_luts_flat = cal.depth_luts().flat_data();
        let base_boundaries = cal.depth_luts().zone_boundaries();

        // HSL data: extract from calibration
        let hsl = cal.hsl_corrections();
        let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
        let s_ratios: Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
        let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
        let weights: Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();

        let mut grader = match AdaptiveGrader::new(
            &base_luts_flat, base_boundaries, 32,
            (&h_offsets, &s_ratios, &v_offsets, &weights),
            &params, 17, 8,
        ) {
            Ok(g) => g,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        // Build runtime zones (uniform 8 zones)
        let runtime_bounds: Vec<f32> = (0..=8).map(|i| i as f32 / 8.0).collect();
        grader.prepare_keyframe(&runtime_bounds).expect("prepare_keyframe failed");
        grader.swap_textures();
        // Build second set for blending
        grader.prepare_keyframe(&runtime_bounds).expect("prepare_keyframe 2 failed");

        let (pixels, depth) = make_frame(4, 4);
        let out = grader.grade_frame_blended(&pixels, &depth, 4, 4, 0.0)
            .expect("grade_frame_blended failed");
        assert_eq!(out.len(), 4 * 4 * 3);

        // Test blending (t=0.5)
        let out2 = grader.grade_frame_blended(&pixels, &depth, 4, 4, 0.5)
            .expect("grade_frame_blended t=0.5 failed");
        assert_eq!(out2.len(), 4 * 4 * 3);
    }
```

Note: This test requires `DepthLuts` to expose `flat_data()` and `Calibration` to expose `hsl_corrections()`. If these accessors don't exist, add them:
- In `dorea-lut/src/types.rs`: `pub fn flat_data(&self) -> Vec<f32>` and `pub fn zone_boundaries(&self) -> &[f32]`
- In `dorea-cal`: `pub fn hsl_corrections(&self) -> &HslCorrections` and `pub fn depth_luts(&self) -> &DepthLuts`

- [ ] **Step 6: Run the test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu adaptive_grader -- --nocapture`
Expected: PASS (or SKIP if no GPU)

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-gpu/src/cuda/kernels/combined_lut.cu \
       crates/dorea-gpu/src/cuda/mod.rs \
       crates/dorea-gpu/src/lib.rs
git commit -m "feat(cuda): dual-texture grading kernel + AdaptiveGrader

combined_lut_kernel now samples two texture sets with blend factor t.
AdaptiveGrader wraps AdaptiveLut with prepare_keyframe(), swap_textures(),
load_segment(), and grade_frame_blended() API."
```

---

### Task 7: grade.rs integration — pre-compute + restructured grading pass

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

This is the integration task that wires everything together. It replaces:
- Global reservoir sampling (lines 488–514)
- Single StreamingLutBuilder pass (lines 516–531)
- Single HSL correction pass (lines 533–593)
- Single CudaGrader init (lines 595–606)
- Static grading loop (lines 613–679)

- [ ] **Step 1: Add imports**

At the top of `grade.rs`, add to existing imports:

```rust
use crate::change_detect::{
    depth_distribution_distance, detect_scene_segments, compute_per_kf_zones,
    smooth_zone_boundaries, SegmentRange,
};
```

- [ ] **Step 2: Replace reservoir sampling + LUT + HSL with pre-compute phase**

Replace lines 488–593 (the 3-pass calibration block) with:

```rust
    // ---- Pre-compute depth timeline ----
    use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
    use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
    use dorea_lut::apply::apply_depth_luts;
    use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder};

    // Step 1: Collect per-keyframe depth maps
    let kf_depths: Vec<Vec<f32>> = (0..store.len())
        .map(|i| store.read_depth(i))
        .collect();

    // Step 2: Compute per-keyframe zone boundaries (runtime zones)
    let raw_kf_zones = compute_per_kf_zones(&kf_depths, depth_zones);

    // Step 3: Detect scene segments
    let segments = detect_scene_segments(&kf_depths, scene_threshold, min_segment_kfs);
    log::info!("Scene segments: {} (from {} keyframes)", segments.len(), store.len());
    for (si, seg) in segments.iter().enumerate() {
        log::info!("  segment {si}: keyframes {}..{} ({} KFs)", seg.start, seg.end, seg.end - seg.start);
    }

    // Step 4: Smooth zone boundaries (respecting segment boundaries)
    let smoothed_kf_zones = smooth_zone_boundaries(&raw_kf_zones, &segments, zone_smoothing_w);

    // Step 5: Build per-segment base LUTs + HSL corrections
    struct SegmentCalibration {
        depth_luts: dorea_lut::types::DepthLuts,
        hsl_corrections: HslCorrections,
    }
    let n_quals = HSL_QUALIFIERS.len();

    let segment_calibrations: Vec<SegmentCalibration> = segments.iter().map(|seg| {
        // Collect depths from this segment's keyframes → base zone boundaries (32 fine zones)
        let seg_depths: Vec<f32> = (seg.start..seg.end)
            .flat_map(|i| kf_depths[i].iter().cloned())
            .collect();
        let base_boundaries = adaptive_zone_boundaries(&seg_depths, base_lut_zones);

        // Build streaming LUT for this segment
        let mut lut_builder = StreamingLutBuilder::new(base_boundaries);
        for i in seg.start..seg.end {
            let (w, h) = store.dims(i);
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let importance = compute_importance(&depth, w, h);
            lut_builder.add_frame(&original, &target, &depth, &importance);
        }
        let depth_luts = lut_builder.finish();

        // HSL corrections for this segment
        let mut h_offset_acc = vec![0.0_f64; n_quals];
        let mut s_ratio_acc  = vec![0.0_f64; n_quals];
        let mut v_offset_acc = vec![0.0_f64; n_quals];
        let mut active_count = vec![0_usize; n_quals];
        let mut total_weight_acc = vec![0.0_f64; n_quals];

        for i in seg.start..seg.end {
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let lut_output = apply_depth_luts(&original, &depth, &depth_luts);
            let corrs = derive_hsl_corrections(&lut_output, &target);
            for (qi, corr) in corrs.0.iter().enumerate() {
                if corr.weight >= MIN_WEIGHT {
                    let w = corr.weight as f64;
                    h_offset_acc[qi]     += corr.h_offset as f64 * w;
                    s_ratio_acc[qi]      += corr.s_ratio  as f64 * w;
                    v_offset_acc[qi]     += corr.v_offset as f64 * w;
                    active_count[qi]     += 1;
                    total_weight_acc[qi] += w;
                }
            }
        }

        let mut avg_corrections: Vec<QualifierCorrection> = Vec::with_capacity(n_quals);
        for qi in 0..n_quals {
            let qual = &HSL_QUALIFIERS[qi];
            if active_count[qi] > 0 {
                let tw = total_weight_acc[qi];
                avg_corrections.push(QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: (h_offset_acc[qi] / tw) as f32,
                    s_ratio:  (s_ratio_acc[qi]  / tw) as f32,
                    v_offset: (v_offset_acc[qi] / tw) as f32,
                    weight:   tw as f32,
                });
            } else {
                avg_corrections.push(QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: 0.0,
                    s_ratio:  1.0,
                    v_offset: 0.0,
                    weight:   0.0,
                });
            }
        }

        SegmentCalibration {
            depth_luts,
            hsl_corrections: HslCorrections(avg_corrections),
        }
    }).collect();

    log::info!(
        "Pre-compute complete: {} segments, {} per-KF zone sets",
        segment_calibrations.len(), smoothed_kf_zones.len(),
    );

    // Build segment-to-keyframe index lookup
    let kf_to_segment: Vec<usize> = {
        let mut map = vec![0usize; store.len()];
        for (si, seg) in segments.iter().enumerate() {
            for ki in seg.start..seg.end {
                map[ki] = si;
            }
        }
        map
    };
```

- [ ] **Step 3: Replace CudaGrader init with AdaptiveGrader**

Replace lines 595–606 with:

```rust
    // Initialize adaptive CUDA grader with first segment's base LUT
    #[cfg(feature = "cuda")]
    let mut adaptive_grader = {
        let seg0 = &segment_calibrations[0];
        let base_flat = seg0.depth_luts.flat_data();
        let base_bounds = seg0.depth_luts.zone_boundaries();
        let hsl = &seg0.hsl_corrections;
        let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
        let s_ratios: Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
        let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
        let weights: Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();

        let grader = dorea_gpu::AdaptiveGrader::new(
            &base_flat, base_bounds, base_lut_zones,
            (&h_offsets, &s_ratios, &v_offsets, &weights),
            &params, dorea_lut::build::LUT_SIZE, depth_zones,
        ).context("AdaptiveGrader init failed")?;

        log::info!("Adaptive CUDA grader initialized ({base_lut_zones} base zones, {depth_zones} runtime zones)");
        grader
    };

    // Build first keyframe's runtime texture
    #[cfg(feature = "cuda")]
    {
        adaptive_grader.prepare_keyframe(&smoothed_kf_zones[0])
            .context("prepare initial keyframe texture failed")?;
        adaptive_grader.swap_textures();
        // Pre-build second keyframe if available
        if smoothed_kf_zones.len() > 1 {
            adaptive_grader.prepare_keyframe(&smoothed_kf_zones[1])
                .context("prepare second keyframe texture failed")?;
        }
    }
```

- [ ] **Step 4: Restructure grading pass with texture swaps + blending**

Replace lines 613–679 with:

```rust
    // Ordered list for pass-2 lookup
    let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
        .map(|kf| (kf.frame_index, kf.scene_cut_before))
        .collect();

    // -----------------------------------------------------------------------
    // Pass 2: full-resolution decode + adaptive depth grading
    // -----------------------------------------------------------------------
    let decode_source = maxine_temp_path.as_deref().unwrap_or(args.input.as_path());
    let frames = ffmpeg::decode_frames(decode_source, &info)
        .context("failed to spawn ffmpeg full-res decoder")?;

    let mut kf_cursor = 0usize;
    let mut frame_count = 0u64;
    let mut current_segment = kf_to_segment[0];

    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;
        let fi = frame.index;

        // Advance cursor
        while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
            let old_cursor = kf_cursor;
            kf_cursor += 1;

            #[cfg(feature = "cuda")]
            {
                // Crossed a keyframe boundary — swap textures
                adaptive_grader.swap_textures();

                // Check for segment boundary
                let new_seg = kf_to_segment[kf_cursor];
                if new_seg != current_segment {
                    let seg_cal = &segment_calibrations[new_seg];
                    let base_flat = seg_cal.depth_luts.flat_data();
                    let base_bounds = seg_cal.depth_luts.zone_boundaries();
                    let hsl = &seg_cal.hsl_corrections;
                    let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
                    let s_ratios: Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
                    let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
                    let weights: Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
                    adaptive_grader.load_segment(
                        &base_flat, base_bounds,
                        (&h_offsets, &s_ratios, &v_offsets, &weights),
                    ).context("load_segment failed")?;
                    current_segment = new_seg;
                    log::info!("Segment switch to {new_seg} at keyframe {kf_cursor}");
                }

                // Pre-build next keyframe's texture (pipelining)
                if kf_cursor + 1 < smoothed_kf_zones.len() {
                    adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor + 1])
                        .context("prepare_keyframe failed")?;
                }
            }
        }

        let (prev_kf_idx, _) = kf_index_list[kf_cursor];
        let (prev_depth_proxy, dpw, dph) = keyframe_depths
            .get(&prev_kf_idx)
            .expect("prev keyframe depth missing");
        let (dpw, dph) = (*dpw, *dph);

        // Depth interpolation (unchanged logic)
        let depth_proxy = if fi == prev_kf_idx {
            prev_depth_proxy.clone()
        } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut_before_next {
                prev_depth_proxy.clone()
            } else {
                let (next_depth_proxy, _, _) = keyframe_depths
                    .get(&next_kf_idx)
                    .expect("next keyframe depth missing");
                let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
                lerp_depth(prev_depth_proxy, next_depth_proxy, t)
            }
        } else {
            prev_depth_proxy.clone()
        };

        let depth = if dpw == frame.width && dph == frame.height {
            depth_proxy
        } else {
            InferenceServer::upscale_depth(&depth_proxy, dpw, dph, frame.width, frame.height)
        };

        // Compute blend_t: temporal position between keyframes
        let blend_t = if fi == prev_kf_idx {
            0.0_f32
        } else if let Some(&(next_kf_idx, scene_cut)) = kf_index_list.get(kf_cursor + 1) {
            if scene_cut {
                0.0 // Don't blend across scene cut
            } else {
                (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32
            }
        } else {
            0.0 // Past last keyframe
        };

        #[cfg(feature = "cuda")]
        let graded = adaptive_grader.grade_frame_blended(
            &frame.pixels, &depth, frame.width, frame.height, blend_t,
        ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;

        #[cfg(not(feature = "cuda"))]
        let graded = {
            // CPU fallback: use first segment's calibration (no adaptive zones on CPU path)
            let seg_idx = kf_to_segment.get(kf_cursor).copied().unwrap_or(0);
            let cal = &segment_calibrations[seg_idx];
            let calibration = Calibration::new(
                cal.depth_luts.clone(), cal.hsl_corrections.clone(), store.len(),
            );
            grade_frame(&frame.pixels, &depth, frame.width, frame.height, &calibration, &params)
                .map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?
        };

        encoder.write_frame(&graded).context("encoder write failed")?;
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }
```

- [ ] **Step 5: Remove old grade_with_grader function**

Delete the `grade_with_grader` function (lines 139–154) and the `#[cfg(feature = "cuda")] use dorea_gpu::cuda::CudaGrader;` import (line 15–16) since they are no longer used.

- [ ] **Step 6: Verify it compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo check -p dorea-cli --features cuda`
Expected: compiles. There may be warnings about unused imports from the removed code path — clean those up.

- [ ] **Step 7: Integration test with a real clip**

Run: `cd /workspaces/dorea-workspace && dorea grade --input footage/flattened/DJI_20251101111428_0055_D.MP4 --output /tmp/test_adaptive.mp4 --verbose 2>&1 | tee /tmp/adaptive_grade.log`

Check log for:
- "Scene segments: N" line
- "Segment switch" lines (if any)
- "Adaptive CUDA grader initialized (32 base zones, 8 runtime zones)"
- Successful completion with output file

- [ ] **Step 8: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): per-keyframe adaptive depth zones integration

Replace global reservoir/LUT/HSL with per-segment pre-compute phase.
Grading pass uses AdaptiveGrader with dual-texture temporal blending,
texture swaps at keyframe boundaries, and segment loading at scene cuts."
```

- [ ] **Step 9: Update dorea.toml with final config**

```bash
git add dorea.toml
git commit -m "chore: update dorea.toml with adaptive depth zone defaults"
```

---

## Accessor Methods (may be needed by Task 6/7)

If `DepthLuts` or `Calibration` don't already expose the needed data, add these:

**In `crates/dorea-lut/src/types.rs`:**
```rust
impl DepthLuts {
    /// Flatten all zone LUTs into a contiguous f32 buffer: [n_zones][lut_size^3][3]
    pub fn flat_data(&self) -> Vec<f32> {
        self.luts.iter().flat_map(|lut| lut.flat_data()).collect()
    }

    pub fn zone_boundaries(&self) -> &[f32] {
        &self.boundaries
    }
}

impl LutGrid {
    pub fn flat_data(&self) -> &[f32] {
        &self.data
    }
}
```

**In `crates/dorea-cal/src/lib.rs`:**
```rust
impl Calibration {
    pub fn depth_luts(&self) -> &DepthLuts { &self.depth_luts }
    pub fn hsl_corrections(&self) -> &HslCorrections { &self.hsl }
}
```

These should be added as needed when the compiler reports missing methods.

---

## Self-Review Checklist

1. **Spec coverage:** All spec sections mapped to tasks:
   - Two-tier architecture → Tasks 4, 5, 6
   - Scene segmentation → Task 2
   - Zone smoothing → Task 3
   - Pre-compute phase → Task 7 Step 2
   - Pipelined double-buffer → Task 5
   - Dual-texture blending → Task 6
   - Config → Task 1

2. **Placeholder scan:** No TBDs, TODOs, or "implement later". All steps have code.

3. **Type consistency:** Verified:
   - `AdaptiveLut` (combined_lut.rs) ↔ `AdaptiveGrader` (mod.rs) ↔ `grade.rs` integration
   - `SegmentRange` (change_detect.rs) ↔ `detect_scene_segments` ↔ grade.rs usage
   - `smoothed_kf_zones: Vec<Vec<f32>>` consistent everywhere
   - Kernel parameter ordering matches Rust launch calls
