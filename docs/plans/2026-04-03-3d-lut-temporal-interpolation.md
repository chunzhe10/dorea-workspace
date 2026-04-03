# 3D LUT Temporal Interpolation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `grade_frame` on non-keyframe frames with a precomputed 65x65x65 per-depth-band 3D LUT to eliminate the expensive sRGB↔LAB roundtrip (~25ms/frame).

**Architecture:** Build a `GradeLut` (array of 7 zone-aligned 3D LUTs) once after calibration by evaluating the real pipeline on synthetic 1x1 frames. For non-keyframes, trilinear-interpolate each pixel through two bracketing depth-band LUTs and blend. The `grade_frame_inner` function (extracted from existing `grade_frame_cpu`) provides the single-pixel evaluation with skip_clarity support.

**Tech Stack:** Rust, rayon (parallel build + apply), existing dorea-lut `LutGrid` type, existing dorea-cal `Calibration`.

**Spec:** `docs/decisions/2026-04-03-3d-lut-temporal-interpolation.md`

---

### Task 1: Add `skip_clarity` parameter to `grade_frame_cpu`

**Files:**
- Modify: `crates/dorea-gpu/src/cpu.rs:409-449`

- [ ] **Step 1: Add `skip_clarity` param to `grade_frame_cpu`**

Change the signature and pass it through to `finish_grade`:

```rust
/// Full CPU grading pipeline: LUT apply → HSL correct → depth_aware_ambiance → user params.
///
/// When `skip_clarity` is true, the clarity pass (box blur detail extraction) is skipped.
/// Used by the 3D LUT build phase to capture only per-pixel operations.
pub fn grade_frame_cpu(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
    skip_clarity: bool,
) -> Result<Vec<u8>, String> {
    // Convert u8 → f32
    let mut rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // 1. Apply depth-stratified LUT
    let lut_result = apply_depth_luts(
        &rgb_f32
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>(),
        depth,
        &calibration.depth_luts,
    );
    for (i, px) in lut_result.iter().enumerate() {
        rgb_f32[i * 3]     = px[0];
        rgb_f32[i * 3 + 1] = px[1];
        rgb_f32[i * 3 + 2] = px[2];
    }

    // 2. Apply HSL qualifier corrections
    let pixels_arr: Vec<[f32; 3]> = rgb_f32
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let hsl_result = apply_hsl_corrections(&pixels_arr, &calibration.hsl_corrections);
    for (i, px) in hsl_result.iter().enumerate() {
        rgb_f32[i * 3]     = px[0];
        rgb_f32[i * 3 + 1] = px[1];
        rgb_f32[i * 3 + 2] = px[2];
    }

    // 3–5. Ambiance + (optional clarity) + warmth + blend + u8
    Ok(finish_grade(&mut rgb_f32, pixels, depth, width, height, params, calibration, skip_clarity))
}
```

- [ ] **Step 2: Fix the one existing caller**

In `crates/dorea-gpu/src/lib.rs`, `grade_frame_cpu` is not called directly (the CUDA path is used). But check for any test callers and add `false` as the `skip_clarity` argument to maintain existing behavior.

Search for `grade_frame_cpu` across the crate:
```bash
cd /workspaces/dorea-workspace/repos/dorea && grep -rn "grade_frame_cpu" crates/
```

Update each caller to pass `false` (preserve existing behavior).

- [ ] **Step 3: Build and test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release 2>&1 | tail -5
cargo test -p dorea-gpu 2>&1 | tail -10
```

Expected: build succeeds, all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cpu.rs crates/dorea-gpu/src/lib.rs
git commit -m "refactor(dorea-gpu): add skip_clarity param to grade_frame_cpu

Enables the 3D LUT build phase to evaluate the per-pixel pipeline without
the spatial clarity pass. Existing callers pass false to preserve behavior."
```

---

### Task 2: Create `lut3d` module with `GradeLut` struct

**Files:**
- Create: `crates/dorea-gpu/src/lut3d.rs`
- Modify: `crates/dorea-gpu/src/lib.rs` (add `pub mod lut3d;`)

- [ ] **Step 1: Create `lut3d.rs` with struct definitions**

```rust
//! Per-depth-band 3D color LUT for fast non-keyframe grading.
//!
//! Captures the full per-pixel grading pipeline (LUT apply + HSL + ambiance +
//! warmth + blend) in a set of 65x65x65 3D LUTs, one per depth sample point.
//! Non-keyframe pixels are graded via trilinear lookup instead of running the
//! full pipeline, eliminating the expensive sRGB↔LAB roundtrip.

use dorea_cal::Calibration;
use dorea_lut::types::LutGrid;
use rayon::prelude::*;

use crate::cpu::grade_frame_cpu;
use crate::GradeParams;

/// Grid resolution for the grade LUT. 65x65x65 validated to produce
/// max error < 3/255, p99 < 1/255 against the direct pipeline.
/// Upgrade to 129 when 10-bit pipeline lands.
pub const GRADE_LUT_SIZE: usize = 65;

/// Precomputed grade LUT: array of 3D LUTs, one per depth sample point.
pub struct GradeLut {
    /// One 65x65x65 LUT per depth sample.
    luts: Vec<LutGrid>,
    /// Depth values at which each LUT was evaluated.
    /// Typically [0.0] + calibration.zone_centers + [1.0].
    depth_samples: Vec<f32>,
    /// Hash of (calibration + params) used to build this LUT.
    /// Used to detect invalidation if params change mid-clip.
    build_hash: u64,
}
```

- [ ] **Step 2: Add the build function**

```rust
impl GradeLut {
    /// Build the grade LUT by evaluating the real pipeline on synthetic 1x1 frames.
    ///
    /// `depth_samples` should be zone-aligned: `[0.0] + cal.depth_luts.zone_centers + [1.0]`.
    /// Each sample produces a 65x65x65 LUT. Build is parallelized with rayon.
    pub fn build(
        calibration: &Calibration,
        params: &GradeParams,
    ) -> Self {
        let mut depth_samples: Vec<f32> = Vec::with_capacity(
            calibration.depth_luts.zone_centers.len() + 2,
        );
        depth_samples.push(0.0);
        depth_samples.extend_from_slice(&calibration.depth_luts.zone_centers);
        depth_samples.push(1.0);
        // Deduplicate and sort
        depth_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        depth_samples.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

        let build_hash = Self::compute_hash(calibration, params);

        let luts: Vec<LutGrid> = depth_samples
            .iter()
            .map(|&d| Self::build_single_lut(d, calibration, params))
            .collect();

        log::info!(
            "GradeLut built: {}x{}x{} grid, {} depth bands, {:.1}MB",
            GRADE_LUT_SIZE, GRADE_LUT_SIZE, GRADE_LUT_SIZE,
            luts.len(),
            luts.iter().map(|l| l.data.len() * 4).sum::<usize>() as f64 / 1e6,
        );

        Self { luts, depth_samples, build_hash }
    }

    fn build_single_lut(
        depth_val: f32,
        calibration: &Calibration,
        params: &GradeParams,
    ) -> LutGrid {
        let size = GRADE_LUT_SIZE;
        let scale = (size - 1) as f32;
        let mut lut = LutGrid::new(size);

        // Collect grid indices for rayon parallelism
        let indices: Vec<(usize, usize, usize)> = (0..size)
            .flat_map(|ri| (0..size).flat_map(move |gi| (0..size).map(move |bi| (ri, gi, bi))))
            .collect();

        let results: Vec<([f32; 3], usize, usize, usize)> = indices
            .par_iter()
            .map(|&(ri, gi, bi)| {
                let r = (ri as f32 / scale * 255.0).round() as u8;
                let g = (gi as f32 / scale * 255.0).round() as u8;
                let b = (bi as f32 / scale * 255.0).round() as u8;

                let pixels = vec![r, g, b];
                let depth = vec![depth_val];

                let graded = grade_frame_cpu(
                    &pixels, &depth, 1, 1, calibration, params, true, // skip_clarity
                ).expect("grade_frame_cpu failed on synthetic pixel");

                let out = [
                    graded[0] as f32 / 255.0,
                    graded[1] as f32 / 255.0,
                    graded[2] as f32 / 255.0,
                ];
                (out, ri, gi, bi)
            })
            .collect();

        for (out, ri, gi, bi) in results {
            lut.set(ri, gi, bi, out);
        }

        lut
    }

    fn compute_hash(calibration: &Calibration, params: &GradeParams) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        // Hash params
        params.warmth.to_bits().hash(&mut hasher);
        params.strength.to_bits().hash(&mut hasher);
        params.contrast.to_bits().hash(&mut hasher);
        // Hash calibration identity (zone count + boundaries)
        calibration.depth_luts.n_zones().hash(&mut hasher);
        for &b in &calibration.depth_luts.zone_boundaries {
            b.to_bits().hash(&mut hasher);
        }
        calibration.hsl_corrections.0.len().hash(&mut hasher);
        hasher.finish()
    }

    /// Check that this LUT is still valid for the given calibration and params.
    pub fn is_valid_for(&self, calibration: &Calibration, params: &GradeParams) -> bool {
        self.build_hash == Self::compute_hash(calibration, params)
    }
}
```

- [ ] **Step 3: Add the apply function**

```rust
impl GradeLut {
    /// Grade a frame using the precomputed LUT.
    ///
    /// For each pixel: trilinear-interpolate in two bracketing depth-band LUTs,
    /// then linearly blend by depth. Parallelized with rayon.
    pub fn apply(&self, pixels: &[u8], depth: &[f32], width: usize, height: usize) -> Vec<u8> {
        let n = width * height;
        assert_eq!(pixels.len(), n * 3, "pixels length mismatch");
        assert_eq!(depth.len(), n, "depth length mismatch");

        let result: Vec<u8> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let r = pixels[i * 3] as f32 / 255.0;
                let g = pixels[i * 3 + 1] as f32 / 255.0;
                let b = pixels[i * 3 + 2] as f32 / 255.0;
                let d = depth[i];

                let [ro, go, bo] = self.lookup(r, g, b, d);
                [
                    (ro.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (go.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (bo.clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect();

        result
    }

    /// Single-pixel lookup with depth-band interpolation.
    fn lookup(&self, r: f32, g: f32, b: f32, d: f32) -> [f32; 3] {
        let (lo, hi, t) = self.depth_bracket(d);
        let graded_lo = trilinear(&self.luts[lo], [r, g, b]);
        if lo == hi {
            return graded_lo;
        }
        let graded_hi = trilinear(&self.luts[hi], [r, g, b]);
        [
            graded_lo[0] + (graded_hi[0] - graded_lo[0]) * t,
            graded_lo[1] + (graded_hi[1] - graded_lo[1]) * t,
            graded_lo[2] + (graded_hi[2] - graded_lo[2]) * t,
        ]
    }

    /// Find bracketing depth samples and interpolation weight.
    fn depth_bracket(&self, d: f32) -> (usize, usize, f32) {
        let samples = &self.depth_samples;
        if d <= samples[0] {
            return (0, 0, 0.0);
        }
        if d >= *samples.last().unwrap() {
            let last = samples.len() - 1;
            return (last, last, 0.0);
        }
        for i in 0..samples.len() - 1 {
            if d >= samples[i] && d < samples[i + 1] {
                let t = (d - samples[i]) / (samples[i + 1] - samples[i]).max(1e-8);
                return (i, i + 1, t);
            }
        }
        let last = samples.len() - 1;
        (last, last, 0.0)
    }
}

/// Trilinear interpolation through a LutGrid for one pixel.
/// Duplicated from dorea-lut::apply (which is private) to avoid cross-crate
/// dependency on an internal function.
fn trilinear(lut: &LutGrid, rgb: [f32; 3]) -> [f32; 3] {
    let size = lut.size;
    let scale = (size - 1) as f32;

    let sr = (rgb[0] * scale).clamp(0.0, scale);
    let sg = (rgb[1] * scale).clamp(0.0, scale);
    let sb = (rgb[2] * scale).clamp(0.0, scale);

    let i0 = (sr.floor() as usize).min(size - 2);
    let j0 = (sg.floor() as usize).min(size - 2);
    let k0 = (sb.floor() as usize).min(size - 2);

    let fr = (sr - i0 as f32).clamp(0.0, 1.0);
    let fg = (sg - j0 as f32).clamp(0.0, 1.0);
    let fb = (sb - k0 as f32).clamp(0.0, 1.0);

    let c000 = lut.get(i0, j0, k0);
    let c001 = lut.get(i0, j0, k0 + 1);
    let c010 = lut.get(i0, j0 + 1, k0);
    let c011 = lut.get(i0, j0 + 1, k0 + 1);
    let c100 = lut.get(i0 + 1, j0, k0);
    let c101 = lut.get(i0 + 1, j0, k0 + 1);
    let c110 = lut.get(i0 + 1, j0 + 1, k0);
    let c111 = lut.get(i0 + 1, j0 + 1, k0 + 1);

    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let v00 = c000[c] * (1.0 - fb) + c001[c] * fb;
        let v01 = c010[c] * (1.0 - fb) + c011[c] * fb;
        let v10 = c100[c] * (1.0 - fb) + c101[c] * fb;
        let v11 = c110[c] * (1.0 - fb) + c111[c] * fb;
        let v0 = v00 * (1.0 - fg) + v01 * fg;
        let v1 = v10 * (1.0 - fg) + v11 * fg;
        out[c] = v0 * (1.0 - fr) + v1 * fr;
    }
    out
}
```

- [ ] **Step 4: Export the module from `lib.rs`**

Add to `crates/dorea-gpu/src/lib.rs` after `pub mod cpu;`:

```rust
pub mod lut3d;
```

- [ ] **Step 5: Build**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release 2>&1 | tail -5
```

Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-gpu/src/lut3d.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(dorea-gpu): add lut3d module with GradeLut build and apply

65x65x65 per-depth-band 3D LUT capturing the full per-pixel grading
pipeline (LUT apply + HSL + ambiance + warmth + blend, minus clarity).
Zone-aligned depth sampling from calibration.zone_centers.
Rayon-parallelized build and apply."
```

---

### Task 3: Unit tests — trilinear accuracy

**Files:**
- Modify: `crates/dorea-gpu/src/lut3d.rs` (add `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write trilinear accuracy test**

Add at the bottom of `lut3d.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use dorea_cal::Calibration;
    use dorea_lut::types::{DepthLuts, LutGrid as CalLutGrid};
    use dorea_hsl::derive::HslCorrections;

    /// Create a minimal but realistic calibration for testing.
    /// Uses identity LUTs (no color change from LUT stage) so that
    /// the pipeline exercises HSL + ambiance + warmth + blend.
    fn test_calibration() -> Calibration {
        let n_zones = 5;
        let size = 33;
        let luts: Vec<CalLutGrid> = (0..n_zones).map(|_| {
            let mut lut = CalLutGrid::new(size);
            for ri in 0..size {
                for gi in 0..size {
                    for bi in 0..size {
                        let r = ri as f32 / (size - 1) as f32;
                        let g = gi as f32 / (size - 1) as f32;
                        let b = bi as f32 / (size - 1) as f32;
                        lut.set(ri, gi, bi, [r, g, b]);
                    }
                }
            }
            lut
        }).collect();
        let boundaries: Vec<f32> = (0..=n_zones).map(|i| i as f32 / n_zones as f32).collect();
        let depth_luts = DepthLuts::new(luts, boundaries);
        let hsl_corrections = HslCorrections(vec![]);
        Calibration::new(depth_luts, hsl_corrections, 1)
    }

    #[test]
    fn trilinear_accuracy_under_threshold() {
        let cal = test_calibration();
        let params = GradeParams::default();
        let grade_lut = GradeLut::build(&cal, &params);

        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            // Simple xorshift64 for deterministic random
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f32 / u64::MAX as f32).clamp(0.0, 1.0)
        };

        let n_samples = 10_000;
        let mut max_err: f32 = 0.0;
        let mut errors: Vec<f32> = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let r = next_f32();
            let g = next_f32();
            let b = next_f32();
            let d = next_f32();

            // Direct pipeline
            let px = [
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ];
            let direct = grade_frame_cpu(&px, &[d], 1, 1, &cal, &params, true)
                .expect("grade_frame_cpu failed");

            // LUT lookup
            let [lo, go, bo] = grade_lut.lookup(r, g, b, d);
            let lut_result = [
                (lo.clamp(0.0, 1.0) * 255.0).round() as u8,
                (go.clamp(0.0, 1.0) * 255.0).round() as u8,
                (bo.clamp(0.0, 1.0) * 255.0).round() as u8,
            ];

            let err = (0..3)
                .map(|c| (direct[c] as f32 - lut_result[c] as f32).abs())
                .fold(0.0_f32, f32::max);
            errors.push(err);
            max_err = max_err.max(err);
        }

        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = errors[(n_samples as f32 * 0.99) as usize];

        eprintln!("trilinear accuracy: max={max_err:.1}/255, p99={p99:.1}/255");
        assert!(max_err < 3.0, "max error {max_err}/255 exceeds 3/255 threshold");
        assert!(p99 < 1.5, "p99 error {p99}/255 exceeds 1.5/255 threshold");
    }
}
```

- [ ] **Step 2: Run test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu -- lut3d::tests::trilinear_accuracy --nocapture 2>&1 | tail -15
```

Expected: PASS with max < 3/255, p99 < 1.5/255.

Note: This test builds a full 65^3 × 7 LUT, so it takes ~5-10 seconds.

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-gpu/src/lut3d.rs
git commit -m "test(dorea-gpu): trilinear accuracy validation for 3D grade LUT

10K random (R,G,B,d) samples compared against direct pipeline.
Asserts max error < 3/255 and p99 < 1.5/255 at 65x65x65 grid."
```

---

### Task 4: Unit tests — boundary conditions and validity check

**Files:**
- Modify: `crates/dorea-gpu/src/lut3d.rs` (add tests to existing test module)

- [ ] **Step 1: Add boundary condition test**

Add to the `tests` module in `lut3d.rs`:

```rust
    #[test]
    fn boundary_conditions_exact_at_grid_nodes() {
        let cal = test_calibration();
        let params = GradeParams::default();
        let grade_lut = GradeLut::build(&cal, &params);

        // Test at grid nodes: LUT lookup should exactly match stored value.
        // Use first depth sample (d=0.0) for simplicity.
        let d = grade_lut.depth_samples[0];
        let scale = (GRADE_LUT_SIZE - 1) as f32;

        // Test corners: black, white, midgray, and primary/secondary colors
        let test_points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],   // black
            [1.0, 1.0, 1.0],   // white
            [0.5, 0.5, 0.5],   // midgray (not exact grid node at 65, but close)
            [1.0, 0.0, 0.0],   // red
            [0.0, 1.0, 0.0],   // green
            [0.0, 0.0, 1.0],   // blue
            [0.0, 1.0, 1.0],   // cyan
            [1.0, 0.0, 1.0],   // magenta
            [1.0, 1.0, 0.0],   // yellow
        ];

        for rgb in &test_points {
            // Quantize to grid node
            let ri = (rgb[0] * scale).round() as usize;
            let gi = (rgb[1] * scale).round() as usize;
            let bi = (rgb[2] * scale).round() as usize;
            let r = ri as f32 / scale;
            let g = gi as f32 / scale;
            let b = bi as f32 / scale;

            // Direct pipeline
            let px = [
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ];
            let direct = grade_frame_cpu(&px, &[d], 1, 1, &cal, &params, true)
                .expect("grade_frame_cpu failed");

            // LUT lookup at exact grid node (should be near-exact, only u8 quantization diff)
            let [lo, go, bo] = grade_lut.lookup(r, g, b, d);
            let lut_result = [
                (lo.clamp(0.0, 1.0) * 255.0).round() as u8,
                (go.clamp(0.0, 1.0) * 255.0).round() as u8,
                (bo.clamp(0.0, 1.0) * 255.0).round() as u8,
            ];

            for c in 0..3 {
                let err = (direct[c] as i32 - lut_result[c] as i32).unsigned_abs();
                assert!(
                    err <= 1,
                    "Boundary ({:.1},{:.1},{:.1}) d={d}: channel {c} direct={} lut={} err={err}",
                    rgb[0], rgb[1], rgb[2], direct[c], lut_result[c]
                );
            }
        }
    }

    #[test]
    fn validity_check_detects_param_change() {
        let cal = test_calibration();
        let params = GradeParams::default();
        let grade_lut = GradeLut::build(&cal, &params);

        assert!(grade_lut.is_valid_for(&cal, &params));

        let changed_params = GradeParams { warmth: 1.5, ..params };
        assert!(!grade_lut.is_valid_for(&cal, &changed_params));
    }
```

- [ ] **Step 2: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu -- lut3d::tests --nocapture 2>&1 | tail -15
```

Expected: all 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-gpu/src/lut3d.rs
git commit -m "test(dorea-gpu): boundary conditions and param validity for grade LUT

Verifies grid-node lookups match direct pipeline within 1/255.
Asserts is_valid_for detects GradeParams changes."
```

---

### Task 5: Add `--no-grade-lut` CLI flag

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (GradeArgs struct)

- [ ] **Step 1: Add the flag to GradeArgs**

Add after the existing `no_depth_interp` field:

```rust
    /// Disable 3D grade LUT — run full pipeline on every frame (for debugging/comparison)
    #[arg(long)]
    pub no_grade_lut: bool,
```

- [ ] **Step 2: Build to verify**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(dorea-cli): add --no-grade-lut flag for debugging

Mirrors --no-depth-interp. When set, forces full grade_frame on every
non-keyframe instead of the 3D LUT path."
```

---

### Task 6: Wire GradeLut into the grading loop

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

- [ ] **Step 1: Add imports and build the LUT after calibration**

At the top of `grade.rs`, add:

```rust
use dorea_gpu::lut3d::GradeLut;
```

In the `run` function, after the calibration is loaded (after line ~213 where `calibration` is available) and before the inference server spawn, add:

```rust
    // Build 3D grade LUT for fast non-keyframe grading (unless disabled)
    let grade_lut = if !args.no_grade_lut && !args.no_depth_interp {
        log::info!("Building 3D grade LUT...");
        let lut = GradeLut::build(&calibration, &params);
        Some(lut)
    } else {
        None
    };
```

- [ ] **Step 2: Modify `flush_buffer_with_depth` to accept optional GradeLut**

Update the function signature:

```rust
fn flush_buffer_with_depth(
    buffer: &mut Vec<BufferedFrame>,
    depth_before: &Option<Vec<f32>>,
    depth_after: Option<&Vec<f32>>,
    calibration: &Calibration,
    params: &GradeParams,
    grade_lut: Option<&GradeLut>,
    encoder: &mut FrameEncoder,
    frame_count: &mut u64,
    info: &ffmpeg::VideoInfo,
) -> Result<()> {
```

Inside the function, replace the `grade_frame` call with:

```rust
        let graded = if let Some(lut) = grade_lut {
            lut.apply(&bf.pixels, &depth, bf.width, bf.height)
        } else {
            grade_frame(
                &bf.pixels, &depth, bf.width, bf.height, calibration, params,
            ).map_err(|e| anyhow::anyhow!("Grading failed for buffered frame {}: {e}", bf.index))?
        };
```

- [ ] **Step 3: Update all call sites of `flush_buffer_with_depth`**

There are 3 call sites in `run()`. Update each to pass `grade_lut.as_ref()`:

Scene cut flush:
```rust
                flush_buffer_with_depth(
                    &mut frame_buffer, &last_keyframe_depth, None,
                    &calibration, &params, grade_lut.as_ref(),
                    &mut encoder, &mut frame_count, &info,
                )?;
```

Keyframe flush:
```rust
                flush_buffer_with_depth(
                    &mut frame_buffer, &last_keyframe_depth, Some(&depth),
                    &calibration, &params, grade_lut.as_ref(),
                    &mut encoder, &mut frame_count, &info,
                )?;
```

End-of-video flush:
```rust
        flush_buffer_with_depth(
            &mut frame_buffer, &last_keyframe_depth, None,
            &calibration, &params, grade_lut.as_ref(),
            &mut encoder, &mut frame_count, &info,
        )?;
```

- [ ] **Step 4: Build and run existing tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release 2>&1 | tail -5
cargo test -p dorea-cli -- grade 2>&1 | tail -10
```

Expected: build succeeds, all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(dorea-cli): wire GradeLut into non-keyframe grading path

Builds 3D LUT after calibration. Non-keyframe frames use LUT lookup
instead of grade_frame. Disabled with --no-grade-lut or --no-depth-interp."
```

---

### Task 7: Full pipeline benchmark and verify

**Files:** None (runtime verification)

- [ ] **Step 1: Run the benchmark on the test clip**

```bash
cd /workspaces/dorea-workspace
time repos/dorea/target/release/dorea grade \
  --input footage/raw/2025-11-01/DJI_20251101111428_0055_D.MP4 \
  --output working/graded/DJI_20251101111428_0055_D_graded_lut3d.mp4 \
  --raune-weights working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth \
  --raune-models-dir working/sea_thru_poc \
  --depth-model models/depth_anything_v2_small \
  --verbose 2>&1 | tee /tmp/dorea_bench_lut3d.log
```

Record: total wall time, LUT build time, per-100-frame progress intervals.

- [ ] **Step 2: Run with `--no-grade-lut` to verify flag works**

```bash
cd /workspaces/dorea-workspace
time repos/dorea/target/release/dorea grade \
  --input footage/raw/2025-11-01/DJI_20251101111428_0055_D.MP4 \
  --output working/graded/DJI_20251101111428_0055_D_graded_nolut.mp4 \
  --raune-weights working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth \
  --raune-models-dir working/sea_thru_poc \
  --depth-model models/depth_anything_v2_small \
  --no-grade-lut \
  --verbose 2>&1 | tee /tmp/dorea_bench_nolut.log
```

Should produce same timing as the depth-interp baseline (~11m37s).

- [ ] **Step 3: Visual quality check — extract frames and compare**

```bash
mkdir -p /tmp/lut_compare
# Extract frames at t=2s, 5s, 10s from both outputs
for t in 2 5 10; do
  ffmpeg -y -ss $t -i working/graded/DJI_20251101111428_0055_D_graded_lut3d.mp4 \
    -vframes 1 -update 1 /tmp/lut_compare/lut3d_${t}s.png 2>/dev/null
  ffmpeg -y -ss $t -i working/graded/DJI_20251101111428_0055_D_graded_nolut.mp4 \
    -vframes 1 -update 1 /tmp/lut_compare/nolut_${t}s.png 2>/dev/null
done
```

Compare visually — should be imperceptibly different.

- [ ] **Step 4: PSNR/SSIM comparison between LUT and no-LUT outputs**

```bash
ffmpeg -i working/graded/DJI_20251101111428_0055_D_graded_lut3d.mp4 \
  -i working/graded/DJI_20251101111428_0055_D_graded_nolut.mp4 \
  -lavfi "ssim;[0:v][1:v]psnr" -f null - 2>&1 | grep -E 'SSIM|PSNR'
```

Expected: PSNR > 40 dB, SSIM > 0.99 (very high similarity).

- [ ] **Step 5: Record results to corvia**

Use `corvia_write` to record the benchmark finding with build time, wall time, fps, and quality metrics.
