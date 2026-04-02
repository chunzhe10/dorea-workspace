# Rayon CPU Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize four embarrassingly-parallel CPU-bound operations in `dorea-lut` and `dorea-hsl` using rayon, achieving ~3–8× speedup on 8–16 core workstations.

**Architecture:** Add `rayon = "1"` as a workspace dependency, then replace the outermost serial pixel/row loops in `compute_importance`, `apply_depth_luts`, the NN fill, and `derive_hsl_corrections` with rayon parallel iterators. No new public APIs, no structural changes — pure internal parallelism.

**Tech Stack:** Rust, rayon 1.x, cargo test

---

## File Map

| File | Change |
|------|--------|
| `repos/dorea/Cargo.toml` | Add `rayon = "1"` to `[workspace.dependencies]` |
| `repos/dorea/crates/dorea-lut/Cargo.toml` | Add `rayon.workspace = true` to `[dependencies]` |
| `repos/dorea/crates/dorea-hsl/Cargo.toml` | Add `rayon.workspace = true` to `[dependencies]` |
| `repos/dorea/crates/dorea-lut/src/build.rs` | Parallelize Sobel row loops, Gaussian blur passes, and NN fill |
| `repos/dorea/crates/dorea-lut/src/apply.rs` | Parallelize pixel loop in `apply_depth_luts` |
| `repos/dorea/crates/dorea-hsl/src/derive.rs` | Parallelize HSV conversion + fold/reduce accumulation |

---

## Task 1: Add rayon workspace dependency

**Files:**
- Modify: `repos/dorea/Cargo.toml`
- Modify: `repos/dorea/crates/dorea-lut/Cargo.toml`
- Modify: `repos/dorea/crates/dorea-hsl/Cargo.toml`

- [ ] **Step 1: Add rayon to workspace dependencies**

In `repos/dorea/Cargo.toml`, add one line to `[workspace.dependencies]`:

```toml
[workspace.dependencies]
dorea-color = { path = "crates/dorea-color" }
dorea-lut = { path = "crates/dorea-lut" }
dorea-hsl = { path = "crates/dorea-hsl" }
dorea-cal = { path = "crates/dorea-cal" }
dorea-gpu = { path = "crates/dorea-gpu" }
dorea-video = { path = "crates/dorea-video" }
serde = { version = "1", features = ["derive"] }
bincode = "1"
thiserror = "1"
anyhow = "1"
log = "0.4"
rayon = "1"
```

- [ ] **Step 2: Wire rayon into dorea-lut**

In `repos/dorea/crates/dorea-lut/Cargo.toml`, add to `[dependencies]`:

```toml
[dependencies]
dorea-color = { workspace = true }
serde = { workspace = true }
log = { workspace = true }
rayon = { workspace = true }
```

- [ ] **Step 3: Wire rayon into dorea-hsl**

In `repos/dorea/crates/dorea-hsl/Cargo.toml`, add to `[dependencies]`:

```toml
[dependencies]
dorea-color = { workspace = true }
serde = { workspace = true }
rayon = { workspace = true }
```

- [ ] **Step 4: Verify workspace compiles**

```bash
cd repos/dorea && cargo build 2>&1
```

Expected: compiles without errors. `Compiling rayon ...` appears in output.

- [ ] **Step 5: Commit**

```bash
cd repos/dorea
git add Cargo.toml crates/dorea-lut/Cargo.toml crates/dorea-hsl/Cargo.toml Cargo.lock
git commit -m "chore: add rayon workspace dependency to dorea-lut and dorea-hsl"
```

---

## Task 2: Parallelize `apply_depth_luts` (pixel-parallel)

**Files:**
- Modify: `repos/dorea/crates/dorea-lut/src/apply.rs`

The entire pixel loop (line 81+) is independent per pixel because `luts` is read-only.
Replace with `par_iter_mut().zip(par_iter().zip(par_iter()))`.

- [ ] **Step 1: Run the existing test to confirm baseline**

```bash
cd repos/dorea && cargo test -p dorea-lut test_apply_identity_lut -- --nocapture 2>&1
```

Expected: `test test_apply_identity_lut ... ok`

- [ ] **Step 2: Add rayon import and rewrite the pixel loop**

Replace the top of `apply.rs` and the pixel loop in `apply_depth_luts`:

```rust
//! Apply depth-stratified LUTs via trilinear interpolation + depth blending.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::apply_depth_luts`.

use rayon::prelude::*;

use crate::types::{DepthLuts, LutGrid};
```

Replace the loop body of `apply_depth_luts` (from `let mut result = ...` through `result`):

```rust
    let n_zones = luts.n_zones();

    let mut result = vec![[0.0_f32; 3]; pixels.len()];

    result
        .par_iter_mut()
        .zip(pixels.par_iter().zip(depth.par_iter()))
        .for_each(|(out, (&px, &d))| {
            let mut acc = [0.0_f32; 3];
            let mut total_w = 0.0_f32;

            for z in 0..n_zones {
                // I1: use actual adaptive zone width from boundaries, not a fixed 1/n_zones.
                let zone_width = luts.zone_boundaries[z + 1] - luts.zone_boundaries[z];
                let dist = (d - luts.zone_centers[z]).abs();
                let w = (1.0 - dist / zone_width.max(1e-6)).max(0.0);
                if w < 1e-7 {
                    continue;
                }
                let lut_out = trilinear(&luts.luts[z], px);
                for (c, lo) in lut_out.iter().enumerate() {
                    acc[c] += lo * w;
                }
                total_w += w;
            }

            if total_w > 1e-6 {
                for c in 0..3 {
                    out[c] = (acc[c] / total_w).clamp(0.0, 1.0);
                }
            } else {
                *out = px;
            }
        });

    result
```

The full function after the change:

```rust
pub fn apply_depth_luts(
    pixels: &[[f32; 3]],
    depth: &[f32],
    luts: &DepthLuts,
) -> Vec<[f32; 3]> {
    assert_eq!(
        pixels.len(),
        depth.len(),
        "pixels ({}) and depth ({}) must have same length",
        pixels.len(),
        depth.len()
    );

    let n_zones = luts.n_zones();

    let mut result = vec![[0.0_f32; 3]; pixels.len()];

    result
        .par_iter_mut()
        .zip(pixels.par_iter().zip(depth.par_iter()))
        .for_each(|(out, (&px, &d))| {
            let mut acc = [0.0_f32; 3];
            let mut total_w = 0.0_f32;

            for z in 0..n_zones {
                let zone_width = luts.zone_boundaries[z + 1] - luts.zone_boundaries[z];
                let dist = (d - luts.zone_centers[z]).abs();
                let w = (1.0 - dist / zone_width.max(1e-6)).max(0.0);
                if w < 1e-7 {
                    continue;
                }
                let lut_out = trilinear(&luts.luts[z], px);
                for (c, lo) in lut_out.iter().enumerate() {
                    acc[c] += lo * w;
                }
                total_w += w;
            }

            if total_w > 1e-6 {
                for c in 0..3 {
                    out[c] = (acc[c] / total_w).clamp(0.0, 1.0);
                }
            } else {
                *out = px;
            }
        });

    result
}
```

- [ ] **Step 3: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-lut 2>&1
```

Expected: all tests pass including `test_apply_identity_lut` and all `test_adaptive_zone_boundaries_*`.

- [ ] **Step 4: Commit**

```bash
cd repos/dorea
git add crates/dorea-lut/src/apply.rs
git commit -m "perf(dorea-lut): parallelize apply_depth_luts pixel loop with rayon"
```

---

## Task 3: Parallelize NN fill in `build_depth_luts` (empty-cell-parallel)

**Files:**
- Modify: `repos/dorea/crates/dorea-lut/src/build.rs`

Extract the inner O(E) search into a `find_nearest` pure function, then replace the serial `for &ec in &empty` loop with `par_iter().map().collect()` + serial write-back.

- [ ] **Step 1: Run the existing test to confirm baseline**

```bash
cd repos/dorea && cargo test -p dorea-lut test_nn_fill_no_identity -- --nocapture 2>&1
```

Expected: `test test_nn_fill_no_identity ... ok`

- [ ] **Step 2: Add rayon import to build.rs**

At the top of `crates/dorea-lut/src/build.rs`, after the existing `use crate::types::{DepthLuts, LutGrid};` line, add:

```rust
use rayon::prelude::*;
```

- [ ] **Step 3: Add `find_nearest` helper function**

Insert the following function anywhere before `build_depth_luts` in `build.rs` (e.g., right before the function at line ~200):

```rust
/// Find the nearest populated LUT cell to `empty_cell` by brute-force L2 in index space.
///
/// `populated` must be non-empty.
fn find_nearest(empty_cell: &[usize; 3], populated: &[[usize; 3]]) -> ([usize; 3], [usize; 3]) {
    let mut best_dist = u64::MAX;
    let mut best_pop = populated[0];
    for &pc in populated {
        let dr = (empty_cell[0] as i64 - pc[0] as i64).pow(2) as u64;
        let dg = (empty_cell[1] as i64 - pc[1] as i64).pow(2) as u64;
        let db = (empty_cell[2] as i64 - pc[2] as i64).pow(2) as u64;
        let dist = dr + dg + db;
        if dist < best_dist {
            best_dist = dist;
            best_pop = pc;
        }
    }
    (*empty_cell, best_pop)
}
```

- [ ] **Step 4: Replace the serial NN fill loop with parallel version**

Find the serial NN fill block in `build_depth_luts` (the comment `// NN fill: brute-force L2 in index space` around line 288). Replace:

```rust
        // NN fill: brute-force L2 in index space
        // TODO: parallelize with rayon (O(E×P) brute force, acceptable for LUT_SIZE=33 but slow for sparse footage)
        if !populated.is_empty() && !empty.is_empty() {
            for &ec in &empty {
                let mut best_dist = u64::MAX;
                let mut best_pop = populated[0];
                for &pc in &populated {
                    let dr = (ec[0] as i64 - pc[0] as i64).pow(2) as u64;
                    let dg = (ec[1] as i64 - pc[1] as i64).pow(2) as u64;
                    let db = (ec[2] as i64 - pc[2] as i64).pow(2) as u64;
                    let dist = dr + dg + db;
                    if dist < best_dist {
                        best_dist = dist;
                        best_pop = pc;
                    }
                }
                let val = lut.get(best_pop[0], best_pop[1], best_pop[2]);
                lut.set(ec[0], ec[1], ec[2], val);
            }
        }
```

With:

```rust
        // NN fill: brute-force L2 in index space (parallel per empty cell)
        if !populated.is_empty() && !empty.is_empty() {
            // Each empty cell search is fully independent — populated is read-only.
            let filled: Vec<([usize; 3], [usize; 3])> = empty
                .par_iter()
                .map(|ec| find_nearest(ec, &populated))
                .collect();
            // Sequential write-back (each cell is distinct, no conflicts).
            for (ec, best_pop) in filled {
                let val = lut.get(best_pop[0], best_pop[1], best_pop[2]);
                lut.set(ec[0], ec[1], ec[2], val);
            }
        }
```

- [ ] **Step 5: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-lut 2>&1
```

Expected: all tests pass including `test_nn_fill_no_identity`.

- [ ] **Step 6: Commit**

```bash
cd repos/dorea
git add crates/dorea-lut/src/build.rs
git commit -m "perf(dorea-lut): parallelize NN fill with rayon par_iter"
```

---

## Task 4: Parallelize `compute_importance` (row-parallel Sobel + Gaussian blur)

**Files:**
- Modify: `repos/dorea/crates/dorea-lut/src/build.rs`

Three loops are independent per output row: the Sobel pass, the horizontal Gaussian blur pass, and the vertical Gaussian blur pass. Replace each with `par_chunks_mut(width).enumerate()`.

The Combine step at the end (O(n) scalar) is already fast; leave serial.
The `local_variance_box` integral-image function is O(n) and correct; leave serial.

- [ ] **Step 1: Parallelize the Sobel edge detection loop**

In `compute_importance`, replace the Sobel double loop (lines ~36–59):

```rust
    // --- Sobel edge detection ---
    let mut edge_mag = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
```

With:

```rust
    // --- Sobel edge detection (row-parallel) ---
    let mut edge_mag = vec![0.0_f32; n];
    edge_mag
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
```

The full parallelized Sobel block:

```rust
    // --- Sobel edge detection (row-parallel) ---
    let mut edge_mag = vec![0.0_f32; n];
    edge_mag
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut dx = 0.0_f32;
                let mut dy = 0.0_f32;
                for ky in -1_i32..=1 {
                    for kx in -1_i32..=1 {
                        let ny = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let nx = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                        let v = depth[ny * width + nx];
                        let kx_w = kx as f32;
                        let ky_w = ky as f32;
                        let wx = kx_w * (2.0 - ky_w.abs());
                        let wy = ky_w * (2.0 - kx_w.abs());
                        dx += v * wx;
                        dy += v * wy;
                    }
                }
                row[x] = (dx * dx + dy * dy).sqrt();
            }
        });
```

Note: `depth` is referenced from the closure but only read, not written — safe for shared reference across threads.

- [ ] **Step 2: Parallelize the horizontal Gaussian blur pass**

Replace the horizontal blur loop (lines ~80–89):

```rust
    // Horizontal pass
    let mut temp = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let nx = (x as i32 + ki as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                acc += edge_mag[y * width + nx] * kv;
            }
            temp[y * width + x] = acc;
        }
    }
```

With:

```rust
    // Horizontal pass (row-parallel)
    let mut temp = vec![0.0_f32; n];
    temp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nx = (x as i32 + ki as i32 - radius as i32)
                        .clamp(0, width as i32 - 1) as usize;
                    acc += edge_mag[y * width + nx] * kv;
                }
                row[x] = acc;
            }
        });
```

- [ ] **Step 3: Parallelize the vertical Gaussian blur pass**

Replace the vertical blur loop (lines ~91–101):

```rust
    // Vertical pass
    let mut edge_dilated = vec![0.0_f32; n];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let ny = (y as i32 + ki as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                acc += temp[ny * width + x] * kv;
            }
            edge_dilated[y * width + x] = acc;
        }
    }
```

With:

```rust
    // Vertical pass (row-parallel)
    let mut edge_dilated = vec![0.0_f32; n];
    edge_dilated
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ny = (y as i32 + ki as i32 - radius as i32)
                        .clamp(0, height as i32 - 1) as usize;
                    acc += temp[ny * width + x] * kv;
                }
                row[x] = acc;
            }
        });
```

- [ ] **Step 4: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-lut 2>&1
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd repos/dorea
git add crates/dorea-lut/src/build.rs
git commit -m "perf(dorea-lut): parallelize compute_importance Sobel + Gaussian blur with rayon"
```

---

## Task 5: Parallelize `derive_hsl_corrections` (HSV conversion + fold/reduce)

**Files:**
- Modify: `repos/dorea/crates/dorea-hsl/src/derive.rs`

Two changes:
1. Convert the two serial `lut_hsv` / `tgt_hsv` Vec builds to `par_iter().map().collect()`.
2. Inside each qualifier, replace the two serial passes (weight build + accumulation) with a single `fold/reduce` parallel pass.

The outer `for qual in HSL_QUALIFIERS` loop (6–8 items) stays serial — not worth parallelizing.

- [ ] **Step 1: Run the existing test to confirm baseline**

```bash
cd repos/dorea && cargo test -p dorea-hsl test_derive_same_image -- --nocapture 2>&1
```

Expected: `test test_derive_same_image ... ok`

- [ ] **Step 2: Add rayon import to derive.rs**

At the top of `crates/dorea-hsl/src/derive.rs`, after the existing use statements, add:

```rust
use rayon::prelude::*;
```

So the top of the file becomes:

```rust
//! Derive per-qualifier HSL corrections from (lut_output, raune_target) pair.
//!
//! Ported from `run_fixed_hsl_lut_poc.py::derive_hsl_corrections`.

use rayon::prelude::*;

use dorea_color::hsv::rgb_to_hsv;

use crate::qualifiers::{HSL_QUALIFIERS, MIN_SATURATION, MIN_WEIGHT};
```

- [ ] **Step 3: Parallelize the HSV conversion Vecs**

Replace:

```rust
    // Convert to HSV
    let lut_hsv: Vec<(f32, f32, f32)> = lut_output
        .iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();
    let tgt_hsv: Vec<(f32, f32, f32)> = target
        .iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();
```

With:

```rust
    // Convert to HSV (parallel — each pixel is independent)
    let lut_hsv: Vec<(f32, f32, f32)> = lut_output
        .par_iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();
    let tgt_hsv: Vec<(f32, f32, f32)> = target
        .par_iter()
        .map(|&[r, g, b]| rgb_to_hsv(r, g, b))
        .collect();
```

- [ ] **Step 4: Replace the two-pass qualifier loop with a single fold/reduce pass**

The current qualifier body does two serial passes: one to build `weights` + compute `total_weight`, then a second to accumulate `h_offset_sum` etc. Replace both with a single `fold/reduce` pass.

Replace the entire qualifier body inside `for qual in HSL_QUALIFIERS { ... }` with:

```rust
    for qual in HSL_QUALIFIERS {
        let hc = qual.h_center;
        let hw = qual.h_width;

        // Single parallel pass: compute weight and accumulate in one scan.
        // fold() builds partial sums per rayon thread; reduce() merges them.
        // Tuple: (total_w, h_sum, lut_s_sum, tgt_s_sum, v_sum)
        let (total_weight, h_offset_sum, lut_s_sum, tgt_s_sum, v_diff_sum) =
            lut_hsv
                .par_iter()
                .zip(tgt_hsv.par_iter())
                .fold(
                    || (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
                    |acc, (&(lh, ls, lv), &(th, ts, tv))| {
                        let h_dist = angular_dist(lh, hc);
                        let mask = (1.0 - h_dist / hw).max(0.0)
                            * if ls > MIN_SATURATION { 1.0_f32 } else { 0.0_f32 };
                        if mask < 1e-7 {
                            return acc;
                        }
                        let w = mask as f64;
                        let h_diff = wrap_hue_diff(th - lh) as f64;
                        (
                            acc.0 + w,
                            acc.1 + h_diff * w,
                            acc.2 + ls as f64 * w,
                            acc.3 + ts as f64 * w,
                            acc.4 + (tv - lv) as f64 * w,
                        )
                    },
                )
                .reduce(
                    || (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
                );

        if total_weight < MIN_WEIGHT as f64 {
            corrections.push(QualifierCorrection {
                h_center: hc,
                h_width: hw,
                h_offset: 0.0,
                s_ratio: 1.0,
                v_offset: 0.0,
                weight: 0.0,
            });
            continue;
        }

        let h_offset = (h_offset_sum / total_weight) as f32;
        let lut_s_mean = (lut_s_sum / total_weight) as f32;
        let tgt_s_mean = (tgt_s_sum / total_weight) as f32;
        let s_ratio = tgt_s_mean / lut_s_mean.max(1e-6);
        let v_offset = (v_diff_sum / total_weight) as f32;

        corrections.push(QualifierCorrection {
            h_center: hc,
            h_width: hw,
            h_offset,
            s_ratio,
            v_offset,
            weight: total_weight as f32,
        });
    }
```

Note: Floating-point reduction order differs from the serial path by ~1e-6 — within existing test tolerances (`h_offset.abs() < 0.5`, `(s_ratio - 1.0).abs() < 0.01`, `v_offset.abs() < 0.01`).

- [ ] **Step 5: Run tests**

```bash
cd repos/dorea && cargo test -p dorea-hsl 2>&1
```

Expected: `test_derive_same_image ... ok`

- [ ] **Step 6: Run full workspace test suite**

```bash
cd repos/dorea && cargo test 2>&1
```

Expected: all tests pass across all crates.

- [ ] **Step 7: Run clippy**

```bash
cd repos/dorea && cargo clippy -- -D warnings 2>&1
```

Expected: no warnings.

- [ ] **Step 8: Commit**

```bash
cd repos/dorea
git add crates/dorea-hsl/src/derive.rs
git commit -m "perf(dorea-hsl): parallelize derive_hsl_corrections HSV conversion + fold/reduce"
```

---

## Self-Review Against Spec

**Spec coverage:**
- [x] `rayon = "1"` workspace dependency — Task 1
- [x] `dorea-lut/Cargo.toml` + `dorea-hsl/Cargo.toml` wiring — Task 1
- [x] `apply_depth_luts` pixel-parallel — Task 2
- [x] NN fill empty-cell-parallel with extracted `find_nearest` — Task 3
- [x] `compute_importance` row-parallel Sobel + Gaussian blur — Task 4
- [x] `derive_hsl_corrections` parallel HSV conversion + fold/reduce — Task 5
- [x] Existing tests pass unchanged — verified at each task
- [x] `dorea-video` and `dorea-cli` unchanged — not touched
- [x] `local_variance_box` left serial (integral image, O(n), fast) — not touched
- [x] outer qualifier loop stays serial — maintained in Task 5
- [x] full test suite + clippy in Task 5

**No placeholders found.**

**Type consistency:** `find_nearest` returns `([usize; 3], [usize; 3])` — destructured in Task 3 step 4 as `(ec, best_pop)`. Consistent.
