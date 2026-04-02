# CPU Parallelization — rayon

**Date:** 2026-04-02  
**Status:** Approved  
**Scope:** `dorea-lut`, `dorea-hsl`

## Problem

The `dorea calibrate` pipeline has four CPU-bound operations that run single-threaded.
For a 5-keyframe FHD run they account for ~2–4s of wall clock time; at 4K they scale
to ~12–15s. All four are embarrassingly parallel and have no inter-pixel dependencies.

| Operation | Location | Estimated time (5× FHD) |
|---|---|---|
| `compute_importance` Gaussian blur σ=8 | `dorea-lut/src/build.rs:30` | 0.5–1.5s |
| `apply_depth_luts` trilinear + blend | `dorea-lut/src/apply.rs:81` | 1–1.75s |
| NN fill (brute-force L2) | `dorea-lut/src/build.rs:290` | 0.3–1s |
| `derive_hsl_corrections` RGB→HSV + accumulation | `dorea-hsl/src/derive.rs:44` | 0.4–0.75s |

The Python inference subprocess (RAUNE-Net, Depth Anything) is intentionally sequential
due to the 6 GB VRAM constraint and is excluded from this work.

## Decision

Add `rayon` as a workspace-level dependency and parallelize all four operations using
`par_iter` on the outermost pixel/cell loops. Three approaches were considered: tile-based
parallelism (rejected — premature, adds boundary complexity) and a `parallel` Cargo feature
flag (rejected — unnecessary for a desktop pipeline tool).

## Dependency wiring

`rayon = "1"` added to `[workspace.dependencies]` in the root `Cargo.toml`, then declared in:

- `crates/dorea-lut/Cargo.toml` — `rayon.workspace = true`
- `crates/dorea-hsl/Cargo.toml` — `rayon.workspace = true`

`dorea-video` and `dorea-cli` are unchanged.

## Per-target strategy

### 1. `compute_importance` — row-parallel passes

The Sobel kernel and both Gaussian blur passes (horizontal + vertical) are independent
per output row. Replace the `for y in 0..height` loops with
`output.par_chunks_mut(width).enumerate()`.

The box-filter integral image is O(n) and fast; left serial.

### 2. `apply_depth_luts` — pixel-parallel

Every pixel is fully independent (LUT is read-only). Replace the flat pixel loop with:

```rust
result.par_iter_mut()
      .zip(pixels.par_iter().zip(depth.par_iter()))
      .for_each(|(out, (&px, &d))| { /* trilinear + zone blend */ });
```

### 3. NN fill — empty-cell-parallel

`populated` list is read-only; each empty cell search is independent. Extract the inner
search into a `find_nearest` pure function, then:

```rust
let filled: Vec<_> = empty.par_iter()
    .map(|&ec| find_nearest(&ec, &populated))
    .collect();
// sequential write-back to lut
```

### 4. `derive_hsl_corrections` — parallel conversion + fold/reduce

The outer qualifier loop (6–8 items) stays serial.

**Before the qualifier loop** (once): parallelize the two HSV conversion Vecs that the
current serial code already computes outside the loop:

```rust
let lut_hsv: Vec<_> = lut_output.par_iter().map(|&[r,g,b]| rgb_to_hsv(r,g,b)).collect();
let tgt_hsv: Vec<_> = target.par_iter().map(|&[r,g,b]| rgb_to_hsv(r,g,b)).collect();
```

**Inside each qualifier**: fold the soft-mask weights build and weighted accumulation into
a single parallel pass (one scan over pixels instead of two):

```rust
let (total_w, h_sum, lut_s_sum, tgt_s_sum, v_sum) =
    lut_hsv.par_iter().zip(tgt_hsv.par_iter())
           .fold(|| (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64), |acc, (lhsv, thsv)| {
               // compute weight, accumulate
           })
           .reduce(|| (0.0, 0.0, 0.0, 0.0, 0.0), |a, b| (a.0+b.0, ...));
```

Accumulated in `f64`. Floating-point reduction order differs from the serial path by
~1e-6 — within the existing test tolerances.

## Correctness bar

The existing test suite is the correctness bar — no new tests required:

- `test_nn_fill_no_identity`
- `test_apply_identity_lut`
- `test_adaptive_zone_boundaries_*`
- `test_derive_same_image`

All must pass unchanged. Parallel output must be numerically identical to serial output
except for the documented ~1e-6 f64 fold/reduce variance in `derive_hsl_corrections`.

A non-identity integration test for `derive_hsl_corrections` (input ≠ target) is recommended
as a follow-up to provide a stronger regression guard for the parallel reduction path.

## Expected outcome

~3–8× speedup on the four CPU steps on 8–16 core workstations (estimates; no benchmarks
yet — `criterion` benchmarks are a recommended follow-up to validate claims).
At 4K (8.3M pixels vs 2.1M), gains scale ~4× larger in absolute terms.
RAUNE-Net inference (dominant at ~7–20s for 5 frames) is unchanged.

`--cpu-only` is unaffected: it controls whether inference uses CUDA, not the Rust thread
count. The rayon thread pool uses all available cores regardless of `--cpu-only`.
