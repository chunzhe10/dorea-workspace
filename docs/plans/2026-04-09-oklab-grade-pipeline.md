# OKLab Grade Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CIELab with OKLab (rescaled to CIELab ranges) in the per-pixel grading pipeline, and switch CUDA `powf` to `__powf` for faster sRGB gamma.

**Architecture:** Two files contain the conversion functions: `dorea-color/src/lab.rs` (Rust CPU) and `dorea-gpu/src/cuda/kernels/grade_pixel.cuh` (CUDA GPU). Both expose `srgb_to_lab` / `lab_to_srgb` with identical signatures. Replace internals with OKLab math, rescale output to match CIELab numeric ranges (L×100, a×300, b×300) so all downstream manipulation constants stay unchanged. CUDA path additionally switches `powf` → `__powf`.

**Tech Stack:** Rust, CUDA C++

**Spec:** `docs/decisions/2026-04-09-oklab-grade-pipeline.md`

---

## File Map

| File | Change |
|------|--------|
| `crates/dorea-color/src/lab.rs` | Replace CIELab internals with OKLab + rescale. Update tests. |
| `crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh` | Replace CUDA CIELab with OKLab + rescale + `__powf`. |

Files that need NO changes (they call the functions above, signatures unchanged):
- `crates/dorea-gpu/src/cpu.rs`
- `crates/dorea-gpu/src/cuda/kernels/postprocess.cu`
- `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu`
- `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu`

---

## Task 1: Replace CIELab with OKLab in Rust (`lab.rs`)

**Files:**
- Modify: `crates/dorea-color/src/lab.rs`

- [ ] **Step 1: Update the module doc comment**

Replace the first 3 lines of `crates/dorea-color/src/lab.rs`:

```rust
//! OKLab colorspace conversion, rescaled to CIELab-compatible ranges.
//!
//! Uses Björn Ottosson's OKLab (2020) internally, with output rescaled to match
//! CIELab numeric ranges so downstream consumers (ambiance, warmth, vibrance)
//! need no constant changes:
//!   L: OKLab [0,1] × 100 → [0,100]
//!   a: OKLab [-0.4,0.4] × 300 → [-120,120]
//!   b: OKLab [-0.4,0.4] × 300 → [-120,120]
//!
//! sRGB linearization uses IEC 61966-2-1 piecewise gamma (unchanged).
```

- [ ] **Step 2: Replace constants and helper functions**

Remove the CIELab constants and `f_lab`/`f_lab_inv` functions. Replace lines 5–53 with:

```rust
// OKLab rescale factors — map OKLab ranges to CIELab-compatible ranges.
const L_SCALE: f32 = 100.0;
const AB_SCALE: f32 = 300.0;

/// sRGB component → linear light (IEC 61966-2-1).
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → sRGB component (IEC 61966-2-1).
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
```

- [ ] **Step 3: Replace `srgb_to_lab`**

Replace the existing `srgb_to_lab` function (was lines 55–79) with:

```rust
/// Convert sRGB [0,1] to OKLab, rescaled to CIELab-compatible ranges.
///
/// Returns (L, a, b) where L ∈ [0,100], a/b ∈ approx [-120, 120].
pub fn srgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB → linear
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);

    // Linear RGB → LMS
    let l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl;
    let m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl;
    let s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl;

    // Cube root
    let l_ = l.max(0.0).cbrt();
    let m_ = m.max(0.0).cbrt();
    let s_ = s.max(0.0).cbrt();

    // LMS' → OKLab
    let ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    // Rescale to CIELab-compatible ranges
    (ok_l * L_SCALE, ok_a * AB_SCALE, ok_b * AB_SCALE)
}
```

- [ ] **Step 4: Replace `lab_to_srgb`**

Replace the existing `lab_to_srgb` function (was lines 81–102) with:

```rust
/// Convert OKLab (rescaled) to sRGB [0,1] (clamped).
pub fn lab_to_srgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Unscale from CIELab-compatible ranges to native OKLab
    let ok_l = l / L_SCALE;
    let ok_a = a / AB_SCALE;
    let ok_b = b / AB_SCALE;

    // OKLab → LMS'
    let l_ = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b;
    let m_ = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b;
    let s_ = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b;

    // Cube (inverse of cube root)
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // LMS → linear RGB
    let rl =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let bl =  0.0208763419 * l - 0.7034186147 * m + 1.6825422694 * s;

    // Linear → sRGB, clamped
    let r = linear_to_srgb(rl.max(0.0));
    let g = linear_to_srgb(gl.max(0.0));
    let b_out = linear_to_srgb(bl.max(0.0));

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b_out.clamp(0.0, 1.0))
}
```

- [ ] **Step 5: Update tests**

Replace the `tests` module (was lines 104–134) with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(r: f32, g: f32, b: f32) {
        let (l, a, lab_b) = srgb_to_lab(r, g, b);
        let (r2, g2, b2) = lab_to_srgb(l, a, lab_b);
        assert!(
            (r2 - r).abs() < 1e-3 && (g2 - g).abs() < 1e-3 && (b2 - b).abs() < 1e-3,
            "Round-trip ({r},{g},{b}) -> Lab({l:.2},{a:.2},{lab_b:.2}) -> ({r2:.5},{g2:.5},{b2:.5})"
        );
    }

    #[test]
    fn test_lab_round_trip() {
        round_trip(1.0, 1.0, 1.0); // white
        round_trip(0.0, 0.0, 0.0); // black
        round_trip(0.5, 0.5, 0.5); // mid grey
        round_trip(1.0, 0.0, 0.0); // red
        round_trip(0.0, 1.0, 0.0); // green
        round_trip(0.0, 0.0, 1.0); // blue
    }

    #[test]
    fn test_white_point() {
        let (l, a, b) = srgb_to_lab(1.0, 1.0, 1.0);
        // OKLab white = (1,0,0), rescaled L = 100, a = 0, b = 0
        assert!((l - 100.0).abs() < 1e-2, "White L: expected 100, got {l}");
        assert!(a.abs() < 1e-2, "White a: expected 0, got {a}");
        assert!(b.abs() < 1e-2, "White b: expected 0, got {b}");
    }

    #[test]
    fn test_black_point() {
        let (l, a, b) = srgb_to_lab(0.0, 0.0, 0.0);
        assert!(l.abs() < 1e-3, "Black L: expected 0, got {l}");
        assert!(a.abs() < 1e-3, "Black a: expected 0, got {a}");
        assert!(b.abs() < 1e-3, "Black b: expected 0, got {b}");
    }

    #[test]
    fn test_l_range() {
        // L should span [0, 100] for sRGB gamut
        let (l_black, _, _) = srgb_to_lab(0.0, 0.0, 0.0);
        let (l_white, _, _) = srgb_to_lab(1.0, 1.0, 1.0);
        assert!(l_black < 1.0, "Black L should be near 0, got {l_black}");
        assert!(l_white > 99.0, "White L should be near 100, got {l_white}");
    }
}
```

- [ ] **Step 6: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-color --lib -- lab
```

Expected: all 4 tests pass.

- [ ] **Step 7: Run downstream crate tests**

Verify `cpu.rs` tests still pass (they call `srgb_to_lab`/`lab_to_srgb` via the unchanged public API):

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu --lib
```

Expected: all tests pass (tolerances may need adjustment if any test checks exact CIELab values — if so, update the expected values to match OKLab output).

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-color/src/lab.rs
git commit -m "feat(color): replace CIELab with OKLab (rescaled to CIELab ranges)

OKLab provides better perceptual uniformity for color manipulation.
Output rescaled to match CIELab numeric ranges (L×100, a×300, b×300)
so all downstream constants (ambiance, warmth, vibrance) stay unchanged.

Ref: docs/decisions/2026-04-09-oklab-grade-pipeline.md"
```

---

## Task 2: Replace CIELab with OKLab + `__powf` in CUDA kernel (`grade_pixel.cuh`)

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh`

- [ ] **Step 1: Replace the colorspace section header and constants**

Replace lines 28–36 of `grade_pixel.cuh` (the CIELab header and constants):

```cuda
// -------------------------------------------------------------------------
// OKLab colorspace, rescaled to CIELab-compatible ranges.
// Matches dorea-color/src/lab.rs exactly.
//
// OKLab (Björn Ottosson, 2020) with output rescaled:
//   L: OKLab [0,1] × 100 → [0,100]
//   a: OKLab [-0.4,0.4] × 300 → [-120,120]
//   b: OKLab [-0.4,0.4] × 300 → [-120,120]
// -------------------------------------------------------------------------

#define L_SCALE  100.0f
#define AB_SCALE 300.0f
```

- [ ] **Step 2: Replace `srgb_to_linear` and `linear_to_srgb` with `__powf`**

Replace lines 38–44 with:

```cuda
__device__ __forceinline__ float srgb_to_linear(float v) {
    return (v <= 0.04045f) ? (v / 12.92f) : __powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return (v <= 0.0031308f) ? (v * 12.92f) : (1.055f * __powf(v, 1.0f / 2.4f) - 0.055f);
}
```

- [ ] **Step 3: Remove CIELab helper functions**

Delete `f_lab` and `f_lab_inv` functions entirely (lines 46–53 in the original). They are no longer needed.

- [ ] **Step 4: Replace `srgb_to_lab`**

Replace the existing `srgb_to_lab` function (was lines 55–72) with:

```cuda
__device__ void srgb_to_lab(float r, float g, float b,
                              float* l_out, float* a_out, float* b_lab_out) {
    float rl = srgb_to_linear(r);
    float gl = srgb_to_linear(g);
    float bl = srgb_to_linear(b);

    // Linear RGB → LMS
    float l = 0.4122214708f*rl + 0.5363325363f*gl + 0.0514459929f*bl;
    float m = 0.2119034982f*rl + 0.6806995451f*gl + 0.1073969566f*bl;
    float s = 0.0883024619f*rl + 0.2817188376f*gl + 0.6299787005f*bl;

    // Cube root
    float l_ = cbrtf(fmaxf(l, 0.0f));
    float m_ = cbrtf(fmaxf(m, 0.0f));
    float s_ = cbrtf(fmaxf(s, 0.0f));

    // LMS' → OKLab, rescaled to CIELab ranges
    *l_out     = (0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_) * L_SCALE;
    *a_out     = (1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_) * AB_SCALE;
    *b_lab_out = (0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_) * AB_SCALE;
}
```

- [ ] **Step 5: Replace `lab_to_srgb`**

Replace the existing `lab_to_srgb` function (was lines 74–91) with:

```cuda
__device__ void lab_to_srgb(float l, float a, float b_lab,
                              float* r_out, float* g_out, float* b_out) {
    // Unscale from CIELab-compatible ranges to native OKLab
    float ok_l = l / L_SCALE;
    float ok_a = a / AB_SCALE;
    float ok_b = b_lab / AB_SCALE;

    // OKLab → LMS'
    float l_ = ok_l + 0.3963377774f*ok_a + 0.2158037573f*ok_b;
    float m_ = ok_l - 0.1055613458f*ok_a - 0.0638541728f*ok_b;
    float s_ = ok_l - 0.0894841775f*ok_a - 1.2914855480f*ok_b;

    // Cube (inverse of cube root)
    float lc = l_ * l_ * l_;
    float mc = m_ * m_ * m_;
    float sc = s_ * s_ * s_;

    // LMS → linear RGB
    float rl =  4.0767416621f*lc - 3.3077115913f*mc + 0.2309699292f*sc;
    float gl = -1.2684380046f*lc + 2.6097574011f*mc - 0.3413193965f*sc;
    float bl =  0.0208763419f*lc - 0.7034186147f*mc + 1.6825422694f*sc;

    *r_out = fminf(fmaxf(linear_to_srgb(fmaxf(rl, 0.0f)), 0.0f), 1.0f);
    *g_out = fminf(fmaxf(linear_to_srgb(fmaxf(gl, 0.0f)), 0.0f), 1.0f);
    *b_out = fminf(fmaxf(linear_to_srgb(fmaxf(bl, 0.0f)), 0.0f), 1.0f);
}
```

- [ ] **Step 6: Build the CUDA kernels**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-gpu
```

Expected: compiles without errors. All three `.cu` files that include `grade_pixel.cuh` (`build_combined_lut.cu`, `combined_lut.cu`, `postprocess.cu`) pick up the new OKLab functions.

- [ ] **Step 7: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu --lib
```

Expected: all tests pass. If any test compares CPU output against CUDA output with tight tolerances, the tolerances may need loosening slightly since OKLab produces different values than CIELab for the same input. Acceptable delta: any pixel-level difference < 3/255 (imperceptible).

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh
git commit -m "feat(gpu): OKLab + __powf in CUDA grade pipeline

Replace CIELab with OKLab in grade_pixel.cuh (matches lab.rs change).
Switch powf → __powf for sRGB gamma (~2× faster on SFU, ~2 ULP accuracy,
invisible at 10-bit).

Ref: docs/decisions/2026-04-09-oklab-grade-pipeline.md"
```

---

## Task 3: Verify CPU/GPU parity

**Files:**
- No file changes — verification only.

- [ ] **Step 1: Run the combined LUT accuracy test**

The existing test `combined_lut_accuracy` in `crates/dorea-gpu/src/cuda/mod.rs` compares `grade_pixel_cpu()` (Rust, uses `lab.rs`) against `grade_pixel_device()` (CUDA, uses `grade_pixel.cuh`). Both now use OKLab, so they should match.

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu -- combined_lut 2>&1
```

Expected: passes. If the test enforces tight per-pixel tolerances, the `__powf` vs `powf` difference (~2 ULP) may cause failures. In that case, loosen the tolerance from 1/255 to 2/255 — this is expected and documented in the spec.

- [ ] **Step 2: Run full workspace tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test --workspace 2>&1
```

Expected: all tests pass.

- [ ] **Step 3: Commit any tolerance adjustments**

If Step 1 or 2 required tolerance changes:

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add -A
git commit -m "test: adjust CPU/GPU parity tolerances for OKLab + __powf

__powf has ~2 ULP vs powf's ~1 ULP accuracy. This is invisible at 10-bit
but may cause 1-count (1/255) differences in CPU vs GPU comparison tests."
```

If no changes were needed, skip this step.
