# depth_aware_ambiance CPU-Only Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `depth_aware_ambiance` (+ warmth + blend) from `grade_frame_cuda` so it runs on CPU *after* GPU resources are released, eliminating the GPU-hold bottleneck.

**Architecture:** Extract a `finish_grade` helper in `cpu.rs` that handles ambiance + warmth + blend + u8 conversion. `grade_frame_cuda` becomes GPU-only (LUT+HSL), returning an `f32` intermediate. `lib.rs` calls `finish_grade` on CPU after CUDA returns, and `grade_frame_cpu` reuses the same helper.

**Tech Stack:** Rust, `dorea-gpu` crate (`cpu.rs`, `cuda/mod.rs`, `lib.rs`)

---

## File Map

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cpu.rs` | Extract `finish_grade` from `grade_frame_cpu`; keep `depth_aware_ambiance` pub |
| `crates/dorea-gpu/src/cuda/mod.rs` | Remove ambiance/warmth/blend/u8 from `grade_frame_cuda`; return `Vec<f32>` |
| `crates/dorea-gpu/src/lib.rs` | Call `cpu::finish_grade` after CUDA returns |

---

### Task 1: Extract `finish_grade` helper in `cpu.rs`

**Files:**
- Modify: `crates/dorea-gpu/src/cpu.rs`

This extracts the last three steps of `grade_frame_cpu` (ambiance + warmth + blend + u8 out) into a reusable `pub(crate) fn finish_grade`. Both the CPU path and the refactored CUDA path will call it.

- [ ] **Step 1: Write the failing test for `finish_grade`**

Add to the `#[cfg(test)]` block at the bottom of `cpu.rs`:

```rust
#[test]
fn finish_grade_roundtrip() {
    use crate::GradeParams;
    use dorea_cal::Calibration;

    let width = 2;
    let height = 2;
    let n = width * height;
    // Grey pixels, f32
    let mut rgb_f32: Vec<f32> = vec![0.5; n * 3];
    let orig_pixels: Vec<u8> = vec![128u8; n * 3];
    let depth: Vec<f32> = vec![0.5; n];
    let params = GradeParams::default();
    let cal = Calibration::default();

    let out = finish_grade(&mut rgb_f32, &orig_pixels, &depth, width, height, &params, &cal);
    assert_eq!(out.len(), n * 3);
    for &v in &out {
        assert!(v <= 255, "out-of-range u8: {v}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu finish_grade_roundtrip 2>&1 | tail -20
```

Expected: `error[E0425]: cannot find function \`finish_grade\``

- [ ] **Step 3: Add `finish_grade` to `cpu.rs`**

In `cpu.rs`, after the closing `}` of `depth_aware_ambiance` (line 115) and before `apply_clarity`, add this function. Also add `use dorea_cal::Calibration;` to the imports if not already present (it is not — `grade_frame_cpu` uses it; check the top of the file).

Add `use dorea_cal::Calibration;` to the existing imports at the top of `cpu.rs` (it's already imported for `grade_frame_cpu` — verify it's there):

```rust
// cpu.rs top imports (verify these are already present — they are):
use dorea_cal::Calibration;
use dorea_color::lab::{srgb_to_lab, lab_to_srgb};
use dorea_lut::apply::apply_depth_luts;
use dorea_hsl::apply::apply_hsl_corrections;
use crate::GradeParams;
```

Insert `finish_grade` between `depth_aware_ambiance` and `apply_clarity` (after line 115, before line 117):

```rust
/// Apply depth_aware_ambiance, warmth, blend, and convert f32 → u8.
///
/// Called by both the CPU and CUDA code paths after LUT+HSL processing.
/// `rgb_f32` is modified in place (ambiance + warmth applied).
/// `orig_pixels` is the original u8 input used for the strength blend.
pub(crate) fn finish_grade(
    rgb_f32: &mut Vec<f32>,
    orig_pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    params: &GradeParams,
    _cal: &Calibration,
) -> Vec<u8> {
    let n = width * height;

    // 1. Depth-aware ambiance (shadow lift, S-curve, clarity, etc.)
    depth_aware_ambiance(rgb_f32, depth, width, height, params.contrast);

    // 2. Warmth (scale LAB a*/b*)
    if (params.warmth - 1.0).abs() > 1e-4 {
        let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_f32[i * 3];
            let g = rgb_f32[i * 3 + 1];
            let b = rgb_f32[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_f32[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }
    }

    // 3. Blend with original using strength
    if params.strength < 1.0 - 1e-4 {
        for i in 0..rgb_f32.len() {
            let orig = orig_pixels[i] as f32 / 255.0;
            rgb_f32[i] = orig * (1.0 - params.strength) + rgb_f32[i] * params.strength;
        }
    }

    // 4. f32 → u8
    rgb_f32.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu finish_grade_roundtrip 2>&1 | tail -20
```

Expected: `test tests::finish_grade_roundtrip ... ok`

- [ ] **Step 5: Refactor `grade_frame_cpu` to use `finish_grade`**

Replace the body of `grade_frame_cpu` in `cpu.rs`. The current code (lines 201–272) does LUT, HSL, ambiance, warmth, blend, u8. After the refactor it does LUT+HSL then calls `finish_grade`:

```rust
/// Full CPU grading pipeline: LUT apply → HSL correct → depth_aware_ambiance → user params.
pub fn grade_frame_cpu(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
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

    // 3–5. Ambiance + warmth + blend + u8 (CPU finish pass)
    Ok(finish_grade(&mut rgb_f32, pixels, depth, width, height, params, calibration))
}
```

- [ ] **Step 6: Run all cpu.rs tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu 2>&1 | tail -30
```

Expected: all existing tests pass (`depth_aware_ambiance_deterministic`, `depth_aware_ambiance_zero_contrast`, `finish_grade_roundtrip`).

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/src/cpu.rs
git commit -m "refactor(dorea-gpu): extract finish_grade helper for CPU post-GPU pass"
```

---

### Task 2: Make `grade_frame_cuda` GPU-only (return `Vec<f32>`)

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`

Remove `depth_aware_ambiance`, warmth, blend, and `f32 → u8` conversion from `grade_frame_cuda`. Return `Vec<f32>` (the LUT+HSL output) instead of `Vec<u8>`.

- [ ] **Step 1: Verify the compile error before editing**

The test in Task 1 Step 6 passes. Now we know `finish_grade` exists. Edit `cuda/mod.rs` next.

- [ ] **Step 2: Rewrite `grade_frame_cuda` signature and body**

Replace the full content of `grade_frame_cuda` in `crates/dorea-gpu/src/cuda/mod.rs` (lines 46–146):

```rust
/// Attempt GPU-accelerated grading: LUT apply + HSL correct only.
///
/// Returns the LUT+HSL-processed pixels as f32 [0,1], interleaved RGB.
/// The caller is responsible for applying `cpu::finish_grade` (depth_aware_ambiance,
/// warmth, blend, u8 conversion) after this function returns, so GPU resources
/// are freed before the CPU-heavy ambiance pass begins.
///
/// Returns `Err` on any CUDA failure so the caller can fall back to full CPU.
#[cfg(feature = "cuda")]
pub fn grade_frame_cuda(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    _params: &GradeParams,
) -> Result<Vec<f32>, GpuError> {
    let n = width * height;

    // --- u8 → f32 ---
    let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // --- Flatten LUT data ---
    let depth_luts = &calibration.depth_luts;
    let n_zones = depth_luts.n_zones();
    let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };

    let luts_flat: Vec<f32> = depth_luts.luts.iter()
        .flat_map(|lg| lg.data.iter().copied())
        .collect();

    // --- GPU: LUT apply ---
    let mut rgb_after_lut = vec![0.0f32; n * 3];
    let status = unsafe {
        dorea_lut_apply_gpu(
            rgb_f32.as_ptr(),
            depth.as_ptr(),
            luts_flat.as_ptr(),
            depth_luts.zone_boundaries.as_ptr(),
            rgb_after_lut.as_mut_ptr(),
            n as i32,
            lut_size as i32,
            n_zones as i32,
        )
    };
    if status != 0 {
        return Err(GpuError::Cuda(format!("dorea_lut_apply_gpu returned CUDA error {status}")));
    }

    // --- Extract HSL arrays (6 qualifiers) ---
    let mut h_offsets = [0.0f32; 6];
    let mut s_ratios  = [1.0f32; 6];
    let mut v_offsets = [0.0f32; 6];
    let mut weights   = [0.0f32; 6];
    for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
        h_offsets[i] = q.h_offset;
        s_ratios[i]  = q.s_ratio;
        v_offsets[i] = q.v_offset;
        weights[i]   = q.weight;
    }

    // --- GPU: HSL correct ---
    let mut rgb_after_hsl = vec![0.0f32; n * 3];
    let status = unsafe {
        dorea_hsl_correct_gpu(
            rgb_after_lut.as_ptr(),
            rgb_after_hsl.as_mut_ptr(),
            h_offsets.as_ptr(),
            s_ratios.as_ptr(),
            v_offsets.as_ptr(),
            weights.as_ptr(),
            n as i32,
        )
    };
    if status != 0 {
        return Err(GpuError::Cuda(format!("dorea_hsl_correct_gpu returned CUDA error {status}")));
    }

    // GPU work done. Return f32 intermediate — caller applies CPU finish pass.
    Ok(rgb_after_hsl)
}
```

Also remove the now-unused `use dorea_color::lab::{srgb_to_lab, lab_to_srgb};` import from the `#[cfg(feature = "cuda")]` block at the top of `cuda/mod.rs`.

- [ ] **Step 3: Update `lib.rs` to call `cpu::finish_grade` after CUDA returns**

Replace the `grade_frame` function body in `crates/dorea-gpu/src/lib.rs`:

```rust
pub fn grade_frame(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    if pixels.len() != width * height * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels length {} != width*height*3 {}",
            pixels.len(),
            width * height * 3
        )));
    }
    if depth.len() != width * height {
        return Err(GpuError::InvalidInput(format!(
            "depth length {} != width*height {}",
            depth.len(),
            width * height
        )));
    }

    #[cfg(feature = "cuda")]
    {
        match cuda::grade_frame_cuda(pixels, depth, width, height, calibration, params) {
            Ok(mut rgb_f32) => {
                // GPU resources are now freed. Apply CPU-only ambiance + warmth + blend.
                return Ok(cpu::finish_grade(
                    &mut rgb_f32,
                    pixels,
                    depth,
                    width,
                    height,
                    params,
                    calibration,
                ));
            }
            Err(e) => {
                log::warn!("CUDA grading failed ({e}), falling back to CPU");
            }
        }
    }

    cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
        .map_err(|e| GpuError::InvalidInput(e.to_string()))
}
```

- [ ] **Step 4: Fix any unused import warnings in `cuda/mod.rs`**

Remove `use dorea_color::lab::{srgb_to_lab, lab_to_srgb};` from the `#[cfg(feature = "cuda")]` imports at the top of `cuda/mod.rs` (it's no longer needed since warmth calculation was removed):

The `use` block at lines 7-12 of `cuda/mod.rs` should become:

```rust
#[cfg(feature = "cuda")]
use crate::{GradeParams, GpuError};
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;
```

(Remove the `dorea_color` import line.)

- [ ] **Step 5: Build and verify**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-gpu 2>&1 | tail -30
```

Expected: compiles cleanly, no errors, possibly no warnings.

- [ ] **Step 6: Run all tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/src/cuda/mod.rs crates/dorea-gpu/src/lib.rs
git commit -m "fix(dorea-gpu): move depth_aware_ambiance out of CUDA path to CPU-only finish pass"
```

---

### Task 3: Run full crate tests + integration check

**Files:** none changed

- [ ] **Step 1: Run full workspace tests touching dorea-gpu**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test --workspace 2>&1 | grep -E "test |FAILED|error" | tail -40
```

Expected: all tests pass.

- [ ] **Step 2: Verify CLI callers still build**

The public API of `dorea-gpu` hasn't changed (`grade_frame` still takes and returns the same types). Verify:

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-cli 2>&1 | tail -20
```

Expected: builds cleanly.

- [ ] **Step 3: Commit if any fixes needed, otherwise done**

If clean, no commit needed. Branch `fix/inference-gpu-bugs` is ready to review.

---

## Self-Review

**Spec coverage:**
- ✅ `depth_aware_ambiance` no longer runs inside `grade_frame_cuda`
- ✅ GPU resources freed before CPU ambiance pass
- ✅ CPU path unchanged in output semantics (both paths call `finish_grade`)
- ✅ Public API of `grade_frame` unchanged

**No placeholders:** All code is complete.

**Type consistency:** `finish_grade` signature used in Task 1 matches usage in Task 2 exactly.
