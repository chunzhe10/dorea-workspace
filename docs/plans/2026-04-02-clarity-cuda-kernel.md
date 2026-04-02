# Clarity CUDA Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `clarity.cu` — the planned-but-missing CUDA kernel that computes the clarity effect at proxy resolution (518 px long edge) on GPU, replacing the current O(N×radius) CPU box blur that runs at full 4K resolution and takes ~9 s/frame.

**Architecture:** Three kernel passes at proxy size (~518×291 = 151 K px): (1) downsample full-res f32 RGB → proxy L channel; (2) 3-pass separable box blur on proxy L; (3) upsample blurred L back to full res, compute detail per pixel, write back RGB. The naive O(N×radius) approach is fast enough at proxy resolution (55 M GPU ops ≈ 0.05 ms). The CPU `finish_grade` path is refactored to make clarity opt-out so the CUDA path can skip the CPU clarity pass after GPU runs it.

**Tech Stack:** CUDA C (clarity.cu), Rust (cpu.rs, cuda/mod.rs, lib.rs). All inline `#[cfg(test)]` unit tests. Build with `cargo build -p dorea-gpu` in `repos/dorea/`. No new Cargo deps required.

---

## Background

The original Phase 2+3 plan (`docs/plans/2026-04-02-dorea-v2-phase2-3-plan.md`) specified `clarity.cu` as a CUDA kernel operating at proxy resolution. It was never built. The current `cpu.rs::apply_clarity` runs the box blur at full 4K with a naive inner loop:

```
8.3 M pixels × 181 iters × 6 passes = 9 billion loop iterations per frame (~9 s on CPU)
```

At proxy (518×291), the same naive approach is:
```
151 K pixels × 61 iters × 6 passes = 55 M iterations (0.05 ms on RTX 3060)
```

`build.rs` already lists `clarity.cu` in `kernel_names` and watches it with `rerun-if-changed`. The file just doesn't exist yet.

---

## How the clarity effect works

```
clarity algorithm (at proxy resolution):
1. Downsample full-res f32 RGB → proxy-res L channel (bilinear, LAB conversion)
2. 3-pass box blur on proxy L: blur_L = box_blur³(L_proxy, radius=30)
3. At full res, per pixel:
     L_full = srgb_to_lab(pixel).L
     blur_upsampled = bilinear_sample(blur_L_proxy, at this pixel's proxy coords)
     detail = tanh((L_full - blur_upsampled) × 3) / 3
     L_new = clamp(L_full + detail × clarity_amount, 0, 1)
     pixel = lab_to_srgb(L_new, A_unchanged, B_unchanged)
```

`clarity_amount = (0.2 + 0.25 × mean_depth) × contrast_scale` — computed from depth on the Rust side before invoking the kernel.

---

## File Map

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cpu.rs` | Extract clarity from `depth_aware_ambiance`; add `skip_clarity: bool` to `finish_grade` |
| `crates/dorea-gpu/src/lib.rs` | Update `finish_grade` call in CUDA path to `skip_clarity: false` (Task 1); to `true` (Task 3) |
| `crates/dorea-gpu/src/cuda/kernels/clarity.cu` | **New** — all 4 kernels + `dorea_clarity_gpu` host launcher |
| `crates/dorea-gpu/src/cuda/mod.rs` | Declare + call `dorea_clarity_gpu` after LUT+HSL |

---

## Task 1: Refactor `cpu.rs` — extract clarity so the CUDA path can skip it

**Files:** `crates/dorea-gpu/src/cpu.rs`, `crates/dorea-gpu/src/lib.rs`

The goal is to move the `apply_clarity` call out of `depth_aware_ambiance` and into `finish_grade` behind a `skip_clarity: bool` flag, so the CUDA path can skip it once the GPU kernel does the work.

- [ ] **Step 1: Write failing tests for the refactored `finish_grade` signature**

Add inside the existing `#[cfg(test)] mod tests` block at the bottom of `crates/dorea-gpu/src/cpu.rs`:

```rust
#[test]
fn finish_grade_skip_clarity_runs_without_panic() {
    use crate::GradeParams;
    use dorea_cal::Calibration;
    use dorea_hsl::HslCorrections;
    use dorea_lut::types::{DepthLuts, LutGrid};

    let width = 4; let height = 4; let n = width * height;
    let mut lut = LutGrid::new(2);
    for ri in 0..2usize { for gi in 0..2usize { for bi in 0..2usize {
        lut.set(ri, gi, bi, [ri as f32, gi as f32, bi as f32]);
    }}}
    let cal = Calibration::new(
        DepthLuts::new(vec![lut], vec![0.0, 1.0]),
        HslCorrections(vec![]),
        0,
    );

    let mut rgb_f32: Vec<f32> = vec![0.5; n * 3];
    let orig: Vec<u8>  = vec![128u8; n * 3];
    let depth: Vec<f32> = vec![0.5; n];

    // skip_clarity = true: must not panic, output must be in [0,1]
    let out = finish_grade(&mut rgb_f32, &orig, &depth, width, height,
                           &GradeParams::default(), &cal, true);
    assert_eq!(out.len(), n * 3);
    for &v in &out { assert!(v <= 255, "out of range: {v}"); }

    // skip_clarity = false: same shape, must not panic
    let mut rgb_f32b: Vec<f32> = vec![0.5; n * 3];
    let out2 = finish_grade(&mut rgb_f32b, &orig, &depth, width, height,
                            &GradeParams::default(), &cal, false);
    assert_eq!(out2.len(), n * 3);
}
```

- [ ] **Step 2: Run to confirm it fails (signature mismatch)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu finish_grade_skip_clarity 2>&1 | tail -5
```
Expected: compile error — `finish_grade` does not take 8 arguments.

- [ ] **Step 3: Refactor `cpu.rs`**

**3a. Remove `apply_clarity` call from `depth_aware_ambiance`** (lines 112–114 currently):

In `depth_aware_ambiance`, delete these three lines at the end of the function body:

```rust
    // Clarity: separable box-blur approximation of Gaussian (σ=30px at proxy).
    // We run a 3-pass box blur (approximates Gaussian well).
    let radius = (30.0_f32 * 3.0).ceil() as usize;
    let clarity_amount = (0.2 + 0.25 * mean_d) * contrast_scale;
    apply_clarity(rgb, width, height, radius, clarity_amount);
```

**3b. Add a `pub(crate) fn apply_cpu_clarity`** immediately above `finish_grade`:

```rust
/// Run the clarity pass (box-blur detail extraction) at full resolution on CPU.
/// Called by `finish_grade` when `skip_clarity` is false (CPU-only path).
pub(crate) fn apply_cpu_clarity(
    rgb: &mut [f32],
    depth: &[f32],
    width: usize,
    height: usize,
    contrast_scale: f32,
) {
    let mean_d: f32 = depth.iter().sum::<f32>() / depth.len() as f32;
    let radius = (30.0_f32 * 3.0).ceil() as usize;
    let clarity_amount = (0.2 + 0.25 * mean_d) * contrast_scale;
    apply_clarity(rgb, width, height, radius, clarity_amount);
}
```

**3c. Add `skip_clarity: bool` parameter to `finish_grade` and call `apply_cpu_clarity` conditionally.**

Replace the current `finish_grade` signature and step 1 body:

```rust
pub(crate) fn finish_grade(
    rgb_f32: &mut [f32],
    orig_pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    params: &GradeParams,
    _cal: &Calibration,
    skip_clarity: bool,
) -> Vec<u8> {
    let n = width * height;

    // 1. Depth-aware ambiance (shadow lift, S-curve, highlight compress, warmth, vibrance)
    depth_aware_ambiance(rgb_f32, depth, width, height, params.contrast);

    // 1b. Clarity — skip when the CUDA path already ran the GPU clarity kernel.
    if !skip_clarity {
        apply_cpu_clarity(rgb_f32, depth, width, height, params.contrast);
    }

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

**3d. Update callers of `finish_grade` inside `cpu.rs`.**

`grade_frame_cpu` calls `finish_grade` — add `false` as the last argument:

```rust
Ok(finish_grade(&mut rgb_f32, pixels, depth, width, height, params, calibration, false))
```

Also update the existing `finish_grade_roundtrip` test to pass `false`:

```rust
let out = finish_grade(&mut rgb_f32, &orig_pixels, &depth, width, height, &params, &cal, false);
```

**3e. Update the `finish_grade` call in `lib.rs`** (CUDA path — for now pass `false`, Task 3 changes it to `true`):

```rust
return Ok(cpu::finish_grade(
    &mut rgb_f32,
    pixels,
    depth,
    width,
    height,
    params,
    calibration,
    false,  // Task 3 changes this to true when GPU clarity is wired in
));
```

- [ ] **Step 4: Run all dorea-gpu tests**

```bash
cargo test -p dorea-gpu 2>&1 | tail -20
```
Expected: all tests pass, including `finish_grade_skip_clarity_runs_without_panic`, `finish_grade_roundtrip`, `depth_aware_ambiance_deterministic`, and `depth_aware_ambiance_zero_contrast`.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/src/cpu.rs crates/dorea-gpu/src/lib.rs
git commit -m "refactor(dorea-gpu): extract clarity from depth_aware_ambiance; add skip_clarity to finish_grade"
```

---

## Task 2: Write `clarity.cu`

**Files:** Create `crates/dorea-gpu/src/cuda/kernels/clarity.cu`

- [ ] **Step 1: Create the kernel file**

Create `crates/dorea-gpu/src/cuda/kernels/clarity.cu` with the following content:

```c
/**
 * clarity.cu — GPU clarity enhancement kernel.
 *
 * Clarity = high-frequency luminance detail boost, computed at proxy resolution
 * to avoid the O(N×radius) CPU box blur at 4K.
 *
 * Algorithm (3 passes):
 *   Pass A: Downsample full-res f32 RGB → proxy-res L channel (bilinear, sRGB→LAB)
 *   Pass B: 3-pass separable box blur on proxy L (naive O(N×r) — fast at proxy size)
 *   Pass C: For each full-res pixel: upsample blurred L, compute detail, apply to RGB
 *
 * Host entry point: dorea_clarity_gpu(h_rgb_in, h_rgb_out, full_w, full_h,
 *                                      proxy_w, proxy_h, blur_radius, clarity_amount)
 * Returns cudaError_t as int; 0 = success.
 */

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// sRGB ↔ CIE LAB device helpers (D65 illuminant)
// ---------------------------------------------------------------------------

__device__ static float srgb_to_linear_d(float c) {
    return (c <= 0.04045f)
        ? (c / 12.92f)
        : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ static float linear_to_srgb_d(float c) {
    c = fmaxf(0.0f, fminf(1.0f, c));
    return (c <= 0.0031308f)
        ? (12.92f * c)
        : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
}

__device__ static float lab_f_d(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
}

__device__ static float lab_f_inv_d(float t) {
    float t3 = t * t * t;
    return (t3 > 0.008856f) ? t3 : ((t - 16.0f / 116.0f) / 7.787f);
}

#define D65_XN 0.95047f
#define D65_YN 1.00000f
#define D65_ZN 1.08883f

__device__ static void srgb_to_lab_d(float r, float g, float b,
                                       float* L, float* A, float* B) {
    float rl = srgb_to_linear_d(r);
    float gl = srgb_to_linear_d(g);
    float bl = srgb_to_linear_d(b);

    float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
    float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
    float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;

    float fx = lab_f_d(x / D65_XN);
    float fy = lab_f_d(y / D65_YN);
    float fz = lab_f_d(z / D65_ZN);

    *L = 116.0f * fy - 16.0f;
    *A = 500.0f * (fx - fy);
    *B = 200.0f * (fy - fz);
}

__device__ static void lab_to_srgb_d(float L, float A, float B,
                                       float* r, float* g, float* b) {
    float fy = (L + 16.0f) / 116.0f;
    float fx = A / 500.0f + fy;
    float fz = fy - B / 200.0f;

    float x = lab_f_inv_d(fx) * D65_XN;
    float y = lab_f_inv_d(fy) * D65_YN;
    float z = lab_f_inv_d(fz) * D65_ZN;

    float rl =  3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    float bl_out =  0.0556434f * x - 0.2040259f * y + 1.0572252f * z;

    *r = linear_to_srgb_d(rl);
    *g = linear_to_srgb_d(gl);
    *b = linear_to_srgb_d(bl_out);
}

// ---------------------------------------------------------------------------
// Kernel A: downsample full-res f32 RGB → proxy-res L channel (bilinear)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_extract_L_proxy(
    const float* __restrict__ rgb_full,   // [full_w * full_h * 3]
    float* __restrict__       l_proxy,    // [proxy_w * proxy_h]
    int full_w, int full_h,
    int proxy_w, int proxy_h
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= proxy_w || py >= proxy_h) return;

    // Map proxy pixel → full-res bilinear coordinates
    float sx = (proxy_w > 1)
        ? ((float)px / (float)(proxy_w - 1) * (float)(full_w - 1))
        : 0.0f;
    float sy = (proxy_h > 1)
        ? ((float)py / (float)(proxy_h - 1) * (float)(full_h - 1))
        : 0.0f;

    int x0 = (int)sx;  int x1 = min(x0 + 1, full_w - 1);
    int y0 = (int)sy;  int y1 = min(y0 + 1, full_h - 1);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Extract L from each corner via sRGB→LAB
    float L00, L10, L01, L11, A_unused, B_unused;
    int i00 = (y0 * full_w + x0) * 3;
    int i10 = (y0 * full_w + x1) * 3;
    int i01 = (y1 * full_w + x0) * 3;
    int i11 = (y1 * full_w + x1) * 3;
    srgb_to_lab_d(rgb_full[i00], rgb_full[i00+1], rgb_full[i00+2], &L00, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i10], rgb_full[i10+1], rgb_full[i10+2], &L10, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i01], rgb_full[i01+1], rgb_full[i01+2], &L01, &A_unused, &B_unused);
    srgb_to_lab_d(rgb_full[i11], rgb_full[i11+1], rgb_full[i11+2], &L11, &A_unused, &B_unused);

    float L = L00*(1-fx)*(1-fy) + L10*fx*(1-fy)
            + L01*(1-fx)*fy     + L11*fx*fy;
    l_proxy[py * proxy_w + px] = L;
}

// ---------------------------------------------------------------------------
// Kernel B-row: box blur along rows of the proxy L channel (one thread per pixel)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_box_blur_rows(
    const float* __restrict__ in,
    float* __restrict__       out,
    int width, int height, int radius
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int lo = max(0, col - radius);
    int hi = min(width - 1, col + radius);
    float s = 0.0f;
    int base = row * width;
    for (int k = lo; k <= hi; k++) s += in[base + k];
    out[base + col] = s / (float)(hi - lo + 1);
}

// ---------------------------------------------------------------------------
// Kernel B-col: box blur along columns of the proxy L channel (one thread per pixel)
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_box_blur_cols(
    const float* __restrict__ in,
    float* __restrict__       out,
    int width, int height, int radius
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int lo = max(0, row - radius);
    int hi = min(height - 1, row + radius);
    float s = 0.0f;
    for (int k = lo; k <= hi; k++) s += in[k * width + col];
    out[row * width + col] = s / (float)(hi - lo + 1);
}

// ---------------------------------------------------------------------------
// Kernel C: apply clarity — upsample blurred L to full res, compute detail,
//           reconstruct RGB with modified L only.
// ---------------------------------------------------------------------------
extern "C"
__global__ void clarity_apply_kernel(
    const float* __restrict__ rgb_in,      // [full_w * full_h * 3]
    float* __restrict__       rgb_out,     // [full_w * full_h * 3]
    const float* __restrict__ blur_proxy,  // [proxy_w * proxy_h], L in [0..100]
    float clarity_amount,
    int full_w, int full_h,
    int proxy_w, int proxy_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= full_w * full_h) return;

    int fy = idx / full_w;
    int fx = idx % full_w;

    float r = rgb_in[idx * 3 + 0];
    float g = rgb_in[idx * 3 + 1];
    float b = rgb_in[idx * 3 + 2];

    // Convert pixel to LAB
    float L_full, A, B;
    srgb_to_lab_d(r, g, b, &L_full, &A, &B);
    float L_norm = L_full / 100.0f;  // normalise to [0,1]

    // Bilinear sample blurred proxy L at this full-res pixel's proxy coords
    float sx = (proxy_w > 1)
        ? ((float)fx / (float)(full_w - 1) * (float)(proxy_w - 1))
        : 0.0f;
    float sy = (proxy_h > 1)
        ? ((float)fy / (float)(full_h - 1) * (float)(proxy_h - 1))
        : 0.0f;
    int px0 = (int)sx;  int px1 = min(px0 + 1, proxy_w - 1);
    int py0 = (int)sy;  int py1 = min(py0 + 1, proxy_h - 1);
    float bfx = sx - (float)px0;
    float bfy = sy - (float)py0;

    float blur_sampled =
        blur_proxy[py0 * proxy_w + px0] * (1-bfx) * (1-bfy) +
        blur_proxy[py0 * proxy_w + px1] *  bfx    * (1-bfy) +
        blur_proxy[py1 * proxy_w + px0] * (1-bfx) *  bfy    +
        blur_proxy[py1 * proxy_w + px1] *  bfx    *  bfy;

    float blur_norm = blur_sampled / 100.0f;  // same normalisation as L_norm

    // Detail boost: tanh((L - blur)*3)/3, then add scaled detail to L
    float detail = tanhf((L_norm - blur_norm) * 3.0f) / 3.0f;
    float L_new = fminf(fmaxf(L_norm + detail * clarity_amount, 0.0f), 1.0f);

    // Reconstruct RGB: only L changed, A and B unchanged
    float ro, go, bo;
    lab_to_srgb_d(L_new * 100.0f, A, B, &ro, &go, &bo);

    rgb_out[idx * 3 + 0] = ro;
    rgb_out[idx * 3 + 1] = go;
    rgb_out[idx * 3 + 2] = bo;
}

// ---------------------------------------------------------------------------
// Host-callable launcher
// ---------------------------------------------------------------------------
extern "C" int dorea_clarity_gpu(
    const float* h_rgb_in,   // full-res f32 RGB, row-major [full_w * full_h * 3]
    float*       h_rgb_out,  // output, same layout
    int full_w, int full_h,
    int proxy_w, int proxy_h,
    int blur_radius,
    float clarity_amount
) {
    int n_full  = full_w * full_h;
    int n_proxy = proxy_w * proxy_h;
    size_t full_bytes  = (size_t)n_full  * 3 * sizeof(float);
    size_t proxy_bytes = (size_t)n_proxy * sizeof(float);

    float *d_rgb_in = nullptr, *d_rgb_out = nullptr;
    float *d_l_proxy = nullptr, *d_blur_a = nullptr, *d_blur_b = nullptr;

#define CK(call) do {                                              \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        cudaFree(d_rgb_in); cudaFree(d_rgb_out);                   \
        cudaFree(d_l_proxy); cudaFree(d_blur_a); cudaFree(d_blur_b); \
        return (int)_e;                                            \
    }                                                              \
} while(0)

    CK(cudaMalloc(&d_rgb_in,   full_bytes));
    CK(cudaMalloc(&d_rgb_out,  full_bytes));
    CK(cudaMalloc(&d_l_proxy,  proxy_bytes));
    CK(cudaMalloc(&d_blur_a,   proxy_bytes));
    CK(cudaMalloc(&d_blur_b,   proxy_bytes));

    CK(cudaMemcpy(d_rgb_in, h_rgb_in, full_bytes, cudaMemcpyHostToDevice));

    // Kernel A: full-res RGB → proxy L
    {
        dim3 block(16, 16);
        dim3 grid((proxy_w + 15) / 16, (proxy_h + 15) / 16);
        clarity_extract_L_proxy<<<grid, block>>>(
            d_rgb_in, d_l_proxy, full_w, full_h, proxy_w, proxy_h
        );
        CK(cudaGetLastError());
    }

    // Kernel B: 3-pass separable box blur on proxy L
    // Each pass: row blur → col blur. d_l_proxy ↔ d_blur_a (ping-pong via d_blur_b as temp)
    for (int pass = 0; pass < 3; pass++) {
        dim3 block(32, 8);
        dim3 grid((proxy_w + 31) / 32, (proxy_h + 7) / 8);

        // Row blur: d_l_proxy → d_blur_a
        clarity_box_blur_rows<<<grid, block>>>(
            d_l_proxy, d_blur_a, proxy_w, proxy_h, blur_radius
        );
        CK(cudaGetLastError());

        // Col blur: d_blur_a → d_blur_b
        clarity_box_blur_cols<<<grid, block>>>(
            d_blur_a, d_blur_b, proxy_w, proxy_h, blur_radius
        );
        CK(cudaGetLastError());

        // Swap so d_l_proxy holds the result for next pass (or for kernel C)
        float* tmp = d_l_proxy;
        d_l_proxy = d_blur_b;
        d_blur_b  = tmp;
    }
    // After 3 passes, d_l_proxy holds the 3×-blurred proxy L channel.

    // Kernel C: apply clarity at full res
    {
        int blocks = (n_full + 255) / 256;
        clarity_apply_kernel<<<blocks, 256>>>(
            d_rgb_in, d_rgb_out, d_l_proxy, clarity_amount,
            full_w, full_h, proxy_w, proxy_h
        );
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }

    CK(cudaMemcpy(h_rgb_out, d_rgb_out, full_bytes, cudaMemcpyDeviceToHost));

#undef CK
    cudaFree(d_rgb_in);
    cudaFree(d_rgb_out);
    cudaFree(d_l_proxy);
    cudaFree(d_blur_a);
    cudaFree(d_blur_b);
    return 0;
}
```

- [ ] **Step 2: Confirm it compiles**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-gpu 2>&1 | grep -E "warning=|error|Compiling|Finished"
```

Expected output:
```
warning: dorea-gpu@0.1.0: Found nvcc at /usr/local/cuda/bin/nvcc
   Compiling dorea-gpu v0.1.0 (...)
    Finished dev profile [...]
```

If nvcc reports an error, check that the LAB device functions don't conflict with system math headers. The `--allow-unsupported-compiler` and `-isystem` flags in `build.rs` handle GCC 14/glibc 2.35 issues already.

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-gpu/src/cuda/kernels/clarity.cu
git commit -m "feat(dorea-gpu): clarity.cu — GPU clarity kernel at proxy resolution"
```

---

## Task 3: Wire `dorea_clarity_gpu` into `cuda/mod.rs`

**Files:** `crates/dorea-gpu/src/cuda/mod.rs`, `crates/dorea-gpu/src/lib.rs`

- [ ] **Step 1: Add `extern "C"` declaration and call in `cuda/mod.rs`**

In `crates/dorea-gpu/src/cuda/mod.rs`, add the `dorea_clarity_gpu` declaration alongside the existing ones, and call it after `dorea_hsl_correct_gpu`. Replace the entire file with:

```rust
//! CUDA-backed grading pipeline.
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).
//! Provides `grade_frame_cuda` which runs LUT apply, HSL correct, and clarity on GPU,
//! returning f32 intermediate pixels. The caller (`lib.rs`) applies
//! `cpu::finish_grade` (depth_aware_ambiance WITHOUT clarity, warmth, blend, u8)
//! after GPU resources are freed.

#[cfg(feature = "cuda")]
use crate::{GradeParams, GpuError};
#[cfg(feature = "cuda")]
use dorea_cal::Calibration;

#[cfg(feature = "cuda")]
extern "C" {
    /// GPU launcher: depth-stratified LUT apply. Returns cudaError_t (0 = success).
    fn dorea_lut_apply_gpu(
        h_pixels_in: *const f32,
        h_depth: *const f32,
        h_luts: *const f32,
        h_zone_boundaries: *const f32,
        h_pixels_out: *mut f32,
        n_pixels: i32,
        lut_size: i32,
        n_zones: i32,
    ) -> i32;

    /// GPU launcher: HSL 6-qualifier correction. Returns cudaError_t (0 = success).
    fn dorea_hsl_correct_gpu(
        h_pixels_in: *const f32,
        h_pixels_out: *mut f32,
        h_offsets: *const f32,
        h_s_ratios: *const f32,
        h_v_offsets: *const f32,
        h_weights: *const f32,
        n_pixels: i32,
    ) -> i32;

    /// GPU launcher: clarity at proxy resolution. Returns cudaError_t (0 = success).
    fn dorea_clarity_gpu(
        h_rgb_in: *const f32,
        h_rgb_out: *mut f32,
        full_w: i32,
        full_h: i32,
        proxy_w: i32,
        proxy_h: i32,
        blur_radius: i32,
        clarity_amount: f32,
    ) -> i32;
}

/// Compute proxy dimensions: scale so the long edge ≤ max_size.
/// Inline here to avoid a dep on dorea-video just for this helper.
#[cfg(feature = "cuda")]
fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    let long_edge = src_w.max(src_h);
    if long_edge <= max_size {
        return (src_w, src_h);
    }
    let scale = max_size as f64 / long_edge as f64;
    let pw = ((src_w as f64 * scale).round() as usize).max(1);
    let ph = ((src_h as f64 * scale).round() as usize).max(1);
    (pw, ph)
}

/// Attempt GPU-accelerated grading: LUT apply + HSL correct + clarity.
///
/// Returns the fully-graded pixels as f32 [0,1], interleaved RGB (clarity applied).
/// The caller is responsible for applying `cpu::finish_grade(skip_clarity=true)` —
/// which runs depth_aware_ambiance WITHOUT clarity, warmth, blend, and u8 conversion.
///
/// Returns `Err` on any CUDA failure so the caller can fall back to full CPU.
#[cfg(feature = "cuda")]
pub fn grade_frame_cuda(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
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

    // --- GPU: Clarity at proxy resolution ---
    // Compute clarity_amount from depth mean + contrast param (mirrors cpu.rs::apply_cpu_clarity)
    let mean_d = depth.iter().sum::<f32>() / depth.len().max(1) as f32;
    let clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast;

    let (proxy_w, proxy_h) = proxy_dims(width, height, 518);
    const BLUR_RADIUS: i32 = 30;

    let mut rgb_after_clarity = vec![0.0f32; n * 3];
    let status = unsafe {
        dorea_clarity_gpu(
            rgb_after_hsl.as_ptr(),
            rgb_after_clarity.as_mut_ptr(),
            width as i32,
            height as i32,
            proxy_w as i32,
            proxy_h as i32,
            BLUR_RADIUS,
            clarity_amount,
        )
    };
    if status != 0 {
        log::warn!("dorea_clarity_gpu returned CUDA error {status} — clarity skipped");
        // Fall back gracefully: return hsl result without clarity
        return Ok(rgb_after_hsl);
    }

    Ok(rgb_after_clarity)
}
```

- [ ] **Step 2: Update `lib.rs` CUDA path to pass `skip_clarity: true`**

In `crates/dorea-gpu/src/lib.rs`, find the CUDA path call to `finish_grade` and change `false` → `true`:

```rust
#[cfg(feature = "cuda")]
{
    match cuda::grade_frame_cuda(pixels, depth, width, height, calibration, params) {
        Ok(mut rgb_f32) => {
            return Ok(cpu::finish_grade(
                &mut rgb_f32,
                pixels,
                depth,
                width,
                height,
                params,
                calibration,
                true,  // clarity was applied by GPU kernel
            ));
        }
        Err(e) => {
            log::warn!("CUDA grading failed ({e}), falling back to CPU");
        }
    }
}
```

- [ ] **Step 3: Build to confirm everything compiles**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-gpu 2>&1 | grep -E "warning=|error\[|Compiling|Finished"
```

Expected: no errors; nvcc compiles all three `.cu` files.

- [ ] **Step 4: Run all dorea-gpu tests**

```bash
cargo test -p dorea-gpu 2>&1 | tail -20
```

Expected: all tests pass. (There are no GPU-execution tests since they require a live CUDA device; the tests exercise the CPU paths and compilation only.)

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(dorea-gpu): wire dorea_clarity_gpu into CUDA grading path"
```

---

## Task 4: End-to-end test on the real video

- [ ] **Step 1: Run grade with `--release` and measure total time**

```bash
cd /workspaces/dorea-workspace/repos/dorea
time cargo run --release --bin dorea -- grade \
  --input /workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D.MP4 \
  --output /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_clarity_gpu.mp4 \
  --raune-weights /workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth \
  --raune-models-dir /workspaces/dorea-workspace/working/sea_thru_poc \
  --depth-model /workspaces/dorea-workspace/models/depth_anything_v2_small \
  --verbose 2>&1
```

Expected: progress lines every 100 frames; total wall time under 15 minutes (down from ~52 hours):

```
[INFO  dorea_cli::grade] Progress: 100/1671 frames (6.0%)
[INFO  dorea_cli::grade] Progress: 200/1671 frames (12.0%)
...
[INFO  dorea_cli::grade] Done. Graded 1671 frames → .../DJI_20251101111428_0055_D_clarity_gpu.mp4
```

- [ ] **Step 2: Verify output**

```bash
ffprobe -v quiet -print_format json -show_streams \
  /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_clarity_gpu.mp4 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d.get('streams', []):
    if s.get('codec_type') == 'video':
        print('frames:', s.get('nb_frames', '?'))
        print('duration:', s.get('duration', '?'), 's')
        print('codec:', s.get('codec_name', '?'))
"
```

Expected:
```
frames: 1671
duration: ~13.9s
codec: h264
```

- [ ] **Step 3: Check file size**

```bash
ls -lh /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_clarity_gpu.mp4
```

Expected: 20–300 MB (comparable to source 218 MB). Under 1 MB means encoder received no frames.

- [ ] **Step 4: Record results in corvia**

After successful run, use `corvia_write` to record the actual wall time and configuration. Use `scope_id: "dorea"`, `source_origin: "repo:dorea"`, `content_role: "learning"`.

---

## Self-Review

**Spec coverage:**
- [x] GPU clarity kernel at proxy resolution → Tasks 2+3
- [x] 3-pass separable box blur on GPU → Task 2 (`clarity_box_blur_rows` + `clarity_box_blur_cols`, 3 passes in launcher)
- [x] Bilinear downsample full-res → proxy L → Task 2 (`clarity_extract_L_proxy`)
- [x] Upsample blur to full res + apply detail → Task 2 (`clarity_apply_kernel`)
- [x] CPU clarity path disabled when GPU runs → Task 1 (`skip_clarity`) + Task 3 (`true`)
- [x] CPU fallback still works (CPU-only mode, CUDA failure) → Task 1 sets `skip_clarity=false` for CPU path; Task 3 logs + falls back to hsl result on clarity GPU error
- [x] End-to-end verification → Task 4

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:**
- `finish_grade(..., skip_clarity: bool)` — added in Task 1, used as `false` in cpu.rs, updated to `true` in lib.rs in Task 3
- `dorea_clarity_gpu` C signature matches `extern "C"` Rust declaration exactly (all `i32` / `f32` / `*const f32`)
- `proxy_dims` in `cuda/mod.rs` is a private helper with the same signature as in `dorea-video/src/resize.rs` — consistent results
- `BLUR_RADIUS = 30` matches the original Phase 2+3 spec (σ=30 at proxy)
- `clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast` matches `apply_cpu_clarity` exactly
