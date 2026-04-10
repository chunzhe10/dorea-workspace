# Replace CIELab with OKLab in CUDA Grade Pipeline

## Problem

The per-pixel grading pipeline uses CIELab D65 for the ambiance, warmth, and vibrance
stages. CIELab has known perceptual non-uniformity — equal-magnitude shifts in a*/b*
produce visually unequal color changes depending on hue region. This affects warmth and
vibrance adjustments, which push a*/b* channels proportional to depth.

OKLab (Björn Ottosson, 2020) is designed for perceptual uniformity in color manipulation
tasks. It has the same L/a/b channel structure as CIELab but with better uniformity,
simpler math (no XYZ intermediate, no conditional piecewise functions), and fewer
operations per pixel.

## Decision

Replace CIELab conversion functions with OKLab in the CUDA grade pipeline. Rescale OKLab
output to match CIELab numeric ranges so all tuned manipulation constants (shadow lift,
warmth push, vibrance thresholds) remain unchanged.

### Rescale mapping

```
OKLab L ∈ [0, 1]       → × 100  → L ∈ [0, 100]    (matches CIELab L)
OKLab a ∈ [-0.4, 0.4]  → × 300  → a ∈ [-120, 120]  (approximates CIELab a range)
OKLab b ∈ [-0.4, 0.4]  → × 300  → b ∈ [-120, 120]  (approximates CIELab b range)
```

The `× 300` factor maps OKLab's native `[-0.4, 0.4]` range to approximately match
CIELab's `[-128, 127]` range. This is close enough that all hardcoded constants
(warmth push magnitudes, vibrance chroma thresholds at `40.0`, a/b clamping at
`[-128, 127]`) remain valid without retuning.

### OKLab conversion path

```
srgb_to_lab:
  sRGB → linear (piecewise gamma, unchanged)
  → LMS via 3×3 matrix
  → cube root (cbrtf, no conditional)
  → OKLab via 3×3 matrix
  → rescale (L×100, a×300, b×300)

lab_to_srgb:
  unscale (L/100, a/300, b/300)
  → inverse OKLab via 3×3 matrix
  → cube (x³, no conditional)
  → inverse LMS via 3×3 matrix
  → linear → sRGB (piecewise gamma, unchanged)
```

Compared to CIELab, this eliminates: the XYZ intermediate step, the `f_lab` / `f_lab_inv`
conditional piecewise functions, and one division per channel (XYZ normalization by
XN/YN/ZN whitepoint).

### `__powf` fast intrinsic for sRGB gamma

The sRGB piecewise gamma (`powf(2.4)` and `powf(1/2.4)`) is the most expensive per-pixel
operation in both CIELab and OKLab paths. Replace `powf` with CUDA's `__powf` fast-math
intrinsic, which uses 2 SFU cycles instead of ~10+ for IEEE-compliant `powf`.

Accuracy drops from ~1 ULP to ~2 ULP — invisible at 10-bit (1024 levels need ~10 bits of
precision, `__powf` provides ~21 bits). Applied to both `srgb_to_linear` and
`linear_to_srgb` in the CUDA kernel. The Rust CPU path (`lab.rs`) keeps standard `powf`
since CPU performance is not a bottleneck.

### Performance impact

Two improvements stack:
1. **OKLab conversion** — fewer ALU ops per pixel (drop XYZ step, drop `f_lab` conditional).
   Estimated 5-15% faster on the LAB stage.
2. **`__powf` intrinsic** — ~2× faster on the sRGB gamma step, which is the dominant cost
   in the conversion. Estimated additional 10-20% on the LAB stage.

Combined: estimated **15-30% faster on the LAB stage** of the per-pixel kernel. Since the
LAB stage is one of four stages (LUT → HSL → LAB → strength), overall per-frame
improvement is estimated at **5-10%**.

The primary motivation remains perceptual uniformity, not speed.

### Benchmark validation

POC benchmark (`working/sea_thru_poc/bench_oklab_transfer.py`) on a 4K frame (3840×2160):

| Approach | ms/frame | Speedup |
|---|---|---|
| CIELab fp32 (PyTorch, baseline) | 33.08 | 1× |
| OKLab + torch.compile + fp16 | 6.81 | 4.9× |
| OKLab fused Triton kernel | 2.89 | 11.4× |

These numbers reflect Python/PyTorch kernel launch overhead elimination, not directly
comparable to the production CUDA kernel which is already fused.

## Files Changed

| File | Change |
|---|---|
| `crates/dorea-color/src/lab.rs` | Replace `srgb_to_lab` / `lab_to_srgb` internals with OKLab + rescale. Keep function signatures identical. Update round-trip tests. |
| `crates/dorea-gpu/src/cuda/kernels/grade_pixel.cuh` | Replace CUDA `srgb_to_lab` / `lab_to_srgb` with OKLab equivalents. Remove XN/YN/ZN constants, `f_lab`/`f_lab_inv`, DELTA constants. Add OKLab matrix constants and rescale factors. Switch `powf` → `__powf` in `srgb_to_linear` / `linear_to_srgb`. |
| `crates/dorea-gpu/src/cpu.rs` | No change — calls `dorea_color::lab::*` which swaps underneath. |
| `crates/dorea-gpu/src/cuda/kernels/postprocess.cu` | No change — includes `grade_pixel.cuh` which swaps underneath. |
| `crates/dorea-gpu/src/cuda/kernels/build_combined_lut.cu` | No change — includes `grade_pixel.cuh`. |

### What does NOT change

- Function signatures (`srgb_to_lab`, `lab_to_srgb`) — callers are unaffected
- Ambiance constants (shadow lift, S-curve, highlight compress)
- Warmth constants (a*/b* push magnitudes, warmth_factor scaling)
- Vibrance constants (chroma threshold `40.0`, boost formula)
- a/b clamping ranges (`[-128, 127]`)
- L clamping range (`[0, 100]`)
- Stage bitmask logic
- 10-bit pipeline path (uses same `grade_pixel_device`)

## Testing

1. `cargo test -p dorea-color` — round-trip accuracy in `lab.rs` (update expected values)
2. `cargo build -p dorea-gpu` — verify CUDA kernel compiles
3. `dorea grade` on a short test clip — visual comparison against CIELab output
4. Existing integration tests pass unchanged (function signatures preserved)

## Visual Impact

The grading output will differ slightly from CIELab. OKLab distributes color corrections
more uniformly across hue regions, so warmth and vibrance adjustments will appear more
natural, particularly in the blue-cyan range common in underwater footage. The rescaling
ensures magnitudes are comparable, so the difference should be subtle — a quality
improvement rather than a visible shift.
