# Dorea v2 — Phase 1 Implementation Plan
**Date:** 2026-04-02
**Issue:** chunzhe10/dorea-workspace#24
**Branch:** feat/24-dorea-v2-rust-rewrite (workspace), main (dorea repo)
**Scope:** Cargo workspace + core algorithms + `dorea calibrate` CLI

---

## Goal

Implement a working `dorea calibrate` command that:
1. Accepts keyframe images + depth maps + RAUNE-Net target images
2. Builds depth-stratified LUTs (σ=0, NN fill, importance-weighted, adaptive zone boundaries)
3. Derives global HSL qualifier corrections
4. Saves a `.dorea-cal` calibration file

All CPU-only. No CUDA, no Python inference, no video I/O in Phase 1.

---

## Cargo Workspace Structure

```
repos/dorea/
├── Cargo.toml              # workspace manifest
├── Cargo.lock
├── README.md               # updated for v2
├── crates/
│   ├── dorea-cli/          # binary: dorea grade/calibrate/preview
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       └── calibrate.rs
│   ├── dorea-color/        # pure color math (no external deps)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dlog_m.rs   # D-Log M ↔ linear
│   │       ├── lab.rs      # RGB ↔ CIELAB (D65)
│   │       └── hsv.rs      # RGB ↔ HSV
│   ├── dorea-lut/          # depth-stratified LUT build + apply
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs    # LutGrid, DepthLuts
│   │       ├── build.rs    # build_depth_luts (importance weighting, NN fill)
│   │       └── apply.rs    # apply_depth_luts (trilinear + depth blending)
│   ├── dorea-hsl/          # HSL 6-vector qualifier correction
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── qualifiers.rs
│   │       ├── derive.rs   # derive_hsl_corrections
│   │       └── apply.rs    # apply_hsl_corrections
│   ├── dorea-cal/          # calibration file format
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── format.rs   # Calibration struct, save/load (bincode)
│   └── dorea-video/        # stub (Phase 2: ffmpeg subprocess)
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
└── python/                 # stub (Phase 2: VDA + RAUNE-Net subprocess)
    └── dorea_inference/
        └── __init__.py
```

---

## Crate Dependencies

| Crate | External Deps |
|-------|---------------|
| dorea-color | — (pure math) |
| dorea-lut | dorea-color |
| dorea-hsl | dorea-color |
| dorea-cal | serde, bincode, dorea-lut, dorea-hsl |
| dorea-cli | clap, image, dorea-color, dorea-lut, dorea-hsl, dorea-cal |
| dorea-video | — (stub) |

---

## Algorithm Specifications

### 1. D-Log M Transfer Function (`dorea-color/src/dlog_m.rs`)

Port from `pipeline_utils.py`:

```
a = 0.9892, b = 0.0108, c = 0.256663, d = 0.584555
cut_encoded = 0.14
cut_linear = (10^((cut_encoded - d) / c) - b) / a
slope = c * a / ((a * cut_linear + b) * ln(10))
intercept = cut_encoded - slope * cut_linear

decode(x):
  if x <= cut_encoded: (x - intercept) / slope
  else: (10^((x - d) / c) - b) / a

encode(x):
  if x <= cut_linear: slope * x + intercept
  else: c * log10(a * x + b) + d
```

Both operate on f32 pixel values [0,1].

### 2. RGB ↔ CIELAB D65 (`dorea-color/src/lab.rs`)

Standard conversion chain:
1. sRGB → linear RGB (IEC 61966-2-1 precise piecewise)
2. Linear RGB → XYZ (D65, ITU-R BT.709 matrix)
3. XYZ → LAB (CIE 1976, D65 whitepoint Xn=0.95047, Yn=1.0, Zn=1.08883)

Port from `ambiance_grade.py`'s `rgb_to_lab` / `lab_to_rgb`.

### 3. RGB ↔ HSV (`dorea-color/src/hsv.rs`)

Standard algorithm. H in [0,360), S in [0,1], V in [0,1].
Input/output RGB in [0,1].

### 4. Depth-Stratified LUT Build (`dorea-lut/src/build.rs`)

Port from `run_fixed_hsl_lut_poc.py::build_fixed_depth_luts()`.

**Constants:**
- `LUT_SIZE = 33`
- `N_DEPTH_ZONES = 5`
- `EDGE_SCALE = 0.3`
- `CONTRAST_SCALE = 0.3`

**Zone boundaries:** Adaptive density-weighted (not linear linspace).
- Pool all depth values from all keyframes
- Sort and compute percentile boundaries so each zone covers ~20% of pixels
- Fall back to linspace if depth distribution is degenerate

**Per-keyframe:**
1. Load original (sRGB) + RAUNE target (sRGB) + depth map (f32 [0,1])
2. Compute importance map = depth × (1 + EDGE_SCALE × edge_norm) × (1 + CONTRAST_SCALE × contrast_norm)
   - Edge: Sobel gradient of depth map
   - Contrast: local variance of depth map (31×31 window)
3. For each depth zone [lo, hi]:
   - Mask pixels in zone
   - Bin original pixels into 33³ grid (floor index)
   - Accumulate: `lut_wsum[r,g,b] += raune_rgb × importance`; `lut_wcount[r,g,b] += importance`
4. Normalize: `lut[r,g,b] = lut_wsum / lut_wcount` for populated cells
5. NN fill for empty cells:
   - Collect populated (i,j,k) coords
   - For each empty cell, find nearest populated cell in L2 distance (brute force — N=33 is small)
   - Fill with that cell's value

**No Gaussian smoothing (σ=0). Critical fix.**

### 5. LUT Application (`dorea-lut/src/apply.rs`)

**Trilinear interpolation:**
- For each pixel (r,g,b) ∈ [0,1]³:
  - Scaled = (r,g,b) × (LUT_SIZE - 1)
  - Floor indices (i0,j0,k0) + fractions (fr,fg,fb)
  - 8-corner trilinear interpolation

**Depth-weighted blending:**
- Compute soft zone weights: `weight_z = max(1 - |depth - zone_center_z| / zone_width, 0)` for each zone z
- Apply each zone's LUT to get `result_z`
- `result = Σ(result_z × weight_z) / Σ(weight_z)`

### 6. HSL Qualifier Derivation (`dorea-hsl/src/derive.rs`)

Port from `run_fixed_hsl_lut_poc.py::derive_hsl_corrections()`.

**6 qualifiers:**
```
Red/Skin:  h_center=0,   h_width=40
Yellow:    h_center=40,  h_width=40
Green:     h_center=100, h_width=50
Cyan:      h_center=170, h_width=40
Blue:      h_center=210, h_width=40
Magenta:   h_center=290, h_width=50
```

**Per qualifier:**
1. Soft mask: `w = max(1 - angular_dist(H, h_center) / h_width, 0.0) × (S > 0.08)`
2. Skip if `total_weight < 100`
3. `h_offset = weighted_circular_mean(H_target - H_lut)`
4. `s_ratio = weighted_mean(S_target) / weighted_mean(S_lut)`
5. `v_offset = weighted_mean(V_target - V_lut)` (in [0,1] units)

### 7. HSL Qualifier Application (`dorea-hsl/src/apply.rs`)

Port from `run_fixed_hsl_lut_poc.py::apply_hsl_corrections()`.

For each pixel:
1. Convert to HSV
2. For each qualifier with weight ≥ 100:
   - Compute soft mask (same as derivation)
   - `H += h_offset × mask`
   - `S = S × (1 + (s_ratio - 1) × mask)`, clamped [0,1]
   - `V += v_offset × mask`, clamped [0,1]
3. Wrap H to [0,360), clamp S and V
4. Convert back to RGB

### 8. Calibration Format (`dorea-cal/src/format.rs`)

Binary format via `bincode` + `serde`:

```rust
struct Calibration {
    version: u8,           // format version (current: 1)
    lut_size: u8,          // 33
    n_zones: u8,           // 5
    zone_boundaries: [f32; 6],  // adaptive zone edges [0,1]
    depth_luts: Vec<LutGrid>,   // n_zones LUTs, each 33³×3
    hsl_corrections: HslCorrections,
    created_at: String,    // ISO 8601
    keyframe_count: u8,
}
```

File extension: `.dorea-cal`

### 9. CLI — `dorea calibrate` (`dorea-cli/src/calibrate.rs`)

```
dorea calibrate
  --keyframes <dir>   Directory of sRGB PNG keyframes
  --depth <dir>       Directory of 16-bit PNG depth maps (matched by name stem)
  --targets <dir>     Directory of RAUNE-Net output PNGs (Phase 1: pre-computed)
  --output <path>     Output .dorea-cal file [default: calibration.dorea-cal]
  --cpu-only          No-op in Phase 1 (placeholder for Phase 2 GPU path)
  -v / --verbose      Enable debug logging
```

**Matching:** keyframe `foo.png` → depth `foo.png` + target `foo.png` (by stem).

---

## Testing Strategy

### Unit Tests (per crate)

- `dorea-color`: round-trip tests (D-Log M encode→decode recovers input; RGB→LAB→RGB round-trip within tolerance)
- `dorea-lut`: build with synthetic data (solid-color keyframes where LUT should be identity-ish); apply tests with known input/output
- `dorea-hsl`: derivation with synthetic same-image (should give zero corrections); application with known offsets
- `dorea-cal`: serialize→deserialize round-trip

### Golden-File Tests

`dorea-lut/tests/golden.rs` and `dorea-hsl/tests/golden.rs`:
- Use the 3 keyframes from `working/sea_thru_poc/frames/` as reference
- Compare Rust LUT output against Python reference within 1-2 LSB at 8-bit

**Note:** Golden files (expected outputs) must be generated by running the Python POC once and saving outputs. Add a `regenerate_goldens` flag or separate script for this.

---

## Tasks

### Task 1: Workspace manifest + crate scaffolding
- Create `Cargo.toml` (workspace)
- Scaffold all 6 crate `Cargo.toml` files and `src/lib.rs` stubs

### Task 2: `dorea-color` — color math
- `dlog_m.rs`: D-Log M encode/decode
- `lab.rs`: RGB↔LAB
- `hsv.rs`: RGB↔HSV
- Unit tests for round-trips

### Task 3: `dorea-lut` — LUT build + apply
- `types.rs`: LutGrid, DepthLuts
- `build.rs`: build_depth_luts (importance weighting, adaptive zones, NN fill, σ=0)
- `apply.rs`: apply_depth_luts (trilinear + depth blending)
- Unit tests

### Task 4: `dorea-hsl` — HSL qualifier correction
- `qualifiers.rs`: HSL_QUALIFIERS constants
- `derive.rs`: derive_hsl_corrections
- `apply.rs`: apply_hsl_corrections
- Unit tests

### Task 5: `dorea-cal` — calibration format
- `format.rs`: Calibration struct, save/load
- Round-trip test

### Task 6: `dorea-cli` — calibrate command
- `main.rs`: CLI entry with clap
- `calibrate.rs`: orchestration (load images → build LUTs → derive HSL → save .dorea-cal)
- Integration test: run with sample data, verify .dorea-cal is readable

### Task 7: Remove dead code + update README
- Delete old Python scripts from main branch
- Update README to describe v2 architecture

---

## Constraints

- RTX 3060 6GB VRAM — Phase 1 is CPU only, no VRAM constraint
- No DaVinci Resolve dependency in Phase 1+
- All tests must pass with `cargo test`
- Clippy clean: `cargo clippy -- -D warnings`
