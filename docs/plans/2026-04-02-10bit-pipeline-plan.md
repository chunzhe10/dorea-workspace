# 10-Bit Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Dorea from 8-bit to 10-bit I/O supporting DJI D-Log M and Insta360 X5 I-Log, with f32 internal processing, 3-stage pipelined grading, and 10-bit output to DaVinci Resolve.

**Architecture:** Trait-based transfer functions abstract camera log curves. 16-bit PNG replaces JPEG for keyframe extraction. Internal f32 math is unchanged. CUDA kernels move from u8 to f32. A 3-stage bounded-channel pipeline (decode → depth → grade+encode) replaces the serial frame loop. Tiny planet clips skip depth estimation.

**Tech Stack:** Rust (dorea crates), CUDA 12.4, ffmpeg (NVDEC/ProRes), Python (inference server), clap (CLI), rayon (CPU parallelism)

**Spec:** `docs/decisions/2026-04-02-10bit-pipeline-design.md`

---

## Phase A: Foundation (Color Math + I/O + LUT Size)

These tasks produce a working 10-bit calibration pipeline. No GPU or grading changes yet.

---

### Task 1: TransferFunction Trait in dorea-color

**Files:**
- Modify: `crates/dorea-color/src/lib.rs`
- Modify: `crates/dorea-color/src/dlog_m.rs`

- [ ] **Step 1: Define the TransferFunction trait**

Add to `crates/dorea-color/src/lib.rs`:

```rust
pub mod dlog_m;
pub mod hsv;
pub mod lab;

/// Camera log curve abstraction. Implementations decode/encode between
/// a camera-specific log encoding and scene-linear light.
pub trait TransferFunction {
    /// Decode a log-encoded value to scene-linear light.
    fn to_linear(&self, encoded: f32) -> f32;
    /// Encode a scene-linear light value to log encoding.
    fn from_linear(&self, linear: f32) -> f32;
    /// Highlight rolloff shoulder (0.0–1.0). Higher = later compression.
    fn shoulder(&self) -> f32;
    /// Human-readable name (e.g. "D-Log M").
    fn name(&self) -> &'static str;
}
```

- [ ] **Step 2: Implement TransferFunction for DLogM**

Add to the bottom of `crates/dorea-color/src/dlog_m.rs` (before `#[cfg(test)]`):

```rust
use crate::TransferFunction;

/// D-Log M transfer function (DJI Action 4).
pub struct DLogM;

impl TransferFunction for DLogM {
    fn to_linear(&self, encoded: f32) -> f32 {
        dlog_m_to_linear(encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        linear_to_dlog_m(linear)
    }

    fn shoulder(&self) -> f32 {
        0.85
    }

    fn name(&self) -> &'static str {
        "D-Log M"
    }
}
```

- [ ] **Step 3: Export DLogM struct from lib.rs**

Update `crates/dorea-color/src/lib.rs` to re-export:

```rust
pub use dlog_m::DLogM;
```

- [ ] **Step 4: Add trait object test**

Add test in `crates/dorea-color/src/dlog_m.rs` inside the `tests` module:

```rust
#[test]
fn test_dlog_m_as_trait_object() {
    let tf: Box<dyn crate::TransferFunction> = Box::new(super::DLogM);
    let encoded = tf.from_linear(0.18);
    let decoded = tf.to_linear(encoded);
    assert!((decoded - 0.18).abs() < 1e-4, "Trait round-trip: expected ~0.18, got {decoded}");
    assert_eq!(tf.shoulder(), 0.85);
    assert_eq!(tf.name(), "D-Log M");
}
```

- [ ] **Step 5: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-color`
Expected: All tests pass including the new trait object test.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-color/src/lib.rs crates/dorea-color/src/dlog_m.rs
git commit -m "feat(color): add TransferFunction trait, implement for DLogM"
```

---

### Task 2: I-Log Transfer Function (S-Log3 Proxy)

**Files:**
- Create: `crates/dorea-color/src/ilog.rs`
- Modify: `crates/dorea-color/src/lib.rs`

The I-Log curve coefficients are not published by Insta360. We use Sony S-Log3 as a
proxy (similar shape: linear toe + log curve, ~12-13 stops). This will be replaced by
empirical extraction later via the LutBased variant (Task 3).

- [ ] **Step 1: Write failing tests**

Create `crates/dorea-color/src/ilog.rs`:

```rust
//! I-Log transfer function (Insta360 X5).
//!
//! Uses S-Log3 as a proxy curve until empirical extraction is available.
//! Replace with `LutBased` once a 1D .cube file is produced from a
//! grayscale chart shot on the X5.

use crate::TransferFunction;

// Sony S-Log3 parameters (BT.2100 / Sony S-Gamut3)
const A: f64 = 0.432699;
const B: f64 = 0.01125;
const C: f64 = 0.01125;
const D: f64 = 0.616596;
const E: f64 = 0.03;
const F: f64 = 0.037584;
// Linear cut point: below this, encoding is linear
const LIN_CUT: f64 = 0.01125000;
// Encoded cut point: S-Log3 value at LIN_CUT
const ENC_CUT: f64 = 0.171260702; // A * log10(LIN_CUT + B) + D

/// Convert I-Log (S-Log3 proxy) encoded value to scene-linear light.
pub fn ilog_to_linear(x: f32) -> f32 {
    let x = x as f64;
    let linear = if x >= ENC_CUT {
        10_f64.powf((x - D) / A) - B
    } else {
        (x * E - F) / E
    };
    linear.max(0.0) as f32
}

/// Convert scene-linear light to I-Log (S-Log3 proxy) encoding.
pub fn linear_to_ilog(x: f32) -> f32 {
    let x = x as f64;
    let encoded = if x >= LIN_CUT {
        A * (x + B).log10() + D
    } else {
        (x * E + F) / E
    };
    encoded.clamp(0.0, 1.0) as f32
}

/// I-Log transfer function (Insta360 X5, S-Log3 proxy).
pub struct ILog;

impl TransferFunction for ILog {
    fn to_linear(&self, encoded: f32) -> f32 {
        ilog_to_linear(encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        linear_to_ilog(linear)
    }

    fn shoulder(&self) -> f32 {
        0.88
    }

    fn name(&self) -> &'static str {
        "I-Log"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ilog_round_trip() {
        let values = [0.0_f32, 0.01, 0.05, 0.18, 0.5, 0.9, 1.0];
        for &v in &values {
            let encoded = linear_to_ilog(v);
            let decoded = ilog_to_linear(encoded);
            assert!(
                (decoded - v).abs() < 1e-4,
                "Round-trip failed for {v}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_ilog_middle_grey() {
        // S-Log3 middle grey: 18% linear ≈ 0.41 encoded
        let encoded = linear_to_ilog(0.18);
        assert!(
            (0.35..0.50).contains(&encoded),
            "Middle grey encode: expected 0.35–0.50, got {encoded}"
        );
    }

    #[test]
    fn test_ilog_monotonic() {
        let mut prev = ilog_to_linear(0.0);
        for i in 1..=100 {
            let x = i as f32 / 100.0;
            let lin = ilog_to_linear(x);
            assert!(lin >= prev, "Non-monotonic at x={x}: {lin} < {prev}");
            prev = lin;
        }
    }

    #[test]
    fn test_ilog_trait() {
        let tf: Box<dyn crate::TransferFunction> = Box::new(ILog);
        assert_eq!(tf.shoulder(), 0.88);
        assert_eq!(tf.name(), "I-Log");
    }
}
```

- [ ] **Step 2: Register module and export**

Update `crates/dorea-color/src/lib.rs`:

```rust
pub mod dlog_m;
pub mod hsv;
pub mod ilog;
pub mod lab;

/// Camera log curve abstraction.
pub trait TransferFunction {
    fn to_linear(&self, encoded: f32) -> f32;
    fn from_linear(&self, linear: f32) -> f32;
    fn shoulder(&self) -> f32;
    fn name(&self) -> &'static str;
}

pub use dlog_m::DLogM;
pub use ilog::ILog;
```

- [ ] **Step 3: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-color`
Expected: All pass (including new ilog tests).

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-color/src/ilog.rs crates/dorea-color/src/lib.rs
git commit -m "feat(color): add I-Log transfer function (S-Log3 proxy)"
```

---

### Task 3: LutBased Transfer Function

**Files:**
- Create: `crates/dorea-color/src/lut_transfer.rs`
- Modify: `crates/dorea-color/src/lib.rs`
- Modify: `crates/dorea-color/Cargo.toml`

This enables swapping in an empirically-extracted I-Log curve from a 1D .cube file
without touching code. The .cube format stores a 1D LUT as a list of output values
for evenly-spaced input values.

- [ ] **Step 1: Create lut_transfer.rs with tests**

Create `crates/dorea-color/src/lut_transfer.rs`:

```rust
//! 1D LUT-based transfer function.
//!
//! Loads a 1D .cube file as a transfer function for cameras whose exact
//! log curve is unknown. Drop in an empirical extraction to replace the
//! S-Log3 proxy for I-Log.

use crate::TransferFunction;
use std::path::Path;

/// A transfer function defined by a 1D lookup table.
///
/// The forward table maps evenly-spaced encoded values [0,1] to linear light.
/// The inverse table maps evenly-spaced linear values [0,1] to encoded.
pub struct LutBased {
    /// Forward LUT: encoded → linear. Length = LUT size.
    forward: Vec<f32>,
    /// Inverse LUT: linear → encoded. Length = LUT size.
    inverse: Vec<f32>,
    shoulder: f32,
    name: String,
}

impl LutBased {
    /// Build from a forward LUT (encoded → linear).
    /// The inverse is computed by table inversion.
    pub fn new(forward: Vec<f32>, shoulder: f32, name: String) -> Self {
        let inverse = invert_1d_lut(&forward);
        Self { forward, inverse, shoulder, name }
    }

    /// Load from a 1D .cube file.
    ///
    /// Expected format:
    /// ```text
    /// # comment
    /// TITLE "curve name"
    /// LUT_1D_SIZE 1024
    /// 0.000000
    /// 0.000123
    /// ...
    /// ```
    pub fn from_cube_file(path: &Path, shoulder: f32) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

        let mut size: Option<usize> = None;
        let mut values: Vec<f32> = Vec::new();
        let mut title = path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "custom".to_string());

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("TITLE") {
                title = rest.trim().trim_matches('"').to_string();
                continue;
            }
            if let Some(rest) = line.strip_prefix("LUT_1D_SIZE") {
                size = Some(rest.trim().parse::<usize>()
                    .map_err(|e| format!("bad LUT_1D_SIZE: {e}"))?);
                continue;
            }
            // Skip 3D LUT directives
            if line.starts_with("LUT_3D_SIZE") || line.starts_with("DOMAIN_") {
                continue;
            }
            // Data line: single float (1D) or three floats (use first channel)
            let first_token = line.split_whitespace().next().unwrap_or("");
            if let Ok(v) = first_token.parse::<f32>() {
                values.push(v);
            }
        }

        if let Some(sz) = size {
            if values.len() != sz {
                return Err(format!(
                    "LUT_1D_SIZE={sz} but found {} values", values.len()
                ));
            }
        }
        if values.len() < 2 {
            return Err("LUT must have at least 2 entries".to_string());
        }

        Ok(Self::new(values, shoulder, title))
    }
}

impl TransferFunction for LutBased {
    fn to_linear(&self, encoded: f32) -> f32 {
        lerp_lut(&self.forward, encoded)
    }

    fn from_linear(&self, linear: f32) -> f32 {
        lerp_lut(&self.inverse, linear)
    }

    fn shoulder(&self) -> f32 {
        self.shoulder
    }

    fn name(&self) -> &'static str {
        // Leak the string to get a 'static lifetime. This is fine because
        // LutBased instances live for the duration of the pipeline.
        // A single allocation per pipeline run.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

/// Linearly interpolate a 1D LUT. Input clamped to [0, 1].
fn lerp_lut(lut: &[f32], x: f32) -> f32 {
    let n = lut.len();
    if n == 0 { return x; }
    let x = x.clamp(0.0, 1.0);
    let pos = x * (n - 1) as f32;
    let lo = (pos as usize).min(n - 2);
    let hi = lo + 1;
    let frac = pos - lo as f32;
    lut[lo] * (1.0 - frac) + lut[hi] * frac
}

/// Invert a monotonic 1D LUT by building a reverse table.
/// Assumes the input LUT is monotonically non-decreasing.
fn invert_1d_lut(forward: &[f32]) -> Vec<f32> {
    let n = forward.len();
    let mut inverse = vec![0.0f32; n];

    let out_min = forward[0];
    let out_max = forward[n - 1];
    let range = (out_max - out_min).max(1e-10);

    for i in 0..n {
        // Target linear value for this inverse table entry
        let target = i as f32 / (n - 1) as f32;
        // Scale target to output range of forward LUT
        let target_scaled = out_min + target * range;

        // Binary search for position in forward table
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if forward[mid] <= target_scaled {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Interpolate
        let denom = (forward[hi] - forward[lo]).max(1e-10);
        let frac = (target_scaled - forward[lo]) / denom;
        inverse[i] = (lo as f32 + frac) / (n - 1) as f32;
    }

    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    #[test]
    fn test_identity_round_trip() {
        let lut = LutBased::new(identity_lut(256), 0.90, "identity".to_string());
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let linear = lut.to_linear(x);
            let encoded = lut.from_linear(linear);
            assert!(
                (encoded - x).abs() < 0.01,
                "Identity round-trip failed at {x}: linear={linear}, encoded={encoded}"
            );
        }
    }

    #[test]
    fn test_gamma_lut_round_trip() {
        // Simulate a gamma 2.2 curve (encoded → linear)
        let n = 1024;
        let forward: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32).powf(2.2))
            .collect();
        let lut = LutBased::new(forward, 0.85, "gamma2.2".to_string());

        let values = [0.0, 0.1, 0.18, 0.5, 0.8, 1.0];
        for &v in &values {
            let linear = lut.to_linear(v);
            let back = lut.from_linear(linear);
            assert!(
                (back - v).abs() < 0.01,
                "Gamma round-trip failed at {v}: linear={linear}, back={back}"
            );
        }
    }

    #[test]
    fn test_lerp_lut_boundaries() {
        let lut = vec![0.0, 0.5, 1.0];
        assert!((lerp_lut(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((lerp_lut(&lut, 0.5) - 0.5).abs() < 1e-6);
        assert!((lerp_lut(&lut, 1.0) - 1.0).abs() < 1e-6);
        assert!((lerp_lut(&lut, 0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_trait_impl() {
        let lut = LutBased::new(identity_lut(256), 0.90, "test".to_string());
        let tf: &dyn crate::TransferFunction = &lut;
        assert_eq!(tf.shoulder(), 0.90);
        assert_eq!(tf.name(), "test");
    }
}
```

- [ ] **Step 2: Register module and export**

Update `crates/dorea-color/src/lib.rs` — add `pub mod lut_transfer;` and `pub use lut_transfer::LutBased;`

- [ ] **Step 3: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-color`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-color/src/lut_transfer.rs crates/dorea-color/src/lib.rs
git commit -m "feat(color): add LutBased transfer function for 1D .cube files"
```

---

### Task 4: 16-Bit PNG Encode/Decode in inference.rs

**Files:**
- Modify: `crates/dorea-video/src/inference.rs:364-408` (encode)
- Modify: `crates/dorea-video/src/inference.rs:480-586` (decode)

The hand-rolled PNG codec currently hardcodes 8-bit. We make it support both 8 and 16.

- [ ] **Step 1: Write failing test for 16-bit PNG round-trip**

Add to the `tests` module in `crates/dorea-video/src/inference.rs`:

```rust
#[test]
fn test_16bit_png_round_trip() {
    // Create a 2x2 image with values that differ at 16-bit but not 8-bit
    let pixels: Vec<[f32; 3]> = vec![
        [257.0 / 65535.0, 0.0, 0.0],          // R=257 (would be 1 in 8-bit)
        [513.0 / 65535.0, 1024.0 / 65535.0, 0.0], // values only distinguishable at 16-bit
        [0.0, 0.0, 32768.0 / 65535.0],
        [1.0, 1.0, 1.0],
    ];
    let encoded = encode_rgb_png_16bit(&pixels, 2, 2).unwrap();
    let (decoded, w, h) = parse_png_rgb(&encoded).unwrap();
    assert_eq!((w, h), (2, 2));
    for (i, (orig, dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                (orig[c] - dec[c]).abs() < 1.0 / 65535.0 + 1e-6,
                "Pixel {i} channel {c}: orig={}, decoded={}",
                orig[c], dec[c]
            );
        }
    }
}

#[test]
fn test_8bit_png_still_works() {
    // Existing 8-bit path must not break
    let pixels: Vec<[f32; 3]> = vec![
        [0.0, 0.5, 1.0],
        [0.25, 0.75, 0.0],
    ];
    let encoded = encode_rgb_png(&pixels, 2, 1).unwrap();
    let (decoded, w, h) = parse_png_rgb(&encoded).unwrap();
    assert_eq!((w, h), (2, 1));
    for (orig, dec) in pixels.iter().zip(decoded.iter()) {
        for c in 0..3 {
            assert!(
                (orig[c] - dec[c]).abs() < 1.0 / 255.0 + 1e-6,
                "8-bit round-trip: orig={}, decoded={}",
                orig[c], dec[c]
            );
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video test_16bit`
Expected: FAIL — `encode_rgb_png_16bit` not found.

- [ ] **Step 3: Add encode_rgb_png_16bit function**

Add alongside the existing `encode_rgb_png` (which stays unchanged for 8-bit proxy path).
In `crates/dorea-video/src/inference.rs`, after the existing `encode_rgb_png` function:

```rust
/// Encode f32 RGB pixels as a 16-bit PNG (rgb48be).
/// Used for high-precision keyframe I/O. The 8-bit `encode_rgb_png`
/// remains for inference proxy frames.
pub(crate) fn encode_rgb_png_16bit(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    if pixels.len() != width * height {
        return Err(format!(
            "pixel count {} != {}x{}", pixels.len(), width, height
        ));
    }

    // Build raw scanlines: filter byte (0x00) + 6 bytes per pixel (RGB16 big-endian)
    let row_bytes = 1 + width * 6;
    let mut raw = Vec::with_capacity(row_bytes * height);
    for y in 0..height {
        raw.push(0x00); // filter: None
        for x in 0..width {
            let px = &pixels[y * width + x];
            for c in 0..3 {
                let v = (px[c].clamp(0.0, 1.0) * 65535.0).round() as u16;
                raw.extend_from_slice(&v.to_be_bytes());
            }
        }
    }

    let compressed = deflate_zlib(&raw);

    let mut out = Vec::new();
    // PNG signature
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    // IHDR: bit_depth=16, color_type=2 (RGB)
    write_png_chunk(&mut out, b"IHDR", &{
        let mut h = Vec::new();
        h.extend_from_slice(&(width as u32).to_be_bytes());
        h.extend_from_slice(&(height as u32).to_be_bytes());
        h.extend_from_slice(&[16u8, 2, 0, 0, 0]); // bit_depth=16, RGB, compression, filter, interlace
        h
    });
    write_png_chunk(&mut out, b"IDAT", &compressed);
    write_png_chunk(&mut out, b"IEND", &[]);

    Ok(out)
}
```

- [ ] **Step 4: Update parse_png_rgb to accept 16-bit**

In `crates/dorea-video/src/inference.rs`, modify the IHDR parsing in `parse_png_rgb`.

Change the bit depth check from:

```rust
if bit_depth != 8 || color_type != 2 || interlace != 0 {
    return Err(format!(
        "unsupported PNG: bit_depth={bit_depth} color_type={color_type} interlace={interlace}"
    ));
}
```

To:

```rust
if (bit_depth != 8 && bit_depth != 16) || color_type != 2 || interlace != 0 {
    return Err(format!(
        "unsupported PNG: bit_depth={bit_depth} color_type={color_type} interlace={interlace}"
    ));
}
let bytes_per_channel: usize = bit_depth as usize / 8; // 1 for 8-bit, 2 for 16-bit
```

Then update the scanline pixel parsing. Find the loop that reads pixel data from
decompressed scanlines and replace the per-pixel decode with:

```rust
// For 8-bit: 1 byte per channel, normalize by /255.0
// For 16-bit: 2 bytes big-endian per channel, normalize by /65535.0
let bytes_per_pixel = bytes_per_channel * 3;
let expected_row = 1 + width * bytes_per_pixel; // filter byte + pixel data
```

And in the pixel reading loop, branch on `bytes_per_channel`:

```rust
if bytes_per_channel == 1 {
    let r = row_data[1 + x * 3] as f32 / 255.0;
    let g = row_data[1 + x * 3 + 1] as f32 / 255.0;
    let b = row_data[1 + x * 3 + 2] as f32 / 255.0;
    pixels.push([r, g, b]);
} else {
    let off = 1 + x * 6;
    let r = u16::from_be_bytes([row_data[off], row_data[off + 1]]) as f32 / 65535.0;
    let g = u16::from_be_bytes([row_data[off + 2], row_data[off + 3]]) as f32 / 65535.0;
    let b = u16::from_be_bytes([row_data[off + 4], row_data[off + 5]]) as f32 / 65535.0;
    pixels.push([r, g, b]);
}
```

- [ ] **Step 5: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video`
Expected: All pass (including new 16-bit test and existing 8-bit tests).

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-video/src/inference.rs
git commit -m "feat(video): 16-bit PNG encode/decode in inference codec"
```

---

### Task 5: load_rgb_image 16-Bit Upgrade

**Files:**
- Modify: `crates/dorea-cli/src/calibrate.rs:371-384`

- [ ] **Step 1: Update load_rgb_image to use into_rgb16**

In `crates/dorea-cli/src/calibrate.rs`, change `load_rgb_image`:

From:

```rust
fn load_rgb_image(path: &Path) -> Result<(Vec<[f32; 3]>, usize, usize)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?
        .decode()
        .with_context(|| format!("Cannot decode {}", path.display()))?
        .into_rgb8();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0])
        .collect();
    Ok((pixels, w, h))
}
```

To:

```rust
fn load_rgb_image(path: &Path) -> Result<(Vec<[f32; 3]>, usize, usize)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?
        .decode()
        .with_context(|| format!("Cannot decode {}", path.display()))?
        .into_rgb16();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixels: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| [p[0] as f32 / 65535.0, p[1] as f32 / 65535.0, p[2] as f32 / 65535.0])
        .collect();
    Ok((pixels, w, h))
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli`
Expected: Compiles. The `image` crate's `into_rgb16()` is available on `DynamicImage`.

- [ ] **Step 3: Run existing tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli`
Expected: All existing tests pass. 8-bit PNGs loaded via `into_rgb16()` produce values
at multiples of 257 (e.g., 255 → 65535, 128 → 32896), which normalize to the same
float range. The math is unchanged.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-cli/src/calibrate.rs
git commit -m "feat(cli): upgrade load_rgb_image to 16-bit precision"
```

---

### Task 6: Configurable LUT Size

**Files:**
- Modify: `crates/dorea-lut/src/build.rs:8` (LUT_SIZE constant)
- Modify: `crates/dorea-lut/src/build.rs` (build_depth_luts signature)
- Modify: `crates/dorea-lut/src/types.rs` (LutGrid::new)
- Modify: `crates/dorea-lut/src/apply.rs` (trilinear uses grid.size)

- [ ] **Step 1: Write failing test for 65^3 LUT**

Add test in `crates/dorea-lut/src/build.rs` inside the `tests` module:

```rust
#[test]
fn test_build_lut_size_65() {
    // Minimal test: 1 keyframe, 1x1 pixel, verify output grid is 65^3
    let kf = KeyframeData {
        original: vec![[0.5, 0.5, 0.5]],
        target: vec![[0.6, 0.4, 0.5]],
        depth: vec![0.5],
        importance: vec![1.0],
        width: 1,
        height: 1,
    };
    let luts = build_depth_luts(&[kf], 65);
    assert_eq!(luts.luts[0].size, 65);
    assert_eq!(luts.luts[0].data.len(), 65 * 65 * 65 * 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-lut test_build_lut_size_65`
Expected: FAIL — `build_depth_luts` does not accept a size argument.

- [ ] **Step 3: Make LUT size a parameter**

In `crates/dorea-lut/src/build.rs`:

Change the constant to a default:

```rust
/// Default LUT grid size. Use 33 for 8-bit sources, 65 for 10-bit.
pub const DEFAULT_LUT_SIZE: usize = 33;
```

Update `build_depth_luts` signature from:

```rust
pub fn build_depth_luts(keyframes: &[KeyframeData]) -> DepthLuts {
```

To:

```rust
pub fn build_depth_luts(keyframes: &[KeyframeData], lut_size: usize) -> DepthLuts {
```

Replace all references to `LUT_SIZE` inside the function with `lut_size`.

Update `LutGrid::new` calls inside `build_depth_luts` to pass `lut_size` instead of
the hardcoded constant.

- [ ] **Step 4: Update all callers of build_depth_luts**

Search for `build_depth_luts(` across the codebase and add the size argument.
The main caller is in `crates/dorea-cli/src/calibrate.rs`. Update it to pass
`DEFAULT_LUT_SIZE` for now (will be made configurable via CLI in a later task):

```rust
use dorea_lut::build::DEFAULT_LUT_SIZE;
// ...
let luts = build_depth_luts(&keyframe_data_vec, DEFAULT_LUT_SIZE);
```

- [ ] **Step 5: Update existing tests**

All existing tests in `build.rs` that call `build_depth_luts` need the size argument.
Add `33` (or `DEFAULT_LUT_SIZE`) as the second argument to each existing test call.

- [ ] **Step 6: Run all tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test`
Expected: All pass, including the new 65^3 test.

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-lut/src/build.rs crates/dorea-lut/src/types.rs crates/dorea-cli/src/calibrate.rs
git commit -m "feat(lut): configurable LUT grid size (33 or 65)"
```

---

## Phase B: GPU + Grading Pipeline

These tasks upgrade the grading path: CUDA f32 signatures, persistent LUT, 3-stage pipeline, 10-bit output.

---

### Task 7: GPU Signature u8 → f32 (CPU Fallback)

**Files:**
- Modify: `crates/dorea-gpu/src/lib.rs:50-97` (grade_frame)
- Modify: `crates/dorea-gpu/src/cpu.rs:321-361` (grade_frame_cpu)
- Modify: `crates/dorea-gpu/src/cpu.rs:132-177` (finish_grade)

The public API changes from `pixels: &[u8]` → `pixels: &[f32]` (interleaved RGB,
3 floats per pixel in [0,1]) and returns `Vec<f32>` instead of `Vec<u8>`.

- [ ] **Step 1: Write failing test**

Add test in `crates/dorea-gpu/src/cpu.rs` inside the `tests` module:

```rust
#[test]
fn test_grade_frame_f32_signature() {
    // Create a 2x2 f32 image (instead of u8)
    let pixels: Vec<f32> = vec![
        0.5, 0.3, 0.2,  // pixel 0
        0.6, 0.4, 0.3,  // pixel 1
        0.4, 0.5, 0.6,  // pixel 2
        0.7, 0.2, 0.1,  // pixel 3
    ];
    let depth = vec![0.5f32; 4];

    // Minimal calibration with 33^3 identity-ish LUT
    let cal = crate::tests::make_test_calibration();
    let params = GradeParams::default();

    let result = grade_frame_cpu(&pixels, &depth, 2, 2, &cal, &params);
    assert!(result.is_ok());
    let out = result.unwrap();
    // Output should be f32, 3 values per pixel
    assert_eq!(out.len(), 4 * 3);
    // Values should be in [0, 1]
    for &v in &out {
        assert!(v >= 0.0 && v <= 1.0, "Output value out of range: {v}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu test_grade_frame_f32`
Expected: FAIL — signature mismatch.

- [ ] **Step 3: Update grade_frame_cpu signature**

In `crates/dorea-gpu/src/cpu.rs`, change `grade_frame_cpu`:

From: `pub fn grade_frame_cpu(pixels: &[u8], ...) -> Result<Vec<u8>, ...>`
To: `pub fn grade_frame_cpu(pixels: &[f32], ...) -> Result<Vec<f32>, ...>`

Remove the `u8 → f32` conversion at the start (it's no longer needed — input is already f32).
Remove the `f32 → u8` conversion at the end (output stays f32).

Update `finish_grade` similarly:
From: `pub fn finish_grade(graded: &mut [f32], ..., from_cuda: bool) -> Vec<u8>`
To: `pub fn finish_grade(graded: &mut [f32], ..., from_cuda: bool) -> Vec<f32>`

Remove the final `(v * 255.0).round() as u8` clamp-and-convert loop. Just clamp to [0,1]
and return the f32 vec.

- [ ] **Step 4: Update grade_frame public API**

In `crates/dorea-gpu/src/lib.rs`, update `grade_frame`:

From: `pub fn grade_frame(pixels: &[u8], ...) -> Result<Vec<u8>, GpuError>`
To: `pub fn grade_frame(pixels: &[f32], ...) -> Result<Vec<f32>, GpuError>`

Update the size validation from `pixels.len() != width * height * 3` (u8, 3 bytes/px)
— this stays the same since f32 is also 3 values per pixel.

- [ ] **Step 5: Update all callers in dorea-cli**

In `crates/dorea-cli/src/grade.rs`, the frame loop currently passes `&frame.pixels`
(which is `Vec<u8>` from ffmpeg). This will need a u8→f32 conversion at the call site:

```rust
let pixels_f32: Vec<f32> = frame.pixels.iter()
    .map(|&b| b as f32 / 255.0)
    .collect();
let graded = grade_frame(&pixels_f32, &depth, frame.width, frame.height, &calibration, &params)
    .unwrap_or_else(|e| {
        log::warn!("Grading failed for frame {}: {e} — passing through", frame.index);
        pixels_f32.clone()
    });
// Convert back to u8 for encoder (temporary — Task 13 adds 10-bit output)
let graded_u8: Vec<u8> = graded.iter()
    .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
    .collect();
encoder.write_frame(&graded_u8).context("encoder write failed")?;
```

Do the same in `crates/dorea-cli/src/preview.rs`.

- [ ] **Step 6: Update existing tests**

All existing tests in `cpu.rs` that call `grade_frame_cpu` with `&[u8]` need to convert
to `&[f32]` input and expect `Vec<f32>` output. Update each test's pixel input from
`vec![128u8, 100, 80, ...]` to `vec![128.0/255.0, 100.0/255.0, 80.0/255.0, ...]`
and assertions from `u8` comparisons to `f32` tolerance checks.

- [ ] **Step 7: Run all tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add crates/dorea-gpu/src/lib.rs crates/dorea-gpu/src/cpu.rs \
       crates/dorea-cli/src/grade.rs crates/dorea-cli/src/preview.rs
git commit -m "feat(gpu): change grade_frame API from u8 to f32"
```

---

### Task 8: CUDA Kernels f32 Signature

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu`
- Modify: `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu`
- Modify: `crates/dorea-gpu/src/cuda/kernels/clarity.cu`
- Modify: `crates/dorea-gpu/src/cuda/mod.rs:17-49` (extern "C" signatures)
- Modify: `crates/dorea-gpu/src/cuda/mod.rs:74-171` (grade_frame_cuda)

- [ ] **Step 1: Update lut_apply.cu**

In `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu`:

Change the kernel signature and host launcher from `unsigned char*` to `float*`.
Remove the `uchar→float / 255.0` conversion at kernel start.
Remove the `float→uchar * 255.0` conversion at kernel end.
Change `h_pixels_in` and `h_pixels_out` from `const unsigned char*` / `unsigned char*`
to `const float*` / `float*`.
Update `pixels_bytes` calculation from `n_pixels * 3 * sizeof(unsigned char)` to
`n_pixels * 3 * sizeof(float)`.

The host launcher signature becomes:

```c
extern "C" int dorea_lut_apply_gpu(
    const float* h_pixels_in,
    const float* h_depth,
    const float* h_luts,
    const float* h_zone_boundaries,
    float* h_pixels_out,
    int width,
    int height,
    int lut_size,
    int n_zones
)
```

- [ ] **Step 2: Update hsl_correct.cu**

Same pattern: `unsigned char*` → `float*`, remove byte↔float conversions.

```c
extern "C" int dorea_hsl_correct_gpu(
    const float* h_pixels_in,
    float* h_pixels_out,
    const float* h_offsets,
    const float* h_s_ratios,
    const float* h_v_offsets,
    const float* h_weights,
    int width,
    int height,
    int n_qualifiers
)
```

- [ ] **Step 3: Update clarity.cu**

Same pattern: `unsigned char*` → `float*`.

```c
extern "C" int dorea_clarity_gpu(
    const float* h_rgb_in,
    float* h_rgb_out,
    int full_w,
    int full_h,
    int proxy_w,
    int proxy_h,
    float strength
)
```

- [ ] **Step 4: Update Rust FFI declarations in cuda/mod.rs**

In `crates/dorea-gpu/src/cuda/mod.rs`, update the `extern "C"` block:

```rust
#[cfg(feature = "cuda")]
extern "C" {
    fn dorea_lut_apply_gpu(
        h_pixels_in: *const f32,
        h_depth: *const f32,
        h_luts: *const f32,
        h_zone_boundaries: *const f32,
        h_pixels_out: *mut f32,
        width: i32,
        height: i32,
        lut_size: i32,
        n_zones: i32,
    ) -> i32;

    fn dorea_hsl_correct_gpu(
        h_pixels_in: *const f32,
        h_pixels_out: *mut f32,
        h_offsets: *const f32,
        h_s_ratios: *const f32,
        h_v_offsets: *const f32,
        h_weights: *const f32,
        width: i32,
        height: i32,
        n_qualifiers: i32,
    ) -> i32;

    fn dorea_clarity_gpu(
        h_rgb_in: *const f32,
        h_rgb_out: *mut f32,
        full_w: i32,
        full_h: i32,
        proxy_w: i32,
        proxy_h: i32,
        strength: f32,
    ) -> i32;
}
```

- [ ] **Step 5: Update grade_frame_cuda**

Change signature from:
```rust
pub fn grade_frame_cuda(pixels: &[u8], ...) -> Result<Vec<f32>, GpuError>
```
To:
```rust
pub fn grade_frame_cuda(pixels: &[f32], ...) -> Result<Vec<f32>, GpuError>
```

Remove the u8→f32 conversion. Update buffer size calculations to use `sizeof::<f32>()`
for pixel buffers.

- [ ] **Step 6: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu && cargo test -p dorea-gpu`
Expected: Compiles. Tests pass (CUDA tests may skip if no GPU in container — that's fine,
the CPU fallback tests exercise the signature).

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-gpu/
git commit -m "feat(gpu): update CUDA kernels from u8 to f32 pixel format"
```

---

### Task 9: Persistent GPU LUT Resources

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`
- Modify: `crates/dorea-gpu/src/lib.rs`

- [ ] **Step 1: Define GpuResources struct**

Add to `crates/dorea-gpu/src/cuda/mod.rs`:

```rust
#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
}

#[cfg(feature = "cuda")]
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

/// Persistent GPU resources for LUT data. Uploaded once, reused across frames.
#[cfg(feature = "cuda")]
pub struct GpuResources {
    d_luts: *mut f32,
    d_zone_boundaries: *mut f32,
    lut_size: usize,
    n_zones: usize,
    luts_bytes: usize,
    bounds_bytes: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuResources {}

#[cfg(feature = "cuda")]
impl GpuResources {
    /// Upload calibration LUT data to GPU. Call once at pipeline start.
    pub fn upload(calibration: &dorea_cal::Calibration) -> Result<Self, crate::GpuError> {
        let luts = &calibration.depth_luts;
        let n_zones = luts.n_zones();
        let lut_size = if n_zones > 0 { luts.luts[0].size } else { 33 };
        let luts_flat: Vec<f32> = luts.luts.iter()
            .flat_map(|g| g.data.iter().copied())
            .collect();
        let luts_bytes = luts_flat.len() * std::mem::size_of::<f32>();
        let bounds_bytes = luts.zone_boundaries.len() * std::mem::size_of::<f32>();

        unsafe {
            let mut d_luts: *mut f32 = std::ptr::null_mut();
            let mut d_bounds: *mut f32 = std::ptr::null_mut();

            let r1 = cudaMalloc(&mut d_luts as *mut *mut f32 as *mut *mut std::ffi::c_void, luts_bytes);
            if r1 != 0 { return Err(crate::GpuError::Cuda(format!("cudaMalloc luts failed: {r1}"))); }

            let r2 = cudaMalloc(&mut d_bounds as *mut *mut f32 as *mut *mut std::ffi::c_void, bounds_bytes);
            if r2 != 0 {
                cudaFree(d_luts as *mut std::ffi::c_void);
                return Err(crate::GpuError::Cuda(format!("cudaMalloc bounds failed: {r2}")));
            }

            let r3 = cudaMemcpy(
                d_luts as *mut std::ffi::c_void,
                luts_flat.as_ptr() as *const std::ffi::c_void,
                luts_bytes, CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if r3 != 0 {
                cudaFree(d_luts as *mut std::ffi::c_void);
                cudaFree(d_bounds as *mut std::ffi::c_void);
                return Err(crate::GpuError::Cuda(format!("cudaMemcpy luts failed: {r3}")));
            }

            let r4 = cudaMemcpy(
                d_bounds as *mut std::ffi::c_void,
                luts.zone_boundaries.as_ptr() as *const std::ffi::c_void,
                bounds_bytes, CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if r4 != 0 {
                cudaFree(d_luts as *mut std::ffi::c_void);
                cudaFree(d_bounds as *mut std::ffi::c_void);
                return Err(crate::GpuError::Cuda(format!("cudaMemcpy bounds failed: {r4}")));
            }

            Ok(Self { d_luts, d_zone_boundaries: d_bounds, lut_size, n_zones, luts_bytes, bounds_bytes })
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuResources {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.d_luts as *mut std::ffi::c_void);
            cudaFree(self.d_zone_boundaries as *mut std::ffi::c_void);
        }
    }
}
```

- [ ] **Step 2: Update grade_frame_cuda to accept optional GpuResources**

Add an overload or modify `grade_frame_cuda` to accept `Option<&GpuResources>`.
When `Some(res)`, skip the per-frame cudaMalloc/cudaMemcpy for LUT data and pass
the device pointers from the persistent resources. When `None`, fall back to
per-frame allocation (backward compat).

- [ ] **Step 3: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu`
Expected: Compiles (CUDA paths only active with feature flag).

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(gpu): persistent GPU LUT resources (upload once, reuse)"
```

---

### Task 10: InputEncoding Extended + Projection Enum

**Files:**
- Modify: `crates/dorea-cli/src/calibrate.rs:163-176`
- Modify: `crates/dorea-cli/src/grade.rs:15-72`

- [ ] **Step 1: Extend InputEncoding enum**

In `crates/dorea-cli/src/calibrate.rs`, update:

```rust
/// Supported input encodings for keyframe images.
#[derive(Debug, Clone, PartialEq)]
pub enum InputEncoding {
    Srgb,
    DlogM,
    ILog,
    Custom(std::path::PathBuf),
}

fn parse_input_encoding(s: &str) -> std::result::Result<InputEncoding, String> {
    match s {
        "srgb" => Ok(InputEncoding::Srgb),
        "dlog_m" => Ok(InputEncoding::DlogM),
        "ilog" => Ok(InputEncoding::ILog),
        other => {
            let path = std::path::PathBuf::from(other);
            if path.exists() {
                Ok(InputEncoding::Custom(path))
            } else {
                Err(format!(
                    "unknown input encoding '{other}'; expected 'srgb', 'dlog_m', 'ilog', or path to 1D .cube"
                ))
            }
        }
    }
}
```

- [ ] **Step 2: Add Projection enum**

Add to `crates/dorea-cli/src/calibrate.rs`:

```rust
/// Projection type of the input clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Rectilinear,
    TinyPlanet,
}

impl Projection {
    pub fn needs_depth(&self) -> bool {
        matches!(self, Projection::Rectilinear)
    }

    pub fn default_lut_zone(&self) -> Option<usize> {
        match self {
            Projection::Rectilinear => None,
            Projection::TinyPlanet => Some(0),
        }
    }
}

fn parse_projection(s: &str) -> std::result::Result<Projection, String> {
    match s {
        "rectilinear" => Ok(Projection::Rectilinear),
        "tiny-planet" | "tiny_planet" => Ok(Projection::TinyPlanet),
        other => Err(format!(
            "unknown projection '{other}'; expected 'rectilinear' or 'tiny-planet'"
        )),
    }
}
```

- [ ] **Step 3: Add CLI flags to CalibrateArgs**

In `crates/dorea-cli/src/calibrate.rs`, add to `CalibrateArgs`:

```rust
/// Projection type of input footage
#[arg(long, default_value = "rectilinear", value_parser = parse_projection)]
pub projection: Projection,

/// Override LUT grid size (default: auto from source bit depth)
#[arg(long)]
pub lut_size: Option<usize>,

/// Override highlight rolloff shoulder (default: per-encoding)
#[arg(long)]
pub shoulder: Option<f32>,

/// Force a single LUT depth zone (for tiny-planet or creative use)
#[arg(long)]
pub lut_zone: Option<usize>,
```

- [ ] **Step 4: Add same flags to GradeArgs**

In `crates/dorea-cli/src/grade.rs`, add matching fields to `GradeArgs`:

```rust
/// Input encoding: srgb, dlog_m, ilog, or path to 1D .cube
#[arg(long, default_value = "dlog_m", value_parser = crate::calibrate::parse_input_encoding)]
pub input_encoding: crate::calibrate::InputEncoding,

/// Projection type
#[arg(long, default_value = "rectilinear", value_parser = crate::calibrate::parse_projection)]
pub projection: crate::calibrate::Projection,

/// Output codec: prores, hevc, h264
#[arg(long, default_value = "prores")]
pub output_codec: String,

/// Depth inference batch size
#[arg(long, default_value = "2")]
pub batch_size: usize,
```

- [ ] **Step 5: Wire up D-Log M decode to use the trait**

In `crates/dorea-cli/src/calibrate.rs`, update the D-Log M decode block to use the
`TransferFunction` trait. Replace the direct `dlog_m_to_linear` calls with a
trait-dispatched decoder:

```rust
use dorea_color::{TransferFunction, DLogM, ILog, LutBased};

// Build transfer function from encoding
let transfer_fn: Box<dyn TransferFunction> = match &args.input_encoding {
    InputEncoding::DlogM => Box::new(DLogM),
    InputEncoding::ILog => Box::new(ILog),
    InputEncoding::Custom(path) => {
        let shoulder = args.shoulder.unwrap_or(0.88);
        Box::new(LutBased::from_cube_file(path, shoulder)
            .with_context(|| format!("Failed to load custom transfer function from {}", path.display()))?)
    }
    InputEncoding::Srgb => {
        // No decode needed for sRGB — skip the block below
        // Use a no-op marker
        Box::new(DLogM) // placeholder, won't be used
    }
};

// Apply log decode
if args.input_encoding != InputEncoding::Srgb {
    log::debug!("Applying {} decode to keyframe {kf_path:?}", transfer_fn.name());
    for px in kf_pixels.iter_mut() {
        px[0] = transfer_fn.to_linear(px[0]);
        px[1] = transfer_fn.to_linear(px[1]);
        px[2] = transfer_fn.to_linear(px[2]);
    }
}
```

- [ ] **Step 6: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build && cargo test`
Expected: Compiles and all tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-cli/src/calibrate.rs crates/dorea-cli/src/grade.rs
git commit -m "feat(cli): extend InputEncoding with ILog/Custom, add Projection enum"
```

---

### Task 11: Container Auto-Detection (dorea probe)

**Files:**
- Create: `crates/dorea-video/src/probe.rs`
- Modify: `crates/dorea-video/src/lib.rs`
- Create: `crates/dorea-cli/src/probe.rs`
- Modify: `crates/dorea-cli/src/main.rs`

- [ ] **Step 1: Create probe module in dorea-video**

Create `crates/dorea-video/src/probe.rs`:

```rust
//! Container/codec sniffing for auto-detection of input encoding.

use crate::ffmpeg;
use std::path::Path;

/// Detected container type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContainerHint {
    Mov,
    Mp4,
    Other,
}

/// Suggested input encoding based on container/codec analysis.
#[derive(Debug)]
pub struct ProbeResult {
    pub info: ffmpeg::VideoInfo,
    pub container: ContainerHint,
    pub codec_name: String,
    pub bit_depth: Option<u8>,
    pub suggested_encoding: String,
    pub suggested_lut_size: usize,
    pub suggested_output_codec: String,
}

/// Probe a video file and suggest pipeline settings.
pub fn probe_file(path: &Path) -> Result<ProbeResult, crate::ffmpeg::FfmpegError> {
    let info = ffmpeg::probe(path)?;

    let ext = path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let container = match ext.as_str() {
        "mov" => ContainerHint::Mov,
        "mp4" | "m4v" => ContainerHint::Mp4,
        _ => ContainerHint::Other,
    };

    // Sniff codec via ffprobe -show_streams
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,bits_per_raw_sample,pix_fmt",
            "-of", "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| crate::ffmpeg::FfmpegError::NotFound(e))?;

    let probe_line = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = probe_line.trim().split(',').collect();

    let codec_name = fields.first().unwrap_or(&"unknown").to_string();
    let bit_depth = fields.get(1).and_then(|s| s.parse::<u8>().ok());

    let (suggested_encoding, suggested_lut_size, suggested_output_codec) =
        match (container, codec_name.as_str()) {
            (ContainerHint::Mov, "prores") => ("ilog".to_string(), 65, "prores".to_string()),
            (ContainerHint::Mp4, "hevc") | (ContainerHint::Mp4, "h265") =>
                ("dlog_m".to_string(), 65, "prores".to_string()),
            _ => ("srgb".to_string(), 33, "h264".to_string()),
        };

    Ok(ProbeResult {
        info,
        container,
        codec_name,
        bit_depth,
        suggested_encoding,
        suggested_lut_size,
        suggested_output_codec,
    })
}
```

- [ ] **Step 2: Register in dorea-video/src/lib.rs**

Add `pub mod probe;` to `crates/dorea-video/src/lib.rs`.

- [ ] **Step 3: Create CLI probe subcommand**

Create `crates/dorea-cli/src/probe.rs`:

```rust
use clap::Args;
use std::path::PathBuf;
use anyhow::{Context, Result};

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Input video file to probe
    #[arg(long)]
    pub input: PathBuf,
}

pub fn run(args: ProbeArgs) -> Result<()> {
    let result = dorea_video::probe::probe_file(&args.input)
        .context("probe failed")?;

    println!("File:       {}", args.input.display());
    println!("Container:  {:?}", result.container);
    println!("Codec:      {}", result.codec_name);
    if let Some(bd) = result.bit_depth {
        println!("Bit depth:  {bd}-bit");
    }
    println!("Resolution: {}x{} @ {:.2}fps", result.info.width, result.info.height, result.info.fps);
    println!("Duration:   {:.1}s ({} frames)", result.info.duration_secs, result.info.frame_count);
    println!("Suggested:  --input-encoding {} --output-codec {} --lut-size {}",
        result.suggested_encoding, result.suggested_output_codec, result.suggested_lut_size);

    Ok(())
}
```

- [ ] **Step 4: Register probe in main.rs**

In `crates/dorea-cli/src/main.rs`, add `Probe(probe::ProbeArgs)` to the `Command` enum
and dispatch it in the match arm:

```rust
Command::Probe(args) => probe::run(args),
```

Add `pub mod probe;` to the module declarations.

- [ ] **Step 5: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli`
Expected: Compiles.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-video/src/probe.rs crates/dorea-video/src/lib.rs \
       crates/dorea-cli/src/probe.rs crates/dorea-cli/src/main.rs
git commit -m "feat(cli): add dorea probe command for container auto-detection"
```

---

### Task 12: 10-Bit Output Encoding

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs` (FrameEncoder)

- [ ] **Step 1: Add OutputCodec enum**

In `crates/dorea-video/src/ffmpeg.rs`, add:

```rust
/// Output codec for the graded video.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputCodec {
    /// ProRes 422 HQ 10-bit (default for DaVinci Resolve)
    ProRes,
    /// HEVC 10-bit via NVENC
    Hevc10,
    /// H.264 8-bit (legacy)
    H264,
}

impl OutputCodec {
    pub fn from_str(s: &str) -> Result<Self, FfmpegError> {
        match s {
            "prores" => Ok(Self::ProRes),
            "hevc" => Ok(Self::Hevc10),
            "h264" => Ok(Self::H264),
            other => Err(FfmpegError::UnsupportedFormat(
                format!("unknown output codec '{other}'; expected prores, hevc, or h264")
            )),
        }
    }

    fn ffmpeg_args(&self) -> Vec<String> {
        match self {
            Self::ProRes => vec![
                "-c:v".into(), "prores_ks".into(),
                "-profile:v".into(), "3".into(), // HQ
                "-pix_fmt".into(), "yuv422p10le".into(),
                "-vendor".into(), "apl0".into(),
            ],
            Self::Hevc10 => vec![
                "-c:v".into(), "hevc_nvenc".into(),
                "-profile:v".into(), "main10".into(),
                "-preset".into(), "p4".into(),
                "-cq".into(), "18".into(),
                "-pix_fmt".into(), "yuv420p10le".into(),
            ],
            Self::H264 => vec![
                "-c:v".into(), "libx264".into(),
                "-crf".into(), "18".into(),
                "-preset".into(), "fast".into(),
                "-pix_fmt".into(), "yuv420p".into(),
            ],
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::ProRes => "mov",
            Self::Hevc10 | Self::H264 => "mp4",
        }
    }
}
```

- [ ] **Step 2: Add FfmpegError::UnsupportedFormat variant if needed**

Check if `FfmpegError` already has an appropriate variant. If not, add:

```rust
#[error("unsupported format: {0}")]
UnsupportedFormat(String),
```

- [ ] **Step 3: Update FrameEncoder to accept OutputCodec**

Modify `FrameEncoder::new` to accept an `OutputCodec` parameter and use its
`ffmpeg_args()` instead of the hardcoded NVENC/libx264 detection.

For ProRes output, the encoder must accept **f32 pixel data** and pipe it as
`-f rawvideo -pix_fmt gbrpf32le` (planar float) or convert to 16-bit on the Rust
side before piping. The simplest approach: convert f32 → u16 in Rust, pipe as
`-f rawvideo -pix_fmt rgb48le`, and let ffmpeg handle the yuv422p10le conversion.

Add a new method:

```rust
/// Write a frame from f32 RGB data [0,1]. Converts to 16-bit for the pipe.
pub fn write_frame_f32(&mut self, pixels: &[f32]) -> Result<(), FfmpegError> {
    let u16_data: Vec<u8> = pixels.iter().flat_map(|&v| {
        let val = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
        val.to_le_bytes()
    }).collect();
    self.write_raw(&u16_data)
}
```

Update the ffmpeg input format from `-pix_fmt rgb24` to `-pix_fmt rgb48le` when
the encoder is created with a 10-bit output codec.

- [ ] **Step 4: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-video`
Expected: Compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): 10-bit output encoding (ProRes HQ, HEVC NVENC, H.264)"
```

---

## Phase C: Pipeline + Tiny Planet + Integration

---

### Task 13: 3-Stage Pipelined Frame Loop

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs:75-209`

This is the largest task. Replace the serial `for frame in frames` loop with
a 3-stage bounded-channel pipeline.

- [ ] **Step 1: Add sync_channel imports**

At the top of `crates/dorea-cli/src/grade.rs`:

```rust
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};
use std::thread;
```

- [ ] **Step 2: Define inter-stage message types**

Add in `grade.rs`:

```rust
/// Message from Stage 1 (decode) to Stage 2 (depth).
struct DecodedFrame {
    index: u64,
    /// f32 RGB pixels [0,1], length = width * height * 3
    pixels_f32: Vec<f32>,
    /// u8 proxy pixels for inference, length = proxy_w * proxy_h * 3
    proxy_u8: Vec<u8>,
    proxy_w: usize,
    proxy_h: usize,
    width: usize,
    height: usize,
}

/// Message from Stage 2 (depth) to Stage 3 (grade+encode).
struct DepthFrame {
    index: u64,
    pixels_f32: Vec<f32>,
    depth: Vec<f32>,
    width: usize,
    height: usize,
}
```

- [ ] **Step 3: Refactor run() into 3-stage pipeline**

Replace the `for frame_result in frames` loop in `run()` with:

```rust
let (tx_decoded, rx_decoded) = sync_channel::<DecodedFrame>(3);
let (tx_depth, rx_depth) = sync_channel::<DepthFrame>(3);

let input_path = args.input.clone();
let proxy_size = args.proxy_size;
let info_clone = info.clone();

// Stage 1: Decode thread
let decode_handle = thread::spawn(move || -> Result<()> {
    let frames = ffmpeg::decode_frames(&input_path, &info_clone)
        .context("failed to spawn ffmpeg decoder")?;
    for frame_result in frames {
        let frame = frame_result.context("frame decode error")?;
        let pixels_f32: Vec<f32> = frame.pixels.iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        let (pw, ph) = dorea_video::resize::proxy_dims(frame.width, frame.height, proxy_size);
        let proxy_u8 = if pw != frame.width || ph != frame.height {
            dorea_video::resize::resize_rgb_bilinear(
                &frame.pixels, frame.width, frame.height, pw, ph,
            )
        } else {
            frame.pixels.clone()
        };
        if tx_decoded.send(DecodedFrame {
            index: frame.index,
            pixels_f32,
            proxy_u8,
            proxy_w: pw,
            proxy_h: ph,
            width: frame.width,
            height: frame.height,
        }).is_err() {
            break; // receiver dropped
        }
    }
    Ok(())
});

// Stage 2: Depth inference thread
let needs_depth = args.projection.needs_depth();
let depth_handle = thread::spawn(move || -> Result<()> {
    if needs_depth {
        let inf_cfg = InferenceConfig {
            skip_raune: true,
            ..build_inference_config_from_grade(&args_clone)
        };
        let mut inf_server = InferenceServer::spawn(&inf_cfg)
            .context("failed to spawn inference server")?;
        for df in rx_decoded {
            let (depth_proxy, dw, dh) = inf_server
                .run_depth(&df.index.to_string(), &df.proxy_u8, df.proxy_w, df.proxy_h, proxy_size)
                .unwrap_or_else(|e| {
                    log::warn!("Depth failed frame {}: {e}", df.index);
                    (vec![0.5f32; df.width * df.height], df.width, df.height)
                });
            let depth = if dw == df.width && dh == df.height {
                depth_proxy
            } else {
                InferenceServer::upscale_depth(&depth_proxy, dw, dh, df.width, df.height)
            };
            if tx_depth.send(DepthFrame {
                index: df.index,
                pixels_f32: df.pixels_f32,
                depth,
                width: df.width,
                height: df.height,
            }).is_err() {
                break;
            }
        }
        let _ = inf_server.shutdown();
    } else {
        // Tiny planet: skip depth, pass through with uniform depth
        for df in rx_decoded {
            let n = df.width * df.height;
            if tx_depth.send(DepthFrame {
                index: df.index,
                pixels_f32: df.pixels_f32,
                depth: vec![0.5f32; n],
                width: df.width,
                height: df.height,
            }).is_err() {
                break;
            }
        }
    }
    Ok(())
});

// Stage 3: Grade + encode (main thread, owns CUDA context)
let mut frame_count = 0u64;
for df in rx_depth {
    let graded = grade_frame(
        &df.pixels_f32, &df.depth, df.width, df.height, &calibration, &params,
    ).unwrap_or_else(|e| {
        log::warn!("Grade failed frame {}: {e}", df.index);
        df.pixels_f32.clone()
    });
    encoder.write_frame_f32(&graded).context("encoder write failed")?;
    frame_count += 1;
    if frame_count % 100 == 0 {
        let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
        log::info!("Progress: {frame_count}/{} ({:.1}%)", info.frame_count, pct);
    }
}

// Join worker threads
decode_handle.join().map_err(|_| anyhow::anyhow!("decode thread panicked"))??;
depth_handle.join().map_err(|_| anyhow::anyhow!("depth thread panicked"))??;

encoder.finish().context("encoder failed to finalize")?;
```

- [ ] **Step 4: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli`
Expected: Compiles. Full integration test requires a video file — defer to manual testing.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(cli): 3-stage pipelined frame loop (decode → depth → grade)"
```

---

### Task 14: Batched Depth Inference

**Files:**
- Modify: `python/dorea_inference/server.py`
- Modify: `crates/dorea-video/src/inference.rs`

- [ ] **Step 1: Add depth_batch handler to Python server**

In `python/dorea_inference/server.py`, add a new command handler in the main loop
(alongside the existing `"depth"` handler):

```python
elif req_type == "depth_batch":
    frames = req.get("frames", [])
    results = []
    for frame_req in frames:
        fid = frame_req["id"]
        img_b64 = frame_req["image"]
        proxy_size = frame_req.get("proxy_size", 518)
        img_bytes = base64.b64decode(img_b64)
        img = decode_png(img_bytes)  # reuse existing decode
        depth = depth_model.infer(img, proxy_size)
        depth_b64 = base64.b64encode(depth.astype(np.float32).tobytes()).decode()
        results.append({"id": fid, "depth_f32_b64": depth_b64,
                        "width": depth.shape[1], "height": depth.shape[0]})
    send_response({"type": "depth_batch_result", "results": results})
```

- [ ] **Step 2: Add run_depth_batch to Rust InferenceServer**

In `crates/dorea-video/src/inference.rs`, add:

```rust
/// Run depth inference on a batch of frames. Falls back to sequential
/// single-frame calls if the server doesn't support batching.
pub fn run_depth_batch(
    &mut self,
    frames: &[(String, &[u8], usize, usize, usize)], // (id, pixels, w, h, proxy_size)
) -> Result<Vec<(Vec<f32>, usize, usize)>, InferenceError> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }

    // Build batch request
    let batch_frames: Vec<serde_json::Value> = frames.iter().map(|(id, pixels, w, h, ps)| {
        let png = encode_rgb_png(
            &pixels.chunks(3).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]).collect::<Vec<_>>(),
            *w, *h,
        ).unwrap();
        serde_json::json!({
            "id": id,
            "image": base64::engine::general_purpose::STANDARD.encode(&png),
            "proxy_size": ps,
        })
    }).collect();

    let request = serde_json::json!({
        "type": "depth_batch",
        "frames": batch_frames,
    });

    self.send_request(&request)?;
    let response = self.read_response()?;

    let results = response["results"].as_array()
        .ok_or(InferenceError::Protocol("missing results array".into()))?;

    let mut output = Vec::new();
    for r in results {
        let depth_b64 = r["depth_f32_b64"].as_str()
            .ok_or(InferenceError::Protocol("missing depth_f32_b64".into()))?;
        let depth_bytes = base64::engine::general_purpose::STANDARD.decode(depth_b64)
            .map_err(|e| InferenceError::Protocol(format!("bad base64: {e}")))?;
        let depth: Vec<f32> = depth_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let w = r["width"].as_u64().unwrap_or(0) as usize;
        let h = r["height"].as_u64().unwrap_or(0) as usize;
        output.push((depth, w, h));
    }

    Ok(output)
}
```

- [ ] **Step 3: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-video`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add python/dorea_inference/server.py crates/dorea-video/src/inference.rs
git commit -m "feat(video): batched depth inference (Python server + Rust client)"
```

---

### Task 15: Tiny Planet Depth-Skip Path

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

The 3-stage pipeline from Task 13 already branches on `projection.needs_depth()`.
This task adds the degenerate depth map warning heuristic and zone-0 LUT forcing.

- [ ] **Step 1: Add depth map variance check**

In `crates/dorea-cli/src/grade.rs`, add a helper:

```rust
/// Check if a depth map looks degenerate (possible tiny planet).
/// Returns true if variance is suspiciously low.
fn depth_looks_degenerate(depth: &[f32]) -> bool {
    if depth.is_empty() { return false; }
    let mean = depth.iter().sum::<f32>() / depth.len() as f32;
    let variance = depth.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / depth.len() as f32;
    // Threshold: typical underwater depth maps have variance > 0.01.
    // Tiny planet or flat scenes have variance < 0.005.
    variance < 0.005
}
```

- [ ] **Step 2: Add warning after first depth inference**

In the Stage 2 depth thread (from Task 13), after the first frame's depth is computed,
check:

```rust
if frame_count_stage2 == 0 && depth_looks_degenerate(&depth) {
    log::warn!(
        "Depth map has very low variance — is this a tiny planet shot? \
         Use --projection tiny-planet to skip depth estimation."
    );
}
```

- [ ] **Step 3: Wire lut_zone override into grading**

When `projection == TinyPlanet` or `--lut-zone N` is set, the grading path should
use only the specified zone. This means passing a `force_zone: Option<usize>` through
to the LUT apply function. In the grading call:

```rust
let graded = if let Some(zone) = args.projection.default_lut_zone().or(args.lut_zone) {
    grade_frame_single_zone(&df.pixels_f32, zone, df.width, df.height, &calibration, &params)
} else {
    grade_frame(&df.pixels_f32, &df.depth, df.width, df.height, &calibration, &params)
};
```

The `grade_frame_single_zone` function applies one LUT zone uniformly (no depth blending).
Add this to `dorea-gpu/src/lib.rs`:

```rust
pub fn grade_frame_single_zone(
    pixels: &[f32],
    zone: usize,
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<f32>, GpuError> {
    // Create uniform depth at the zone center
    let n = width * height;
    let zone_center = calibration.depth_luts.zone_centers
        .get(zone)
        .copied()
        .unwrap_or(0.5);
    let uniform_depth = vec![zone_center; n];
    grade_frame(pixels, &uniform_depth, width, height, calibration, params)
}
```

- [ ] **Step 4: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build && cargo test`
Expected: Compiles, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(cli): tiny planet depth-skip path with degenerate depth warning"
```

---

### Task 16: Calibration Format Upgrade

**Files:**
- Modify: `crates/dorea-cal/src/format.rs`

- [ ] **Step 1: Add camera profile to Calibration**

In `crates/dorea-cal/src/format.rs`, update the struct:

```rust
const FORMAT_VERSION: u8 = 3; // bumped from 2

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub version: u8,
    pub depth_luts: DepthLuts,
    pub hsl_corrections: HslCorrections,
    pub created_at_unix_secs: u64,
    pub keyframe_count: usize,
    pub source_description: String,
    /// Camera encoding used during calibration (e.g. "dlog_m", "ilog").
    #[serde(default = "default_encoding")]
    pub source_encoding: String,
    /// LUT grid size (33 or 65).
    #[serde(default = "default_lut_size")]
    pub lut_size: usize,
}

fn default_encoding() -> String { "dlog_m".to_string() }
fn default_lut_size() -> usize { 33 }
```

The `#[serde(default)]` annotations ensure old .dorea-cal files (version 2) load
successfully with defaults.

- [ ] **Step 2: Update load() to accept old versions**

In `Calibration::load`, change version check from `!= FORMAT_VERSION` to
`> FORMAT_VERSION`:

```rust
if cal.version > FORMAT_VERSION {
    return Err(CalError::UnsupportedVersion(cal.version, FORMAT_VERSION));
}
```

This allows loading v2 and v3 files.

- [ ] **Step 3: Update new() to set new fields**

Update the constructor and callers to pass `source_encoding` and `lut_size`.

- [ ] **Step 4: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cal`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cal/src/format.rs
git commit -m "feat(cal): bump format to v3 with source_encoding and lut_size"
```

---

### Task 17: ProRes Frame Selection (All-Intra)

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

- [ ] **Step 1: Add codec-aware frame extraction**

In `crates/dorea-video/src/ffmpeg.rs`, update `spawn_decoder` to detect ProRes and
skip the I-frame select filter. Add a helper:

```rust
/// Check if a codec is all-intra (every frame is a keyframe).
fn is_all_intra_codec(codec_name: &str) -> bool {
    matches!(codec_name, "prores" | "dnxhd" | "dnxhr" | "mjpeg" | "ffv1")
}
```

In the frame extraction path (used by auto-calibration in grade.rs), when building
the ffmpeg `-vf` filter string, skip `select=eq(pict_type,I)` for all-intra codecs.
The interval/scene-change selection still applies.

- [ ] **Step 2: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-video`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): codec-aware frame selection (skip I-frame filter for ProRes)"
```

---

### Task 18: Integration — Wire Everything Together

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`
- Modify: `crates/dorea-cli/src/calibrate.rs`

This task connects all the pieces: CLI flags flow through to the pipeline, auto-detection
prints mismatch warnings, output codec is respected, LUT size is auto-defaulted.

- [ ] **Step 1: Wire output codec in grade.rs**

In `run()`, use `args.output_codec` to create the encoder:

```rust
let output_codec = ffmpeg::OutputCodec::from_str(&args.output_codec)
    .context("invalid output codec")?;

let output = args.output.clone().unwrap_or_else(|| {
    let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
    args.input.with_file_name(format!("{stem}_graded.{}", output_codec.file_extension()))
});

let mut encoder = FrameEncoder::new_with_codec(
    &output, info.width, info.height, info.fps, audio_src, output_codec
).context("failed to spawn encoder")?;
```

- [ ] **Step 2: Wire auto-detection mismatch warning**

At the start of `run()`, after probing:

```rust
let probe_result = dorea_video::probe::probe_file(&args.input)
    .context("probe failed")?;

if probe_result.suggested_encoding != args.input_encoding.to_string() {
    log::warn!(
        "Container suggests --input-encoding {} but you specified {}. \
         Use --force to suppress this warning.",
        probe_result.suggested_encoding,
        args.input_encoding.to_string()
    );
}
```

Add a `to_string()` method on `InputEncoding` for display.

- [ ] **Step 3: Wire LUT size auto-default in calibrate.rs**

In the calibration path:

```rust
let lut_size = args.lut_size.unwrap_or_else(|| {
    match &args.input_encoding {
        InputEncoding::DlogM | InputEncoding::ILog | InputEncoding::Custom(_) => 65,
        InputEncoding::Srgb => 33,
    }
});
```

Pass `lut_size` to `build_depth_luts`.

- [ ] **Step 4: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build && cargo test`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/calibrate.rs
git commit -m "feat(cli): wire 10-bit pipeline end-to-end (codec, detection, LUT size)"
```

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|------------------|
| A (Foundation) | 1–6 | TransferFunction trait, I-Log, LutBased, 16-bit I/O, configurable LUT |
| B (GPU + Pipeline) | 7–12 | f32 grading, persistent GPU LUT, extended CLI, 10-bit output |
| C (Integration) | 13–18 | 3-stage pipeline, batched depth, tiny planet, format upgrade, wiring |

**Total tasks:** 18
**Estimated commits:** 18 (one per task)
**Dependencies:** Tasks are sequential within each phase. Phase B depends on Phase A. Phase C depends on Phase B.
