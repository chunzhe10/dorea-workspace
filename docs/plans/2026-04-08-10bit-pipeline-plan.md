# 10-Bit Pipeline Implementation Plan (Updated)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Dorea from 8-bit to 10-bit I/O supporting DJI D-Log M and Insta360 X5 I-Log, with 10-bit decode/encode and an optional u16 CUDA kernel path for full precision.

**Architecture:** The pipeline already has a CombinedLut CUDA 3D texture grader, 3-thread pipeline (decode → grade → encode), YOLO-seg diver detection, and depth dithering/output dithering. This plan upgrades the I/O boundaries: ffmpeg decodes as rgb48le (u16), a new CUDA kernel variant processes u16→f32→u16, and the encoder outputs ProRes 422 HQ or HEVC 10-bit. Transfer functions abstract camera log curves. The existing 8-bit path is preserved for backward compat.

**Tech Stack:** Rust (dorea crates), CUDA 12.4, ffmpeg (rgb48le decode, ProRes/HEVC encode), Python (inference stays 8-bit proxy)

**Spec:** `docs/decisions/2026-04-02-10bit-pipeline-design.md`

**What changed since the original plan (2026-04-02):**
- CombinedLut replaced 3 separate kernels (lut_apply, hsl_correct, clarity all deleted)
- 3-thread pipeline already exists (decode/grade/encode with crossbeam channels)
- 97³ LUT grid already in use
- YOLO-seg diver detection added
- Depth dithering + triangular output dithering added
- GPU-side bilinear depth sampling added
- Keyframes are in-memory Vec<u8>, not JPEG files on disk
- AdaptiveGrader with dual-texture blending and per-keyframe zone boundaries

---

## File Structure

### New files:
- `crates/dorea-color/src/ilog.rs` — I-Log transfer function (S-Log3 proxy)
- `crates/dorea-color/src/lut_transfer.rs` — 1D .cube LUT-based transfer function
- `crates/dorea-cli/src/probe.rs` — `dorea probe` subcommand

### Modified files:
- `crates/dorea-color/src/lib.rs` — TransferFunction trait, module exports
- `crates/dorea-color/src/dlog_m.rs` — Implement TransferFunction for DLogM
- `crates/dorea-video/src/ffmpeg.rs` — Frame16, decode_frames_16bit, OutputCodec, FrameEncoder::new_10bit, VideoInfo enhancements
- `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu` — Add combined_lut_kernel_16 entry point (u16 I/O)
- `crates/dorea-gpu/src/cuda/mod.rs` — FrameBuffers16, grade_frame_blended_16 on AdaptiveGrader
- `crates/dorea-gpu/src/lib.rs` — grade_frame_with_adaptive_grader_16 wrapper
- `crates/dorea-cli/src/grade.rs` — --input-encoding, --output-codec CLI flags
- `crates/dorea-cli/src/config.rs` — output_codec, input_encoding config fields
- `crates/dorea-cli/src/pipeline/mod.rs` — OutputCodec + InputEncoding in PipelineConfig
- `crates/dorea-cli/src/pipeline/grading.rs` — 10-bit frame path (Frame16 → u16 grade → u16 encode)
- `crates/dorea-cli/src/main.rs` — Register Probe subcommand

---

## Phase A: Color Science Foundation

---

### Task 1: TransferFunction Trait + DLogM Implementation

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

- [ ] **Step 1: Create ilog.rs with implementation and tests**

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
const ENC_CUT: f64 = 0.171260702;

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

### Task 3: LutBased Transfer Function (1D .cube)

**Files:**
- Create: `crates/dorea-color/src/lut_transfer.rs`
- Modify: `crates/dorea-color/src/lib.rs`

- [ ] **Step 1: Create lut_transfer.rs with full implementation and tests**

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
pub struct LutBased {
    forward: Vec<f32>,
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
            if line.starts_with("LUT_3D_SIZE") || line.starts_with("DOMAIN_") {
                continue;
            }
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
        // Leak to get 'static. LutBased lives for the pipeline duration.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

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

fn invert_1d_lut(forward: &[f32]) -> Vec<f32> {
    let n = forward.len();
    let mut inverse = vec![0.0f32; n];
    let out_min = forward[0];
    let out_max = forward[n - 1];
    let range = (out_max - out_min).max(1e-10);

    for i in 0..n {
        let target = i as f32 / (n - 1) as f32;
        let target_scaled = out_min + target * range;
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

## Phase B: 10-Bit I/O (Decode + Encode)

---

### Task 4: VideoInfo Enhancement — Detect Bit Depth and Codec

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

The current `probe()` function returns width/height/fps/duration/frame_count/has_audio.
We need to also detect the video codec name, pixel format, and bit depth so that
downstream code can auto-detect 10-bit input and suggest the correct `InputEncoding`.

- [ ] **Step 1: Add fields to VideoInfo**

In `crates/dorea-video/src/ffmpeg.rs`, extend the `VideoInfo` struct:

```rust
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub duration_secs: f64,
    pub frame_count: u64,
    pub has_audio: bool,
    /// Video codec name (e.g. "hevc", "prores", "h264").
    pub codec_name: String,
    /// Pixel format (e.g. "yuv420p10le", "yuv422p10le", "yuv420p").
    pub pix_fmt: String,
    /// Bits per component (8, 10, 12). 0 if unknown.
    pub bits_per_component: u8,
}
```

- [ ] **Step 2: Extract new fields from ffprobe JSON**

In the `probe()` function, after extracting width/height, add:

```rust
let codec_name = video["codec_name"].as_str().unwrap_or("unknown").to_string();
let pix_fmt = video["pix_fmt"].as_str().unwrap_or("unknown").to_string();

// Detect bit depth: prefer bits_per_raw_sample, fall back to pix_fmt heuristic
let bits_per_component = video["bits_per_raw_sample"]
    .as_str()
    .and_then(|s| s.parse::<u8>().ok())
    .unwrap_or_else(|| {
        if pix_fmt.contains("10") { 10 }
        else if pix_fmt.contains("12") { 12 }
        else { 8 }
    });
```

Update the `Ok(VideoInfo { ... })` return to include the three new fields.

- [ ] **Step 3: Add test for 10-bit detection**

Add to the existing `tests` module in `crates/dorea-video/src/ffmpeg.rs`:

```rust
#[test]
fn video_info_has_codec_fields() {
    // Verify the struct can be constructed with the new fields
    let info = VideoInfo {
        width: 3840,
        height: 2160,
        fps: 29.97,
        duration_secs: 10.0,
        frame_count: 300,
        has_audio: true,
        codec_name: "hevc".to_string(),
        pix_fmt: "yuv420p10le".to_string(),
        bits_per_component: 10,
    };
    assert_eq!(info.bits_per_component, 10);
    assert_eq!(info.codec_name, "hevc");
}
```

- [ ] **Step 4: Fix any compilation errors from new fields**

Any code that constructs `VideoInfo` directly (e.g. tests) needs the new fields.
Search for `VideoInfo {` in the codebase and add defaults where needed.

- [ ] **Step 5: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): add codec_name, pix_fmt, bits_per_component to VideoInfo"
```

---

### Task 5: InputEncoding Enum and Auto-Detection

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

- [ ] **Step 1: Define InputEncoding enum**

Add near the top of `crates/dorea-video/src/ffmpeg.rs` (after the error types):

```rust
/// Camera input encoding — determines the transfer function applied during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEncoding {
    /// DJI Action 4 D-Log M (default for .mp4 H.265 10-bit)
    DLogM,
    /// Insta360 X5 I-Log (default for .mov ProRes 10-bit)
    ILog,
    /// Standard sRGB (8-bit consumer cameras)
    Srgb,
}

impl InputEncoding {
    /// Auto-detect encoding from codec and container metadata.
    ///
    /// Heuristic:
    /// - ProRes in .mov → ILog (Insta360 X5)
    /// - H.265 10-bit in .mp4 → DLogM (DJI Action 4)
    /// - Everything else → Srgb
    pub fn auto_detect(info: &VideoInfo, path: &std::path::Path) -> Self {
        let ext = path.extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        if info.codec_name == "prores" || ext == "mov" {
            return Self::ILog;
        }
        if info.bits_per_component >= 10
            && (info.codec_name == "hevc" || info.codec_name == "h265")
        {
            return Self::DLogM;
        }
        Self::Srgb
    }

    /// Whether this encoding comes from a 10-bit source.
    pub fn is_10bit(&self) -> bool {
        matches!(self, Self::DLogM | Self::ILog)
    }
}

impl std::fmt::Display for InputEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DLogM => write!(f, "dlog-m"),
            Self::ILog => write!(f, "ilog"),
            Self::Srgb => write!(f, "srgb"),
        }
    }
}

impl std::str::FromStr for InputEncoding {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "dlog-m" | "dlogm" | "dlog_m" => Ok(Self::DLogM),
            "ilog" | "i-log" => Ok(Self::ILog),
            "srgb" | "rec709" => Ok(Self::Srgb),
            other => Err(format!("unknown encoding '{other}'; expected dlog-m, ilog, or srgb")),
        }
    }
}
```

- [ ] **Step 2: Add tests**

```rust
#[test]
fn input_encoding_auto_detect_dji() {
    let info = VideoInfo {
        width: 3840, height: 2160, fps: 29.97, duration_secs: 3.0,
        frame_count: 90, has_audio: true,
        codec_name: "hevc".to_string(),
        pix_fmt: "yuv420p10le".to_string(),
        bits_per_component: 10,
    };
    let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
    assert_eq!(enc, InputEncoding::DLogM);
}

#[test]
fn input_encoding_auto_detect_x5() {
    let info = VideoInfo {
        width: 5760, height: 2880, fps: 30.0, duration_secs: 5.0,
        frame_count: 150, has_audio: false,
        codec_name: "prores".to_string(),
        pix_fmt: "yuv422p10le".to_string(),
        bits_per_component: 10,
    };
    let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mov"));
    assert_eq!(enc, InputEncoding::ILog);
}

#[test]
fn input_encoding_parse() {
    assert_eq!("dlog-m".parse::<InputEncoding>().unwrap(), InputEncoding::DLogM);
    assert_eq!("ilog".parse::<InputEncoding>().unwrap(), InputEncoding::ILog);
    assert_eq!("srgb".parse::<InputEncoding>().unwrap(), InputEncoding::Srgb);
    assert!("bogus".parse::<InputEncoding>().is_err());
}
```

- [ ] **Step 3: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): add InputEncoding enum with auto-detection"
```

---

### Task 6: 10-Bit Decode Path — Frame16 and decode_frames_16bit

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

ffmpeg outputs `rgb48le` (little-endian, 16-bit per component, 6 bytes per pixel) for
10-bit sources. We add a `Frame16` struct and `decode_frames_16bit()` function.

- [ ] **Step 1: Add Frame16 struct**

Add after the existing `Frame` struct:

```rust
/// A decoded video frame in 16-bit precision.
/// Used for 10-bit source material (DJI D-Log M, Insta360 X5 I-Log).
#[derive(Debug)]
pub struct Frame16 {
    pub index: u64,
    pub pixels: Vec<u16>, // RGB48: width * height * 3 u16 values
    pub width: usize,
    pub height: usize,
}
```

- [ ] **Step 2: Add FrameReader16**

Add a 16-bit variant of the frame reader:

```rust
struct FrameReader16 {
    child: Child,
    frame_index: u64,
    width: usize,
    height: usize,
    frame_bytes: usize, // width * height * 6 (2 bytes per component × 3 channels)
    done: bool,
}

impl FrameReader16 {
    fn new(child: Child, width: usize, height: usize) -> Self {
        Self {
            child,
            frame_index: 0,
            width,
            height,
            frame_bytes: width * height * 6,
            done: false,
        }
    }
}

impl Drop for FrameReader16 {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Iterator for FrameReader16 {
    type Item = Result<Frame16, FfmpegError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut buf = vec![0u8; self.frame_bytes];
        let stdout = self.child.stdout.as_mut()?;

        match read_exact(stdout, &mut buf) {
            Ok(0) | Err(_) => {
                self.done = true;
                return None;
            }
            Ok(_) => {}
        }

        // Convert LE byte pairs to u16
        let pixels: Vec<u16> = buf.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        let frame = Frame16 {
            index: self.frame_index,
            pixels,
            width: self.width,
            height: self.height,
        };
        self.frame_index += 1;
        Some(Ok(frame))
    }
}
```

- [ ] **Step 3: Add spawn_decoder_16bit and decode_frames_16bit**

```rust
fn spawn_decoder_16bit(input: &Path, info: &VideoInfo) -> Result<Child, FfmpegError> {
    let input_str = input.to_str().unwrap_or("");
    let size_str = format!("{}x{}", info.width, info.height);

    // Try hardware decode first
    let hw_result = Command::new("ffmpeg")
        .args([
            "-hwaccel", "nvdec",
            "-i", input_str,
            "-vf", &format!("scale={size_str}"),
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();

    if let Ok(child) = hw_result {
        return Ok(child);
    } else if let Err(ref e) = hw_result {
        log::debug!("nvdec 16-bit spawn failed ({e}), falling back to software decode");
    }

    Command::new("ffmpeg")
        .args([
            "-i", input_str,
            "-f", "rawvideo",
            "-pix_fmt", "rgb48le",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(FfmpegError::NotFound)
}

/// Decode all frames from a 10-bit video file as 16-bit RGB.
///
/// Uses `-pix_fmt rgb48le` for full 10-bit precision preservation.
pub fn decode_frames_16bit(
    input: &Path,
    info: &VideoInfo,
) -> Result<impl Iterator<Item = Result<Frame16, FfmpegError>>, FfmpegError> {
    let child = spawn_decoder_16bit(input, info)?;
    Ok(FrameReader16::new(child, info.width, info.height))
}
```

- [ ] **Step 4: Add u16→u8 downscale helper for proxy path**

The proxy path (inference: RAUNE, depth) stays 8-bit. Add a helper:

```rust
impl Frame16 {
    /// Downscale to 8-bit for proxy inference paths.
    /// Maps [0, 65535] → [0, 255] with rounding.
    pub fn to_8bit(&self) -> Vec<u8> {
        self.pixels.iter().map(|&v| (v >> 8) as u8).collect()
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): add Frame16 and 10-bit decode path (rgb48le)"
```

---

### Task 7: OutputCodec Enum and 10-Bit Encoder

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

- [ ] **Step 1: Define OutputCodec enum**

Add after `InputEncoding`:

```rust
/// Output video codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputCodec {
    /// ProRes 422 HQ — 10-bit 4:2:2, DaVinci Resolve compatible (default for 10-bit)
    ProRes,
    /// HEVC 10-bit (NVENC) — fast, smaller files, 4:2:0
    Hevc10,
    /// H.264 8-bit (NVENC or libx264) — legacy, existing default
    H264,
}

impl OutputCodec {
    /// Whether this codec carries 10-bit precision.
    pub fn is_10bit(&self) -> bool {
        matches!(self, Self::ProRes | Self::Hevc10)
    }

    /// Bytes per pixel for the raw frame input to the encoder.
    /// ProRes/HEVC10: 6 (rgb48le). H264: 3 (rgb24).
    pub fn bytes_per_pixel(&self) -> usize {
        if self.is_10bit() { 6 } else { 3 }
    }
}

impl std::fmt::Display for OutputCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProRes => write!(f, "prores"),
            Self::Hevc10 => write!(f, "hevc10"),
            Self::H264 => write!(f, "h264"),
        }
    }
}

impl std::str::FromStr for OutputCodec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "prores" | "prores422" => Ok(Self::ProRes),
            "hevc10" | "hevc-10" | "hevc_10bit" => Ok(Self::Hevc10),
            "h264" | "h.264" | "avc" => Ok(Self::H264),
            other => Err(format!("unknown codec '{other}'; expected prores, hevc10, or h264")),
        }
    }
}
```

- [ ] **Step 2: Add FrameEncoder::new_10bit**

Add a new constructor to `FrameEncoder`:

```rust
impl FrameEncoder {
    /// Spawn a 10-bit encoder subprocess.
    ///
    /// Accepts raw rgb48le input (6 bytes per pixel).
    /// Output format depends on `codec`:
    /// - ProRes: prores_ks -profile:v 3 (422 HQ), .mov container
    /// - Hevc10: hevc_nvenc -profile:v main10, .mp4 container
    pub fn new_10bit(
        output: &Path,
        width: usize,
        height: usize,
        fps: f64,
        codec: OutputCodec,
        input_for_audio: Option<&Path>,
    ) -> Result<Self, FfmpegError> {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(FfmpegError::EncodeFailed(format!(
                    "output directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        let w_s = width.to_string();
        let h_s = height.to_string();
        let fps_s = format!("{fps:.3}");
        let size_s = format!("{w_s}x{h_s}");
        let out_s = output.to_str().unwrap_or("output.mov");

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb48le",
            "-s", &size_s,
            "-r", &fps_s,
            "-i", "pipe:0",
        ]);

        if let Some(audio_src) = input_for_audio {
            cmd.args(["-i", audio_src.to_str().unwrap_or("")]);
            cmd.args(["-map", "0:v", "-map", "1:a", "-c:a", "copy"]);
        } else {
            cmd.args(["-map", "0:v"]);
        }

        match codec {
            OutputCodec::ProRes => {
                cmd.args([
                    "-c:v", "prores_ks",
                    "-profile:v", "3",     // 422 HQ
                    "-pix_fmt", "yuv422p10le",
                ]);
            }
            OutputCodec::Hevc10 => {
                // Try NVENC first
                static HEVC_NVENC_AVAILABLE: OnceLock<bool> = OnceLock::new();
                let nvenc = *HEVC_NVENC_AVAILABLE.get_or_init(|| {
                    Command::new("ffmpeg")
                        .args(["-hide_banner", "-encoders"])
                        .output()
                        .map(|o| String::from_utf8_lossy(&o.stdout).contains("hevc_nvenc"))
                        .unwrap_or(false)
                });
                if nvenc {
                    cmd.args([
                        "-c:v", "hevc_nvenc",
                        "-profile:v", "main10",
                        "-preset", "p4",
                        "-cq", "18",
                    ]);
                } else {
                    cmd.args([
                        "-c:v", "libx265",
                        "-profile:v", "main10",
                        "-crf", "18",
                        "-preset", "medium",
                        "-pix_fmt", "yuv420p10le",
                    ]);
                }
            }
            OutputCodec::H264 => {
                // Downconvert internally — ffmpeg handles rgb48le → yuv420p
                cmd.args(["-c:v", "libx264", "-crf", "18", "-preset", "fast"]);
            }
        }

        cmd.arg(out_s);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(FfmpegError::NotFound)?;
        let stdin = child.stdin.take().ok_or_else(|| {
            FfmpegError::EncodeFailed("could not open 10-bit encoder stdin".to_string())
        })?;
        let stderr = child.stderr.take();

        if let Ok(Some(status)) = child.try_wait() {
            let stderr_msg = stderr.map(|mut s| {
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            }).unwrap_or_default();
            return Err(FfmpegError::EncodeFailed(format!(
                "10-bit encoder exited immediately (code {:?}): {}",
                status.code(), stderr_msg.trim()
            )));
        }

        Ok(Self {
            child,
            stdin,
            stderr,
            frame_bytes: width * height * 6, // rgb48le = 6 bytes per pixel
        })
    }

    /// Write one RGB48LE frame (u16 data) to the 10-bit encoder.
    pub fn write_frame_16(&mut self, pixels: &[u16]) -> Result<(), FfmpegError> {
        let bytes: Vec<u8> = pixels.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        if bytes.len() != self.frame_bytes {
            return Err(FfmpegError::EncodeFailed(format!(
                "16-bit frame size mismatch: expected {} bytes, got {}",
                self.frame_bytes, bytes.len()
            )));
        }
        self.write_frame(&bytes)
    }
}
```

- [ ] **Step 3: Add tests**

```rust
#[test]
fn output_codec_parse() {
    assert_eq!("prores".parse::<OutputCodec>().unwrap(), OutputCodec::ProRes);
    assert_eq!("hevc10".parse::<OutputCodec>().unwrap(), OutputCodec::Hevc10);
    assert_eq!("h264".parse::<OutputCodec>().unwrap(), OutputCodec::H264);
    assert!("bogus".parse::<OutputCodec>().is_err());
}

#[test]
fn output_codec_10bit_flag() {
    assert!(OutputCodec::ProRes.is_10bit());
    assert!(OutputCodec::Hevc10.is_10bit());
    assert!(!OutputCodec::H264.is_10bit());
}
```

- [ ] **Step 4: Run tests**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-video`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(video): add OutputCodec enum and 10-bit ProRes/HEVC encoder"
```

---

## Phase C: CUDA 10-Bit Kernel

---

### Task 8: CUDA Kernel — combined_lut_kernel_16 (u16 I/O)

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu`

Add a second kernel entry point for u16 I/O. The internal f32 math is identical.
Only the input load (u16→f32, /65535.0) and output store (f32→u16, *65535.0) change.

- [ ] **Step 1: Add the u16 kernel entry point**

Append to `crates/dorea-gpu/src/cuda/kernels/combined_lut.cu`, after the existing kernel:

```cuda
/**
 * 10-bit variant: u16 input/output for full 10-bit precision.
 * Internal math is identical to combined_lut_kernel.
 * Input: rgb48le u16 pixels. Output: rgb48le u16 pixels.
 */
extern "C"
__global__ void combined_lut_kernel_16(
    const unsigned short* __restrict__ pixels_in,
    const float*          __restrict__ depth,
    const unsigned long long* __restrict__ textures_a,
    const float*              __restrict__ zone_boundaries_a,
    const unsigned long long* __restrict__ textures_b,
    const float*              __restrict__ zone_boundaries_b,
    float blend_t,
    unsigned short*      __restrict__ pixels_out,
    int n_pixels,
    int n_zones,
    int grid_size,
    int frame_w,
    int frame_h,
    int depth_w,
    int depth_h,
    const unsigned char* __restrict__ class_mask,
    int mask_w,
    int mask_h,
    float diver_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    // u16 → f32 (10-bit source uses [0, 65535] range)
    float r = pixels_in[idx * 3 + 0] * (1.0f / 65535.0f);
    float g = pixels_in[idx * 3 + 1] * (1.0f / 65535.0f);
    float b = pixels_in[idx * 3 + 2] * (1.0f / 65535.0f);

    // --- Bilinear sample depth (identical to u8 kernel) ---
    float d;
    if (depth_w == frame_w && depth_h == frame_h) {
        d = depth[idx];
    } else {
        int px = idx % frame_w;
        int py = idx / frame_w;
        float sx = (float)px * (float)(depth_w - 1) / (float)(frame_w - 1);
        float sy = (float)py * (float)(depth_h - 1) / (float)(frame_h - 1);
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = min(x0 + 1, depth_w - 1);
        int y1 = min(y0 + 1, depth_h - 1);
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        d = depth[y0 * depth_w + x0] * (1.0f - fx) * (1.0f - fy)
          + depth[y0 * depth_w + x1] * fx * (1.0f - fy)
          + depth[y1 * depth_w + x0] * (1.0f - fx) * fy
          + depth[y1 * depth_w + x1] * fx * fy;
    }

    // --- Depth dithering (identical) ---
    {
        unsigned int hash = (unsigned int)(idx * 2654435761u);
        float noise = ((float)(hash & 0xFFFF) / 65535.0f - 0.5f);
        float avg_zone_width = 1.0f / (float)n_zones;
        d += noise * avg_zone_width * 0.5f;
        d = fminf(fmaxf(d, 0.0f), 1.0f);
    }

    // --- YOLO-seg mask (identical) ---
    if (class_mask != 0) {
        int px = idx % frame_w;
        int py = idx / frame_w;
        float mx = (float)px * (float)(mask_w - 1) / (float)(frame_w - 1);
        float my = (float)py * (float)(mask_h - 1) / (float)(frame_h - 1);
        int mi = min((int)my, mask_h - 1) * mask_w + min((int)mx, mask_w - 1);
        if (class_mask[mi] > 0) {
            d = diver_depth;
        }
    }

    float gs = (float)(grid_size - 1);

    // --- Sample texture set A (identical) ---
    float total_w_a = 0.0f;
    float4 blended_a = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int z = 0; z < n_zones; z++) {
        float z_lo = zone_boundaries_a[z];
        float z_hi = zone_boundaries_a[z + 1];
        float z_width = z_hi - z_lo;
        if (z_width < 1e-6f) continue;
        float z_center = 0.5f * (z_lo + z_hi);
        float dist = fabsf(d - z_center);
        float blend_radius = z_width * 2.0f;
        float w = fmaxf(1.0f - dist / blend_radius, 0.0f);
        w = w * w;
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

    // --- Early out + set B (identical) ---
    float r_out, g_out, b_out;
    if (blend_t < 1e-4f) {
        r_out = r_a; g_out = g_a; b_out = b_a;
    } else {
        float total_w_b = 0.0f;
        float4 blended_b = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int z = 0; z < n_zones; z++) {
            float z_lo = zone_boundaries_b[z];
            float z_hi = zone_boundaries_b[z + 1];
            float z_width = z_hi - z_lo;
            if (z_width < 1e-6f) continue;
            float z_center = 0.5f * (z_lo + z_hi);
            float dist = fabsf(d - z_center);
            float blend_radius = z_width * 2.0f;
            float w = fmaxf(1.0f - dist / blend_radius, 0.0f);
            w = w * w;
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

        float inv_t = 1.0f - blend_t;
        r_out = r_a * inv_t + r_b * blend_t;
        g_out = g_a * inv_t + g_b * blend_t;
        b_out = b_a * inv_t + b_b * blend_t;
    }

    // --- Depth-dependent strength (identical) ---
    float depth_strength = fminf(d / 0.15f, 1.0f);

    float max_shift = 0.35f;
    r_out = r + fminf(fmaxf(r_out - r, -max_shift), max_shift) * depth_strength;
    g_out = g + fminf(fmaxf(g_out - g, -max_shift), max_shift) * depth_strength;
    b_out = b + fminf(fmaxf(b_out - b, -max_shift), max_shift) * depth_strength;

    // --- Output dithering — scaled for 16-bit (±1/65535 instead of ±1/255) ---
    {
        unsigned int h1 = (unsigned int)(idx * 2654435761u);
        unsigned int h2 = (unsigned int)(idx * 340573321u + 1013904223u);
        float u1 = (float)(h1 & 0xFFFF) / 65536.0f;
        float u2 = (float)(h2 & 0xFFFF) / 65536.0f;
        float tri = (u1 + u2 - 1.0f);
        float dither = tri / 65535.0f;
        r_out += dither;
        unsigned int h3 = h1 ^ (h2 << 13);
        unsigned int h4 = h2 ^ (h1 >> 7);
        float u3 = (float)(h3 & 0xFFFF) / 65536.0f;
        float u4 = (float)(h4 & 0xFFFF) / 65536.0f;
        g_out += (u3 + u4 - 1.0f) / 65535.0f;
        unsigned int h5 = h1 ^ (h2 >> 5) ^ 0xDEADBEEF;
        unsigned int h6 = h2 ^ (h1 << 11) ^ 0xCAFEBABE;
        float u5 = (float)(h5 & 0xFFFF) / 65536.0f;
        float u6 = (float)(h6 & 0xFFFF) / 65536.0f;
        b_out += (u5 + u6 - 1.0f) / 65535.0f;
    }

    // f32 → u16 output
    pixels_out[idx * 3 + 0] = (unsigned short)(__float2uint_rn(fminf(fmaxf(r_out, 0.0f), 1.0f) * 65535.0f));
    pixels_out[idx * 3 + 1] = (unsigned short)(__float2uint_rn(fminf(fmaxf(g_out, 0.0f), 1.0f) * 65535.0f));
    pixels_out[idx * 3 + 2] = (unsigned short)(__float2uint_rn(fminf(fmaxf(b_out, 0.0f), 1.0f) * 65535.0f));
}
```

- [ ] **Step 2: Register the new kernel in build.rs and PTX loader**

In `crates/dorea-gpu/src/cuda/mod.rs`, update the PTX loading to register the new kernel:

```rust
device.load_ptx(
    Ptx::from_src(COMBINED_LUT_PTX),
    "combined_lut",
    &["combined_lut_kernel", "combined_lut_kernel_16"],
).map_err(|e| GpuError::ModuleLoad(format!("load combined_lut PTX: {e}")))?;
```

Do the same in `AdaptiveGrader::new()`.

- [ ] **Step 3: Build to verify CUDA compilation**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu --features cuda 2>&1 | tail -20`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/cuda/kernels/combined_lut.cu crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat(gpu): add combined_lut_kernel_16 for 10-bit u16 I/O"
```

---

### Task 9: CudaGrader and AdaptiveGrader — u16 Methods

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs`
- Modify: `crates/dorea-gpu/src/lib.rs`

- [ ] **Step 1: Add FrameBuffers16 struct**

Add alongside existing `FrameBuffers`:

```rust
#[cfg(feature = "cuda")]
struct FrameBuffers16 {
    width: usize,
    height: usize,
    d_pixels_in: CudaSlice<u16>,   // n * 3
    d_depth: CudaSlice<f32>,        // n
    d_pixels_out: CudaSlice<u16>,   // n * 3
}

#[cfg(feature = "cuda")]
fn alloc_frame_buffers_16(dev: &Arc<CudaDevice>, width: usize, height: usize) -> Result<FrameBuffers16, GpuError> {
    let n = width.checked_mul(height).ok_or_else(|| {
        GpuError::InvalidInput("frame dimensions overflow usize".into())
    })?;
    Ok(FrameBuffers16 {
        width,
        height,
        d_pixels_in: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
        d_depth: dev.alloc_zeros(n).map_err(map_cudarc_error)?,
        d_pixels_out: dev.alloc_zeros(n * 3).map_err(map_cudarc_error)?,
    })
}
```

- [ ] **Step 2: Add frame_bufs_16 field to AdaptiveGrader**

Add to the `AdaptiveGrader` struct:

```rust
frame_bufs_16: RefCell<Option<FrameBuffers16>>,
```

Initialize it as `RefCell::new(None)` in `AdaptiveGrader::new()`.

- [ ] **Step 3: Add grade_frame_blended_16 to AdaptiveGrader**

Add a u16 variant of `grade_frame_blended`:

```rust
/// Grade one frame at 10-bit precision (u16 I/O).
///
/// Same semantics as `grade_frame_blended` but accepts and returns u16 pixels.
pub fn grade_frame_blended_16(
    &self,
    pixels: &[u16],
    depth: &[f32],
    width: usize,
    height: usize,
    depth_w: usize,
    depth_h: usize,
    blend_t: f32,
    class_mask: Option<&[u8]>,
    mask_w: usize,
    mask_h: usize,
    diver_depth: f32,
) -> Result<Vec<u16>, GpuError> {
    let n = width.checked_mul(height).ok_or_else(|| {
        GpuError::InvalidInput("frame dimensions overflow usize".into())
    })?;
    if pixels.len() != n * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels len {} != {}*3", pixels.len(), n
        )));
    }
    let dn = depth_w.checked_mul(depth_h).ok_or_else(|| {
        GpuError::InvalidInput("depth dimensions overflow usize".into())
    })?;
    if depth.len() != dn {
        return Err(GpuError::InvalidInput(format!(
            "depth len {} != depth_w*depth_h {}", depth.len(), dn
        )));
    }
    let dev = &self.device;

    // Allocate/reuse 16-bit frame buffers
    {
        let mut slot = self.frame_bufs_16.borrow_mut();
        let needs = slot.as_ref().map_or(true, |b| b.width != width || b.height != height);
        if needs {
            *slot = Some(alloc_frame_buffers_16(dev, width, height)?);
        }
        let bufs = slot.as_mut().expect("frame_bufs_16 allocated above");
        dev.htod_sync_copy_into(pixels, &mut bufs.d_pixels_in).map_err(map_cudarc_error)?;
    }

    let d_depth = dev.htod_sync_copy(depth).map_err(map_cudarc_error)?;

    let active_idx = self.adaptive_lut.active_index();
    let (d_tex_active, d_bounds_active, d_tex_inactive, d_bounds_inactive) = if active_idx == 0 {
        (&self.d_textures_a, &self.d_bounds_a, &self.d_textures_b, &self.d_bounds_b)
    } else {
        (&self.d_textures_b, &self.d_bounds_b, &self.d_textures_a, &self.d_bounds_a)
    };

    {
        let slot = self.frame_bufs_16.borrow();
        let bufs = slot.as_ref().expect("frame_bufs_16 allocated above");
        let bounds_a = d_bounds_active.borrow();
        let bounds_b = d_bounds_inactive.borrow();

        let func = dev.get_func("combined_lut", "combined_lut_kernel_16")
            .ok_or_else(|| GpuError::ModuleLoad("combined_lut_kernel_16 not found".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (div_ceil(n as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let nz_i32 = self.adaptive_lut.runtime_n_zones as i32;
        let gs_i32 = self.adaptive_lut.grid_size as i32;
        let fw_i32 = width as i32;
        let fh_i32 = height as i32;
        let dw_i32 = depth_w as i32;
        let dh_i32 = depth_h as i32;
        let mw_i32 = mask_w as i32;
        let mh_i32 = mask_h as i32;
        use cudarc::driver::{DeviceRepr, DevicePtr};

        let d_mask: Option<CudaSlice<u8>> = class_mask.map(|m| {
            dev.htod_sync_copy(m).expect("mask upload failed")
        });
        let mask_ptr: u64 = match &d_mask {
            Some(s) => *s.device_ptr() as u64,
            None => 0u64,
        };

        let mut args: [*mut std::ffi::c_void; 19] = [
            (&bufs.d_pixels_in).as_kernel_param(),
            (&d_depth).as_kernel_param(),
            d_tex_active.as_kernel_param(),
            (&*bounds_a).as_kernel_param(),
            d_tex_inactive.as_kernel_param(),
            (&*bounds_b).as_kernel_param(),
            blend_t.as_kernel_param(),
            (&bufs.d_pixels_out).as_kernel_param(),
            n_i32.as_kernel_param(),
            nz_i32.as_kernel_param(),
            gs_i32.as_kernel_param(),
            fw_i32.as_kernel_param(),
            fh_i32.as_kernel_param(),
            dw_i32.as_kernel_param(),
            dh_i32.as_kernel_param(),
            mask_ptr.as_kernel_param(),
            mw_i32.as_kernel_param(),
            mh_i32.as_kernel_param(),
            diver_depth.as_kernel_param(),
        ];
        unsafe {
            func.launch(cfg, &mut args[..])
        }.map_err(map_cudarc_error)?;
    }

    let slot = self.frame_bufs_16.borrow();
    let bufs = slot.as_ref().expect("frame_bufs_16 allocated above");
    let result = dev.dtoh_sync_copy(&bufs.d_pixels_out).map_err(map_cudarc_error)?;
    Ok(result)
}
```

- [ ] **Step 4: Add wrapper in lib.rs**

Add to `crates/dorea-gpu/src/lib.rs`:

```rust
/// Grade a single 10-bit frame using an existing `AdaptiveGrader` (u16 I/O).
#[cfg(feature = "cuda")]
pub fn grade_frame_with_adaptive_grader_16(
    grader: &cuda::AdaptiveGrader,
    pixels: &[u16],
    depth: &[f32],
    width: usize,
    height: usize,
    depth_w: usize,
    depth_h: usize,
    blend_t: f32,
) -> Result<Vec<u16>, GpuError> {
    if pixels.len() != width * height * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels length {} != width*height*3 {}",
            pixels.len(), width * height * 3
        )));
    }
    grader.grade_frame_blended_16(pixels, depth, width, height, depth_w, depth_h, blend_t, None, 0, 0, 0.5)
}
```

- [ ] **Step 5: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu --features cuda 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(gpu): add u16 grade methods for 10-bit pipeline"
```

---

## Phase D: Pipeline Integration

---

### Task 10: CLI Flags — --input-encoding, --output-codec

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`
- Modify: `crates/dorea-cli/src/config.rs`
- Modify: `crates/dorea-cli/src/pipeline/mod.rs`

- [ ] **Step 1: Add config fields**

In `crates/dorea-cli/src/config.rs`, add to `GradeDefaults`:

```rust
/// Input encoding override (default: auto-detect from container/codec)
pub input_encoding: Option<String>,
/// Output codec (default: "h264" for 8-bit, "prores" for 10-bit)
pub output_codec: Option<String>,
```

- [ ] **Step 2: Add CLI args**

In `crates/dorea-cli/src/grade.rs`, add to `GradeArgs`:

```rust
/// Input encoding: dlog-m, ilog, srgb (default: auto-detect from container)
#[arg(long)]
pub input_encoding: Option<String>,

/// Output codec: prores, hevc10, h264 (default: auto for source bit depth)
#[arg(long)]
pub output_codec: Option<String>,
```

- [ ] **Step 3: Add fields to PipelineConfig**

In `crates/dorea-cli/src/pipeline/mod.rs`, add to `PipelineConfig`:

```rust
pub input_encoding: dorea_video::ffmpeg::InputEncoding,
pub output_codec: dorea_video::ffmpeg::OutputCodec,
```

- [ ] **Step 4: Wire up in grade.rs run()**

In the `run()` function in `grade.rs`, after existing config resolution:

```rust
use dorea_video::ffmpeg::{InputEncoding, OutputCodec};

// After probing video info:
let input_encoding = args.input_encoding.as_deref()
    .or(cfg.grade.input_encoding.as_deref())
    .map(|s| s.parse::<InputEncoding>())
    .transpose()
    .context("invalid --input-encoding")?
    .unwrap_or_else(|| InputEncoding::auto_detect(&info, &args.input));

let output_codec = args.output_codec.as_deref()
    .or(cfg.grade.output_codec.as_deref())
    .map(|s| s.parse::<OutputCodec>())
    .transpose()
    .context("invalid --output-codec")?
    .unwrap_or_else(|| {
        if input_encoding.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 }
    });

log::info!(
    "Encoding: input={input_encoding}, output={output_codec}, 10-bit={}",
    output_codec.is_10bit()
);
```

Update the output filename default to use `.mov` for ProRes:

```rust
let output = args.output.clone().unwrap_or_else(|| {
    let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
    let ext = if output_codec == OutputCodec::ProRes { "mov" } else { "mp4" };
    args.input.with_file_name(format!("{stem}_graded.{ext}"))
});
```

Update the encoder creation to use the 10-bit path when appropriate:

```rust
let encoder = if output_codec.is_10bit() {
    FrameEncoder::new_10bit(&output, info.width, info.height, info.fps, output_codec, audio_src)
        .context("failed to spawn 10-bit ffmpeg encoder")?
} else {
    FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
        .context("failed to spawn ffmpeg encoder")?
};
```

Add `input_encoding` and `output_codec` to the `PipelineConfig` struct construction.

- [ ] **Step 5: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/config.rs crates/dorea-cli/src/pipeline/mod.rs
git commit -m "feat(cli): add --input-encoding and --output-codec flags"
```

---

### Task 11: Grading Stage — 10-Bit Frame Path

**Files:**
- Modify: `crates/dorea-cli/src/pipeline/grading.rs`

This is the core integration: the 3-thread grading pipeline gains a 10-bit path.
When `output_codec.is_10bit()`, the decoder produces `Frame16`, the GPU grades with
`grade_frame_blended_16`, and the encoder receives u16 data.

- [ ] **Step 1: Update run_grading_stage signature**

Add `PipelineConfig` already has `input_encoding` and `output_codec`. The grading
stage uses `output_codec.is_10bit()` to decide which path to run.

- [ ] **Step 2: Add 10-bit decode → grade → encode path**

In `run_grading_stage`, after the existing code that sets up channels and threads,
add a branching point. The key changes:

**Decoder thread (10-bit path):** Use `decode_frames_16bit()` instead of `decode_frames()`.
Send `Frame16` through the channel instead of `Frame`.

**GPU thread (10-bit path):** Call `grader.grade_frame_blended_16()` with `&frame.pixels`
(u16). Produce `Vec<u16>` output.

**Encoder thread (10-bit path):** Call `encoder.write_frame_16()` with the `&graded_u16`.

The channel types need to support both 8-bit and 16-bit frames. Use an enum:

```rust
enum GradedFrame {
    Rgb8(Vec<u8>),
    Rgb16(Vec<u16>),
}
```

Or simpler: since the bit depth is known at pipeline start, use two separate code paths
(the 3-thread structure is duplicated but with different types). This avoids runtime
enum overhead on every frame.

The cleanest approach: extract the common pipeline logic into a helper, parameterized
by decode/grade/encode closures. But for clarity and to avoid over-engineering,
implement as two branches in `run_grading_stage`:

```rust
pub fn run_grading_stage(
    cfg: &PipelineConfig,
    info: &VideoInfo,
    encoder: FrameEncoder,
    cal_out: CalibrationStageOutput,
) -> Result<u64> {
    if cfg.output_codec.is_10bit() {
        run_grading_stage_16bit(cfg, info, encoder, cal_out)
    } else {
        run_grading_stage_8bit(cfg, info, encoder, cal_out)
    }
}
```

Move the existing pipeline logic into `run_grading_stage_8bit()` (rename, no changes).

Create `run_grading_stage_16bit()` with the same structure but:
1. Uses `decode_frames_16bit()` for the decoder
2. Uses `grade_frame_blended_16()` for grading
3. Uses `encoder.write_frame_16()` for encoding
4. Channels carry `Frame16` and `Vec<u16>` respectively
5. Proxy pixels for depth/mask lookup: call `frame.to_8bit()` to get u8 for proxy path

**Critical detail:** The feature/calibration stages still use 8-bit proxy frames
(RAUNE, depth, YOLO-seg all run on 8-bit). Only the grading render path is 10-bit.

- [ ] **Step 3: Build**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-cli/src/pipeline/grading.rs
git commit -m "feat(grading): add 10-bit render path (u16 decode → grade → encode)"
```

---

### Task 12: dorea probe Subcommand

**Files:**
- Create: `crates/dorea-cli/src/probe.rs`
- Modify: `crates/dorea-cli/src/main.rs`
- Modify: `crates/dorea-cli/src/lib.rs`

- [ ] **Step 1: Create probe.rs**

Create `crates/dorea-cli/src/probe.rs`:

```rust
//! `dorea probe` — detect container, codec, bit depth, and suggest flags.

use std::path::PathBuf;
use clap::Args;
use anyhow::{Context, Result};
use dorea_video::ffmpeg::{self, InputEncoding};

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Input video file
    #[arg(long)]
    pub input: PathBuf,
}

pub fn run(args: ProbeArgs) -> Result<()> {
    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed — is ffmpeg installed?")?;

    let encoding = InputEncoding::auto_detect(&info, &args.input);

    println!("File:       {}", args.input.display());
    println!("Resolution: {}x{}", info.width, info.height);
    println!("FPS:        {:.3}", info.fps);
    println!("Duration:   {:.1}s ({} frames)", info.duration_secs, info.frame_count);
    println!("Codec:      {}", info.codec_name);
    println!("Pixel fmt:  {}", info.pix_fmt);
    println!("Bit depth:  {}-bit", info.bits_per_component);
    println!("Audio:      {}", if info.has_audio { "yes" } else { "no" });
    println!();
    println!("Detected encoding: {encoding}");

    if encoding.is_10bit() {
        println!();
        println!("Suggested command:");
        println!(
            "  dorea grade --input {} --input-encoding {} --output-codec prores",
            args.input.display(), encoding
        );
    } else {
        println!();
        println!("Suggested command:");
        println!(
            "  dorea grade --input {}",
            args.input.display()
        );
    }

    Ok(())
}
```

- [ ] **Step 2: Register in main.rs**

In `crates/dorea-cli/src/main.rs`, add `Probe` variant:

```rust
#[derive(Subcommand)]
enum Command {
    Calibrate(dorea_cli::calibrate::CalibrateArgs),
    Grade(dorea_cli::grade::GradeArgs),
    Preview(dorea_cli::preview::PreviewArgs),
    /// Detect container, codec, bit depth, and suggest flags
    Probe(dorea_cli::probe::ProbeArgs),
}
```

Add the match arm:

```rust
Command::Probe(args) => dorea_cli::probe::run(args),
```

Note: `Probe` doesn't need `&config` — it only probes, doesn't grade.

- [ ] **Step 3: Register module in lib.rs**

In `crates/dorea-cli/src/lib.rs`, add `pub mod probe;`

- [ ] **Step 4: Build and test**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/probe.rs crates/dorea-cli/src/main.rs crates/dorea-cli/src/lib.rs
git commit -m "feat(cli): add dorea probe subcommand"
```

---

## Phase E: Integration and Backward Compatibility

---

### Task 13: Integration Test — 8-Bit Backward Compatibility

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (test module)

- [ ] **Step 1: Verify auto-detection defaults to 8-bit for existing clips**

Add test to `crates/dorea-cli/src/grade.rs`:

```rust
#[test]
fn default_encoding_for_8bit_h264() {
    use dorea_video::ffmpeg::{InputEncoding, OutputCodec, VideoInfo};
    let info = VideoInfo {
        width: 1920, height: 1080, fps: 30.0, duration_secs: 10.0,
        frame_count: 300, has_audio: true,
        codec_name: "h264".to_string(),
        pix_fmt: "yuv420p".to_string(),
        bits_per_component: 8,
    };
    let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
    assert_eq!(enc, InputEncoding::Srgb);
    assert!(!enc.is_10bit());
    // Default output codec for 8-bit should be H264
    let codec = if enc.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 };
    assert_eq!(codec, OutputCodec::H264);
}
```

- [ ] **Step 2: Run full test suite**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo test 2>&1 | tail -30`
Expected: All existing tests still pass. No regressions.

- [ ] **Step 3: Run clippy**

Run: `cd /workspaces/dorea-workspace/repos/dorea && cargo clippy -- -D warnings 2>&1 | tail -30`
Expected: No warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "test: backward compat — 8-bit auto-detection and defaults"
```

---

### Task 14: Update dorea.toml Defaults

**Files:**
- Modify: `dorea.toml` (project root, if it exists — may need to recreate since git shows it as deleted)

- [ ] **Step 1: Add 10-bit config defaults**

The `dorea.toml` was deleted in working tree. If restoring it or creating fresh, add:

```toml
[grade]
# ... existing fields ...
# input_encoding = "auto"  # auto-detect from container/codec
# output_codec = "auto"    # "prores" for 10-bit, "h264" for 8-bit
```

These are commented out since auto-detection is the default. Document them for users.

- [ ] **Step 2: Commit**

```bash
git add dorea.toml
git commit -m "chore: document 10-bit config options in dorea.toml"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - ✅ TransferFunction trait (Tasks 1-3)
   - ✅ 10-bit decode (Tasks 4-6)
   - ✅ 10-bit encode (Task 7)
   - ✅ CUDA u16 kernel (Tasks 8-9)
   - ✅ CLI flags (Task 10)
   - ✅ Pipeline integration (Task 11)
   - ✅ dorea probe (Task 12)
   - ✅ Backward compat (Task 13)
   - ⏳ Tiny planet projection — DEFERRED. Not needed for MVP. Can be added later as a separate issue.
   - ⏳ LUT grid bump 97→129 — DEFERRED. Current 97³ with output dithering works well. Bump if testing reveals banding at 10-bit.

2. **Placeholder scan:** No TBD/TODO/placeholders found.

3. **Type consistency:**
   - `InputEncoding` — defined in Task 5, used in Tasks 10, 11, 12
   - `OutputCodec` — defined in Task 7, used in Tasks 10, 11, 12
   - `Frame16` — defined in Task 6, used in Task 11
   - `grade_frame_blended_16` — defined in Task 9, used in Task 11
   - `write_frame_16` — defined in Task 7, used in Task 11
   - `VideoInfo` new fields — defined in Task 4, used in Tasks 5, 12
