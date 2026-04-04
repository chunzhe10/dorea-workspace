# Unified Keyframe Sampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse two independent keyframe-selection paths (uniform calibration + MSE-adaptive grading) into one: a single proxy decode pass with abstracted change detection feeds a fused RAUNE+depth inference batch whose outputs populate both the calibration store and the grading depth cache.

**Architecture:** `ChangeDetector` trait (MSE impl today, optical flow later) drives pass 1. After pass 1, a single `InferenceServer` runs `run_raune_depth_batch`; results split into `PagedCalibrationStore` (for 3-pass inline calibration) and `keyframe_depths` HashMap (for pass 2 grading). `auto_calibrate()` is deleted; `--keyframe-interval` is removed.

**Tech Stack:** Rust (`dorea-cli`, `dorea-video`), existing JSON-lines IPC, `memmap2`, `dorea-lut`, `dorea-hsl`.

---

## Prerequisites

**This plan requires `docs/plans/2026-04-03-inference-gpu-saturation.md` to be fully implemented first.**

Specifically, the following must exist before starting Task 3:
- `RauneDepthBatchItem` struct in `crates/dorea-video/src/inference_subprocess.rs`
- `InferenceServer::run_raune_depth_batch` method in `inference_subprocess.rs`
- `FUSED_BATCH_SIZE: usize = 8` constant in `grade.rs`
- Updated import: `use dorea_video::inference::{DepthBatchItem, RauneDepthBatchItem, InferenceConfig, InferenceServer};`

Verify:
```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-video 2>&1 | tail -3
grep -n "RauneDepthBatchItem\|run_raune_depth_batch" crates/dorea-video/src/inference_subprocess.rs | head -5
```

---

## File Map

| File | Change |
|---|---|
| `crates/dorea-cli/src/change_detect.rs` | **New** — `ChangeDetector` trait, `MseDetector` impl, `frame_mse` helper |
| `crates/dorea-cli/src/lib.rs` | Add `pub mod change_detect;` |
| `crates/dorea-cli/src/grade.rs` | Major refactor — see tasks below |

---

## Task 1: Add `change_detect.rs` — `ChangeDetector` trait + `MseDetector`

**Files:**
- Create: `crates/dorea-cli/src/change_detect.rs`

- [ ] **Step 1.1: Write the failing tests**

Create `crates/dorea-cli/src/change_detect.rs` with just the test module:

```rust
/// Abstracted change detection between video frames.
///
/// Implementations hold a reference frame and compute a scalar change score
/// against it. The caller decides when to update the reference (i.e. on keyframe
/// detection) — this is intentional: we compare each frame to the *last keyframe*,
/// not to the immediately preceding frame.

pub trait ChangeDetector: Send {
    /// Change score from `pixels` vs the stored reference frame.
    /// Returns `f32::MAX` when no reference has been set yet.
    fn score(&self, pixels: &[u8]) -> f32;

    /// Accept `pixels` as the new reference for future `score()` calls.
    fn set_reference(&mut self, pixels: &[u8]);

    /// Clear the reference — next `score()` will return `f32::MAX`.
    fn reset(&mut self);
}

/// Mean-squared-error change detector.
pub struct MseDetector {
    reference: Option<Vec<u8>>,
}

impl Default for MseDetector {
    fn default() -> Self {
        Self { reference: None }
    }
}

/// Normalized MSE between two equal-length u8 slices.
/// Returns value in [0, 1] where 0 = identical.
pub fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av as f64 - bv as f64;
            d * d
        })
        .sum();
    (sum_sq / (n * 255.0 * 255.0)) as f32
}

impl ChangeDetector for MseDetector {
    fn score(&self, pixels: &[u8]) -> f32 {
        match self.reference.as_ref() {
            Some(r) => frame_mse(pixels, r),
            None => f32::MAX,
        }
    }

    fn set_reference(&mut self, pixels: &[u8]) {
        self.reference = Some(pixels.to_vec());
    }

    fn reset(&mut self) {
        self.reference = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_mse_identical_is_zero() {
        let a: Vec<u8> = vec![100, 150, 200, 50, 75, 125];
        assert_eq!(frame_mse(&a, &a), 0.0);
    }

    #[test]
    fn frame_mse_empty_is_zero() {
        assert_eq!(frame_mse(&[], &[]), 0.0);
    }

    #[test]
    fn frame_mse_opposite_is_one() {
        let a: Vec<u8> = vec![0, 0, 0];
        let b: Vec<u8> = vec![255, 255, 255];
        let mse = frame_mse(&a, &b);
        assert!((mse - 1.0).abs() < 1e-5, "expected ~1.0, got {mse}");
    }

    #[test]
    fn frame_mse_known_value() {
        let a: Vec<u8> = vec![100, 100, 100];
        let b: Vec<u8> = vec![101, 101, 101];
        let mse = frame_mse(&a, &b);
        let expected = 1.0 / (255.0 * 255.0);
        assert!((mse as f64 - expected).abs() < 1e-8, "expected {expected}, got {mse}");
    }

    #[test]
    fn mse_detector_no_reference_returns_max() {
        let det = MseDetector::default();
        let pixels = vec![128u8; 100];
        assert_eq!(det.score(&pixels), f32::MAX);
    }

    #[test]
    fn mse_detector_same_as_reference_returns_zero() {
        let mut det = MseDetector::default();
        let pixels = vec![200u8; 300];
        det.set_reference(&pixels);
        assert_eq!(det.score(&pixels), 0.0);
    }

    #[test]
    fn mse_detector_set_reference_compares_to_set_frame_not_latest() {
        let mut det = MseDetector::default();
        let ref_frame = vec![0u8; 3];
        let other_frame = vec![255u8; 3];
        det.set_reference(&ref_frame);

        // score returns MSE vs ref_frame — not updated by calling score itself
        let s1 = det.score(&other_frame);
        let s2 = det.score(&other_frame);
        assert_eq!(s1, s2, "score should be deterministic without set_reference");
        assert!((s1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mse_detector_reset_clears_reference() {
        let mut det = MseDetector::default();
        det.set_reference(&vec![0u8; 3]);
        det.reset();
        assert_eq!(det.score(&vec![0u8; 3]), f32::MAX);
    }
}
```

- [ ] **Step 1.2: Run tests — expect failure (module not wired yet)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -10
```
Expected: compile error — `crates/dorea-cli/src/change_detect.rs` exists but is not declared as a module.

- [ ] **Step 1.3: Wire module in `lib.rs`**

Edit `crates/dorea-cli/src/lib.rs` — add one line:

```rust
pub mod calibrate;
pub mod change_detect;   // ← add this
pub mod grade;
pub mod preview;
```

- [ ] **Step 1.4: Run tests — expect pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli change_detect 2>&1 | tail -15
```
Expected: 8 tests pass.

- [ ] **Step 1.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/change_detect.rs crates/dorea-cli/src/lib.rs
git commit -m "feat(grade): add ChangeDetector trait + MseDetector in change_detect.rs"
```

---

## Task 2: Replace inline MSE in pass 1 with `MseDetector`

The pass 1 loop in `run()` currently uses `last_proxy: Option<Vec<u8>>` and calls `frame_mse()` directly. This task rewires it to use `MseDetector`, keeping identical runtime behaviour.

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

- [ ] **Step 2.1: Add import at the top of `grade.rs`**

After line 11 (`use dorea_gpu::{grade_frame, GradeParams};`), add:

```rust
use crate::change_detect::{ChangeDetector, MseDetector};
```

- [ ] **Step 2.2: Replace the pass 1 block**

Find the block starting at line 215:
```rust
    let interp_enabled = !args.no_depth_interp;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    // -----------------------------------------------------------------------
    // Pass 1: proxy decode + MSE keyframe detection
    // -----------------------------------------------------------------------
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);
    let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)
        .context("failed to spawn ffmpeg proxy decoder")?;

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut last_proxy: Option<Vec<u8>> = None;
    let mut frames_since_kf = 0usize;

    for frame_result in proxy_frames {
        let frame = frame_result.context("proxy frame decode error")?;
        let mse = last_proxy.as_ref().map(|lp| frame_mse(&frame.pixels, lp));
        let scene_cut = mse.map_or(false, |m| m > scene_cut_threshold);
        let is_keyframe = !interp_enabled
            || last_proxy.is_none()
            || scene_cut
            || frames_since_kf >= args.depth_max_interval
            || mse.map_or(false, |m| m > args.depth_skip_threshold);

        if is_keyframe {
            if scene_cut {
                log::info!(
                    "Scene cut at frame {} (MSE={:.6})",
                    frame.index,
                    mse.unwrap_or(0.0),
                );
            }
            keyframes.push(KeyframeEntry {
                frame_index: frame.index,
                proxy_pixels: frame.pixels.clone(),
                scene_cut_before: scene_cut,
            });
            last_proxy = Some(frame.pixels);
            frames_since_kf = 0;
        } else {
            frames_since_kf += 1;
        }
    }
    log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());
```

Replace with:

```rust
    let interp_enabled = !args.no_depth_interp;

    // -----------------------------------------------------------------------
    // Pass 1: proxy decode + change detection → keyframe list
    // -----------------------------------------------------------------------
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);
    let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)
        .context("failed to spawn ffmpeg proxy decoder")?;

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());
    let mut frames_since_kf = 0usize;

    for frame_result in proxy_frames {
        let frame = frame_result.context("proxy frame decode error")?;
        let change = detector.score(&frame.pixels);
        let scene_cut = change < f32::MAX && change > args.depth_skip_threshold * 10.0;
        let is_keyframe = !interp_enabled
            || keyframes.is_empty()
            || scene_cut
            || frames_since_kf >= args.depth_max_interval
            || change > args.depth_skip_threshold;

        if is_keyframe {
            if scene_cut {
                log::info!(
                    "Scene cut at frame {} (change={:.6})",
                    frame.index,
                    change,
                );
                detector.reset();
            }
            keyframes.push(KeyframeEntry {
                frame_index: frame.index,
                proxy_pixels: frame.pixels.clone(),
                scene_cut_before: scene_cut,
            });
            detector.set_reference(&frame.pixels);
            frames_since_kf = 0;
        } else {
            frames_since_kf += 1;
        }
    }
    log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());
```

- [ ] **Step 2.3: Remove the now-dead `frame_mse` fn and its tests from `grade.rs`**

Delete the `frame_mse` function (lines 90–105):
```rust
/// Compute normalized MSE between two same-length u8 slices.
/// Returns value in [0, 1] where 0 = identical.
fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    ...
}
```

Delete the `frame_mse_*` tests in the `#[cfg(test)]` block at the bottom of `grade.rs` (they've been migrated to `change_detect.rs`):
```rust
    #[test]
    fn frame_mse_identical_is_zero() { ... }

    #[test]
    fn frame_mse_empty_is_zero() { ... }

    #[test]
    fn frame_mse_opposite_is_one() { ... }

    #[test]
    fn frame_mse_known_value() { ... }
```

- [ ] **Step 2.4: Cargo check**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-cli 2>&1 | tail -10
```
Expected: `Finished` with no errors. Fix any "unused import" warnings if `frame_mse` import path needs adjustment.

- [ ] **Step 2.5: Cargo test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -15
```
Expected: all existing tests pass.

- [ ] **Step 2.6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "refactor(grade): wire ChangeDetector in pass 1; move frame_mse to change_detect"
```

---

## Task 3: Restructure `run()` — unified inference block

This is the core of the plan. It rewires the control flow of `run()` so that:
1. Pass 1 runs first (already done in Task 2)
2. After pass 1: a branch on `args.calibration` drives either the fused RAUNE+depth path or the legacy depth-only path
3. The calibration 3-pass streaming runs inline (replacing the `auto_calibrate` call)
4. The second `InferenceServer::spawn` for depth-only grading is removed when auto-calibrating

**Requires:** GPU saturation plan fully implemented (see Prerequisites).

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

- [ ] **Step 3.1: Move pass 1 before calibration derivation**

Currently in `run()`, the order is:
1. Setup (output path, probe, params, encoder) — lines ~158–183
2. Calibration load/derive — lines ~185–192
3. Inference server (depth-only) spawn — lines ~194–200
4. CUDA grader init — lines ~202–213
5. Pass 1 (proxy decode + change detection) — lines ~215–258
6. Batch depth inference — lines ~260–309
7. `inf_server.shutdown()` — line 309
8. `kf_index_list` — lines 312–314
9. Pass 2 — lines 316–393

The new order is:
1. Setup (unchanged)
2. Pass 1 (unchanged from Task 2)
3. Calibration + depth cache block (new unified branch)
4. CUDA grader init (unchanged, just moved down)
5. `kf_index_list` (unchanged)
6. Pass 2 (unchanged)

Find and **move** the entire pass 1 block (from the `let interp_enabled =` line through `log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());`) to **after** the encoder initialisation (`let mut encoder = ...`) and **before** the calibration block. The simplest way: cut lines 215–258, paste before line 185.

After moving, `run()` structure becomes:

```
... setup → encoder init ...
let interp_enabled = ...;
// Pass 1 (moved here)
let (proxy_w, proxy_h) = ...;
// ...
log::info!("Pass 1 complete: ...");

// Calibration — now references keyframes from pass 1
let calibration = if let Some(cal_path) = ...
```

- [ ] **Step 3.2: Replace the calibration+depth block with the unified branch**

After pass 1, `keyframes` is populated. Replace:

```rust
    // Load or derive calibration
    let calibration = if let Some(cal_path) = &args.calibration {
        log::info!("Loading calibration from {}", cal_path.display());
        Calibration::load(cal_path).context("failed to load .dorea-cal file")?
    } else {
        log::info!("No calibration provided — auto-calibrating from keyframes");
        auto_calibrate(&args, &info)?
    };

    // Spawn inference server (depth-only; RAUNE is not needed for grading).
    let inf_cfg = InferenceConfig {
        skip_raune: true,
        ..build_inference_config(&args)
    };
    let mut inf_server = InferenceServer::spawn(&inf_cfg)
        .context("failed to spawn inference server — check --python and --depth-model")?;
```

AND the entire depth batch block that follows (lines ~260–309 in the current layout after moving pass 1 above it):

```rust
    // -----------------------------------------------------------------------
    // Batch depth inference
    // -----------------------------------------------------------------------
    let batch_items: Vec<DepthBatchItem> = keyframes.iter().map(|kf| { ... }).collect();

    let mut keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();
    for (chunk_kfs, chunk_items) in keyframes
        .chunks(DEPTH_BATCH_SIZE)
        .zip(batch_items.chunks(DEPTH_BATCH_SIZE))
    {
        ... // run_depth_batch, pad, insert into keyframe_depths
    }
    log::info!("Batch depth inference complete ({} keyframes)", keyframes.len());

    // Shut down inference server before pass 2 ...
    let _ = inf_server.shutdown();
```

Replace both blocks with this unified branch:

```rust
    // -----------------------------------------------------------------------
    // Calibration + depth cache
    // -----------------------------------------------------------------------
    // When a pre-computed calibration is supplied: load it, run depth-only batch
    // for grading (same path as before).
    // When auto-calibrating: one fused RAUNE+depth server fills both the
    // calibration store and the grading depth cache from the same keyframes.
    let calibration: Calibration;
    let keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)>;

    if let Some(cal_path) = &args.calibration {
        log::info!("Loading calibration from {}", cal_path.display());
        calibration = Calibration::load(cal_path)
            .context("failed to load .dorea-cal file")?;

        // Depth-only inference for grading depth cache.
        let inf_cfg = InferenceConfig {
            skip_raune: true,
            ..build_inference_config(&args)
        };
        let mut inf_server = InferenceServer::spawn(&inf_cfg)
            .context("failed to spawn depth-only inference server")?;

        let batch_items: Vec<DepthBatchItem> = keyframes.iter().map(|kf| DepthBatchItem {
            id: format!("kf_f{}", kf.frame_index),
            pixels: kf.proxy_pixels.clone(),
            width: proxy_w,
            height: proxy_h,
            max_size: args.proxy_size,
        }).collect();

        let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();
        for (chunk_kfs, chunk_items) in keyframes
            .chunks(DEPTH_BATCH_SIZE)
            .zip(batch_items.chunks(DEPTH_BATCH_SIZE))
        {
            let mut results = inf_server.run_depth_batch(chunk_items)
                .unwrap_or_else(|e| {
                    log::warn!("Depth batch failed: {e} — using uniform depth 0.5");
                    chunk_items.iter().map(|it| {
                        (it.id.clone(), vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h)
                    }).collect()
                });

            if results.len() < chunk_items.len() {
                log::warn!(
                    "Depth batch returned {} results for {} items — padding",
                    results.len(), chunk_items.len()
                );
                for it in &chunk_items[results.len()..] {
                    results.push((it.id.clone(), vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h));
                }
            }

            for (kf, (_, depth_raw, dw, dh)) in chunk_kfs.iter().zip(results.iter()) {
                kf_depths.insert(kf.frame_index, (depth_raw.clone(), *dw, *dh));
            }
        }
        log::info!("Depth batch complete ({} keyframes)", keyframes.len());
        let _ = inf_server.shutdown();
        keyframe_depths = kf_depths;

    } else {
        // Auto-calibrate: fused RAUNE+depth, dual output.
        log::info!(
            "No calibration — auto-calibrating from {} keyframes (fused RAUNE+depth)",
            keyframes.len()
        );

        let inf_cfg = build_inference_config(&args);  // skip_raune: false
        let mut inf_server = InferenceServer::spawn(&inf_cfg)
            .context("failed to spawn fused inference server")?;

        let fused_items: Vec<RauneDepthBatchItem> = keyframes.iter().map(|kf| {
            RauneDepthBatchItem {
                id: format!("kf_f{}", kf.frame_index),
                pixels: kf.proxy_pixels.clone(),
                width: proxy_w,
                height: proxy_h,
                raune_max_size: proxy_w,
                depth_max_size: args.proxy_size,
            }
        }).collect();

        let mut store = PagedCalibrationStore::new()
            .context("failed to create paged calibration store")?;
        let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

        for (chunk_kfs, chunk_items) in keyframes
            .chunks(FUSED_BATCH_SIZE)
            .zip(fused_items.chunks(FUSED_BATCH_SIZE))
        {
            let results = inf_server.run_raune_depth_batch(chunk_items)
                .unwrap_or_else(|e| {
                    log::warn!(
                        "Fused RAUNE+depth batch failed: {e} — using originals + uniform depth"
                    );
                    chunk_items.iter().map(|item| {
                        (item.id.clone(), item.pixels.clone(),
                         item.width, item.height,
                         vec![0.5f32; item.width * item.height],
                         item.width, item.height)
                    }).collect()
                });

            for (kf, (_, enhanced, enh_w, enh_h, depth, dw, dh)) in
                chunk_kfs.iter().zip(results.into_iter())
            {
                store.push(&kf.proxy_pixels, &enhanced, &depth, enh_w, enh_h)
                    .context("failed to page fused result to store")?;
                kf_depths.insert(kf.frame_index, (depth, dw, dh));
            }
        }
        log::info!("Fused inference complete ({} keyframes)", keyframes.len());
        let _ = inf_server.shutdown();
        store.seal().context("failed to seal calibration store")?;

        // ---- 3-pass calibration (inline) ----
        use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
        use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
        use dorea_lut::apply::apply_depth_luts;
        use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder, N_DEPTH_ZONES};

        // Pass 1: reservoir-sample depths → adaptive zone boundaries
        const RESERVOIR_CAP: usize = 1_000_000;
        let mut reservoir: Vec<f32> = Vec::with_capacity(RESERVOIR_CAP);
        let mut total_seen: u64 = 0;
        let mut rng: u64 = 0x853c49e6748fea9b_u64;

        for i in 0..store.len() {
            for d in store.depth_bytes(i).chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            {
                total_seen += 1;
                if reservoir.len() < RESERVOIR_CAP {
                    reservoir.push(d);
                } else {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let j = (rng % total_seen) as usize;
                    if j < RESERVOIR_CAP {
                        reservoir[j] = d;
                    }
                }
            }
        }
        let zone_boundaries = adaptive_zone_boundaries(&reservoir, N_DEPTH_ZONES);
        drop(reservoir);

        // Pass 2: stream frames → build LUT
        let mut lut_builder = StreamingLutBuilder::new(zone_boundaries);
        for i in 0..store.len() {
            let (w, h) = store.dims(i);
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let importance = compute_importance(&depth, w, h);
            lut_builder.add_frame(&original, &target, &depth, &importance);
        }
        let depth_luts = lut_builder.finish();

        // Pass 3: stream frames → HSL corrections
        let n_quals = HSL_QUALIFIERS.len();
        let mut h_offset_acc    = vec![0.0_f64; n_quals];
        let mut s_ratio_acc     = vec![0.0_f64; n_quals];
        let mut v_offset_acc    = vec![0.0_f64; n_quals];
        let mut active_count    = vec![0_usize;  n_quals];
        let mut total_weight_acc = vec![0.0_f64; n_quals];

        for i in 0..store.len() {
            let (pixels_u8, target_u8) = store.pixtar_slices(i);
            let depth = store.read_depth(i);
            let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                .collect();
            let lut_output = apply_depth_luts(&original, &depth, &depth_luts);
            let corrs = derive_hsl_corrections(&lut_output, &target);
            for (qi, corr) in corrs.0.iter().enumerate() {
                if corr.weight >= MIN_WEIGHT {
                    let w = corr.weight as f64;
                    h_offset_acc[qi]    += corr.h_offset as f64 * w;
                    s_ratio_acc[qi]     += corr.s_ratio  as f64 * w;
                    v_offset_acc[qi]    += corr.v_offset as f64 * w;
                    active_count[qi]    += 1;
                    total_weight_acc[qi] += w;
                }
            }
        }

        let mut avg_corrections: Vec<QualifierCorrection> = Vec::with_capacity(n_quals);
        for qi in 0..n_quals {
            let qual = &HSL_QUALIFIERS[qi];
            if active_count[qi] > 0 {
                let tw = total_weight_acc[qi];
                avg_corrections.push(QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: (h_offset_acc[qi] / tw) as f32,
                    s_ratio:  (s_ratio_acc[qi]  / tw) as f32,
                    v_offset: (v_offset_acc[qi] / tw) as f32,
                    weight:   tw as f32,
                });
            } else {
                avg_corrections.push(QualifierCorrection {
                    h_center: qual.h_center,
                    h_width:  qual.h_width,
                    h_offset: 0.0,
                    s_ratio:  1.0,
                    v_offset: 0.0,
                    weight:   0.0,
                });
            }
        }

        calibration = Calibration::new(depth_luts, HslCorrections(avg_corrections), store.len());
        keyframe_depths = kf_depths;
        log::info!(
            "Auto-calibration complete ({} keyframes → {} depth zones)",
            store.len(), N_DEPTH_ZONES
        );
    }
```

- [ ] **Step 3.3: Cargo check**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-cli 2>&1 | tail -15
```
Expected: `Finished` with no errors. If there are type errors:
- `Calibration::new` signature: check `crates/dorea-cal/src/lib.rs` — it takes `(depth_luts, hsl_corrections, keyframe_count: usize)`.
- If `dorea_cal::Calibration` needs `use dorea_cal::Calibration as _Calibration;` (the existing alias), use the same pattern already in `auto_calibrate` (soon to be deleted).

- [ ] **Step 3.4: Cargo test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -15
```
Expected: all tests pass.

- [ ] **Step 3.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): unified keyframe sampling — fused RAUNE+depth fills both calibration and grading depth cache"
```

---

## Task 4: Delete `auto_calibrate` and remove `--keyframe-interval`

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

- [ ] **Step 4.1: Delete the `auto_calibrate` function**

Delete the entire function from `fn auto_calibrate(args: &GradeArgs, info: &ffmpeg::VideoInfo) -> Result<Calibration> {` through its closing `}` (currently lines ~522–691). All of its logic is now inline in `run()`.

- [ ] **Step 4.2: Remove `--keyframe-interval` from `GradeArgs`**

In the `GradeArgs` struct, delete:

```rust
    /// Frames between keyframe samples for auto-calibration
    #[arg(long, default_value = "30")]
    pub keyframe_interval: usize,
```

- [ ] **Step 4.3: Remove any remaining `use` imports that were only used by `auto_calibrate`**

The `auto_calibrate` function had local `use` declarations inside it. Since those are now inline in `run()` in the `else` branch, nothing needs to change in the top-level imports. Verify no dead `use` warnings:

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-cli 2>&1 | grep "unused import\|warning"
```

- [ ] **Step 4.4: Cargo check + test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo check -p dorea-cli 2>&1 | tail -10
cargo test -p dorea-cli 2>&1 | tail -15
```
Expected: clean build, all tests pass.

- [ ] **Step 4.5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): delete auto_calibrate(), remove --keyframe-interval — subsumed by unified pass"
```

---

## Task 5: Open GitHub issue + smoke test

- [ ] **Step 5.1: Open GitHub issue**

```bash
gh issue create \
  --repo chunzhe10/dorea \
  --title "feat(grade): unified keyframe sampling — single MSE pass for calibration + grading" \
  --body "$(cat <<'EOF'
## Summary

Replaces two independent keyframe-selection paths (uniform `--keyframe-interval` for calibration, MSE-adaptive `--depth-max-interval` for grading) with a single unified pass.

## New flow

1. Pass 1: proxy decode → `ChangeDetector.score()` → `Vec<KeyframeEntry>`
2. Fused RAUNE+depth batch on those keyframes (single `InferenceServer`)
3. Results split to `PagedCalibrationStore` (calibration) AND `keyframe_depths` HashMap (grading)
4. 3-pass calibration runs inline → `Calibration`
5. Pass 2: full-res grade using `Calibration` + `keyframe_depths`

## What is removed
- `auto_calibrate()` function
- `--keyframe-interval` CLI arg
- Second `InferenceServer::spawn` for grading when auto-calibrating

## What is added
- `crates/dorea-cli/src/change_detect.rs` — `ChangeDetector` trait + `MseDetector`
- Optical flow can replace `MseDetector` without touching decision logic

## Depends on
`docs/plans/2026-04-03-inference-gpu-saturation.md` (fused RAUNE+depth batch API)

## Spec
`docs/plans/2026-04-04-unified-keyframe-sampling.md`
`docs/plans/2026-04-04-unified-keyframe-sampling-impl.md`
EOF
)"
```

- [ ] **Step 5.2: Smoke test (if dive footage is available)**

```bash
time dorea grade \
  --input /workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101112849_0057_D.MP4 \
  --output /workspaces/dorea-workspace/working/graded/unified_test_$(date +%Y%m%d).mp4 \
  --raune-weights /workspaces/dorea-workspace/models/raune_net.pth \
  --raune-models-dir /workspaces/dorea-workspace/models/RAUNE-Net \
  --depth-model /workspaces/dorea-workspace/models/depth_anything_v2_vits \
  --depth-max-interval 5 \
  2>&1 | tee /tmp/unified_run.log
```

Expected: single `[dorea-inference] ready` message (not two), calibration log shows `Auto-calibration complete (N keyframes)`, grading completes successfully.

- [ ] **Step 5.3: Save result to corvia**

```
corvia_write: record result — calibration keyframe count, wall time, any anomalies
source_origin: repo:dorea, content_role: decision
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Single proxy decode pass (pass 1 only) | Task 2 (pass 1 wired with ChangeDetector), Task 3 (calibration rides same keyframes) |
| ChangeDetector trait with MSE impl | Task 1 |
| Fused RAUNE+depth fills both outputs | Task 3 (else branch) |
| 3-pass calibration inline | Task 3 (inline in else branch) |
| Pre-computed calibration path unchanged | Task 3 (if branch — depth-only, identical to current) |
| `auto_calibrate()` deleted | Task 4 |
| `--keyframe-interval` removed | Task 4 |
| GitHub issue | Task 5 |

**Placeholder scan:** None. All code blocks are complete. 3-pass calibration code is copied verbatim from existing `auto_calibrate()`.

**Type consistency:**
- `run_raune_depth_batch` returns `Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>` — destructured as `(_, enhanced, enh_w, enh_h, depth, dw, dh)` ✓
- `PagedCalibrationStore::push` signature: `(&mut self, pixels: &[u8], target: &[u8], depth: &[f32], width: usize, height: usize)` — called with `(&kf.proxy_pixels, &enhanced, &depth, enh_w, enh_h)` ✓
- `Calibration::new(depth_luts, HslCorrections(avg_corrections), store.len())` — matches existing call in `auto_calibrate` ✓
- `kf_depths` declared as `HashMap<u64, (Vec<f32>, usize, usize)>` — assigned to `keyframe_depths` at end of each branch ✓

**Note on `Calibration` import:** `auto_calibrate` uses a local alias `use dorea_cal::Calibration as _Calibration;` to avoid name collision with the outer-scope import. In Task 3, the inline calibration code assigns to `calibration: Calibration` (declared in outer scope). Use `Calibration::new(...)` directly — no alias needed since the declaration is in outer scope.
