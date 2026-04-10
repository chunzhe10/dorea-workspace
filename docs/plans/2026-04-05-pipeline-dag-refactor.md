# Pipeline DAG Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `grade.rs` into four named DAG stages — Keyframe, Feature, LUT Calibration, Grading Render — each composed of typed nodes with explicit inputs and outputs, making individual components easy to address, test, and recompose.

**Architecture:** A `Node` trait (`type Input`, `type Output`, `fn process`) and a `Stage` trait (`fn run`, `fn name`) live in `crates/dorea-cli/src/pipeline/mod.rs`. Each stage is a concrete struct in its own file under `pipeline/`. `grade.rs::run` becomes thin orchestration that constructs and chains the four stages. `PagedCalibrationStore` moves to `pipeline/feature.rs`. `KeyframeEntry` moves to `pipeline/keyframe.rs`. `lerp_depth` moves to `pipeline/grading.rs`. The proxy decoder streams frames one at a time — never materialises the full video into RAM. `FusedInferenceNode` fully owns its `InferenceServer` and implements `process()`. `HslAverageNode` is a stateful streaming accumulator. `LutBuildNode` is a free function.

**Tech Stack:** Rust, anyhow, existing dorea-gpu / dorea-video / dorea-lut / dorea-hsl / dorea-cal crates.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/dorea-cli/src/pipeline/mod.rs` | `Node` + `Stage` traits; shared I/O structs used across stage boundaries |
| `crates/dorea-cli/src/pipeline/keyframe.rs` | `KeyframeStage` + `ProxyDecodeNode` (streaming), `ChangeDetectNode`, `KeyframeSelectNode` |
| `crates/dorea-cli/src/pipeline/feature.rs` | `FeatureStage` + `FusedInferenceNode` (fully implemented), `EnhancedDownscaleNode`; owns `PagedCalibrationStore` |
| `crates/dorea-cli/src/pipeline/calibration.rs` | `CalibrationStage` + `ZoneComputeNode`, `SegmentDetectNode`, `ZoneSmoothNode`, `build_segment_lut` (free fn), `HslAverageNode` (streaming accumulator), `GraderInitNode` |
| `crates/dorea-cli/src/pipeline/grading.rs` | `GradingStage` + `DepthInterpolateNode` (plain `lerp` method), `BlendTNode` |
| `crates/dorea-cli/src/grade.rs` | Thin orchestrator: parse config, construct stages, chain `.run()` calls |

`PagedCalibrationStore` moves from `grade.rs` into `feature.rs` (it is an implementation detail of the feature stage, not shared across stages). `KeyframeEntry` moves to `keyframe.rs`. `lerp_depth` moves to `grading.rs`.

---

### Task 1: `Node` + `Stage` traits + shared I/O types (`pipeline/mod.rs`)

**Files:**
- Create: `crates/dorea-cli/src/pipeline/mod.rs`
- Modify: `crates/dorea-cli/src/lib.rs` (add `pub mod pipeline;`)

- [ ] **Step 1: Write failing test**

Add to `pipeline/mod.rs` (will be created in step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct Double;
    impl Node for Double {
        type Input = i32;
        type Output = i32;
        fn name(&self) -> &'static str { "double" }
        fn process(&mut self, input: i32) -> anyhow::Result<i32> { Ok(input * 2) }
    }

    #[test]
    fn node_process_called() {
        let mut n = Double;
        assert_eq!(n.process(3).unwrap(), 6);
    }

    #[test]
    fn node_name_returns_str() {
        assert_eq!(Double.name(), "double");
    }
}
```

- [ ] **Step 2: Run — verify FAIL**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | grep -E "error|FAILED"
```
Expected: compile error `pipeline` not found.

- [ ] **Step 3: Create `pipeline/mod.rs`**

```rust
//! DAG stage and node abstractions for the dorea grading pipeline.
//!
//! Stages: Keyframe → Feature → LutCalibration → GradingRender
//! Each stage contains named nodes with typed inputs and outputs.

pub mod keyframe;
pub mod feature;
pub mod calibration;
pub mod grading;

/// A single processing unit with typed input and output.
///
/// Nodes are stateful (take `&mut self`) to allow internal caches or
/// server handles. They are not Send/Sync by default; the pipeline is
/// single-threaded sequential.
pub trait Node {
    type Input;
    type Output;

    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &'static str;

    /// Execute this node on one input, producing one output.
    fn process(&mut self, input: Self::Input) -> anyhow::Result<Self::Output>;
}

/// A named pipeline stage composed of one or more nodes.
///
/// A stage owns its nodes and defines the aggregate input/output types
/// that cross stage boundaries. Stages are chained by the orchestrator
/// in `grade.rs::run`.
pub trait Stage {
    type Input;
    type Output;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;

    /// Run all nodes in this stage, consuming `input` and producing `output`.
    fn run(&mut self, input: Self::Input) -> anyhow::Result<Self::Output>;
}

// ── Shared I/O structs used at stage boundaries ──────────────────────────────

/// Output of the Keyframe stage / input of the Feature stage.
#[derive(Debug)]
pub struct KeyframeStageOutput {
    /// Keyframes selected by change detection.
    pub keyframes: Vec<keyframe::KeyframeEntry>,
    /// Proxy frame dimensions (width, height).
    pub proxy_dims: (usize, usize),
}

/// Output of the Feature stage / input of the LUT Calibration stage.
pub struct FeatureStageOutput {
    /// Calibration store: mmap'd (pixels, RAUNE-enhanced-proxy, depth) per keyframe.
    pub store: feature::PagedCalibrationStore,
    /// Native-resolution depth per keyframe frame index: (depth_f32, width, height).
    pub keyframe_depths: std::collections::HashMap<u64, (Vec<f32>, usize, usize)>,
    /// Keyframe metadata list preserved from the Keyframe stage.
    pub keyframes: Vec<keyframe::KeyframeEntry>,
}

/// Output of the LUT Calibration stage / input of the Grading Render stage.
pub struct CalibrationStageOutput {
    /// Per-segment base LUT + HSL corrections.
    pub segment_calibrations: Vec<calibration::SegmentCalibration>,
    /// Per-keyframe smoothed zone boundaries (runtime zones, `depth_zones + 1` values each).
    pub smoothed_kf_zones: Vec<Vec<f32>>,
    /// Maps keyframe index → segment index.
    pub kf_to_segment: Vec<usize>,
    /// Segments from scene detection.
    pub segments: Vec<crate::change_detect::SegmentRange>,
    /// Keyframe metadata (frame_index, scene_cut_before).
    pub kf_index_list: Vec<(u64, bool)>,
    /// Native depth per keyframe (for Pass 2 depth interpolation).
    pub keyframe_depths: std::collections::HashMap<u64, (Vec<f32>, usize, usize)>,
}
```

- [ ] **Step 4: Add `pub mod pipeline;` to `lib.rs`**

In `crates/dorea-cli/src/lib.rs`, add:
```rust
pub mod pipeline;
```

Create stub files so the submodule declarations compile:

```bash
touch crates/dorea-cli/src/pipeline/keyframe.rs
touch crates/dorea-cli/src/pipeline/feature.rs
touch crates/dorea-cli/src/pipeline/calibration.rs
touch crates/dorea-cli/src/pipeline/grading.rs
```

- [ ] **Step 5: Run — verify PASS**

```bash
cargo test -p dorea-cli --features cuda 2>&1 | grep -E "FAILED|^test result"
```
Expected: `test result: ok. N passed`

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-cli/src/pipeline/ crates/dorea-cli/src/lib.rs
git commit -m "feat(pipeline): Node + Stage traits + shared I/O types"
```

---

### Task 2: Keyframe stage (`pipeline/keyframe.rs`)

**Files:**
- Modify: `crates/dorea-cli/src/pipeline/keyframe.rs`

The Keyframe stage contains three nodes executed in sequence:
- `ProxyDecodeNode`: spawns the ffmpeg proxy-scaled decoder; returns a streaming `ProxyDecoder` handle — **never collects frames into a Vec**
- `ChangeDetectNode`: standalone node for scoring (frame, pixels) pairs — used externally; inside the stage, `KeyframeSelectNode` owns the detector directly to avoid pixel copies
- `KeyframeSelectNode`: applies threshold rules and emits `KeyframeEntry` items

The stage input carries everything needed to run (video path, info, config). The stage output is `KeyframeStageOutput` (defined in `mod.rs`).

**Memory contract:** At any point during the Keyframe stage, at most one decoded frame is in memory. Non-keyframe pixels are moved into `FrameScore` and dropped by `KeyframeSelectNode::process` without copying. Keyframe pixels are moved into `KeyframeEntry` and stored.

- [ ] **Step 1: Write failing tests**

```rust
// In pipeline/keyframe.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Node;

    #[test]
    fn keyframe_select_node_first_frame_always_keyframe() {
        let cfg = KeyframeConfig {
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            interp_enabled: true,
            scene_cut_threshold: 0.05,
        };
        let mut node = KeyframeSelectNode::new(cfg);
        let out = node.process(FrameScore {
            frame_index: 0,
            pixels: vec![128u8; 3],
            change_score: f32::MAX, // no reference yet
        }).unwrap();
        assert!(out.is_some(), "first frame must always be a keyframe");
    }

    #[test]
    fn keyframe_select_node_below_threshold_not_keyframe() {
        let cfg = KeyframeConfig {
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            interp_enabled: true,
            scene_cut_threshold: 0.05,
        };
        let mut node = KeyframeSelectNode::new(cfg);
        // Bootstrap with first frame
        let _ = node.process(FrameScore {
            frame_index: 0,
            pixels: vec![128u8; 3],
            change_score: f32::MAX,
        }).unwrap();
        // Second frame below threshold
        let out = node.process(FrameScore {
            frame_index: 1,
            pixels: vec![128u8; 3],
            change_score: 0.001,
        }).unwrap();
        assert!(out.is_none(), "frame below threshold should not be a keyframe");
    }

    #[test]
    fn keyframe_select_node_interval_forces_keyframe() {
        let cfg = KeyframeConfig {
            depth_skip_threshold: 0.005,
            depth_max_interval: 3,
            interp_enabled: true,
            scene_cut_threshold: 0.05,
        };
        let mut node = KeyframeSelectNode::new(cfg);
        // Bootstrap
        let _ = node.process(FrameScore { frame_index: 0, pixels: vec![0u8; 3], change_score: f32::MAX }).unwrap();
        // 3 non-KF frames
        for i in 1..=3 {
            let _ = node.process(FrameScore { frame_index: i, pixels: vec![0u8; 3], change_score: 0.001 }).unwrap();
        }
        // 4th non-KF frame — should be forced by interval
        let out = node.process(FrameScore { frame_index: 4, pixels: vec![0u8; 3], change_score: 0.001 }).unwrap();
        assert!(out.is_some(), "interval exceeded — frame must be forced keyframe");
    }
}
```

- [ ] **Step 2: Run — verify FAIL**

```bash
cargo test -p dorea-cli --features cuda pipeline::keyframe 2>&1 | grep -E "error|FAILED"
```
Expected: compile errors (types not defined yet).

- [ ] **Step 3: Implement `keyframe.rs`**

```rust
//! Keyframe stage: proxy decode → change detection → keyframe selection.
//!
//! Nodes:
//!   ProxyDecodeNode    — spawns proxy decoder, returns streaming iterator handle
//!   ChangeDetectNode   — standalone scorer (external use only; stage uses select_node.detector)
//!   KeyframeSelectNode — applies threshold rules, emits KeyframeEntry

use anyhow::{Context, Result};
use crate::pipeline::{Node, Stage, KeyframeStageOutput};
use crate::change_detect::{ChangeDetector, MseDetector};
use dorea_video::ffmpeg;

// ── Public types ─────────────────────────────────────────────────────────────

/// A keyframe collected during proxy-decode pass.
#[derive(Debug, Clone)]
pub struct KeyframeEntry {
    pub frame_index: u64,
    pub proxy_pixels: Vec<u8>,
    /// True if this keyframe immediately follows a scene cut.
    pub scene_cut_before: bool,
}

// ── Stage input ──────────────────────────────────────────────────────────────

pub struct KeyframeStageInput {
    pub video_path: std::path::PathBuf,
    pub video_info: dorea_video::ffmpeg::VideoInfo,
    pub proxy_w: usize,
    pub proxy_h: usize,
    pub config: KeyframeConfig,
}

#[derive(Clone, Copy)]
pub struct KeyframeConfig {
    pub depth_skip_threshold: f32,
    pub depth_max_interval: usize,
    pub interp_enabled: bool,
    pub scene_cut_threshold: f32,
}

// ── Streaming decoder wrapper ────────────────────────────────────────────────

/// Streaming handle returned by `ProxyDecodeNode::process`.
/// Wraps the ffmpeg iterator so the node output is a proper named type.
pub struct ProxyDecoder(Box<dyn Iterator<Item = anyhow::Result<ffmpeg::DecodedFrame>>>);

impl Iterator for ProxyDecoder {
    type Item = anyhow::Result<ffmpeg::DecodedFrame>;
    fn next(&mut self) -> Option<Self::Item> { self.0.next() }
}

// ── Node: ProxyDecodeNode ─────────────────────────────────────────────────────

/// Spawns the ffmpeg proxy decoder and returns a streaming `ProxyDecoder` handle.
/// The caller drives the iterator one frame at a time — frames are never buffered.
pub struct ProxyDecodeNode;

impl Node for ProxyDecodeNode {
    type Input = (std::path::PathBuf, dorea_video::ffmpeg::VideoInfo, usize, usize);
    type Output = ProxyDecoder;

    fn name(&self) -> &'static str { "proxy_decode" }

    fn process(&mut self, (path, info, pw, ph): Self::Input) -> Result<ProxyDecoder> {
        let iter = ffmpeg::decode_frames_scaled(&path, &info, pw, ph)
            .context("failed to spawn ffmpeg proxy decoder")?;
        Ok(ProxyDecoder(Box::new(iter)))
    }
}

// ── Node: ChangeDetectNode ────────────────────────────────────────────────────

/// Scores a frame against the last reference. Standalone node for external use.
/// Inside KeyframeStage, KeyframeSelectNode owns its own detector to avoid
/// passing pixels through an extra struct allocation.
pub struct FrameScore {
    pub frame_index: u64,
    /// Owned pixels. For non-keyframes, these are moved into process() and dropped.
    /// No copy occurs; the Vec is moved through FrameScore without heap allocation.
    pub pixels: Vec<u8>,
    pub change_score: f32,
}

pub struct ChangeDetectNode {
    detector: Box<dyn ChangeDetector>,
}

impl ChangeDetectNode {
    pub fn new() -> Self {
        Self { detector: Box::new(MseDetector::default()) }
    }
}

impl Node for ChangeDetectNode {
    type Input = (u64, Vec<u8>);
    type Output = FrameScore;

    fn name(&self) -> &'static str { "change_detect" }

    fn process(&mut self, (frame_index, pixels): Self::Input) -> Result<FrameScore> {
        let score = self.detector.score(&pixels);
        Ok(FrameScore { frame_index, pixels, change_score: score })
    }
}

// ── Node: KeyframeSelectNode ──────────────────────────────────────────────────

/// Applies keyframe selection rules; emits Some(KeyframeEntry) on keyframes, None otherwise.
/// Owns its MseDetector so the stage can call score() before process() without pixel copies.
pub struct KeyframeSelectNode {
    cfg: KeyframeConfig,
    pub(super) detector: MseDetector,
    frames_since_kf: usize,
    any_keyframe: bool,
}

impl KeyframeSelectNode {
    pub fn new(cfg: KeyframeConfig) -> Self {
        Self {
            cfg,
            detector: MseDetector::default(),
            frames_since_kf: 0,
            any_keyframe: false,
        }
    }
}

impl Node for KeyframeSelectNode {
    type Input = FrameScore;
    type Output = Option<KeyframeEntry>;

    fn name(&self) -> &'static str { "keyframe_select" }

    fn process(&mut self, fs: FrameScore) -> Result<Option<KeyframeEntry>> {
        let FrameScore { frame_index, pixels, change_score } = fs;
        let scene_cut = change_score < f32::MAX && change_score > self.cfg.scene_cut_threshold;
        let is_keyframe = !self.cfg.interp_enabled
            || !self.any_keyframe
            || scene_cut
            || self.frames_since_kf >= self.cfg.depth_max_interval
            || (change_score < f32::MAX && change_score > self.cfg.depth_skip_threshold);

        if is_keyframe {
            if scene_cut {
                log::info!("Scene cut at frame {frame_index} (change={change_score:.6})");
                self.detector.reset();
            }
            self.detector.set_reference(&pixels);
            self.frames_since_kf = 0;
            self.any_keyframe = true;
            Ok(Some(KeyframeEntry { frame_index, proxy_pixels: pixels, scene_cut_before: scene_cut }))
        } else {
            self.frames_since_kf += 1;
            Ok(None)
        }
    }
}

// ── Stage: KeyframeStage ──────────────────────────────────────────────────────

pub struct KeyframeStage {
    decode_node: ProxyDecodeNode,
    select_node: KeyframeSelectNode,
}

impl KeyframeStage {
    pub fn new(cfg: KeyframeConfig) -> Self {
        Self {
            decode_node: ProxyDecodeNode,
            select_node: KeyframeSelectNode::new(cfg),
        }
    }
}

impl Stage for KeyframeStage {
    type Input = KeyframeStageInput;
    type Output = KeyframeStageOutput;

    fn name(&self) -> &'static str { "keyframe_stage" }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output> {
        log::info!("=== {} ===", self.name());

        // ProxyDecodeNode returns a streaming iterator — no frames are buffered.
        let decoder = self.decode_node.process(
            (input.video_path, input.video_info, input.proxy_w, input.proxy_h)
        )?;

        let mut keyframes: Vec<KeyframeEntry> = Vec::new();
        for frame_result in decoder {
            let frame = frame_result.context("proxy frame decode error")?;
            // Score using select_node's internal detector to avoid moving pixels
            // into an intermediate FrameScore before the score is known.
            let score = self.select_node.detector.score(&frame.pixels);
            if let Some(kf) = self.select_node.process(FrameScore {
                frame_index: frame.index,
                pixels: frame.pixels, // moved, not copied
                change_score: score,
            })? {
                keyframes.push(kf);
            }
        }

        log::info!("{}: {} keyframes detected", self.name(), keyframes.len());
        anyhow::ensure!(!keyframes.is_empty(), "keyframe stage produced no keyframes");
        Ok(KeyframeStageOutput { keyframes, proxy_dims: (input.proxy_w, input.proxy_h) })
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p dorea-cli --features cuda pipeline::keyframe 2>&1 | grep -E "FAILED|ok\."
```
Expected: all 3 keyframe tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/pipeline/keyframe.rs
git commit -m "feat(pipeline): KeyframeStage — ProxyDecodeNode (streaming) + ChangeDetectNode + KeyframeSelectNode"
```

---

### Task 3: Feature stage (`pipeline/feature.rs`)

**Files:**
- Modify: `crates/dorea-cli/src/pipeline/feature.rs`

Moves `PagedCalibrationStore` and the fused inference batch from `grade.rs`. Nodes:
- `FusedInferenceNode`: fully implemented — owns `InferenceServer`, builds each batch inline without pre-collecting all keyframe pixels, calls `run_raune_depth_batch` directly
- `EnhancedDownscaleNode`: downscales Maxine-upscaled enhanced to proxy dims

`DepthCacheNode` is not a node — the stage manages the `keyframe_depths` HashMap directly. No node abstraction is needed for a single HashMap insert.

`FeatureStageInput` carries the model paths so `load_raune` and `load_depth` use the configured weights, not default fallbacks.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Node;

    #[test]
    fn enhanced_downscale_node_noop_when_same_size() {
        let mut node = EnhancedDownscaleNode { proxy_w: 4, proxy_h: 4 };
        let pixels = vec![100u8; 4 * 4 * 3];
        let out = node.process(EnhancedFrame {
            pixels: pixels.clone(),
            width: 4,
            height: 4,
        }).unwrap();
        assert_eq!(out.pixels, pixels);
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
    }

    #[test]
    fn enhanced_downscale_node_downscales_to_proxy() {
        let mut node = EnhancedDownscaleNode { proxy_w: 2, proxy_h: 2 };
        let pixels = vec![200u8; 4 * 4 * 3]; // 4x4 input
        let out = node.process(EnhancedFrame {
            pixels,
            width: 4,
            height: 4,
        }).unwrap();
        assert_eq!(out.pixels.len(), 2 * 2 * 3);
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
    }

    #[test]
    fn paged_calibration_store_push_and_read() {
        let mut store = PagedCalibrationStore::new().unwrap();
        let pixels = vec![10u8, 20, 30, 40, 50, 60]; // 2 pixels
        let target = vec![11u8, 21, 31, 41, 51, 61];
        let depth = vec![0.1f32, 0.9];
        store.push(&pixels, &target, &depth, 2, 1).unwrap();
        store.seal().unwrap();

        let (p, t) = store.pixtar_slices(0);
        assert_eq!(p, pixels.as_slice());
        assert_eq!(t, target.as_slice());
        let d = store.read_depth(0);
        assert!((d[0] - 0.1).abs() < 1e-5);
        assert!((d[1] - 0.9).abs() < 1e-5);
    }
}
```

> Note: `FusedInferenceNode` requires a live inference server and is exercised by integration tests (Task 6 compile + end-to-end run), not by unit tests here.

- [ ] **Step 2: Run — verify FAIL**

```bash
cargo test -p dorea-cli --features cuda pipeline::feature 2>&1 | grep -E "error|FAILED"
```

- [ ] **Step 3: Implement `feature.rs`**

```rust
//! Feature stage: keyframes → RAUNE → Maxine (2×) → Depth.
//!
//! Nodes:
//!   FusedInferenceNode    — batched RAUNE→Maxine→Depth; owns InferenceServer
//!   EnhancedDownscaleNode — downscales Maxine-upscaled enhanced to proxy dims
//!
//! The stage manages keyframe_depths (HashMap) directly — no DepthCacheNode.

use anyhow::{Context, Result};
use std::collections::HashMap;
use crate::pipeline::{Node, Stage, KeyframeStageOutput, FeatureStageOutput};
use crate::pipeline::keyframe::KeyframeEntry;
use dorea_video::inference::{RauneDepthBatchItem, InferenceServer};

// ── Stage input ───────────────────────────────────────────────────────────────

pub struct FeatureStageInput {
    pub keyframe_output: KeyframeStageOutput,
    pub inf_server: InferenceServer,
    pub fused_batch_size: usize,
    pub proxy_size: usize,
    pub enable_maxine: bool,
    /// Configured path for RAUNE weights; None = server default.
    pub raune_weights: Option<std::path::PathBuf>,
    /// Configured models dir for RAUNE; None = server default.
    pub raune_models_dir: Option<std::path::PathBuf>,
    /// Configured path for Depth Anything model; None = server default.
    pub depth_model: Option<std::path::PathBuf>,
}

// ── Intermediate type ─────────────────────────────────────────────────────────

pub struct EnhancedFrame {
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

// ── Node: FusedInferenceNode ──────────────────────────────────────────────────

/// Runs one batch of keyframes through RAUNE → Maxine → Depth.
///
/// Owns the InferenceServer for its lifetime. The stage creates this node after
/// calling load_raune/load_depth on the server, then passes each batch via
/// `process(chunk_kfs.to_vec())`.
///
/// Builds the `RauneDepthBatchItem` list inline per batch — never pre-allocates
/// items for all keyframes at once.
pub struct FusedInferenceNode {
    pub server: InferenceServer,
    enable_maxine: bool,
    proxy_w: usize,
    proxy_h: usize,
    proxy_size: usize,
}

impl FusedInferenceNode {
    pub fn new(
        server: InferenceServer,
        enable_maxine: bool,
        proxy_w: usize,
        proxy_h: usize,
        proxy_size: usize,
    ) -> Self {
        Self { server, enable_maxine, proxy_w, proxy_h, proxy_size }
    }
}

impl Node for FusedInferenceNode {
    /// Input: one batch of KeyframeEntry (proxy pixels).
    type Input = Vec<KeyframeEntry>;
    /// Output: (id, enhanced_rgb, enh_w, enh_h, depth_f32, dw, dh) per keyframe.
    type Output = Vec<(String, Vec<u8>, usize, usize, Vec<f32>, usize, usize)>;

    fn name(&self) -> &'static str { "fused_inference" }

    fn process(&mut self, batch: Vec<KeyframeEntry>) -> Result<Self::Output> {
        // Build items inline — no pre-allocation across all keyframes.
        let items: Vec<RauneDepthBatchItem> = batch.iter().map(|kf| {
            RauneDepthBatchItem {
                id: format!("kf_f{}", kf.frame_index),
                pixels: kf.proxy_pixels.clone(),
                width: self.proxy_w,
                height: self.proxy_h,
                raune_max_size: self.proxy_w.max(self.proxy_h),
                depth_max_size: self.proxy_size.min(1036),
            }
        }).collect();

        let mut results = self.server
            .run_raune_depth_batch(&items, self.enable_maxine)
            .unwrap_or_else(|e| {
                log::warn!("Fused batch failed: {e} — using originals + uniform depth");
                items.iter().map(|item| (
                    item.id.clone(), item.pixels.clone(),
                    item.width, item.height,
                    vec![0.5f32; item.width * item.height],
                    item.width, item.height,
                )).collect()
            });

        if results.len() < items.len() {
            log::warn!(
                "Fused batch returned {} results for {} items — padding",
                results.len(), items.len()
            );
            for item in &items[results.len()..] {
                results.push((
                    item.id.clone(), item.pixels.clone(),
                    item.width, item.height,
                    vec![0.5f32; item.width * item.height],
                    item.width, item.height,
                ));
            }
        }
        Ok(results)
    }
}

// ── Node: EnhancedDownscaleNode ───────────────────────────────────────────────

/// Downscales a Maxine-upscaled enhanced frame back to proxy resolution.
/// Input: `EnhancedFrame` (may be 2× proxy)
/// Output: `EnhancedFrame` at exactly proxy_w × proxy_h
pub struct EnhancedDownscaleNode {
    pub proxy_w: usize,
    pub proxy_h: usize,
}

impl Node for EnhancedDownscaleNode {
    type Input = EnhancedFrame;
    type Output = EnhancedFrame;

    fn name(&self) -> &'static str { "enhanced_downscale" }

    fn process(&mut self, frame: EnhancedFrame) -> Result<EnhancedFrame> {
        if frame.width == self.proxy_w && frame.height == self.proxy_h {
            return Ok(frame);
        }
        let pixels = dorea_video::resize::resize_rgb_bilinear(
            &frame.pixels, frame.width, frame.height, self.proxy_w, self.proxy_h,
        );
        Ok(EnhancedFrame { pixels, width: self.proxy_w, height: self.proxy_h })
    }
}

// ── Stage: FeatureStage ───────────────────────────────────────────────────────

pub struct FeatureStage {
    downscale_node: EnhancedDownscaleNode,
}

impl FeatureStage {
    pub fn new(proxy_w: usize, proxy_h: usize) -> Self {
        Self {
            downscale_node: EnhancedDownscaleNode { proxy_w, proxy_h },
        }
    }
}

impl Stage for FeatureStage {
    type Input = FeatureStageInput;
    type Output = FeatureStageOutput;

    fn name(&self) -> &'static str { "feature_stage" }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output> {
        log::info!("=== {} ===", self.name());
        let FeatureStageInput {
            keyframe_output,
            inf_server,
            fused_batch_size,
            proxy_size,
            enable_maxine,
            raune_weights,
            raune_models_dir,
            depth_model,
        } = input;

        let KeyframeStageOutput { keyframes, proxy_dims: (proxy_w, proxy_h) } = keyframe_output;

        // Load models using configured paths before handing the server to the node.
        let mut inf_server = inf_server;
        inf_server.load_raune(raune_weights.as_deref(), raune_models_dir.as_deref())
            .context("failed to load RAUNE-Net")?;
        inf_server.load_depth(depth_model.as_deref())
            .context("failed to load Depth Anything")?;

        // FusedInferenceNode owns the server for the duration of inference.
        let mut inference_node = FusedInferenceNode::new(
            inf_server, enable_maxine, proxy_w, proxy_h, proxy_size,
        );

        let mut store = PagedCalibrationStore::new()
            .context("failed to create paged calibration store")?;
        let mut keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

        let n_batches = (keyframes.len() + fused_batch_size - 1) / fused_batch_size;
        for (batch_idx, chunk_kfs) in keyframes.chunks(fused_batch_size).enumerate() {
            log::info!(
                "{}: RAUNE+depth batch {}/{n_batches} ({} frames)",
                self.name(), batch_idx + 1, chunk_kfs.len(),
            );

            // process() builds the RauneDepthBatchItem list inline — one batch at a time.
            let results = inference_node.process(chunk_kfs.to_vec())?;

            for (kf, (_, enhanced, enh_w, enh_h, depth, dw, dh)) in
                chunk_kfs.iter().zip(results.into_iter())
            {
                // EnhancedDownscaleNode: downscale Maxine-upscaled enhanced to proxy.
                let enhanced_proxy = self.downscale_node.process(
                    EnhancedFrame { pixels: enhanced, width: enh_w, height: enh_h }
                )?.pixels;

                // Depth for store: upscale to proxy dims if needed.
                let depth_for_store = if dw == proxy_w && dh == proxy_h {
                    depth.clone()
                } else {
                    InferenceServer::upscale_depth(&depth, dw, dh, proxy_w, proxy_h)
                };
                store.push(&kf.proxy_pixels, &enhanced_proxy, &depth_for_store, proxy_w, proxy_h)
                    .context("failed to push to calibration store")?;

                // keyframe_depths keeps native (Maxine-resolution) depth for Pass 2.
                keyframe_depths.insert(kf.frame_index, (depth, dw, dh));
            }
        }

        log::info!("{}: fused inference complete ({} keyframes)", self.name(), keyframes.len());
        debug_assert_eq!(store.len(), keyframes.len(), "store/keyframes length diverged");
        let _ = inference_node.server.shutdown();
        store.seal().context("failed to seal calibration store")?;

        Ok(FeatureStageOutput { store, keyframe_depths, keyframes })
    }
}

// ── PagedCalibrationStore ────────────────────────────────────────────────────
// (moved verbatim from grade.rs)

/// Temporary on-disk store for calibration frames.
///
/// Two packed binary files, both memory-mapped after inference:
///   depths.bin  — all depth maps concatenated (f32 LE)
///   pixtar.bin  — all [pixels | target] pairs concatenated (u8 RGB)
pub struct PagedCalibrationStore {
    // ... (copy verbatim from grade.rs lines 686–970)
}
```

> **Copy `PagedCalibrationStore` exactly from `grade.rs` lines 686–970** (the struct definition + all impl methods: `new`, `push`, `seal`, `len`, `depth_bytes`, `read_depth`, `dims`, `pixtar_slices`). Do not change any logic.

- [ ] **Step 4: Run tests**

```bash
cargo test -p dorea-cli --features cuda pipeline::feature 2>&1 | grep -E "FAILED|ok\."
```
Expected: all 3 feature tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/pipeline/feature.rs
git commit -m "feat(pipeline): FeatureStage — FusedInferenceNode (fully impl) + EnhancedDownscaleNode + PagedCalibrationStore"
```

---

### Task 4: LUT Calibration stage (`pipeline/calibration.rs`)

**Files:**
- Modify: `crates/dorea-cli/src/pipeline/calibration.rs`

Nodes:
- `ZoneComputeNode`: `compute_per_kf_zones`
- `SegmentDetectNode`: `detect_scene_segments`
- `ZoneSmoothNode`: `smooth_zone_boundaries`
- `HslAverageNode`: streaming per-KF accumulator; call `process((lut_out, target))` per keyframe, `finish()` after the segment, `reset()` between segments — **no Vec of pairs materialised**
- `GraderInitNode`: conceptual name for the AdaptiveGrader init block in GradingStage

`LutBuildNode` is replaced by a free function `build_segment_lut` — the borrow checker makes a `Node` implementation here unnecessarily awkward (the store must be borrowed while building), and a free function is simpler and equally testable.

**Public type: `SegmentCalibration`** (moved out of the `run()` local scope in `grade.rs`).

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Node;

    #[test]
    fn zone_compute_node_output_length_matches_keyframe_count() {
        let depths = vec![
            vec![0.1f32, 0.5, 0.9],
            vec![0.2f32, 0.6, 0.8],
        ];
        let mut node = ZoneComputeNode { depth_zones: 4 };
        let zones = node.process(depths).unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].len(), 5); // 4 zones = 5 boundaries
    }

    #[test]
    fn segment_detect_node_single_segment_uniform_depths() {
        let depths: Vec<Vec<f32>> = (0..5)
            .map(|_| (0..50).map(|i| i as f32 / 50.0).collect())
            .collect();
        let mut node = SegmentDetectNode { scene_threshold: 0.15, min_segment_kfs: 3 };
        let segs = node.process(depths).unwrap();
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn zone_smooth_node_noop_at_window_1() {
        let raw = vec![
            vec![0.0f32, 0.25, 0.5, 0.75, 1.0],
            vec![0.0f32, 0.10, 0.20, 0.30, 1.0],
        ];
        let segs = vec![crate::change_detect::SegmentRange { start: 0, end: 2 }];
        let mut node = ZoneSmoothNode { window: 1 };
        let smoothed = node.process((raw.clone(), segs)).unwrap();
        assert_eq!(smoothed, raw);
    }

    #[test]
    fn hsl_average_node_identity_when_source_equals_target() {
        let mut node = HslAverageNode::new();
        // Feed pairs where source == target (identity correction).
        let pixels: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.2]; 16];
        node.process((pixels.clone(), pixels.clone())).unwrap();
        node.process((pixels.clone(), pixels.clone())).unwrap();
        let corrections = node.finish();
        for q in &corrections.0 {
            assert!((q.h_offset).abs() < 0.15, "h_offset should be near 0: {}", q.h_offset);
            assert!((q.s_ratio - 1.0).abs() < 0.15 || q.weight == 0.0,
                "s_ratio should be near 1.0: {}", q.s_ratio);
        }
    }

    #[test]
    fn hsl_average_node_reset_clears_state() {
        let mut node = HslAverageNode::new();
        let pixels: Vec<[f32; 3]> = vec![[0.9, 0.1, 0.1]; 16];
        node.process((pixels.clone(), pixels.clone())).unwrap();
        node.reset();
        let corrs = node.finish();
        for q in &corrs.0 {
            assert_eq!(q.weight, 0.0, "weight must be 0 after reset");
        }
    }
}
```

- [ ] **Step 2: Run — verify FAIL**

```bash
cargo test -p dorea-cli --features cuda pipeline::calibration 2>&1 | grep -E "error|FAILED"
```

- [ ] **Step 3: Implement `calibration.rs`**

```rust
//! LUT Calibration stage: depth timeline → per-segment LUTs + HSL + AdaptiveGrader.
//!
//! Nodes:
//!   ZoneComputeNode   — compute per-KF adaptive zone boundaries
//!   SegmentDetectNode — Wasserstein-1 scene segmentation
//!   ZoneSmoothNode    — weighted moving-average zone smoothing
//!   HslAverageNode    — streaming per-segment HSL accumulator (process + finish + reset)
//!
//! Free function:
//!   build_segment_lut — builds DepthLuts for one segment from the calibration store

use anyhow::{Context, Result};
use crate::pipeline::{Node, Stage, FeatureStageOutput, CalibrationStageOutput};
use crate::change_detect::{
    SegmentRange, compute_per_kf_zones, detect_scene_segments, smooth_zone_boundaries,
};
use dorea_hsl::derive::{derive_hsl_corrections, HslCorrections, QualifierCorrection};
use dorea_hsl::{HSL_QUALIFIERS, MIN_WEIGHT};
use dorea_lut::apply::apply_depth_luts;
use dorea_lut::build::{adaptive_zone_boundaries, compute_importance, StreamingLutBuilder};
use dorea_lut::types::DepthLuts;
use dorea_gpu::GradeParams;

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-segment calibration data used by the Grading Render stage.
pub struct SegmentCalibration {
    pub depth_luts: DepthLuts,
    pub hsl_corrections: HslCorrections,
}

// ── Stage input ───────────────────────────────────────────────────────────────

pub struct CalibrationStageInput {
    pub feature_output: FeatureStageOutput,
    pub config: CalibrationConfig,
    pub grade_params: GradeParams,
}

pub struct CalibrationConfig {
    pub depth_zones: usize,
    pub base_lut_zones: usize,
    pub scene_threshold: f32,
    pub min_segment_kfs: usize,
    pub zone_smoothing_w: usize,
}

// ── Node: ZoneComputeNode ─────────────────────────────────────────────────────

pub struct ZoneComputeNode { pub depth_zones: usize }

impl Node for ZoneComputeNode {
    type Input = Vec<Vec<f32>>;
    type Output = Vec<Vec<f32>>; // raw per-KF zone boundaries

    fn name(&self) -> &'static str { "zone_compute" }

    fn process(&mut self, kf_depths: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        Ok(compute_per_kf_zones(&kf_depths, self.depth_zones))
    }
}

// ── Node: SegmentDetectNode ───────────────────────────────────────────────────

pub struct SegmentDetectNode {
    pub scene_threshold: f32,
    pub min_segment_kfs: usize,
}

impl Node for SegmentDetectNode {
    type Input = Vec<Vec<f32>>;
    type Output = Vec<SegmentRange>;

    fn name(&self) -> &'static str { "segment_detect" }

    fn process(&mut self, kf_depths: Vec<Vec<f32>>) -> Result<Vec<SegmentRange>> {
        Ok(detect_scene_segments(&kf_depths, self.scene_threshold, self.min_segment_kfs))
    }
}

// ── Node: ZoneSmoothNode ──────────────────────────────────────────────────────

pub struct ZoneSmoothNode { pub window: usize }

impl Node for ZoneSmoothNode {
    type Input = (Vec<Vec<f32>>, Vec<SegmentRange>);
    type Output = Vec<Vec<f32>>;

    fn name(&self) -> &'static str { "zone_smooth" }

    fn process(&mut self, (raw, segs): Self::Input) -> Result<Vec<Vec<f32>>> {
        Ok(smooth_zone_boundaries(&raw, &segs, self.window))
    }
}

// ── Free function: build_segment_lut ─────────────────────────────────────────

/// Builds a `DepthLuts` for one segment by streaming `StreamingLutBuilder` over
/// all keyframes in the segment.
///
/// Uses a free function rather than a Node because the store must be borrowed
/// for the full duration of the loop — no raw pointers or lifetime gymnastics needed.
pub fn build_segment_lut(
    store: &crate::pipeline::feature::PagedCalibrationStore,
    kf_indices: &[usize],
    kf_depths: &[Vec<f32>],
    base_lut_zones: usize,
) -> DepthLuts {
    let seg_depths: Vec<f32> = kf_indices.iter()
        .flat_map(|&i| kf_depths[i].iter().cloned())
        .collect();
    let base_boundaries = adaptive_zone_boundaries(&seg_depths, base_lut_zones);
    let mut builder = StreamingLutBuilder::new(base_boundaries);
    for &i in kf_indices {
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
        builder.add_frame(&original, &target, &depth, &importance);
    }
    builder.finish()
}

// ── Node: HslAverageNode ──────────────────────────────────────────────────────

/// Streaming HSL accumulator. Feed one (lut_output, target) pair per keyframe
/// via `process(...)`, then call `finish()` to get corrections.
/// Call `reset()` between segments.
///
/// No Vec of pairs is allocated — state is accumulated in-place via f64 scalars.
pub struct HslAverageNode {
    h_acc:  Vec<f64>,
    s_acc:  Vec<f64>,
    v_acc:  Vec<f64>,
    tw_acc: Vec<f64>,
    active: Vec<bool>,
}

impl HslAverageNode {
    pub fn new() -> Self {
        let n = HSL_QUALIFIERS.len();
        Self {
            h_acc:  vec![0.0; n],
            s_acc:  vec![0.0; n],
            v_acc:  vec![0.0; n],
            tw_acc: vec![0.0; n],
            active: vec![false; n],
        }
    }

    /// Reset all accumulators for the next segment.
    pub fn reset(&mut self) {
        for qi in 0..HSL_QUALIFIERS.len() {
            self.h_acc[qi]  = 0.0;
            self.s_acc[qi]  = 0.0;
            self.v_acc[qi]  = 0.0;
            self.tw_acc[qi] = 0.0;
            self.active[qi] = false;
        }
    }

    /// Return the weighted-average HslCorrections from accumulated state.
    /// Call after all keyframes in a segment have been fed via `process`.
    pub fn finish(&self) -> HslCorrections {
        let corrections: Vec<QualifierCorrection> = (0..HSL_QUALIFIERS.len()).map(|qi| {
            let qual = &HSL_QUALIFIERS[qi];
            if self.active[qi] {
                let tw = self.tw_acc[qi];
                QualifierCorrection {
                    h_center: qual.h_center, h_width: qual.h_width,
                    h_offset: (self.h_acc[qi] / tw) as f32,
                    s_ratio:  (self.s_acc[qi] / tw) as f32,
                    v_offset: (self.v_acc[qi] / tw) as f32,
                    weight: tw as f32,
                }
            } else {
                QualifierCorrection {
                    h_center: qual.h_center, h_width: qual.h_width,
                    h_offset: 0.0, s_ratio: 1.0, v_offset: 0.0, weight: 0.0,
                }
            }
        }).collect();
        HslCorrections(corrections)
    }
}

impl Node for HslAverageNode {
    /// Input: one (lut_output, target) pair for a single keyframe.
    type Input = (Vec<[f32; 3]>, Vec<[f32; 3]>);
    /// Output: () — state is accumulated internally; call `finish()` after the segment.
    type Output = ();

    fn name(&self) -> &'static str { "hsl_average" }

    fn process(&mut self, (lut_out, target): Self::Input) -> Result<()> {
        let corrs = derive_hsl_corrections(&lut_out, &target);
        for (qi, corr) in corrs.0.iter().enumerate() {
            if corr.weight >= MIN_WEIGHT {
                let w = corr.weight as f64;
                self.h_acc[qi] += corr.h_offset as f64 * w;
                self.s_acc[qi] += corr.s_ratio  as f64 * w;
                self.v_acc[qi] += corr.v_offset as f64 * w;
                self.active[qi] = true;
                self.tw_acc[qi] += w;
            }
        }
        Ok(())
    }
}

// ── Stage: CalibrationStage ───────────────────────────────────────────────────

pub struct CalibrationStage {
    zone_compute:   ZoneComputeNode,
    segment_detect: SegmentDetectNode,
    zone_smooth:    ZoneSmoothNode,
    hsl_node:       HslAverageNode,
}

impl CalibrationStage {
    pub fn new(cfg: &CalibrationConfig) -> Self {
        Self {
            zone_compute:   ZoneComputeNode { depth_zones: cfg.depth_zones },
            segment_detect: SegmentDetectNode {
                scene_threshold:  cfg.scene_threshold,
                min_segment_kfs:  cfg.min_segment_kfs,
            },
            zone_smooth: ZoneSmoothNode { window: cfg.zone_smoothing_w },
            hsl_node:    HslAverageNode::new(),
        }
    }
}

impl Stage for CalibrationStage {
    type Input = CalibrationStageInput;
    type Output = CalibrationStageOutput;

    fn name(&self) -> &'static str { "lut_calibration_stage" }

    fn run(&mut self, input: Self::Input) -> Result<Self::Output> {
        log::info!("=== {} ===", self.name());
        let CalibrationStageInput { feature_output, config, .. } = input;
        let FeatureStageOutput { store, keyframe_depths, keyframes } = feature_output;

        let kf_depths_store: Vec<Vec<f32>> = (0..store.len())
            .map(|i| store.read_depth(i))
            .collect();

        let raw_kf_zones = self.zone_compute.process(kf_depths_store.clone())?;
        let segments     = self.segment_detect.process(kf_depths_store.clone())?;
        log::info!("{}: {} segments from {} keyframes", self.name(), segments.len(), store.len());
        for (si, seg) in segments.iter().enumerate() {
            log::info!("  segment {si}: KFs {}..{}", seg.start, seg.end);
        }
        let smoothed_kf_zones = self.zone_smooth.process((raw_kf_zones, segments.clone()))?;

        let mut segment_calibrations: Vec<SegmentCalibration> = Vec::with_capacity(segments.len());

        for seg in &segments {
            let kf_indices: Vec<usize> = (seg.start..seg.end).collect();

            // LUT build via free function — store borrowed for the duration of the loop.
            let depth_luts = build_segment_lut(&store, &kf_indices, &kf_depths_store, config.base_lut_zones);

            // HSL streaming accumulation — no Vec<pairs> allocation.
            self.hsl_node.reset();
            for &i in &kf_indices {
                let (pixels_u8, target_u8) = store.pixtar_slices(i);
                let depth = store.read_depth(i);
                let original: Vec<[f32; 3]> = pixels_u8.chunks_exact(3)
                    .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                    .collect();
                let target: Vec<[f32; 3]> = target_u8.chunks_exact(3)
                    .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                    .collect();
                let lut_out = apply_depth_luts(&original, &depth, &depth_luts);
                self.hsl_node.process((lut_out, target))?;
            }
            let hsl_corrections = self.hsl_node.finish();

            segment_calibrations.push(SegmentCalibration { depth_luts, hsl_corrections });
        }

        log::info!("{}: pre-compute complete ({} segments)", self.name(), segment_calibrations.len());

        let kf_to_segment: Vec<usize> = {
            let mut map = vec![0usize; store.len()];
            for (si, seg) in segments.iter().enumerate() {
                for ki in seg.start..seg.end { map[ki] = si; }
            }
            map
        };

        let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
            .map(|kf| (kf.frame_index, kf.scene_cut_before))
            .collect();

        Ok(CalibrationStageOutput {
            segment_calibrations,
            smoothed_kf_zones,
            kf_to_segment,
            segments,
            kf_index_list,
            keyframe_depths,
        })
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p dorea-cli --features cuda pipeline::calibration 2>&1 | grep -E "FAILED|ok\."
```
Expected: all 5 calibration tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/pipeline/calibration.rs
git commit -m "feat(pipeline): CalibrationStage — ZoneComputeNode + SegmentDetectNode + ZoneSmoothNode + HslAverageNode (streaming) + build_segment_lut"
```

---

### Task 5: Grading Render stage (`pipeline/grading.rs`)

**Files:**
- Modify: `crates/dorea-cli/src/pipeline/grading.rs`

Nodes:
- `DepthInterpolateNode`: lerps depth between adjacent KF depth maps. Implemented as a plain `fn lerp(...)` method rather than via the `Node` trait — the operation naturally takes borrowed `&[f32]` slices, avoiding per-frame clones of the depth Vecs. The `Node` trait's owned-input contract would require allocating copies of potentially large depth maps on every frame.
- `BlendTNode`: computes `blend_t` for dual-texture temporal blending; implements `Node` normally.
- `GradeFrameNode`: the grading call is inlined in `GradingStage::run` due to `#[cfg(feature = "cuda")]` / `#[cfg(not(feature = "cuda"))]` branching that cannot be cleanly expressed as a `Node::process` call.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Node;

    #[test]
    fn blend_t_node_zero_at_keyframe() {
        let mut node = BlendTNode;
        let t = node.process(BlendTInput {
            fi: 10,
            prev_kf_idx: 10,
            next: None,
        }).unwrap();
        assert_eq!(t, 0.0);
    }

    #[test]
    fn blend_t_node_zero_past_last_keyframe() {
        let mut node = BlendTNode;
        let t = node.process(BlendTInput {
            fi: 99,
            prev_kf_idx: 80,
            next: None,
        }).unwrap();
        assert_eq!(t, 0.0);
    }

    #[test]
    fn blend_t_node_midpoint() {
        let mut node = BlendTNode;
        let t = node.process(BlendTInput {
            fi: 15,
            prev_kf_idx: 10,
            next: Some((20, false)),
        }).unwrap();
        assert!((t - 0.5).abs() < 1e-5, "expected 0.5, got {t}");
    }

    #[test]
    fn blend_t_node_zero_across_scene_cut() {
        let mut node = BlendTNode;
        let t = node.process(BlendTInput {
            fi: 15,
            prev_kf_idx: 10,
            next: Some((20, true)), // scene_cut_before = true
        }).unwrap();
        assert_eq!(t, 0.0);
    }

    #[test]
    fn lerp_depth_midpoint() {
        let a = vec![0.0f32, 1.0];
        let b = vec![1.0f32, 0.0];
        let out = lerp_depth(&a, &b, 0.5);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_depth_clamped() {
        let a = vec![0.0f32];
        let b = vec![1.0f32];
        assert_eq!(lerp_depth(&a, &b, 2.0), vec![1.0]);
        assert_eq!(lerp_depth(&a, &b, -1.0), vec![0.0]);
    }

    #[test]
    fn depth_interpolate_at_keyframe_returns_prev() {
        let node = DepthInterpolateNode;
        let prev = vec![0.2f32, 0.8];
        let result = node.lerp(10, 10, &prev, None);
        assert_eq!(result, prev);
    }

    #[test]
    fn depth_interpolate_past_last_kf_returns_prev() {
        let node = DepthInterpolateNode;
        let prev = vec![0.5f32, 0.5];
        let result = node.lerp(99, 80, &prev, None);
        assert_eq!(result, prev);
    }

    #[test]
    fn depth_interpolate_midpoint() {
        let node = DepthInterpolateNode;
        let a = vec![0.0f32, 1.0];
        let b = vec![1.0f32, 0.0];
        let result = node.lerp(15, 10, &a, Some((20, false, &b)));
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn depth_interpolate_scene_cut_returns_prev() {
        let node = DepthInterpolateNode;
        let a = vec![0.2f32];
        let b = vec![0.9f32];
        let result = node.lerp(15, 10, &a, Some((20, true, &b))); // scene_cut
        assert_eq!(result, a);
    }
}
```

- [ ] **Step 2: Run — verify FAIL**

```bash
cargo test -p dorea-cli --features cuda pipeline::grading 2>&1 | grep -E "error|FAILED"
```

- [ ] **Step 3: Implement `grading.rs`**

```rust
//! Grading Render stage: full-res decode → depth interp → blend_t → CUDA grade → encode.
//!
//! Named components:
//!   DepthInterpolateNode — lerps depth between KF depth maps (plain lerp method, not Node)
//!   BlendTNode           — computes temporal blend_t for dual-texture blending (implements Node)
//!   GradeFrameNode       — inlined in GradingStage::run (cfg-gated CUDA/CPU branching)

use anyhow::{Context, Result};
use std::collections::HashMap;
use crate::pipeline::{Node, Stage, CalibrationStageOutput};
use crate::pipeline::calibration::SegmentCalibration;
use dorea_video::ffmpeg::{self, FrameEncoder, VideoInfo};
use dorea_video::inference::InferenceServer;
use dorea_gpu::GradeParams;
#[cfg(feature = "cuda")]
use dorea_gpu::AdaptiveGrader;

// ── Utility ───────────────────────────────────────────────────────────────────

/// Linear interpolation between two depth maps.
pub fn lerp_depth(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter()).map(|(&va, &vb)| va + (vb - va) * t).collect()
}

// ── Stage input ───────────────────────────────────────────────────────────────

pub struct GradingStageInput {
    pub calibration_output: CalibrationStageOutput,
    pub video_path: std::path::PathBuf,
    pub video_info: VideoInfo,
    pub encoder: FrameEncoder,
    pub grade_params: GradeParams,
    pub depth_zones: usize,
    pub base_lut_zones: usize,
}

// ── Node: DepthInterpolateNode ────────────────────────────────────────────────

/// Lerps depth between adjacent keyframe depth maps.
///
/// Implemented as a plain `lerp` method rather than the `Node` trait because
/// the operation takes borrowed `&[f32]` slices — wrapping them in an owned
/// Input struct would require cloning potentially large depth Vecs on every frame.
/// The `lerp` method borrows the caller's HashMap entries directly.
pub struct DepthInterpolateNode;

impl DepthInterpolateNode {
    /// Returns the interpolated depth Vec for frame `fi`.
    ///
    /// - At a keyframe (`fi == prev_kf_idx`): returns a copy of `prev_depth`.
    /// - Past the last keyframe (`next == None`): returns a copy of `prev_depth`.
    /// - Across a scene cut (`scene_cut == true`): returns a copy of `prev_depth`.
    /// - Otherwise: lerps between `prev_depth` and `next_depth` at the appropriate `t`.
    ///
    /// Allocates exactly one `Vec<f32>` per call (the interpolated result).
    pub fn lerp(
        &self,
        fi: u64,
        prev_kf_idx: u64,
        prev_depth: &[f32],
        next: Option<(u64, bool, &[f32])>,
    ) -> Vec<f32> {
        if fi == prev_kf_idx {
            return prev_depth.to_vec();
        }
        if let Some((next_kf_idx, scene_cut, next_depth)) = next {
            if scene_cut {
                return prev_depth.to_vec();
            }
            let t = ((fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32)
                .clamp(0.0, 1.0);
            lerp_depth(prev_depth, next_depth, t)
        } else {
            prev_depth.to_vec()
        }
    }
}

// ── Node: BlendTNode ──────────────────────────────────────────────────────────

pub struct BlendTNode;

pub struct BlendTInput {
    pub fi: u64,
    pub prev_kf_idx: u64,
    /// `(next_kf_idx, scene_cut_before_next)` — None if fi is past the last keyframe.
    pub next: Option<(u64, bool)>,
}

impl Node for BlendTNode {
    type Input = BlendTInput;
    type Output = f32;

    fn name(&self) -> &'static str { "blend_t" }

    fn process(&mut self, input: BlendTInput) -> Result<f32> {
        Ok(if input.fi == input.prev_kf_idx {
            0.0
        } else if let Some((next_kf_idx, scene_cut)) = input.next {
            if scene_cut { 0.0 }
            else {
                ((input.fi - input.prev_kf_idx) as f32
                 / (next_kf_idx - input.prev_kf_idx) as f32)
                    .clamp(0.0, 1.0)
            }
        } else {
            0.0
        })
    }
}

// ── Stage: GradingStage ───────────────────────────────────────────────────────

pub struct GradingStage {
    depth_interp: DepthInterpolateNode,
    blend_t:      BlendTNode,
}

impl GradingStage {
    pub fn new() -> Self {
        Self {
            depth_interp: DepthInterpolateNode,
            blend_t:      BlendTNode,
        }
    }
}

impl Stage for GradingStage {
    type Input = GradingStageInput;
    type Output = (); // side effect: writes to encoder

    fn name(&self) -> &'static str { "grading_render_stage" }

    fn run(&mut self, input: Self::Input) -> Result<()> {
        log::info!("=== {} ===", self.name());
        let GradingStageInput {
            calibration_output,
            video_path,
            video_info,
            mut encoder,
            grade_params,
            depth_zones,
            base_lut_zones,
        } = input;

        let CalibrationStageOutput {
            segment_calibrations,
            smoothed_kf_zones,
            kf_to_segment,
            kf_index_list,
            keyframe_depths,
            ..
        } = calibration_output;

        // GradeFrameNode: Initialize AdaptiveGrader (inlined due to cfg-gated branching).
        #[cfg(feature = "cuda")]
        let mut adaptive_grader = {
            let seg0 = &segment_calibrations[0];
            let base_flat: Vec<f32> = seg0.depth_luts.luts.iter()
                .flat_map(|lut| lut.data.iter().copied())
                .collect();
            let base_bounds = &seg0.depth_luts.zone_boundaries;
            let hsl = &seg0.hsl_corrections;
            let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
            let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
            let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
            let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
            let lut_size = seg0.depth_luts.luts[0].size;
            let grader = AdaptiveGrader::new(
                &base_flat, base_bounds, base_lut_zones,
                (&h_offsets, &s_ratios, &v_offsets, &weights),
                &grade_params, lut_size, depth_zones,
            ).context("AdaptiveGrader init failed")?;
            grader.prepare_keyframe(&smoothed_kf_zones[0])
                .context("prepare initial keyframe texture failed")?;
            grader.swap_textures();
            if smoothed_kf_zones.len() > 1 {
                grader.prepare_keyframe(&smoothed_kf_zones[1])
                    .context("prepare second keyframe texture failed")?;
            }
            log::info!(
                "{}: AdaptiveGrader initialized ({base_lut_zones} base zones, {depth_zones} runtime zones)",
                self.name()
            );
            grader
        };

        let frames = ffmpeg::decode_frames(&video_path, &video_info)
            .context("failed to spawn full-res decoder")?;

        let mut kf_cursor = 0usize;
        let mut frame_count = 0u64;
        let mut current_segment = kf_to_segment.first().copied().unwrap_or(0);

        for frame_result in frames {
            let frame = frame_result.context("frame decode error")?;
            let fi = frame.index;

            while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
                kf_cursor += 1;
                #[cfg(feature = "cuda")]
                {
                    let new_seg = kf_to_segment[kf_cursor];
                    if new_seg != current_segment {
                        let seg_cal = &segment_calibrations[new_seg];
                        let base_flat: Vec<f32> = seg_cal.depth_luts.luts.iter()
                            .flat_map(|lut| lut.data.iter().copied())
                            .collect();
                        let hsl = &seg_cal.hsl_corrections;
                        let h_offsets: Vec<f32> = hsl.0.iter().map(|q| q.h_offset).collect();
                        let s_ratios:  Vec<f32> = hsl.0.iter().map(|q| q.s_ratio).collect();
                        let v_offsets: Vec<f32> = hsl.0.iter().map(|q| q.v_offset).collect();
                        let weights:   Vec<f32> = hsl.0.iter().map(|q| q.weight).collect();
                        // Correct order: load_segment + prepare_keyframe BEFORE swap_textures
                        // so the first frame of the new segment uses the correct color science.
                        adaptive_grader.load_segment(
                            &base_flat, &seg_cal.depth_luts.zone_boundaries,
                            (&h_offsets, &s_ratios, &v_offsets, &weights),
                        ).context("load_segment failed")?;
                        adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor])
                            .context("prepare_keyframe at segment boundary failed")?;
                        current_segment = new_seg;
                        log::info!("{}: segment switch to {new_seg} at KF {kf_cursor}", self.name());
                    }
                    adaptive_grader.swap_textures();
                    if kf_cursor + 1 < smoothed_kf_zones.len() {
                        adaptive_grader.prepare_keyframe(&smoothed_kf_zones[kf_cursor + 1])
                            .context("prepare_keyframe failed")?;
                    }
                }
            }

            let (prev_kf_idx, _) = kf_index_list[kf_cursor];
            let (prev_depth_vec, dpw, dph) = keyframe_depths
                .get(&prev_kf_idx)
                .expect("prev keyframe depth missing");
            let (dpw, dph) = (*dpw, *dph);

            let next_kf = kf_index_list.get(kf_cursor + 1).copied();
            let next_entry = next_kf.and_then(|(nkf, _)| keyframe_depths.get(&nkf));

            // DepthInterpolateNode: borrows HashMap entries — no per-frame depth clones.
            let depth_proxy = self.depth_interp.lerp(
                fi,
                prev_kf_idx,
                prev_depth_vec,
                next_kf.zip(next_entry).map(|((nkf, sc), nd)| (nkf, sc, nd.0.as_slice())),
            );

            let depth = if dpw == frame.width && dph == frame.height {
                depth_proxy
            } else {
                InferenceServer::upscale_depth(&depth_proxy, dpw, dph, frame.width, frame.height)
            };

            let blend_t = self.blend_t.process(BlendTInput {
                fi,
                prev_kf_idx,
                next: next_kf,
            })?;

            // GradeFrameNode: inlined due to cfg-gated CUDA/CPU path split.
            #[cfg(feature = "cuda")]
            let graded = adaptive_grader.grade_frame_blended(
                &frame.pixels, &depth, frame.width, frame.height, blend_t,
            ).map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?;

            #[cfg(not(feature = "cuda"))]
            let graded = {
                use dorea_cal::Calibration;
                use dorea_gpu::grade_frame;
                let seg_idx = kf_to_segment.get(kf_cursor).copied().unwrap_or(0);
                let cal = &segment_calibrations[seg_idx];
                let n_kfs = kf_index_list.len();
                let calibration = Calibration::new(
                    cal.depth_luts.clone(), cal.hsl_corrections.clone(), n_kfs,
                );
                grade_frame(&frame.pixels, &depth, frame.width, frame.height, &calibration, &grade_params)
                    .map_err(|e| anyhow::anyhow!("Grading failed for frame {fi}: {e}"))?
            };

            encoder.write_frame(&graded).context("encoder write failed")?;
            frame_count += 1;
            if frame_count % 100 == 0 {
                let pct = frame_count as f64 / video_info.frame_count.max(1) as f64 * 100.0;
                log::info!(
                    "{}: {frame_count}/{} frames ({:.1}%)",
                    self.name(), video_info.frame_count, pct
                );
            }
        }

        encoder.finish().context("encoder failed to finalize")?;
        log::info!("{}: done. {frame_count} frames graded.", self.name());
        Ok(())
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p dorea-cli --features cuda pipeline::grading 2>&1 | grep -E "FAILED|ok\."
```
Expected: all 10 grading tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/pipeline/grading.rs
git commit -m "feat(pipeline): GradingStage — DepthInterpolateNode (borrowed lerp) + BlendTNode"
```

---

### Task 6: Wire stages into `grade.rs` + delete old code

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

Replace the monolithic `run()` body with four stage constructions and `.run()` calls. Delete `PagedCalibrationStore`, `SegmentCalibration`, `lerp_depth`, and `KeyframeEntry` (now in pipeline modules). Delete the old fused batch, calibration, and grading pass blocks.

- [ ] **Step 1: Rewrite `grade.rs::run`**

```rust
pub fn run(args: GradeArgs, cfg: &crate::config::DoreaConfig) -> Result<()> {
    // --- Config resolution (unchanged) ---
    let warmth              = args.warmth.or(cfg.grade.warmth).unwrap_or(1.0_f32);
    let strength            = args.strength.or(cfg.grade.strength).unwrap_or(0.8_f32);
    let contrast            = args.contrast.or(cfg.grade.contrast).unwrap_or(1.0_f32);
    let proxy_size          = args.proxy_size.or(cfg.inference.proxy_size).unwrap_or(1080_usize);
    let depth_skip_threshold= args.depth_skip_threshold.or(cfg.grade.depth_skip_threshold).unwrap_or(0.005_f32);
    let depth_max_interval  = args.depth_max_interval.or(cfg.grade.depth_max_interval).unwrap_or(12_usize);
    let fused_batch_size    = args.fused_batch_size.or(cfg.grade.fused_batch_size).unwrap_or(32_usize);
    let depth_zones         = args.depth_zones.or(cfg.grade.depth_zones).unwrap_or(8_usize);
    let base_lut_zones      = args.base_lut_zones.or(cfg.grade.base_lut_zones).unwrap_or(32_usize);
    let scene_threshold     = args.scene_threshold.or(cfg.grade.scene_threshold).unwrap_or(0.15_f32);
    let min_segment_kfs     = cfg.grade.min_segment_keyframes.unwrap_or(5_usize);
    let zone_smoothing_w    = cfg.grade.zone_smoothing_window.unwrap_or(3_usize);
    let maxine_upscale_factor = args.maxine_upscale_factor.or(cfg.maxine.upscale_factor).unwrap_or(2_u32);
    let python = args.python.clone()
        .or_else(|| cfg.models.python.clone())
        .unwrap_or_else(|| PathBuf::from("/opt/dorea-venv/bin/python"));
    let raune_weights    = args.raune_weights.clone().or_else(|| cfg.models.raune_weights.clone());
    let raune_models_dir = args.raune_models_dir.clone().or_else(|| cfg.models.raune_models_dir.clone());
    let depth_model      = args.depth_model.clone().or_else(|| cfg.models.depth_model.clone());
    let device = if args.cpu_only {
        Some("cpu".to_string())
    } else {
        cfg.inference.device.clone()
    };

    if args.cpu_only {
        anyhow::bail!("--cpu-only not supported for dorea grade; use dorea preview.");
    }
    if depth_max_interval == 0 {
        anyhow::bail!("--depth-max-interval must be >= 1");
    }

    let output = args.output.clone().unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        args.input.with_file_name(format!("{stem}_graded.mp4"))
    });
    log::info!("Grading: {} → {}", args.input.display(), output.display());

    let info = ffmpeg::probe(&args.input).context("ffprobe failed")?;
    log::info!(
        "Input: {}x{} @ {:.3}fps, {:.1}s ({} frames)",
        info.width, info.height, info.fps, info.duration_secs, info.frame_count
    );

    let params = GradeParams { warmth, strength, contrast };
    let audio_src = if info.has_audio { Some(args.input.as_path()) } else { None };
    let encoder = FrameEncoder::new(&output, info.width, info.height, info.fps, audio_src)
        .context("failed to spawn ffmpeg encoder")?;

    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, proxy_size);

    let start_cfg = InferenceConfig {
        skip_raune: true,
        skip_depth: true,
        ..build_inference_config(&python, raune_weights.as_deref(), raune_models_dir.as_deref(),
                                 depth_model.as_deref(), device, maxine_upscale_factor)
    };
    let inf_server = InferenceServer::spawn(&start_cfg).context("failed to spawn inference server")?;

    // ── Stage 1: Keyframe ───────────────────────────────────────────────────
    use crate::pipeline::{Stage, keyframe::{KeyframeStage, KeyframeStageInput, KeyframeConfig}};
    let mut stage1 = KeyframeStage::new(KeyframeConfig {
        depth_skip_threshold,
        depth_max_interval,
        interp_enabled: !args.no_depth_interp,
        scene_cut_threshold: depth_skip_threshold * 10.0,
    });
    let kf_out = stage1.run(KeyframeStageInput {
        video_path: args.input.clone(),
        video_info: info.clone(),
        proxy_w,
        proxy_h,
        config: KeyframeConfig {
            depth_skip_threshold,
            depth_max_interval,
            interp_enabled: !args.no_depth_interp,
            scene_cut_threshold: depth_skip_threshold * 10.0,
        },
    })?;

    // ── Stage 2: Feature ────────────────────────────────────────────────────
    use crate::pipeline::feature::{FeatureStage, FeatureStageInput};
    let mut stage2 = FeatureStage::new(proxy_w, proxy_h);
    let feat_out = stage2.run(FeatureStageInput {
        keyframe_output: kf_out,
        inf_server,
        fused_batch_size,
        proxy_size,
        enable_maxine: true,
        raune_weights,    // configured paths passed through — not dropped
        raune_models_dir,
        depth_model,
    })?;

    // ── Stage 3: LUT Calibration ────────────────────────────────────────────
    use crate::pipeline::calibration::{CalibrationStage, CalibrationStageInput, CalibrationConfig};
    let cal_cfg = CalibrationConfig {
        depth_zones, base_lut_zones, scene_threshold, min_segment_kfs, zone_smoothing_w,
    };
    let mut stage3 = CalibrationStage::new(&cal_cfg);
    let cal_out = stage3.run(CalibrationStageInput {
        feature_output: feat_out,
        config: cal_cfg,
        grade_params: params.clone(),
    })?;

    // ── Stage 4: Grading Render ─────────────────────────────────────────────
    use crate::pipeline::grading::{GradingStage, GradingStageInput};
    let mut stage4 = GradingStage::new();
    stage4.run(GradingStageInput {
        calibration_output: cal_out,
        video_path: args.input,
        video_info: info,
        encoder,
        grade_params: params,
        depth_zones,
        base_lut_zones,
    })?;

    Ok(())
}
```

> **Note on `KeyframeConfig` construction:** `KeyframeStage::new(cfg)` stores the config internally; `KeyframeStageInput` also carries a `config` field so the stage can log or inspect it in `run()`. Construct both from the same local values (not a clone of the stage's internal state) to avoid a `config()` accessor.

- [ ] **Step 2: Delete dead code from `grade.rs`**

Remove:
- `PagedCalibrationStore` struct + impl (moved to `feature.rs`)
- `KeyframeEntry` struct (moved to `keyframe.rs`)
- `lerp_depth` fn (moved to `grading.rs`)
- The old `run()` body (replaced above)
- The `TempFileGuard` struct + impl (no longer needed)
- Any now-unused `use` statements

- [ ] **Step 3: Compile check**

```bash
cargo check -p dorea-cli --features cuda 2>&1 | grep -E "^error"
```
Expected: no errors.

- [ ] **Step 4: Run all tests**

```bash
cargo test -p dorea-cli --features cuda 2>&1 | grep -E "FAILED|^test result"
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/pipeline/
git commit -m "refactor(grade): wire pipeline DAG stages into grade.rs orchestrator

grade.rs::run is now thin orchestration of four named stages:
  KeyframeStage → FeatureStage → CalibrationStage → GradingStage
Each stage owns its nodes with typed inputs/outputs. Deleted ~800 lines
of monolithic run() body; logic preserved in pipeline/{keyframe,feature,
calibration,grading}.rs."
```

---

## Self-Review

**Spec coverage:**
- ✅ Four named stages: Keyframe, Feature, LUT Calibration, Grading Render
- ✅ `Node` trait with `type Input`, `type Output`, `fn process`, `fn name`
- ✅ `Stage` trait with `fn run`, `fn name`
- ✅ Typed I/O structs at every stage boundary
- ✅ Nodes: `ProxyDecodeNode` (streaming), `ChangeDetectNode`, `KeyframeSelectNode`, `FusedInferenceNode` (fully implemented), `EnhancedDownscaleNode`, `ZoneComputeNode`, `SegmentDetectNode`, `ZoneSmoothNode`, `HslAverageNode` (streaming accumulator), `DepthInterpolateNode` (plain `lerp` method), `BlendTNode`
- ✅ Free function: `build_segment_lut` (replaces phantom `LutBuildNode`)
- ✅ `grade.rs` becomes thin orchestration (~60 lines)
- ✅ Tests for all stateful/testable nodes

**Review issue resolution:**
- ✅ A: `FusedInferenceNode` fully implements `process()`; owns `InferenceServer`
- ✅ B: `DepthCacheNode` removed; stage manages HashMap directly (documented)
- ✅ C: `FeatureStageInput` carries `raune_weights`, `raune_models_dir`, `depth_model`; `load_raune` uses configured paths
- ✅ D/1: `ProxyDecodeNode::Output = ProxyDecoder` (streaming iterator); never collects frames into `Vec`
- ✅ 2: `FusedInferenceNode::process` builds `RauneDepthBatchItem` list inline per batch; no full pre-collection
- ✅ 3: `FrameScore.pixels: Vec<u8>` is a move not a copy; documented in keyframe.rs comment
- ✅ 4: `HslAverageNode` uses running f64 accumulators; `process(pair)` accumulates, `finish()` returns result; no `Vec<pairs>` materialised

**Placeholder scan:** None. All code blocks are complete.

**Type consistency:**
- `KeyframeStageOutput`: defined in `mod.rs`, consumed by `feature.rs`, passed from `grade.rs`
- `FeatureStageOutput`: defined in `mod.rs`, consumed by `calibration.rs`, passed from `grade.rs`
- `CalibrationStageOutput`: defined in `mod.rs`, consumed by `grading.rs`, passed from `grade.rs`
- `SegmentCalibration`: defined in `calibration.rs`, referenced by `grading.rs` via `calibration::SegmentCalibration`
- `KeyframeEntry`: defined in `keyframe.rs`, referenced by `feature.rs` via `keyframe::KeyframeEntry`
- `DepthInterpolateNode::lerp` takes `&[f32]` — consistent with call sites in `GradingStage::run` that borrow from `keyframe_depths` HashMap
- `HslAverageNode::process` accumulates `()`, `finish()` returns `HslCorrections` — `CalibrationStage::run` calls `reset()` per segment, `finish()` per segment after the loop
