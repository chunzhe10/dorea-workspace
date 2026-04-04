# Unified Keyframe Sampling — Design Spec

**Status:** Design approved, pending implementation plan  
**Depends on:** `docs/plans/2026-04-03-inference-gpu-saturation.md` (GPU saturation / fused RAUNE+depth batch — must land first)  
**Closes:** (issue TBD — not yet filed)  
**Related:** #41 (validate depth-on-RAUNE), #44 (FUSED_BATCH_SIZE tuning), #45 (streaming decode)

---

## Problem

The current `dorea grade` pipeline runs two independent keyframe selection passes:

| Pass | Selection method | Keyframes | Inference |
|---|---|---|---|
| `auto_calibrate` | Uniform `--keyframe-interval` | ~988 (at interval 5) | RAUNE + depth, sequential |
| Grading pass 1 | MSE-adaptive `--depth-max-interval` | ~941 (at interval 5) | depth-only batch |

Two proxy decodes. Two inference servers. Two independent keyframe sets. Depth inferred twice on different frames. The calibration LUT is built from frames the grading pass never even looked at.

---

## Goal

One change detection pass → one keyframe list → one fused RAUNE+depth inference run → results feed **both** calibration **and** the grading depth cache.

---

## Architecture

```
Pass 1: proxy decode → ChangeDetector.score() → Vec<KeyframeEntry>
                                ↓
        fused RAUNE+depth batch (single InferenceServer)
                /                          \
  PagedCalibrationStore              keyframe_depths HashMap
  (orig, enhanced, depth)            (depth at proxy res, keyed by frame_index)
                ↓
  3-pass calibration streaming → Calibration
                ↓
Pass 2: full-res decode → grade (Calibration + keyframe_depths)
```

Properties:
- One proxy decode
- One inference server startup (RAUNE + depth enabled)
- One fused inference run — both outputs consumed simultaneously
- `auto_calibrate()` function deleted
- No second server startup before grading pass
- `--keyframe-interval` CLI arg removed

When `--calibration` is supplied, the fused inference path is skipped entirely. A depth-only server starts for grading (same as today).

---

## Change Detection Abstraction

New file: `crates/dorea-cli/src/change_detect.rs`

```rust
/// Produces a change score [0.0, ∞) for each frame relative to the previous.
/// 0.0 = identical. Higher = more change.
/// Implementations must be stateful (hold previous frame for comparison).
pub trait ChangeDetector: Send {
    fn score(&mut self, pixels: &[u8], width: usize, height: usize) -> f32;
    /// Called on scene cut — resets internal state so the next frame is
    /// treated as having no valid predecessor.
    fn reset(&mut self);
}

pub struct MseDetector {
    last_frame: Option<Vec<u8>>,
}

impl ChangeDetector for MseDetector {
    fn score(&mut self, pixels: &[u8], _width: usize, _height: usize) -> f32 {
        let score = self.last_frame.as_ref()
            .map(|lp| frame_mse(pixels, lp))
            .unwrap_or(f32::MAX);
        self.last_frame = Some(pixels.to_vec());
        score
    }
    fn reset(&mut self) {
        self.last_frame = None;
    }
}
```

The keyframe *decision* stays explicit in `run()`:

```rust
let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());

// In the pass 1 loop:
let change = detector.score(&frame.pixels, proxy_w, proxy_h);
let scene_cut = change > args.depth_skip_threshold * 10.0;
let is_keyframe = last_kf.is_none()
    || scene_cut
    || change > args.depth_skip_threshold
    || frames_since_kf >= args.depth_max_interval;
if scene_cut { detector.reset(); }
```

When optical flow is added, it becomes `Box::new(OpticalFlowDetector::new(...))` — the decision logic and all downstream code are unchanged.

---

## Fused Batch → Dual Output

```rust
// In run(), after pass 1 keyframe collection:
let inf_cfg = build_inference_config(&args);  // skip_raune: false
let mut inf_server = InferenceServer::spawn(&inf_cfg)?;

let fused_items: Vec<RauneDepthBatchItem> = keyframes.iter().map(|kf| {
    RauneDepthBatchItem {
        id: format!("kf_f{}", kf.frame_index),
        pixels: kf.proxy_pixels.clone(),
        width: proxy_w,
        height: proxy_h,
        raune_max_size: proxy_w,        // proxy res; tunable later via --raune-proxy-size
        depth_max_size: args.proxy_size, // 518 default
    }
}).collect();

let mut store = PagedCalibrationStore::new()?;
let mut keyframe_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

for (chunk_kfs, chunk_items) in keyframes
    .chunks(FUSED_BATCH_SIZE)
    .zip(fused_items.chunks(FUSED_BATCH_SIZE))
{
    let results = inf_server.run_raune_depth_batch(chunk_items)
        .unwrap_or_else(|e| {
            log::warn!("Fused batch failed: {e} — padding with originals + uniform depth");
            // fallback: original pixels, flat 0.5 depth
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
        store.push(&kf.proxy_pixels, &enhanced, &depth, enh_w, enh_h)?;
        keyframe_depths.insert(kf.frame_index, (depth, dw, dh));
        // store.push borrows &depth; HashMap insert moves the owned Vec
    }
}

inf_server.shutdown()?;   // VRAM freed; both calibration passes and pass 2 are CPU-only

// 3-pass calibration streaming (reservoir → LUT → HSL) — unchanged from current impl
// Pass 2 (full-res grading) — unchanged from current impl
```

---

## Resolution

RAUNE runs at proxy resolution (`--proxy-size`, default 518px) in the unified path. Current `auto_calibrate` uses 1024px for RAUNE. The quality difference is tracked under issue #41. A `--raune-proxy-size` override can be added if the delta is significant. For now proxy res is the default — the architecture is resolution-agnostic.

`FUSED_BATCH_SIZE = 8` retained conservatively (issue #44 tracks tuning).

---

## CLI Changes

| Arg | Change |
|---|---|
| `--keyframe-interval` | **Removed** — subsumed by `--depth-max-interval` |
| `--depth-skip-threshold` | Unchanged — governs both change detection and scene cut |
| `--depth-max-interval` | Unchanged — caps interval between keyframes |
| `--no-depth-interp` | Unchanged — forces every frame as a keyframe |

---

## Files Changed

| File | Change |
|---|---|
| `crates/dorea-cli/src/change_detect.rs` | **New** — `ChangeDetector` trait + `MseDetector` |
| `crates/dorea-cli/src/grade.rs` | Major refactor: delete `auto_calibrate`, merge into `run()`, use `ChangeDetector` in pass 1, add dual-output fused batch block, move `PagedCalibrationStore` init and 3-pass calibration inline |
| `crates/dorea-cli/src/main.rs` | Remove `--keyframe-interval` from CLI args (if declared there); `mod change_detect` |
| `crates/dorea-cli/Cargo.toml` | No new deps |

Files from the GPU saturation plan (prerequisite, not re-listed here):
- `python/dorea_inference/raune_net.py`
- `python/dorea_inference/depth_anything.py`
- `python/dorea_inference/protocol.py`
- `python/dorea_inference/server.py`
- `crates/dorea-video/src/inference_subprocess.rs`

---

## What This Closes / Enables

| Issue | Impact |
|---|---|
| Duplicate depth inference | Eliminated — one inference run feeds both outputs |
| `--keyframe-interval` vs `--depth-max-interval` divergence | Eliminated — single param set |
| Issue #45 (N ffmpeg seeks in auto_calibrate) | Eliminated — proxy decode already scans whole video |
| Issue #41 (validate depth-on-RAUNE) | Unblocked — architecture now always uses RAUNE-enhanced depth |
| Issue #44 (FUSED_BATCH_SIZE tuning) | Unblocked — single batch path to tune |
| Optical flow change detection | Unblocked — `ChangeDetector` trait ready |
