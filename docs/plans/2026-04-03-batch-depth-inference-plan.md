# Batch Depth Inference — Implementation Plan

**Issue:** chunzhe10/dorea#34  
**Branch:** `feat/batch-depth-inference`  
**Date:** 2026-04-03

## Problem

After PR #33 (combined LUT), GPU grading is no longer the bottleneck.  
Depth inference is: ~300 sequential `run_depth()` calls at ~0.35s each = ~70–100s of the ~127s grading phase.

Benchmark (4K 120fps 14s, 1671 frames):
- Total: 2m7s
- Auto-calibration: ~35s  
- Per-frame grading incl. depth: ~1m49s  
- GPU grading kernel only: <5s

## Solution

**Two-pass grading with batched depth inference.**

1. **Pass 1 (proxy decode):** Decode video at proxy resolution, run MSE keyframe detection, collect proxy images for all keyframes.
2. **Batch depth inference:** Send all keyframe images in a single `depth_batch` IPC request. Python server calls `infer_batch()` — one GPU forward pass for all keyframes.
3. **Pass 2 (full-res grade):** Decode at full res, look up pre-computed depth by frame index, lerp between keyframes, grade + encode.

Expected: depth inference ~0.5s (one batched forward pass) vs ~70–100s (sequential).

## Protocol

**Request (Rust → Python):**
```json
{
  "type": "depth_batch",
  "items": [
    {"id": "kf_000042", "image_b64": "<base64 raw RGB>", "format": "raw_rgb",
     "width": 518, "height": 291, "max_size": 518},
    ...
  ]
}
```

**Response (Python → Rust):**
```json
{
  "type": "depth_batch_result",
  "results": [
    {"id": "kf_000042", "depth_f32_b64": "<base64 raw f32 LE>", "width": 518, "height": 291},
    ...
  ]
}
```

Results are in the same order as items. Each depth map normalized independently to [0,1].

## Task A — Python side

**Files:** `python/dorea_inference/depth_anything.py`, `bridge.py`, `protocol.py`, `server.py`

### `depth_anything.py`: add `infer_batch`

```python
def infer_batch(self, imgs: list[np.ndarray], max_size: int = 518) -> list[np.ndarray]:
    """Run depth estimation on a batch of uint8 HxWx3 RGB images.

    All images must have the same source dimensions (guaranteed when called
    with proxy-size keyframes from a single video). Stacks into one
    (N, 3, H, W) forward pass. Each output normalized independently to [0,1].

    Returns list of (H, W) float32 depth maps at inference resolution.
    Falls back to sequential infer() if images have different post-resize dims.
    """
```

Implementation:
1. Resize all images with `_resize_for_depth(pil, max_size)`
2. Check all have same (tw, th) — if not, fallback to `[self.infer(img, max_size) for img in imgs]`
3. Stack: `tensor = torch.stack([...]).to(self.device)` shape `(N, 3, H, W)`
4. One forward pass: `outputs = self.model(pixel_values=tensor)`
5. Split + normalize each: `outputs.predicted_depth` is `(N, H, W)` — normalize each slice independently

### `bridge.py`: add `run_depth_batch_cpu`

```python
def run_depth_batch_cpu(imgs: list[np.ndarray], max_size: int = 518) -> list[np.ndarray]:
    """Run batch depth inference, returning list of numpy arrays."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded")
    return _depth_model.infer_batch(imgs, max_size=max_size)
```

### `protocol.py`: add `DepthBatchResult`

```python
@dataclass
class DepthBatchResult:
    results: list  # list of DepthResult.to_dict()
    type: str = "depth_batch_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "results": self.results}
```

### `server.py`: add `depth_batch` handler

```python
elif req_type == "depth_batch":
    if depth_model is None:
        raise RuntimeError("Depth Anything model not loaded")
    items = req.get("items", [])
    imgs = []
    for item in items:
        fmt = item.get("format", "png")
        if fmt == "raw_rgb":
            imgs.append(decode_raw_rgb(item["image_b64"], int(item["width"]), int(item["height"])))
        else:
            imgs.append(decode_png(item["image_b64"]))
    max_size = int(items[0].get("max_size", 518)) if items else 518
    depths = depth_model.infer_batch(imgs, max_size=max_size)
    results = [
        DepthResult.from_array(item.get("id"), depth).to_dict()
        for item, depth in zip(items, depths)
    ]
    resp = DepthBatchResult(results=results)
```

Also import `DepthBatchResult` in server.py.

---

## Task B — Rust IPC side

**Files:** `crates/dorea-video/src/inference_subprocess.rs`, `crates/dorea-video/src/inference/pyo3_backend.rs`

### `inference_subprocess.rs`: add `DepthBatchItem` + `run_depth_batch`

```rust
pub struct DepthBatchItem {
    pub id: String,
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub max_size: usize,
}

// On InferenceServer:
pub fn run_depth_batch(
    &mut self,
    items: &[DepthBatchItem],
) -> Result<Vec<(String, Vec<f32>, usize, usize)>, InferenceError> {
    // Encode each item, send one JSON request, parse one response
    // Returns Vec<(id, depth_f32, width, height)> in same order
}
```

### `pyo3_backend.rs`: add same `DepthBatchItem` + `run_depth_batch`

```rust
pub fn run_depth_batch(
    &self,
    items: &[DepthBatchItem],
) -> Result<Vec<(String, Vec<f32>, usize, usize)>, InferenceError> {
    // Build list of numpy arrays
    // Call bridge.run_depth_batch_cpu(imgs, max_size)
    // Decode each result
}
```

---

## Task C — Two-pass refactor

**Files:** `crates/dorea-video/src/ffmpeg.rs`, `crates/dorea-cli/src/grade.rs`

### `ffmpeg.rs`: add `decode_frames_scaled`

```rust
/// Decode all frames from a video file at a custom (scaled) resolution.
///
/// Uses ffmpeg -vf scale=WxH. Useful for proxy-resolution passes.
pub fn decode_frames_scaled(
    input: &Path,
    info: &VideoInfo,
    width: usize,
    height: usize,
) -> Result<impl Iterator<Item = Result<Frame, FfmpegError>>, FfmpegError> { ... }
```

New internal `spawn_decoder_at(input, width, height)` that adds `-vf scale=WxH` to both hw and sw decode paths. If `width == info.width && height == info.height`, delegates to existing `spawn_decoder`.

### `grade.rs`: refactor `run()` to two-pass

**Remove:**
- `BufferedFrame` struct
- `flush_buffer_with_depth` function
- `frame_buffer: Vec<BufferedFrame>`
- `last_keyframe_proxy`, `frames_since_keyframe`, `max_buffer` variables

**Add:**
```rust
const DEPTH_BATCH_SIZE: usize = 32;

struct KeyframeEntry {
    frame_index: u64,
    proxy_pixels: Vec<u8>,
    scene_cut_before: bool,
}
```

**Pass 1 — proxy decode + keyframe detection:**
```rust
let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);
let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)?;

let mut keyframes: Vec<KeyframeEntry> = Vec::new();
let mut last_proxy: Option<Vec<u8>> = None;
let mut frames_since_kf = 0usize;

for frame_result in proxy_frames {
    let frame = frame_result?;
    let mse = last_proxy.as_ref().map(|lp| frame_mse(&frame.pixels, lp));
    let scene_cut = mse.map_or(false, |m| m > scene_cut_threshold);
    let is_keyframe = !interp_enabled
        || last_proxy.is_none()
        || scene_cut
        || frames_since_kf >= args.depth_max_interval
        || mse.map_or(false, |m| m > args.depth_skip_threshold);

    if is_keyframe {
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
```

**Batch depth inference:**
```rust
let batch_items: Vec<DepthBatchItem> = keyframes.iter().enumerate().map(|(i, kf)| {
    DepthBatchItem {
        id: format!("kf_{i:06}"),
        pixels: kf.proxy_pixels.clone(),
        width: proxy_w,
        height: proxy_h,
        max_size: args.proxy_size,
    }
}).collect();

let mut keyframe_depths: HashMap<u64, Vec<f32>> = HashMap::new();
for (chunk_kfs, chunk_items) in keyframes.chunks(DEPTH_BATCH_SIZE)
    .zip(batch_items.chunks(DEPTH_BATCH_SIZE))
{
    let results = inf_server.run_depth_batch(chunk_items)
        .unwrap_or_else(|e| {
            log::warn!("Depth batch failed: {e} — using uniform depth 0.5");
            chunk_items.iter().map(|it| {
                (it.id.clone(), vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h)
            }).collect()
        });

    for (kf, (_, depth_raw, dw, dh)) in chunk_kfs.iter().zip(results.iter()) {
        let depth = if *dw == info.width && *dh == info.height {
            depth_raw.clone()
        } else {
            InferenceServer::upscale_depth(depth_raw, *dw, *dh, info.width, info.height)
        };
        keyframe_depths.insert(kf.frame_index, depth);
    }
}

// Ordered keyframe index list for lerp computation
let kf_index_list: Vec<(u64, bool)> = keyframes.iter()
    .map(|kf| (kf.frame_index, kf.scene_cut_before))
    .collect();
```

**Pass 2 — full-res grade:**
```rust
let frames = ffmpeg::decode_frames(&args.input, &info)?;
let mut kf_cursor = 0usize;
let mut frame_count = 0u64;

for frame_result in frames {
    let frame = frame_result?;
    let fi = frame.index;

    // Advance cursor: kf_index_list[kf_cursor].0 is the most recent keyframe ≤ fi
    while kf_cursor + 1 < kf_index_list.len() && kf_index_list[kf_cursor + 1].0 <= fi {
        kf_cursor += 1;
    }

    let (prev_kf_idx, _) = kf_index_list[kf_cursor];
    let prev_depth = keyframe_depths.get(&prev_kf_idx)
        .expect("prev keyframe depth missing — logic error");

    let depth = if fi == prev_kf_idx {
        prev_depth.clone()
    } else if let Some(&(next_kf_idx, scene_cut_before_next)) = kf_index_list.get(kf_cursor + 1) {
        if scene_cut_before_next {
            prev_depth.clone()  // Don't lerp across scene cut
        } else {
            let next_depth = keyframe_depths.get(&next_kf_idx)
                .expect("next keyframe depth missing — logic error");
            let t = (fi - prev_kf_idx) as f32 / (next_kf_idx - prev_kf_idx) as f32;
            lerp_depth(prev_depth, next_depth, t)
        }
    } else {
        prev_depth.clone()  // Past last keyframe
    };

    // Grade
    #[cfg(feature = "cuda")]
    let graded = grade_with_grader(
        cuda_grader.as_ref(), &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
    ).map_err(|e| anyhow::anyhow!("Grading failed for frame {}: {e}", fi))?;
    #[cfg(not(feature = "cuda"))]
    let graded = grade_frame(
        &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
    ).map_err(|e| anyhow::anyhow!("Grading failed for frame {}: {e}", fi))?;

    encoder.write_frame(&graded)?;
    frame_count += 1;

    if frame_count % 100 == 0 {
        let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
        log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
    }
}

inf_server.shutdown();
encoder.finish()?;
log::info!("Done. Graded {frame_count} frames → {}", output.display());
Ok(())
```

---

## Notes

- `auto_calibrate` is NOT batched (uses both RAUNE+depth, n_kf ≤ 20, not the bottleneck)
- `no_depth_interp` case: every frame becomes a keyframe in pass 1; chunked batching handles memory
- `BufferedFrame` and `flush_buffer_with_depth` are removed entirely
- `DepthBatchItem` must be `pub` and re-exported via `inference/mod.rs`
- Both `inference_subprocess.rs` AND `pyo3_backend.rs` must implement `run_depth_batch` with the same signature
- `decode_frames_scaled` should handle the edge case where proxy_w == info.width (call existing `decode_frames`)

## Expected outcome

- Depth inference: ~0.5s (one batched forward pass) vs ~70–100s (sequential)  
- Total pipeline: ~45–60s vs 2m7s for the 4K 120fps 14s benchmark clip
