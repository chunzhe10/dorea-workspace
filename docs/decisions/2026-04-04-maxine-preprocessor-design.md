# Maxine Pre-Processing Integration — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Issues:** #15 (parent), #16, #17, #18, #19, #20, #21, #22
**Branch:** `feat/15-maxine-enhancement`

---

## Summary

Add NVIDIA Maxine VFX SDK as the **first-stage preprocessor** in the dorea grading pipeline.
Maxine runs on **every frame at full resolution** in Pass 1. The Maxine-enhanced frames are
written to a **temporary lossless video file** (ffv1/mkv) and reused in Pass 2 — Maxine runs
exactly once per frame, never twice.

All derivative algorithms (RAUNE, Depth Anything, LUT calibration, GPU grading) operate on
Maxine-enhanced frames, never on raw camera frames.

This corrects the original design (post-grading, keyframes only) to match the actual intent:
Maxine is an input-enhancement stage, not an output-beautification stage.

---

## Architecture

### Pipeline (Approach B — single server, temp-file reuse)

```
[spawn server: Maxine + RAUNE + Depth]

Pass 1 — full-res decode, Maxine per frame, write temp:
  decode_frames(raw_input)
    → enhance(frame)           [IPC — Maxine, once per frame]
    → maxine_full_res
    → write to temp_encoder    [lossless ffv1/mkv temp file]
    → resize_rgb_bilinear → maxine_proxy
    → MSE change detection
    → if keyframe: store maxine_proxy in KeyframeEntry

Calibration (after Pass 1):
  raune_depth_batch(maxine_proxy_pixels) → depth cache + calibration
  [shutdown server]            ← VRAM freed here

Pass 2 — decode temp file, grade:
  decode_frames(temp_file)     [no Maxine, no inference server]
    → maxine_full_res from temp
    → depth lookup + lerp
    → grade_with_grader(maxine_full_res, depth)
    → encoder.write_frame

[delete temp file]
```

**Key properties:**
- Maxine runs exactly **once** per frame (Pass 1 only)
- Inference server shuts down after calibration — full 6 GB VRAM available to CUDA grader in Pass 2
- Pass 2 is: decode + CPU lerp + CUDA grade. No Python subprocess.
- Without `--maxine`: zero overhead. Pass 1 reverts to proxy decode; Pass 2 grades raw frames.

---

## Component Design

### Python: MaxineEnhancer (`maxine_enhancer.py`)

Wraps nvvfx with:
- Lazy effect init on first `enhance()` call (needs input dimensions)
- Shared PyTorch CUDA stream: `NvVFX_SetCudaStream(effect, torch.cuda.current_stream().cuda_stream)`
- Pipeline: RGB u8 → BGRA → (ArtifactReduction if ≤1080p) → 2× VideoSuperRes → INTER_AREA
  downsample → BGRA → RGB u8 at original resolution
- Error handling: exceptions caught, returns original frame, increments passthrough counter
- Mock mode: `DOREA_MAXINE_MOCK=1` replaces nvvfx with identity passthrough

### Python: server.py

New args: `--maxine`, `--maxine-upscale-factor <N>` (default 2).

When `--maxine` is set, `MaxineEnhancer` is loaded at startup (hard error if nvvfx unavailable
and not in mock mode). The `enhance` request handler:

```
req:  { type: "enhance", id, format: "raw_rgb", image_b64, width, height,
        artifact_reduce, upscale_factor }
resp: { type: "enhance_result", id, image_b64, width, height }
```

### Python: protocol.py

New: `EnhanceResult` dataclass + `encode_raw_rgb()` helper.
`encode_raw_rgb`: encodes HxWx3 RGB u8 ndarray to base64 raw bytes.
`EnhanceResult.from_array(id, img)`: convenience constructor → `to_dict()` for JSON serialisation.

### Rust: InferenceConfig + build_args()

New fields: `maxine: bool`, `maxine_upscale_factor: u32`.

`build_args()` extracts CLI-arg building from `spawn()`. When `maxine: true`, appends
`--maxine --maxine-upscale-factor <N>`.

`build_inference_config()` in grade.rs sets `maxine` from the `--maxine` CLI flag.
The same config is used for the single inference server covering Pass 1 and calibration.

### Rust: InferenceServer::enhance()

```rust
pub fn enhance(
    &mut self, id: &str, image_rgb: &[u8], width: usize, height: usize,
    artifact_reduce: bool, upscale_factor: u32,
) -> Result<Vec<u8>, InferenceError>
```

Encodes frame as base64 raw_rgb, sends IPC request, parses `enhance_result` response,
validates returned dimensions match request. Returns enhanced RGB u8 at same resolution.

### Rust: FrameEncoder — lossless temp mode

New constructor:

```rust
impl FrameEncoder {
    pub fn new_lossless_temp(path: &Path, width: usize, height: usize, fps: f64)
        -> Result<Self>
}
```

Uses ffmpeg with `-c:v ffv1 -level 3` (lossless, deterministic). Output is `.mkv`.
No audio stream. Used only as an intermediary; deleted after Pass 2 via Drop guard.

### Rust: grade.rs restructure (Task #21 — major change)

**When `--maxine` is enabled:**

**Server lifecycle:** Spawn ONE `InferenceServer` before Pass 1 (Maxine + RAUNE + Depth).
Use for Pass 1 enhance calls and calibration batch. Shut down after calibration.

**Pass 1:**
1. Change `decode_frames_scaled` → `decode_frames` (full-res decode)
2. Create temp `FrameEncoder` (ffv1/mkv, path: `$TMPDIR/dorea_maxine_<pid>.mkv`)
3. Per frame:
   - `inf_server.enhance(frame.pixels, ...)` → `maxine_full_res`
   - `temp_encoder.write_frame(&maxine_full_res)`
   - `resize_rgb_bilinear(&maxine_full_res, proxy_w, proxy_h)` → `maxine_proxy`
   - MSE on `maxine_proxy` → change detection
   - If keyframe: `KeyframeEntry { proxy_pixels: maxine_proxy }`
4. `temp_encoder.finish()` → temp file sealed

**Calibration:** Unchanged in structure. `kf.proxy_pixels` is now Maxine-proxy automatically.

**After calibration:** `inf_server.shutdown()`. VRAM freed.

**Pass 2:**
1. `decode_frames(&temp_file_path)` instead of `decode_frames(&args.input)`
2. Per frame: grade `frame.pixels` (already Maxine-enhanced, decoded from temp)
3. After `encoder.finish()`: delete temp file

**When `--maxine` is disabled:** Pass 1 uses `decode_frames_scaled` (proxy), Pass 2 grades
`frame.pixels` from the original input. Byte-identical to pre-Maxine pipeline.

**Startup validation:**
- Reject `maxine_upscale_factor` not in `[2, 3, 4]`
- Log Maxine status and estimated temp file size before Pass 1

---

## Data Flow Correctness

| Algorithm | Input with `--maxine` | Input without `--maxine` |
|-----------|----------------------|--------------------------|
| MSE change detect | Maxine-proxy | raw proxy |
| RAUNE | Maxine-proxy | raw proxy |
| Depth Anything | RAUNE(Maxine-proxy) | RAUNE(raw proxy) |
| LUT calibration "original" | Maxine-proxy | raw proxy |
| LUT calibration "target" | RAUNE(Maxine-proxy) | RAUNE(raw proxy) |
| GPU grading input | Maxine-full-res (from temp) | raw full-res |

LUT maps Maxine-enhanced → RAUNE-target. Grading applies that LUT to Maxine-full-res.
Calibration and grading operate on the same enhanced input domain — fully consistent.

---

## VRAM Budget

**Pass 1 + calibration (inference server running):**

| Component | VRAM |
|-----------|------|
| CUDA context (shared) | ~500 MB |
| Depth Anything V2 Small | ~1.8 GB |
| Maxine VideoSuperRes | ~1.0–1.5 GB |
| Maxine ArtifactReduction | ~0.5 GB |
| Maxine inference workspace | ~200–500 MB |
| Depth batch buffers | ~500 MB |
| **Peak** | **~4.8–5.6 GB** |

**Pass 2 (server shut down):**

| Component | VRAM |
|-----------|------|
| Rust CUDA grader | ~300 MB |
| **Peak** | **~300 MB** |

Escape hatches: `--no-maxine-artifact-reduction` (−500 MB), reduce depth batch size (−250 MB).

---

## Temp File

- **Format:** ffv1 in mkv — lossless, ~2–4 GB for 5-min 1080p
- **Location:** `$TMPDIR/dorea_maxine_<pid>.mkv`
- **Lifecycle:** Created at Pass 1 start, deleted after Pass 2 (or on error via RAII Drop guard)
- **Size note:** Log estimated size before Pass 1 so users can abort if disk is tight

---

## Performance

- **Pass 1:** 9,000 Maxine calls at 1080p (~20–40 ms each) = 3–6 min for 5-min clip
- **Calibration:** unchanged from current pipeline
- **Pass 2:** decode temp + CUDA grade. No inference. ~same speed as current Pass 2.
- **Total overhead vs no Maxine:** ~3–6 min for 5-min 1080p clip

---

## Testing

- `DOREA_MAXINE_MOCK=1`: identity passthrough — full IPC and temp-file path exercised without SDK
- Unit tests (Rust): `spawn_command_includes_maxine_flags`, `spawn_command_omits_maxine_when_disabled`,
  `enhance_parses_valid_response`, `enhance_rejects_dimension_mismatch`
- E2E smoke: `DOREA_MAXINE_MOCK=1 dorea grade --input test.mp4 --maxine` completes,
  temp file created and deleted, output produced
- Regression: `dorea grade --input test.mp4` produces byte-identical output (no `--maxine`)

---

## CLI

```bash
dorea grade --input footage.mp4 --maxine
dorea grade --input footage.mp4 --maxine --maxine-upscale-factor 2
dorea grade --input footage.mp4 --maxine --no-maxine-artifact-reduction
dorea grade --input footage.mp4   # Maxine disabled, zero overhead
```

---

## Files Changed

| File | Change |
|------|--------|
| `python/dorea_inference/protocol.py` | Add `EnhanceResult` + `encode_raw_rgb` |
| `python/dorea_inference/maxine_enhancer.py` | New: MaxineEnhancer class |
| `python/dorea_inference/server.py` | `--maxine` flag, enhance handler |
| `crates/dorea-video/src/inference.rs` | `InferenceConfig` fields, `build_args()`, `enhance()` IPC |
| `crates/dorea-video/src/ffmpeg.rs` | `FrameEncoder::new_lossless_temp()` for ffv1/mkv |
| `crates/dorea-cli/src/grade.rs` | Full restructure: one server, full-res Pass 1, temp-file Pass 2 |
| `docs/guides/maxine-setup.md` | New: SDK install guide (pre-processing role, temp file note) |
