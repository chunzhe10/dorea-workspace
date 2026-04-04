# Maxine Pre-Processing Integration — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Issues:** #15 (parent), #16, #17, #18, #19, #20, #21, #22
**Branch:** `feat/15-maxine-enhancement`

---

## Summary

Add NVIDIA Maxine VFX SDK as the **first-stage preprocessor** in the dorea grading pipeline.
Maxine runs on **every frame at full resolution** before any other algorithm. All derivative
algorithms (RAUNE, Depth Anything, LUT calibration, GPU grading) operate on the Maxine-enhanced
frames, never on raw camera frames.

This corrects the original design (post-grading, keyframes only) to match the actual intent:
Maxine is an input-enhancement stage, not an output-beautification stage.

---

## Architecture

### Pipeline (Approach B — single inference server)

```
[spawn: Maxine + RAUNE + Depth server]

Pass 1 — full-res decode, Maxine on every frame:
  decode_frames(full_res)
    → enhance(frame) [IPC per frame]
    → maxine_full_res
    → resize_rgb_bilinear(maxine_full_res, proxy_w, proxy_h)
    → maxine_proxy
    → MSE change detection
    → if keyframe: store KeyframeEntry { maxine_proxy_pixels }

Calibration (after Pass 1 keyframes collected):
  if auto-calibrate:
    run_raune_depth_batch(maxine_proxy_pixels) → enhanced + depth
    calibration store: (pixels=maxine_proxy, target=RAUNE(maxine_proxy), depth)
  if pre-computed calibration:
    run_depth_batch(maxine_proxy_pixels) → depth cache

Pass 2 — full-res decode, Maxine on every frame:
  decode_frames(full_res)
    → enhance(frame) [IPC per frame]
    → maxine_full_res
    → depth lookup + lerp from cache
    → grade_with_grader(maxine_full_res, depth) [CUDA]
    → encoder.write_frame

[shutdown server]
```

### What changes from the original issues

| Issue | Original intent | Updated scope |
|-------|----------------|---------------|
| #16 Protocol | `EnhanceResult` + `encode_raw_rgb` | Unchanged |
| #17 MaxineEnhancer | nvvfx wrapper, mock mode | Unchanged |
| #18 server.py | enhance handler + `--maxine` | Unchanged |
| #19 InferenceConfig | maxine fields + `build_args()` | One server (always has Maxine+RAUNE+Depth when `--maxine`) |
| #20 enhance() IPC | Rust IPC method | Unchanged — used per-frame in Pass 1 and Pass 2 |
| #21 grade.rs | Insert after grade_frame() ← OLD | Full restructure: Pass 1 full-res + enhance; Pass 2 enhance before grade |
| #22 Docs | Setup guide | Reflect pre-processing role, full-res operation |

---

## Component Design

### Python: MaxineEnhancer (`maxine_enhancer.py`)

Wraps nvvfx with:
- Lazy effect init on first `enhance()` call (needs input dimensions)
- Shared PyTorch CUDA stream: `NvVFX_SetCudaStream(effect, torch.cuda.current_stream().cuda_stream)`
- Pipeline: RGB u8 → BGRA → (ArtifactReduction if ≤1080p) → 2× VideoSuperRes → INTER_AREA
  downsample → BGRA → RGB u8 at original resolution
- Error handling: exceptions caught in `enhance()`, returns original frame, increments passthrough counter
- Mock mode: `DOREA_MAXINE_MOCK=1` replaces nvvfx with identity passthrough

### Python: server.py

New args: `--maxine`, `--maxine-upscale-factor <N>` (default 2).

When `--maxine` is set, `MaxineEnhancer` is loaded at startup (hard error if nvvfx unavailable and
not in mock mode). The `enhance` request handler:

```
req: { type: "enhance", id, format: "raw_rgb", image_b64, width, height,
       artifact_reduce, upscale_factor }
resp: { type: "enhance_result", id, image_b64, width, height }
```

### Rust: InferenceConfig + build_args()

New fields: `maxine: bool`, `maxine_upscale_factor: u32`.

`build_args()` extracts CLI-arg building from `spawn()`. When `maxine: true`, appends
`--maxine --maxine-upscale-factor <N>`.

`InferenceConfig` is constructed **once** in grade.rs. The same config is used for both the
calibration RAUNE+depth batch and the per-frame enhance calls.

### Rust: InferenceServer::enhance()

```rust
pub fn enhance(
    &mut self, id: &str, image_rgb: &[u8], width: usize, height: usize,
    artifact_reduce: bool, upscale_factor: u32,
) -> Result<Vec<u8>, InferenceError>
```

Encodes frame as base64 raw_rgb, sends IPC request, parses `enhance_result` response,
validates returned dimensions match request. Returns enhanced RGB u8 at same resolution.

### Rust: grade.rs restructure (Task #21)

**Server lifecycle:** Spawn ONE `InferenceServer` before Pass 1. Use for all three phases
(Pass 1 enhance, calibration batch, Pass 2 enhance). Shutdown after Pass 2.

**Pass 1 change:** Replace `decode_frames_scaled` (proxy decode) with `decode_frames`
(full-res decode). Per frame:
```
enhance(full_res_pixels) → maxine_full_res
resize_rgb_bilinear(maxine_full_res, proxy_w, proxy_h) → maxine_proxy
MSE on maxine_proxy → keyframe detection
if keyframe: KeyframeEntry { proxy_pixels: maxine_proxy }
```

`KeyframeEntry.proxy_pixels` now holds Maxine-enhanced proxy pixels, not raw proxy pixels.
All downstream code that reads `kf.proxy_pixels` automatically uses Maxine-enhanced input.

**Calibration change:** No code change needed — `kf.proxy_pixels` is already Maxine-enhanced.
The calibration store records `(maxine_proxy, RAUNE(maxine_proxy), depth)`.

**Pass 2 change:** Per frame:
```
enhance(frame.pixels) → maxine_full_res
grade_with_grader(maxine_full_res, depth, ...) → graded
encoder.write_frame(graded)
```

Grade operates on `maxine_full_res` instead of `frame.pixels`.

**`--maxine` CLI flag:** Added to `GradeArgs`. When present, `build_inference_config()` sets
`maxine: true`. When absent, `enhance()` is never called and `frame.pixels` is used directly
(zero overhead path).

**Startup validation:** Reject `maxine_upscale_factor` not in `[2, 3, 4]`. Log Maxine status
at startup. Warn that Pass 1 will decode full-res (slower than proxy-only Pass 1 without Maxine).

---

## Data Flow Correctness

| Algorithm | Input (with Maxine) | Input (without Maxine) |
|-----------|---------------------|------------------------|
| MSE change detect | Maxine-proxy | raw proxy |
| RAUNE | Maxine-proxy | raw proxy |
| Depth Anything | RAUNE(Maxine-proxy) | RAUNE(raw proxy) |
| LUT calibration original | Maxine-proxy | raw proxy |
| LUT calibration target | RAUNE(Maxine-proxy) | RAUNE(raw proxy) |
| GPU grading | Maxine-full-res | raw full-res |

The LUT maps Maxine-enhanced → RAUNE-target, and grading applies that LUT to Maxine-enhanced
full-res frames. This is self-consistent: calibration and grading both operate on the same
enhanced input domain.

---

## VRAM Budget

All models resident simultaneously in one Python process. Shared CUDA context via
`NvVFX_SetCudaStream`.

| Component | VRAM |
|-----------|------|
| CUDA context (shared) | ~500 MB |
| Rust CUDA grader | ~300 MB |
| Depth Anything V2 Small | ~1.8 GB |
| Maxine VideoSuperRes model | ~1.0–1.5 GB |
| Maxine ArtifactReduction model | ~0.5 GB |
| Maxine inference workspace | ~200–500 MB |
| Depth batch buffers | ~500 MB |
| **Total peak** | **~4.8–5.6 GB** |
| **Headroom on RTX 3060 6 GB** | **~0.4–1.2 GB** |

Escape hatches: `--maxine-artifact-reduction false` (−500 MB), reduce depth batch size (−250 MB).

---

## Performance Characteristics

Maxine runs on every frame at full resolution in both passes. For a 30 fps, 5-minute clip
(9,000 frames):

- Pass 1: 9,000 Maxine calls at full-res (e.g., 1080p ~20–40 ms/frame) = 3–6 min
- Calibration RAUNE+depth: ~750 keyframes at proxy-res (batched) = same as before
- Pass 2: 9,000 Maxine calls at full-res = 3–6 min additional

Total overhead vs Maxine-disabled: ~6–12 min for a 5-min clip. This is the expected cost
of per-frame AI preprocessing. Users should be informed.

Without `--maxine`: zero overhead. Pass 1 reverts to proxy decode (`decode_frames_scaled`),
Pass 2 grades raw frames. Byte-identical output to the pre-Maxine pipeline.

---

## Testing

- `DOREA_MAXINE_MOCK=1`: passthrough mode — full IPC round-trip exercised without nvvfx SDK.
  All unit and integration tests use this env var.
- Unit tests (Rust): `spawn_command_includes_maxine_flags`, `spawn_command_omits_maxine_when_disabled`,
  `enhance_parses_valid_response`, `enhance_rejects_dimension_mismatch`
- E2E smoke test: `DOREA_MAXINE_MOCK=1 dorea grade --input test.mp4 --maxine` completes successfully
  and produces byte-identical output to mock passthrough (Maxine identity transform).

---

## CLI

```
dorea grade --input footage.mp4 --maxine
dorea grade --input footage.mp4 --maxine --maxine-upscale-factor 2
dorea grade --input footage.mp4 --maxine --no-maxine-artifact-reduction
dorea grade --input footage.mp4  # Maxine disabled, zero overhead
```

---

## Files Changed

| File | Change |
|------|--------|
| `python/dorea_inference/protocol.py` | Add `EnhanceResult` + `encode_raw_rgb` |
| `python/dorea_inference/maxine_enhancer.py` | New: MaxineEnhancer class |
| `python/dorea_inference/server.py` | `--maxine` flag, enhance handler |
| `crates/dorea-video/src/inference.rs` | `InferenceConfig` fields, `build_args()`, `enhance()` IPC |
| `crates/dorea-cli/src/grade.rs` | Full restructure: one server, full-res Pass 1, Maxine in Pass 2 |
| `docs/guides/maxine-setup.md` | New: SDK install guide (updated for pre-processing role) |
