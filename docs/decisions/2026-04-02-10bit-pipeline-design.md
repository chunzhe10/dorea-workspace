# 10-Bit Pipeline Design: DJI D-Log M + Insta360 X5 I-Log

**Date:** 2026-04-02
**Status:** Approved
**Scope:** dorea pipeline (repos/dorea)

## Problem

Dorea's pipeline quantizes 10-bit camera footage to 8-bit at two points before any
color math happens:

1. ffmpeg extracts keyframes as JPEG (8-bit)
2. `load_rgb_image()` calls `.into_rgb8()` then divides by 255.0

This discards 75% of the tonal resolution from both cameras (DJI Action 4 D-Log M
10-bit H.265, Insta360 X5 I-Log 10-bit ProRes 422 HQ). The result is visible banding
in underwater blue/green gradients and reduced precision in shadow/highlight regions.

## Decision

**Approach A: 10-bit I/O, f32 internal.** Upgrade the I/O boundaries to handle 10-bit
(16-bit PNG for keyframes, ProRes for X5), keep internal processing in f32, add I-Log
transfer function, output 10-bit to DaVinci Resolve. Resolve owns final SDR tone mapping,
dithering, and 8-bit conversion.

## Camera Input

### Two input paths, converging to linear light

| Camera | Capture | Container | Bit depth | Log profile |
|--------|---------|-----------|-----------|-------------|
| DJI Action 4 | D-Log M | .mp4 H.265 | 10-bit | D-Log M |
| Insta360 X5 | I-Log 5.7K HDR | .mov ProRes 422 HQ | 10-bit | I-Log |

**X5 workflow:** Shoot I-Log 5.7K HDR -> Insta360 Studio (Color Plus OFF, no LUTs) ->
reframe to flat perspective -> export ProRes 422 HQ.

### Transfer function abstraction (trait-based)

```rust
trait TransferFunction {
    fn to_linear(encoded: f32) -> f32;
    fn from_linear(linear: f32) -> f32;
    fn shoulder() -> f32;
    fn name() -> &'static str;
}
```

Concrete implementations:

| Impl | Module | Shoulder | Notes |
|------|--------|----------|-------|
| DLogM | `dorea-color/src/dlog_m.rs` | 0.85 | Existing, unchanged |
| ILog | `dorea-color/src/ilog.rs` | 0.88 | Initially S-Log3 proxy |
| Srgb | `dorea-color/src/lib.rs` | 0.95 | Standard EOTF |
| LutBased | `dorea-color/src/lut_transfer.rs` | configurable | 1D .cube for empirical curves |

**I-Log empirical extraction (deferred):** Shoot a grayscale ramp chart in I-Log on the
X5, export via Studio, read pixel values, fit the curve. Drop the result into a 1D .cube
file and swap into the `LutBased` variant without touching code.

### Clip metadata

```rust
struct ClipMeta {
    encoding: InputEncoding,       // DlogM | ILog | Srgb | Custom(PathBuf)
    projection: Projection,        // Rectilinear | TinyPlanet
    source_bit_depth: u8,          // 8 | 10 (validation/warning only)
    container_hint: ContainerHint, // Mov | Mp4 | Other
}

enum Projection {
    Rectilinear,  // standard wide reframe, depth estimation OK
    TinyPlanet,   // stereographic, skip depth, flat correction only
}
```

### Auto-detection with mismatch warning

On ingest, sniff container/codec via `ffprobe`:
- `.mov` + ProRes -> suggest ILog, warn if user specified DlogM
- `.mp4` + H.265 -> suggest DlogM, warn if user specified ILog
- Mismatch -> `WARN` (not a hard block, suppressible with `--force`)

### ProRes frame selection (all-intra aware)

ProRes is intra-only (every frame is a keyframe). The extraction path:
- **H.265 input:** select I-frames, then subsample by interval/scene-change
- **ProRes input:** skip I-frame filter, subsample by interval/scene-change only

### Camera-aware shoulder rolloff

| Encoding | Shoulder | Headroom | Rationale |
|----------|----------|----------|-----------|
| D-Log M | 0.85 | 0.15 | 14+ stops, needs early compression |
| I-Log | 0.88 | 0.12 | ~12-13 stops, less headroom needed |
| sRGB | 0.95 | 0.05 | Already display-referred |

Overridable via `--shoulder` flag.

## 16-Bit I/O and f32 Internal Pipeline

### Precision zones

Not every stage needs 16-bit precision. The pipeline splits into zones:

- **HIGH PRECISION (f32):** Transfer function decode, LUT build/apply, HSL correct,
  highlight rolloff, grade output. This is the color-critical path.
- **LOW PRECISION (8-bit proxy):** Depth inference only. Depth Anything V2 quantizes
  to fp16 tensors internally. Sending 16-bit pixels doubles IPC for zero quality gain.

### Keyframe extraction

```bash
# Current (8-bit JPEG)
ffmpeg -i input.mp4 -vf "select=..." -qscale:v 2 frame_%04d.jpg

# New (16-bit PNG, both cameras)
ffmpeg -i input.mp4 -vf "select=..." -pix_fmt rgb48be frame_%04d.png
```

### load_rgb_image upgrade

```rust
// Current
let img = ImageReader::open(path)?.decode()?.into_rgb8();   // 8-bit
let pixels = img.pixels()
    .map(|p| [p[0] as f32 / 255.0, ...]).collect();

// New
let img = ImageReader::open(path)?.decode()?.into_rgb16();  // 16-bit
let pixels = img.pixels()
    .map(|p| [p[0] as f32 / 65535.0, ...]).collect();
```

### inference.rs PNG codec

The hand-rolled PNG encoder/decoder in `inference.rs` supports both bit depths:
- **Encoder:** IHDR `bit_depth` = 8 or 16, scanline data sized accordingly
- **Decoder:** Accept `bit_depth == 8 || bit_depth == 16`, normalize to `[0.0, 1.0]`
- 8-bit backward compat preserved

### LUT size: 33^3 -> 65^3

| Input bits | Standard LUT size | Grid points |
|------------|-------------------|-------------|
| 8-bit | 33^3 | 35,937 |
| 10-bit | 65^3 | 274,625 |

Configurable via `--lut-size 33|65`. Auto-defaults based on `source_bit_depth`.
DaVinci Resolve handles 65^3 .cube files natively.

## Performance Architecture

### 3-Stage pipelined frame loop

Replace the serial `for frame in frames` loop with bounded-channel pipeline:

```
Stage 1 (CPU thread)        Stage 2 (GPU + Python)       Stage 3 (GPU, main thread)
ffmpeg decode               batch depth inference         grade_frame_cuda
  NVDEC 10-bit (H.265)       2-3 frames/call               persistent LUT (65^3)
  CPU (ProRes)                8-bit proxy IPC               f32 throughout
proxy resize (u8)             depth upscale                10-bit encode (ProRes/HEVC)
        |                          |                          |
        v                          v                          v
   channel A (cap 3)          channel B (cap 3)           output file
```

**Why 3 stages:** Depth inference (~1.3s/frame) dominates. While Stage 2 processes
frame N, Stage 1 decodes N+1 and Stage 3 encodes N-1. Grading (~50ms) is hidden
behind inference latency. Expected ~2x throughput improvement.

### Persistent GPU LUT allocation

The 65^3 LUT is constant across all frames. Upload once (16.5 MB), reuse across all
frames. Eliminates 27.6 GB of redundant cudaMemcpy for a 1671-frame clip.

```rust
struct GpuResources {
    d_luts: *mut f32,
    d_zone_boundaries: *mut f32,
    lut_size: usize,
}
impl Drop for GpuResources {
    fn drop(&mut self) { /* cudaFree */ }
}
```

### CUDA kernel signature: u8 -> f32

All CUDA kernels (`lut_apply.cu`, `hsl_correct.cu`, `clarity.cu`) and the CPU fallback
change from `unsigned char*` / `&[u8]` to `float*` / `&[f32]`. No double conversion.

### Batched depth inference

Extend Python inference server protocol with `depth_batch` command (2-3 frames per call).
Amortizes IPC overhead. VRAM safe: batch=2 uses ~2.5 GB, batch=3 uses ~3.5 GB, both
within 6 GB budget since depth and grading never share GPU simultaneously.

### ffmpeg decode: NVDEC for H.265 10-bit

RTX 3060 NVDEC supports HEVC 10-bit natively. ProRes has no NVDEC support (CPU only,
but ProRes all-intra decode is fast enough at ~200+ fps).

### ffmpeg encode: 10-bit output

`--output-codec` flag:

| Value | Codec | Bit depth | GPU | Use case |
|-------|-------|-----------|-----|----------|
| `prores` (default) | `prores_ks -profile:v 3` | 10-bit 4:2:2 | No | DaVinci Resolve |
| `hevc` | `hevc_nvenc -profile:v main10` | 10-bit 4:2:0 | Yes | Fast preview |
| `h264` | `libx264 -crf 18` | 8-bit | No | Legacy |

### VRAM budget (RTX 3060, 6 GB)

| Operation | VRAM | When |
|-----------|------|------|
| Depth Anything V2 (batch=2) | ~2.5 GB | Stage 2 |
| CUDA grading kernels | ~200 MB | Stage 3 |
| Persistent LUT (65^3 x 5 zones) | ~16.5 MB | Entire run |
| NVDEC decoder context | ~50 MB | Entire run |
| **Peak** | **~2.8 GB** | During depth batch |

## Tiny Planet Handling

### Projection-aware pipeline branching

```
Rectilinear  -> 3-stage pipeline (depth + stratified LUT)
TinyPlanet   -> 2-stage pipeline (no depth, zone 0 LUT, uniform depth=0.5)
```

When `projection == TinyPlanet`:
- No inference server spawned (no Python subprocess)
- Single zone LUT (zone 0 = shallowest depth)
- Uniform depth = 0.5 placeholder
- Expected >15 fps (no depth bottleneck)

**Why skip depth:** Depth Anything V2 was trained on rectilinear images. Stereographic
projection (tiny planet) produces garbage depth maps. The model cannot interpret the
radial distortion.

### Detection safety net

After depth inference on the first frame, check if depth map is degenerate (very low
variance or radial pattern). If detected, emit warning suggesting `--projection tiny-planet`.
Not auto-switch — just a warning.

### Zone override

`--lut-zone N` forces a specific depth zone regardless of projection. Default for tiny
planet is zone 0 (shallowest). Overridable for creative control.

## CLI Interface

### Extended commands

```
dorea calibrate --input-encoding ilog --projection rectilinear --lut-size 65
dorea grade --input clip.mov --input-encoding ilog --output-codec prores --batch-size 2
dorea probe --input clip.mov   # NEW: sniff container, suggest flags
```

### Config file (config.yaml)

```yaml
cameras:
  dji_action4:
    encoding: dlog_m
    shoulder: 0.85
    source_bit_depth: 10
    container_hint: mp4
    frame_selection: keyframe
  insta360_x5:
    encoding: ilog
    shoulder: 0.88
    source_bit_depth: 10
    container_hint: mov
    frame_selection: interval

pipeline:
  lut_size: auto
  output_codec: prores
  batch_size: 2
  proxy_size: 518

gpu_device: cuda:0
```

CLI flags override config.yaml.

## Module Changes

```
dorea-color/
  src/dlog_m.rs          -- unchanged
  src/ilog.rs            -- NEW: I-Log transfer function (trait-based)
  src/lut_transfer.rs    -- NEW: LutBased transfer function (1D .cube)
  src/lib.rs             -- NEW: TransferFunction trait

dorea-video/
  src/inference.rs       -- 16-bit PNG encode/decode (accept both 8/16)
  src/ffmpeg.rs          -- NVDEC 10-bit, ProRes decode, 10-bit encode
  src/probe.rs           -- NEW: container/codec sniffing

dorea-lut/
  src/build.rs           -- configurable LUT size (33/65)
  src/apply.rs           -- accept f32 pixels

dorea-gpu/
  src/lib.rs             -- grade_frame: &[u8] -> &[f32]
  src/cpu.rs             -- same signature change
  src/cuda/mod.rs        -- persistent GpuResources, f32 I/O
  src/cuda/kernels/*.cu  -- unsigned char* -> float*

dorea-cli/
  src/calibrate.rs       -- load_rgb_image -> into_rgb16, InputEncoding extended
  src/grade.rs           -- 3-stage pipeline, ClipMeta, output-codec
  src/probe.rs           -- NEW: dorea probe subcommand

config.yaml              -- camera profiles, pipeline defaults
```

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Existing 8-bit JPEG keyframes | into_rgb16() upsamples (values at multiples of 257) |
| Existing 33^3 .cube LUTs | Still valid in Resolve. New calibrations default to 65^3 |
| Existing .dorea-cal files | Load normally. Missing camera profile defaults to dji_action4 |
| --input-encoding srgb | Works as before. Shoulder=0.95, LUT size defaults to 33 |
| CUDA not available | CPU fallback unchanged, same f32 signature |
| Old inference server protocol | Single-frame still works. Batch is additive |

## Expected Performance

| Metric | Current | After |
|--------|---------|-------|
| Frame rate (rectilinear) | 0.74 fps | ~1.4 fps |
| Frame rate (tiny planet) | 0.74 fps | ~20+ fps |
| VRAM peak | ~1.5 GB | ~2.8 GB (batch=2) |
| LUT generation | ~0.5s | ~3-4s (65^3) |
| IPC bandwidth | 24.9 MB/frame | 24.9 MB/frame (stays 8-bit proxy) |
| GPU idle time | ~95% | ~50% (pipelined) |

## Test Plan

### Section 1: Camera Input Abstraction
- `test_ilog_round_trip` -- I-Log encode/decode proxy curve
- `test_lut_based_round_trip` -- 1D LUT transfer function round-trip
- `test_container_detection` -- .mov+ProRes -> ILog; .mp4+H.265 -> DlogM
- `test_mismatch_warning` -- wrong encoding for container -> warning
- `test_prores_frame_selection` -- ProRes skips I-frame filter
- `test_projection_enum_parse` -- CLI parses rectilinear|tiny-planet

### Section 2: 16-bit I/O and Performance
- `test_16bit_png_encode_decode` -- inference.rs 16-bit round-trip
- `test_8bit_png_still_works` -- backward compat
- `test_load_rgb_image_precision` -- value 1024 -> correct f32
- `test_65_cube_lut_format` -- valid .cube file
- `test_lut_size_auto_default` -- 10-bit -> 65^3; 8-bit -> 33^3
- `test_inference_receives_8bit` -- proxy frames are u8
- `test_pipeline_stages_concurrent` -- decode overlaps with depth
- `test_batch_depth_inference` -- 2-3 frame batch returns correct depths
- `test_persistent_lut_gpu` -- 1 cudaMalloc for N frames
- `test_cuda_f32_signature` -- f32 in, f32 out
- `test_10bit_prores_encode` -- output verified 10-bit via ffprobe
- `test_dlogm_golden_file` -- bit-identical regression

### Section 3: Tiny Planet
- `test_tiny_planet_skips_depth` -- no inference server spawned
- `test_tiny_planet_uses_zone_0` -- zone 0 LUT only
- `test_tiny_planet_uniform_depth` -- all pixels depth=0.5
- `test_rectilinear_full_pipeline` -- normal path unchanged
- `test_degenerate_depth_warning` -- low-variance depth -> warning
- `test_lut_zone_override` -- --lut-zone 2 forces zone 2
- `test_two_stage_pipeline_perf` -- tiny planet >15 fps

### Section 4: Integration
- `test_probe_dji_mp4` -- correct detection and suggestion
- `test_probe_x5_prores` -- correct detection and suggestion
- `test_config_camera_profiles` -- config.yaml loads, CLI overrides
- `test_8bit_backward_compat_e2e` -- old 8-bit workflow still works
- `test_dorea_cal_migration` -- old .dorea-cal loads with defaults
- `test_vram_budget` -- peak VRAM under 4 GB

## Alternatives Rejected

**Approach B (Full ProRes pipeline):** Bake LUT into re-encoded video. Rejected because
Resolve applies LUTs natively — baking removes grading flexibility.

**Approach C (Resolve-native LUT only):** Only upgrade LUT precision, keep 8-bit keyframes.
Rejected because LUT quality depends on calibration data — analyzing 8-bit keyframes to
build a 10-bit LUT interpolates precision never measured.

**Unified linear pipeline for both cameras:** Inverse Rec.709 EOTF on X5 footage to bring
into linear, then share correction chain. Rejected because X5 exports I-Log (not Rec.709)
when shot in I-Log with Color Plus disabled — separate transfer functions are the correct
approach.
