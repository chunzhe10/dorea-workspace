# Dorea v2 — Phase 2+3 Implementation Plan
**Date:** 2026-04-02
**Issue:** chunzhe10/dorea-workspace#24
**Branch:** feat/24-dorea-v2-phase2-3
**Scope:** GPU grading + Python inference bridge + end-to-end video pipeline

---

## Phase 2: GPU Grading + Python Inference Bridge

### Goal
- Automate depth + RAUNE-Net generation via Python subprocess (no pre-computed inputs required)
- CPU `depth_aware_ambiance` ported from POC script
- CUDA kernel stubs for fused LUT+HSL+clarity grading (compiled when nvcc present)

---

## Phase 3: End-to-End Video

### Goal
- `dorea grade` — full pipeline: video in → graded video out
- `dorea preview` — before/after contact sheet (5-10 frames)
- ffmpeg decode/encode (NVDEC/NVENC, audio passthrough)
- Scene change detection (histogram distance → re-calibrate)
- User controls: --warmth, --strength, --contrast

---

## New Crate: `dorea-gpu`

```
crates/dorea-gpu/
├── Cargo.toml
├── build.rs               # nvcc detection; compiles .cu → .fatbin; skips if no nvcc
└── src/
    ├── lib.rs             # feature-gated: re-exports cpu::* always; cuda::* with feature "cuda"
    ├── cpu.rs             # depth_aware_ambiance (CPU, no deps beyond dorea-color)
    │                      # grade_frame_cpu (LUT apply + HSL + ambiance in sequence)
    └── cuda/              # guarded by cfg(feature = "cuda")
        ├── mod.rs         # CUDA context init, device query
        ├── kernels/
        │   ├── lut_apply.cu       # fused trilinear LUT + depth blending
        │   ├── hsl_correct.cu     # HSL qualifier correction
        │   └── clarity.cu         # clarity at proxy + bilinear upscale
        └── grade.rs       # grade_frame_cuda: launches kernels, falls back to CPU on error
```

### CPU `depth_aware_ambiance` Algorithm

Port from `run_fixed_hsl_lut_poc.py::depth_aware_ambiance()`:

```
1. RGB f32 → LAB (via dorea-color)
2. Shadow lift: lift_amount = 0.2 + 0.15*d; shadow_mask = clamp((0.15 - L/100) / 0.15, 0, 1)
   L/100 += shadow_mask * lift_amount * 0.15
3. S-curve: strength = 0.3 + 0.3*d; s = 1/(1+exp(-(x-0.5)*(4+4*strength))); L = x + (s-x)*strength
4. Highlight compress: knee=0.88; over = max(L-knee,0); L = knee + (1-knee)*tanh(over/(1-knee)*(1+compress))
   compress = 0.4 + 0.2*(1-d)
5. Clarity: sigma proportional to image height (full res) / proxy res; at proxy: sigma=30px
   blur = gaussian_blur(L, sigma); detail = tanh((L-blur)*3)/3; L += detail * (0.2+0.25*mean(d))
6. Warmth: a += (1+5*d)*lum_weight; b += 4*d*lum_weight; lum_weight = 4*L*(1-L)
7. Vibrance: chroma = sqrt(a²+b²); boost = vibrance*(1-chroma/40)*clamp(L/0.25,0,1)
   a *= (1+boost); b *= (1+boost); vibrance = 0.4+0.5*d
8. LAB → RGB (via dorea-color)
9. Final knee: knee=0.92; over=max(rgb-knee,0); rgb = knee+(1-knee)*tanh(over/(1-knee))
10. Clamp [0,1]
```

### CUDA Kernels (build time, optional)

**`lut_apply.cu`** — one thread per pixel:
- Read pixel (r,g,b) and depth value
- Compute soft zone weights (n_zones triangles over [0,1])
- Trilinear interp from each zone LUT, weighted sum
- Write output (r,g,b)

**`hsl_correct.cu`** — one thread per pixel:
- Read (r,g,b) → HSV
- For each of 6 qualifiers: soft mask, apply H/S/V corrections
- HSV → (r,g,b) write

**`clarity.cu`** — operates at proxy resolution:
- Separable Gaussian blur (σ=30 at proxy)
- detail = tanh((L - blur)*3)/3
- Add detail*clarity back to L
- Bilinear upscale to full res

### build.rs Strategy
```rust
// If nvcc found in PATH or CUDA_HOME/bin/nvcc:
//   compile .cu files with nvcc -ptx or nvcc -arch=sm_86 -c
//   set cfg(feature = "cuda") via println!("cargo:rustc-cfg=feature=\"cuda\"")
// Else:
//   emit warning, skip; CPU fallback is always compiled
```

---

## Updated `dorea-video` Crate

```
crates/dorea-video/src/
├── lib.rs
├── ffmpeg.rs          # decode/encode subprocess
├── scene.rs           # histogram-distance scene change detection
└── inference.rs       # Python inference subprocess manager
```

### `ffmpeg.rs` Design

**Decode** (`ffmpeg_decode_frames`):
```
spawn: ffmpeg -hwaccel nvdec -i <input> -vf scale=<W>:<H> -f image2pipe -vcodec png pipe:1
  fallback: ffmpeg -i <input> -f image2pipe -vcodec png pipe:1  (if NVDEC fails)
parse: read 8-byte PNG header + PNG data for each frame
yield: (frame_index, rgb_pixels: Vec<u8>, width, height)
```

**Encode** (`ffmpeg_encode_frames`):
```
spawn: ffmpeg -f rawvideo -pixel_format rgb24 -s WxH -framerate fps -i pipe:0
         -i <input> -map 0:v -map 1:a -c:a copy
         -c:v h264_nvenc -preset p4 -cq 18 <output>
  fallback: -c:v libx264 -crf 18  (if NVENC fails)
write: raw RGB frames to stdin
```

### `scene.rs` Design

```
histogram_distance(frame_a, frame_b) -> f32:
  compute RGB histograms (64 bins each channel)
  chi-squared distance between frame_a and frame_b histograms
  return chi2_dist / (3 * 64 * n_pixels)

is_scene_change(dist) -> bool:
  threshold = 0.15  (tunable via env DOREA_SCENE_THRESHOLD)
```

### `inference.rs` Design — IPC Protocol

**Protocol**: JSON lines over stdin/stdout. One request per line in, one response per line out.

```json
// Request types:
{"type": "ping"}
{"type": "raune", "id": "kf001", "image_b64": "<base64 PNG>", "max_size": 1024}
{"type": "depth", "id": "kf001", "image_b64": "<base64 PNG>", "max_size": 518}
{"type": "shutdown"}

// Response types:
{"type": "pong", "version": "0.1.0"}
{"type": "raune_result", "id": "kf001", "image_b64": "<base64 PNG>", "width": W, "height": H}
{"type": "depth_result", "id": "kf001", "depth_f32_b64": "<base64 raw f32 LE array>", "width": W, "height": H}
{"type": "error", "id": "kf001", "message": "..."}
{"type": "ok"}
```

**Rust subprocess manager** (`InferenceServer` struct):
- `spawn(python_exe, script_path, model_args) -> Result<InferenceServer>`
- `ping() -> Result<()>` — health check with 5s timeout
- `run_raune(id, image_rgb: &[u8], width, height) -> Result<Vec<u8>>` — returns sRGB PNG bytes
- `run_depth(id, image_rgb: &[u8], width, height) -> Result<Vec<f32>>` — returns f32 depth [0,1]
- `shutdown() -> Result<()>` — send shutdown, wait for process exit

---

## Python Inference Server (`python/dorea_inference/`)

```
python/dorea_inference/
├── __init__.py
├── protocol.py        # dataclasses for request/response types
├── raune_net.py       # RauneNetInference class
├── depth_anything.py  # DepthAnythingInference class
└── server.py          # main: parse args, load models, serve requests
```

### `server.py` Entrypoint

```
usage: python -m dorea_inference.server
  --raune-weights PATH    path to RAUNE-Net weights .pth
  --depth-model PATH      path to Depth Anything V2 Small dir (or .pth)
  --device cpu|cuda       (default: cuda if available, else cpu)

Loop:
  line = sys.stdin.readline()
  if not line: break  (EOF = parent died)
  req = json.loads(line)
  resp = dispatch(req)
  print(json.dumps(resp), flush=True)
```

### `raune_net.py`

```python
class RauneNetInference:
    def __init__(self, weights_path: str, device: str):
        # Load RauneNet from working/sea_thru_poc/models/raune_net.py
        # The model file is in working/sea_thru_poc/models/ — add to sys.path

    def infer(self, img_rgb: np.ndarray, max_size: int = 1024) -> np.ndarray:
        # Resize maintaining aspect to max_size
        # Normalize: (x - 0.5) / 0.5
        # Run model.forward(), de-normalize: (out + 1) / 2
        # Return uint8 RGB array at inference resolution
```

### `depth_anything.py`

```python
class DepthAnythingInference:
    def __init__(self, model_path: str, device: str):
        # Load Depth Anything V2 Small via transformers AutoModel or direct load
        # Model: depth-anything/Depth-Anything-V2-Small-hf or local weights

    def infer(self, img_rgb: np.ndarray, max_size: int = 518) -> np.ndarray:
        # Resize to max_size (must be multiple of 14 for ViT patch alignment)
        # Run inference → raw depth map
        # Normalize to [0, 1]: (d - d.min()) / (d.max() - d.min())
        # Return float32 array at inference resolution
```

---

## CLI Commands

### `dorea grade`

```
dorea grade
  --input <path>          Input video file (MP4/MOV/MKV)
  --output <path>         Output video file [default: <input>_graded.mp4]
  --calibration <path>    .dorea-cal file (if not provided, auto-calibrate)
  --warmth <f32>          Warmth multiplier [0.0–2.0, default: 1.0]
  --strength <f32>        LUT/HSL blend strength [0.0–1.0, default: 0.8]
  --contrast <f32>        Ambiance contrast multiplier [0.0–1.0, default: 1.0]
  --proxy-size <u32>      Proxy resolution for inference [default: 518]
  --raune-weights <path>  RAUNE-Net weights path
  --depth-model <path>    Depth Anything V2 model path
  --cpu-only              Disable CUDA (CPU fallback for all steps)
  --keyframe-interval <n> Frames between keyframe re-samples [default: 30]
  -v / --verbose
```

**Pipeline:**
1. Probe input (width, height, fps, duration) via `ffprobe`
2. Extract keyframes at `--keyframe-interval`
3. Auto-calibrate if no `--calibration`:
   - Spawn inference server
   - Run RAUNE-Net on each keyframe
   - Run Depth Anything on each keyframe
   - `build_depth_luts` + `derive_hsl_corrections` → `.dorea-cal`
4. For each frame:
   - Decode via ffmpeg
   - Scene change detection → re-calibrate if triggered
   - Run Depth Anything at proxy resolution
   - Apply LUT + HSL + depth_aware_ambiance (CUDA or CPU)
   - Scale by warmth/strength/contrast multipliers
   - Write to encoder
5. Encode via ffmpeg NVENC (fallback libx264) with audio passthrough

### `dorea preview`

```
dorea preview
  --input <path>          Input video file
  --calibration <path>    .dorea-cal file
  --output <path>         Output contact sheet PNG [default: preview.png]
  --frames <n>            Number of frames to sample [default: 8]
  --raune-weights <path>
  --depth-model <path>
  --cpu-only
  -v / --verbose
```

**Pipeline:**
1. Probe input for duration/fps
2. Sample `--frames` evenly-spaced frame timestamps
3. Extract frames via ffmpeg seeking
4. Load or auto-derive calibration from sampled keyframes
5. Run depth inference on each frame
6. Apply grading pipeline
7. Compose 2×N grid (top: original, bottom: graded) → PNG

---

## User Parameter Scaling

Applied after LUT+HSL+ambiance:

- **warmth** multiplier: scale a* and b* channels in LAB space by (1 + (warmth-1)*0.3)
- **strength** multiplier: linear blend between original and graded: `out = orig * (1-strength) + graded * strength`
- **contrast** multiplier: scale ambiance `strength` param by contrast value

---

## Testing Strategy

### Unit Tests
- `dorea-gpu`: CPU depth_aware_ambiance round-trip (same input → same output deterministically)
- `dorea-video`: scene change detection (identical frames → low distance; high-contrast frames → high distance)
- `dorea-video/inference`: IPC message serialization/deserialization

### Integration Tests
- `dorea calibrate` without `--targets`: spawn Python server, use real RAUNE-Net on POC keyframes, verify .dorea-cal written
- `dorea preview` on a short test clip

### Python Tests
- `python/dorea_inference/`: test server.py startup + ping + shutdown
- Test RAUNE-Net inference output shape + value range

---

## Constraints

- RTX 3060 6GB VRAM — GPU passthrough available after container rebuild
- CUDA 12.4 (matches PyTorch cu124)
- `cargo test` must pass with CPU-only fallback (no GPU required for tests)
- Clippy clean: `cargo clippy -- -D warnings`
- Python tests via `/opt/dorea-venv/bin/python -m pytest`
