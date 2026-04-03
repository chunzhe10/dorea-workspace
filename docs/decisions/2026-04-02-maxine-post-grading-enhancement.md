# Maxine Post-Grading Enhancement Stage

**Date:** 2026-04-02
**Status:** Accepted
**Scope:** dorea pipeline — optional post-grading AI enhancement using NVIDIA Maxine VFX SDK

## Context

The dorea 10-bit pipeline outputs graded footage (ProRes 422 HQ or HEVC 10-bit) through a 3-stage
pipelined frame loop: decode → depth inference → grade+encode. The grading runs on an RTX 3060 with
6GB VRAM and sequential GPU scheduling.

Underwater footage often suffers from compression artifacts, banding, and softness — especially
H.265 D-Log M footage from the DJI Action 4. An optional AI enhancement stage can improve output
quality without changing the core grading pipeline.

## Decision

Add an optional post-grading enhancement stage using NVIDIA Maxine Video Effects SDK, injected
between `grade_frame()` and `encoder.write_frame()` in the existing frame loop. The stage performs
an **in-resolution oversample**: upscale to a higher resolution using Maxine's AI super-resolution,
then downsample back to the original resolution for a cleaner, sharper result.

## Design

### Pipeline Integration

The current pipeline in `grade.rs` is a sequential for loop — decode, depth, grade, and encode
happen in order for each frame. There is no multi-threading between pipeline stages. The Maxine
enhancement call is inserted sequentially, the same way `run_depth()` is called today:

```
Per-frame loop (sequential, single-threaded):
  1. decode frame → RGB u8
  2. run_depth(proxy_frame) → depth f32          [via inference server]
  3. grade_frame(pixels, depth, calibration) → graded RGB u8
  4. enhance(graded_frame) → enhanced RGB u8     [via same inference server, NEW]
  5. encoder.write_frame(enhanced_frame)
```

When Maxine is disabled, step 4 is skipped — zero overhead. No mutex is needed because the
pipeline is single-threaded; depth and enhance calls are never concurrent.

**Two inference server instances:** `grade.rs` spawns two inference servers — one for
auto-calibration (RAUNE + depth) and one for per-frame grading (depth only). The `--maxine` flag
is passed ONLY to the grading server. The calibration server never loads Maxine models. This
avoids VRAM pressure during calibration (RAUNE + depth + Maxine would exceed 6GB).

### Temporal Interpolation Interaction

The pipeline uses temporal grade interpolation (perf v3 decision): ~83% of frames skip the full
grading pipeline and lerp graded pixel outputs between bracketing keyframes. Maxine enhancement
runs ONLY on keyframes:

```
Keyframe:     decode → depth → grade → enhance → encode
Non-keyframe: decode → MSE check → lerp(enhanced_kf_before, enhanced_kf_after) → encode
```

Interpolation operates on *enhanced* keyframe outputs, not pre-enhancement outputs. This means:
- Non-keyframes cost ~12ms (decode + MSE + lerp + encode) — no Maxine overhead
- Only ~17% of frames pay the Maxine cost (50-100ms per frame)
- No flickering between enhanced keyframes and unenhanced interpolated frames

### Inference Server Extension

The existing `dorea_inference` Python server is extended with a new `enhance` message type. Both
the depth model and Maxine models live in the same Python process.

**CUDA context sharing (critical implementation detail):** PyTorch and nvvfx create CUDA contexts
independently. To share a single context and avoid ~500MB duplicate overhead, `MaxineEnhancer`
must explicitly pass PyTorch's CUDA stream to nvvfx:
```python
NvVFX_SetCudaStream(effect, torch.cuda.current_stream().cuda_stream)
```
This must be done during `MaxineEnhancer.__init__()` after PyTorch has initialized CUDA.

**Request (Rust → Python):**
```json
{
  "type": "enhance",
  "id": "frame_042",
  "format": "raw_rgb",
  "image_b64": "<base64 raw RGB u8 bytes>",
  "width": 1920,
  "height": 1080,
  "artifact_reduce": true,
  "upscale_factor": 2
}
```

**Response (Python → Rust):**
```json
{
  "type": "enhance_result",
  "id": "frame_042",
  "format": "raw_rgb",
  "image_b64": "<base64 raw RGB u8 bytes>",
  "width": 1920,
  "height": 1080
}
```

IPC uses raw RGB format (`"format": "raw_rgb"`), consistent with the existing depth and RAUNE
protocol. Response dimensions match the input — the Python side handles the full
upscale→downsample round-trip internally. The Rust `enhance()` method validates that response
width/height match the request, returning an error on mismatch.

**Startup:**
- New CLI flags: `--maxine` (enable model loading), `--maxine-upscale-factor <N>`
- Maxine models loaded at startup only when `--maxine` is passed
- If `--maxine` passed but `nvvfx` import fails: hard error with install instructions
  (user explicitly enabled Maxine but SDK is missing — fail fast, don't silently degrade)
- If `--maxine` NOT passed: `nvvfx` is never imported, zero overhead

### Python Enhancement Pipeline

A new `MaxineEnhancer` class in `python/dorea_inference/maxine_enhancer.py`:

```python
class MaxineEnhancer:
    def __init__(self, upscale_factor=2):
        # Import nvvfx (hard error if unavailable — caller checks before constructing)
        # Obtain PyTorch CUDA stream
        # Create VideoSuperRes effect with NvVFX_SetCudaStream(effect, stream)
        # Create ArtifactReduction effect with shared stream
        # Load both models onto GPU

    def enhance(self, rgb_u8, width, height, artifact_reduce, upscale_factor):
        # 1. RGB → BGRA (numpy: add alpha=255, swap R↔B)
        # 2. Upload to CUDA tensor (torch → .cuda())
        # 3. If artifact_reduce AND width <= 1920 AND height <= 1080:
        #        run ArtifactReduction (same resolution)
        #    Else if artifact_reduce AND resolution > 1080p:
        #        log warning once ("artifact reduction skipped: input exceeds 1080p")
        # 4. Run VideoSuperRes → 2x resolution
        # 5. cv2.resize with INTER_AREA → original resolution
        # 6. BGRA → RGB (strip alpha, swap B↔R)
        # 7. Return RGB u8 numpy array
```

**Downsample method:** `cv2.resize` with `INTER_AREA`. For a 2x integer downsample, area-based
interpolation is equivalent to box filtering — it cannot produce ringing artifacts (unlike
Lanczos), which is preferable for AI-upscaled content that has distinctive sharp-edge frequency
profiles.

**Error handling:** Errors are caught inside `MaxineEnhancer.enhance()`. On failure, the method
logs the exception to stderr and returns the original `rgb_u8` input unchanged. The server sends
a normal `enhance_result` response with the original frame — the Rust side never sees the error.
This means:
- Rust `enhance()` method needs zero special-casing for failures
- A single frame failure does not kill a multi-hour pipeline
- A counter tracks passthrough events; logged at shutdown ("N of M frames fell through to
  passthrough")

**CI mock mode:** When `DOREA_MAXINE_MOCK=1` is set, `MaxineEnhancer` replaces nvvfx calls with
a passthrough (returns input unchanged). This allows CI to exercise the full orchestration path
(color conversion, IPC round-trip, dimension validation) without the proprietary SDK.

### VRAM Budget

Both models (Depth Anything + Maxine) remain resident in GPU memory for the pipeline duration.
The sequential call pattern means they never *execute* simultaneously, but their weights occupy
VRAM at the same time.

The grading CUDA kernels run in the Rust process (separate CUDA context from the Python
subprocess). Their VRAM usage is additive with the subprocess models — not time-sliced.

| Component | VRAM | Notes |
|-----------|------|-------|
| Python subprocess CUDA context | ~500MB | Shared via NvVFX_SetCudaStream |
| Rust process CUDA context | ~300MB | Separate context for grading kernels |
| Depth Anything V2 Small | ~1.8GB | Resident in subprocess |
| Depth batch buffers (batch=2) | ~500MB | During inference |
| Maxine VideoSuperRes model | ~1.0-1.5GB | Resident in subprocess |
| Maxine inference workspace | ~200-500MB | TensorRT scratch, scales with resolution |
| Maxine ArtifactReduction model | ~0.5GB | Resident in subprocess |
| Grading CUDA kernels | ~60MB | Per-frame malloc/free in Rust context |
| **Total peak** | **~4.9-5.7GB** | |
| **Headroom on 6GB** | **~0.3-1.1GB** | |

**Escape hatches if VRAM is tight:**
- Set `artifact_reduction: false` → drop ~500MB
- Reduce depth `batch_size` to 1 → save ~250MB
- Set `enabled: false` → revert to current 2.8GB peak

**Upscale factor cap:** On GPUs with ≤6GB VRAM, `upscale_factor` is capped at 2. Factors 3x and
4x produce large intermediates (3240p, 4320p) whose TensorRT workspace exceeds available
headroom. The startup validator rejects `upscale_factor > 2` unless total VRAM exceeds 8GB.

### Configuration

New section in `repos/dorea/config.yaml`:

```yaml
maxine:
  enabled: false            # Master switch — false = zero overhead
  artifact_reduction: true  # Clean compression artifacts before upscale
  upscale_factor: 2         # Supported: 2 (default). 1.33, 1.5, 3, 4 require >8GB VRAM.
```

**CLI overrides:** `dorea grade --maxine` / `--no-maxine` to toggle per-run without editing config.

**Validation at startup:**
- `upscale_factor` not in `[1.33, 1.5, 2, 3, 4]` → hard error
- `upscale_factor > 2` on GPU with ≤6GB VRAM → hard error with explanation
- `enabled: true` + `nvvfx` not importable → hard error:
  "Maxine SDK not found. Install from NGC (nvidia-maxine-vfx-sdk): see docs/guides/maxine-setup.md,
  or set maxine.enabled: false"
- Log nvvfx version at startup for debugging SDK compatibility issues

### User Experience

**Throughput impact:** Maxine enhancement adds 50-100ms per *keyframe*. With temporal interpolation
(~17% keyframes at N=6 interval), the effective per-frame overhead is ~8-17ms averaged. For a
1080p/30fps clip where the current pipeline runs at 6-12 fps:
- Without Maxine: ~80-180ms/frame → 6-12 fps
- With Maxine: ~90-200ms/frame (averaged) → 5-11 fps (~10-15% throughput reduction)

**Progress feedback:**
- Startup log: "Maxine enhancement enabled: upscale_factor=2, artifact_reduction=on (nvvfx v1.x.x)"
- Per-frame timing in verbose mode includes enhancement duration
- Shutdown summary: "Enhanced N keyframes, M passthrough failures" (if any)

**Expected visual quality:** Most noticeable on gradient regions (water column, blue backgrounds),
fine detail (coral texture, fish scales), and compression artifact zones (blocking in low-light
areas). Visually similar to a moderate sharpening pass with artifact cleanup. The oversample
technique recovers detail that was lost to compression and quantization.

### Known Limitations

**8-bit bottleneck:** Maxine SDK accepts only uint8 BGRA input. When enhancement is enabled, the
graded f32 output is quantized to u8 before entering Maxine, and the enhanced u8 output is what
gets encoded. This effectively forces 8-bit precision through the enhancement stage, even in a
10-bit pipeline. The encoder receives u8 data in a 10-bit container. Users should understand that
enabling Maxine trades bit-depth precision for AI-enhanced spatial quality. For footage where
banding is a primary concern, disabling Maxine and relying on the 10-bit grading path may be
preferable.

**Artifact reduction training domain:** NVIDIA's artifact reduction filter is documented as
"optimized for H.264 encoder artifacts." The dorea pipeline processes H.265/HEVC footage. H.264
and H.265 share the same fundamental block-based transform coding approach, so the filter will
still reduce common artifacts (blocking, ringing, mosquito noise). However, effectiveness on
H.265-specific artifacts (larger CTU boundaries, SAO interactions) is unvalidated. A/B test on
real footage early in implementation to confirm benefit.

**Processing order tradeoff:** Enhancement runs post-grade, meaning the grading pipeline may
amplify compression artifacts before Maxine tries to remove them (LUT stretching expands
quantized steps, contrast boost amplifies blocking). Running artifact reduction pre-grade would
avoid this, but pre-grade footage is D-Log M (flat, low-contrast) — likely outside the model's
training distribution. Post-grade is the pragmatic choice; the tradeoff is acknowledged.

### Licensing & Distribution

NVIDIA Maxine EULA prohibits bundling SDK binaries under an open source license. Maxine is
handled as a fully optional external dependency:

1. **Lazy import** — `import nvvfx` only in `MaxineEnhancer.__init__()`. The module exists in
   the codebase but does nothing without the SDK installed.
2. **No pip dependency** — `nvvfx` is NOT listed in dorea's requirements.txt. Users install
   separately per NVIDIA instructions.
3. **Setup guide** — `docs/guides/maxine-setup.md`: install driver 570.190+, download SDK from
   NGC (`nvidia-maxine-vfx-sdk`), install Python bindings, verify with
   `python -c "import nvvfx; print(nvvfx.__version__)"`.
4. **CI/tests** — `DOREA_MAXINE_MOCK=1` enables passthrough mode for CI. IPC protocol tests
   exercise the full round-trip without the proprietary SDK.
5. **No license contamination** — dorea's license is unaffected. Same pattern as ffmpeg-based
   projects handling proprietary codecs.

### Maxine SDK Requirements

- **GPU:** NVIDIA Turing, Ampere, Ada, or Blackwell with Tensor Cores (RTX 3060 Ampere: compatible)
- **Driver:** Linux 570.190+, 580.82+, or 590.44+
- **Input format:** BGRA interleaved uint8
- **Super-resolution input range:** 90p–2160p
- **Artifact reduction input range:** 90p–1080p (skipped with warning for larger inputs)
- **Supported scale factors:** 1.33x, 1.5x, 2x, 3x, 4x (3x/4x require >8GB VRAM)

## Files Changed

| File | Change |
|------|--------|
| `crates/dorea-cli/src/grade.rs` | Sequential `enhance()` call after `grade_frame()`, `--maxine`/`--no-maxine` CLI args |
| `crates/dorea-video/src/inference.rs` | New `enhance()` method, `--maxine` CLI flags for subprocess, response dimension validation |
| `python/dorea_inference/server.py` | Handle `enhance` message type, dispatch to MaxineEnhancer |
| `python/dorea_inference/maxine_enhancer.py` | New file: MaxineEnhancer class with CUDA stream sharing, CI mock mode |
| `python/dorea_inference/protocol.py` | Add enhance request/response types |
| `config.yaml` | New `maxine:` section |
| `docs/guides/maxine-setup.md` | New file: Maxine SDK installation guide |

## Risks

1. **VRAM estimate is approximate** — Maxine does not publish VRAM figures. Headroom is tight
   (~0.3-1.1GB on 6GB). First hardware test must validate via `nvidia-smi`. Escape hatches
   documented above.
2. **Maxine SDK stability** — proprietary SDK may change APIs between versions. Version logged at
   startup for debugging. The lazy-import + passthrough pattern limits blast radius.
3. **CUDA context sharing** — if `NvVFX_SetCudaStream` is not called correctly, duplicate contexts
   consume ~500MB extra VRAM, potentially causing OOM. Must be validated on first hardware test.
4. **IPC throughput** — raw RGB at 1080p is ~5.9MB per frame, ~7.9MB base64-encoded. At 50-100ms
   Maxine inference, IPC overhead (~2-3ms raw encode/decode) is negligible. Would require shared
   memory or CUDA IPC at 4K.
5. **8-bit precision loss** — documented in Known Limitations. Users must choose between AI
   enhancement quality and 10-bit precision.
