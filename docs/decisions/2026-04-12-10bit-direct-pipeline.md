# 10-Bit Direct Pipeline + GPU Frame Cache

**Date:** 2026-04-12
**Status:** Draft — awaiting user approval before implementation plan
**Scope:** `repos/dorea/python/dorea_inference/raune_filter.py` — single file change
**Supersedes:** `2026-04-02-10bit-pipeline-design.md` (designed for deleted LUT pipeline)
**Related:**
- `2026-04-11-minimum-direct-rewrite.md` — stripped repo to direct-mode only
- `2026-04-10-direct-mode-fp16-batch.md` — fp16 RAUNE + batch sizing (#67)
- `2026-04-10-direct-mode-3thread-pipeline.md` — 3-thread producer-consumer (#65)

## Problem

The direct-mode pipeline drops 10-bit footage to 8-bit at three points in
`raune_filter.py`, then writes the 8-bit data into a 10-bit ProRes container
(padding the extra 2 bits with zeros):

1. **Decode** (`decoder_thread`, `frame.to_ndarray(format="rgb24")`) — 10-bit
   source truncated to 8-bit immediately.
2. **GPU download** (`_process_batch` apply loop, `* 255.0` → `torch.uint8`) —
   crushes fp32 result back to 8-bit.
3. **Encode handoff** (`encoder_thread`, `from_ndarray(result_np, format="rgb24")`)
   — PyAV receives 8-bit, upconverts to 10-bit for ProRes.

Additionally, `_process_batch` uploads each full-resolution frame to the GPU
**twice**: once to build the proxy (line 485, then deleted), and again to apply
the transfer (line 527). For 4K at batch=8 this is ~192MB of redundant PCIe
transfer per batch.

## Decision

**Approach B: Bit-depth fix + GPU frame cache.** Fix all three 8-bit drops to
16-bit, and eliminate the double GPU upload by caching decoded frames as uint16
tensors on the GPU. Also delete the dead `run_pipe_mode` function.

## Why 10-bit through the full-res path works with an 8-bit trained model

RAUNE was trained on 8-bit images, but the pipeline uses it as a **delta
generator**, not a direct pixel output:

```
original_proxy  → rgb_to_lab → orig_lab
                                    │
raune(proxy)    → rgb_to_lab → raune_lab  ─  subtract  → delta_lab
                                                              │
                                                        upscale to full-res
                                                              │
full_res_frame  → OKLab → + delta → OKLab⁻¹ → output
```

RAUNE only touches the **proxy-resolution** image to compute a smooth color
correction delta. The delta is a low-frequency signal — 8-bit precision at
proxy resolution is more than sufficient. The full-resolution original frame
flows through at 10-bit precision, has the smooth delta added in fp32 OKLab
space, and encodes to 10-bit output.

The normalize-to-[-1,1] step and fp16 cast for RAUNE inference wash out any
difference between 8-bit and 10-bit proxy input, so we feed 10-bit data to
RAUNE without any quantization step.

## Design

### Scope

All changes in one file: `python/dorea_inference/raune_filter.py`. No Rust
changes — the Rust CLI already passes correct codec args and the output stream
is already configured as `yuv422p10le` for ProRes.

### Data flow (after)

```
Decode (PyAV)         GPU upload (once)     Proxy build           RAUNE
rgb48le uint16 → .cuda() as uint16  →  .float()/65535  →  interpolate  →  model()
                      │                                                      │
                 kept on GPU                                           OKLab delta
                 as uint16 cache                                             │
                      │                                              interpolate to full-res
                      │
                 .float()/65535  →  Triton transfer  →  *65535 uint16
                                                              │
                                                  from_ndarray("rgb48le")
                                                              │
                                                  PyAV encode (yuv422p10le)
                                                     ↑ real 10-bit
```

### Change 1: Decoder thread

Decode as `rgb48le` (uint16) instead of `rgb24` (uint8):

```python
# Before:
rgb = frame.to_ndarray(format="rgb24")  # uint8

# After:
rgb = frame.to_ndarray(format="rgb48le")  # uint16
```

The resize fallback (currently uses PIL `.to_image().resize()` which is 8-bit)
changes to use PyAV's reformatter:

```python
# Before:
if rgb.shape[1] != fw or rgb.shape[0] != fh:
    rgb = np.array(frame.to_image().resize((fw, fh)), dtype=np.uint8)

# After:
if rgb.shape[1] != fw or rgb.shape[0] != fh:
    frame_resized = frame.reformat(width=fw, height=fh, format="rgb48le")
    rgb = frame_resized.to_ndarray(format="rgb48le")
```

### Change 2: `_process_batch` — GPU frame cache + 16-bit math

Upload each frame once as uint16, keep on GPU for reuse:

```python
def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype):
    n = len(batch_frames_np)
    results = []

    with torch.no_grad():
        # Upload once, keep uint16 on GPU
        full_u16_gpu = []
        proxy_tensors = []
        for rgb_np in batch_frames_np:
            t_u16 = torch.from_numpy(rgb_np).cuda()              # uint16 on GPU
            full_u16_gpu.append(t_u16)
            full_f32 = t_u16.float() / 65535.0                   # (H,W,3) fp32 [0,1]
            full_f32 = full_f32.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
            proxy_t = F.interpolate(full_f32, size=(ph, pw), mode="bilinear", align_corners=False)
            proxy_norm = (proxy_t - 0.5) / 0.5
            proxy_tensors.append(proxy_norm.squeeze(0))
            del full_f32                                          # free fp32, keep uint16

        # ... RAUNE inference + OKLab delta unchanged ...

        # Apply transfer per frame — reuse GPU cache, no re-upload
        for i in range(n):
            full_t = full_u16_gpu[i].float() / 65535.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)
            result = transfer_fn(full_t, delta_full[i:i+1])
            del full_t

            # GPU → CPU → uint16
            result_u16 = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 65535.0
                         ).to(torch.int32).cpu().numpy().astype(np.uint16)
            results.append(result_u16)
            del result

        del full_u16_gpu, delta_full

    return results
```

Note: PyTorch has no native uint16 dtype, so the download path uses int32 →
numpy uint16. This matches the pattern already used in the (now-deleted) pipe
mode at line 615.

### Change 3: Encoder thread

```python
# Before:
out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb24")

# After:
out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb48le")
```

PyAV handles rgb48le → yuv422p10le conversion internally for ProRes. For HEVC
10-bit (`yuv420p10le`), same. For H264 8-bit (`yuv420p`), PyAV downconverts
from 16-bit to 8-bit at encode time — correct since H264 output is 8-bit by
definition.

### Change 4: Delete pipe mode

Delete:
- `run_pipe_mode` function (~90 lines)
- The `else` dispatch branch in `main()` that calls it

The `main()` function becomes:

```python
def main():
    # ... parse args, load model (unchanged) ...

    if not args.input or not args.output:
        print("error: --input and --output required", file=sys.stderr)
        sys.exit(1)
    run_single_process(args, model, normalize, model_dtype)
```

### Triton kernel — no change

The Triton kernel stays at fp16. Its 11-bit mantissa is sufficient for 10-bit
output. The kernel receives fp32 input (from the uint16 → float / 65535.0
conversion), internally casts to fp16 for the fused OKLab transfer, and outputs
fp16 which is cast back to fp32 by `triton_oklab_transfer`. The calling code
then scales to uint16.

### What's NOT changing

- Triton kernel `_oklab_transfer_kernel` — fp16 stays
- `rgb_to_lab` / `lab_to_rgb` — already fp32
- `triton_oklab_transfer` / `pytorch_oklab_transfer` — receive fp32, unchanged
- RAUNE model loading, fp16 conversion, InstanceNorm handling
- 3-thread queue structure, backpressure, error handling
- Rust CLI (`dorea-cli`) — already passes correct codec args
- `dorea-video` crate — unchanged

## VRAM budget (8 frames, 4K 3840x2160, RTX 3060 6 GB)

| Allocation | Size | Lifetime |
|---|---|---|
| uint16 GPU cache (8 frames) | 380 MB | Full batch |
| fp32 working copy (1 frame at a time) | 95 MB | Per-frame, freed |
| Proxy batch fp16 (8 frames, 1440p) | 56 MB | Through RAUNE |
| RAUNE model fp16 | ~100 MB | Entire run |
| Delta full-res fp32 (8 frames) | 760 MB | After RAUNE |
| **Peak** | **~1.4 GB** | During delta upscale |

Well within 6 GB. The uint16 cache adds 380 MB compared to the previous
pipeline, but the double-upload elimination removes transient fp32 allocations
that would have overlapped.

## Bit-depth trace (after)

| Stage | Format | Bits | Drop? |
|---|---|---|---|
| Decode | `rgb48le` | 16-bit (10-bit source preserved) | No |
| GPU upload | `.cuda()` uint16 | 16-bit | No |
| Float conversion | `.float() / 65535.0` | fp32 | No |
| Proxy downscale | `F.interpolate` | fp32 | No |
| RAUNE normalize | `(x - 0.5) / 0.5` → fp16 cast | fp16 (11-bit mantissa) | No (proxy only) |
| RAUNE inference | `model()` | fp16 | No (proxy only) |
| OKLab delta | `rgb_to_lab` × 2, subtract | fp32 | No |
| Delta upscale | `F.interpolate` | fp32 | No |
| Reuse GPU cache | uint16 → `.float() / 65535.0` | fp32 | No |
| Triton transfer | fp32 input → fp16 internal → fp32 output | fp16 kernel | Negligible (11 > 10 bits) |
| GPU download | `* 65535.0` → int32 → uint16 | 16-bit | No |
| Encode handoff | `from_ndarray("rgb48le")` | 16-bit | No |
| ProRes output | `yuv422p10le` | 10-bit | Correct (16→10 at codec level) |

Zero precision drops from 10-bit source to 10-bit output.

## Performance impact

- **PCIe bandwidth**: Net savings. Eliminated one full-res upload per frame
  (~24 MB at 4K uint8). The single upload is now uint16 (~48 MB) instead of
  uint8 (~24 MB), so net change is 48 MB vs. 2×24 MB = 48 MB — **same total
  bandwidth, but one transfer instead of two** (fewer kernel launches, less
  synchronization overhead).
- **Decode CPU cost**: `rgb48le` conversion is slightly more expensive than
  `rgb24` in libswscale. Negligible — decode thread is not the bottleneck.
- **Encode CPU cost**: `rgb48le` → `yuv422p10le` conversion is marginally more
  work than `rgb24` → `yuv422p10le`. Negligible.
- **GPU memory**: +380 MB for uint16 cache. Peak ~1.4 GB, well within budget.
- **GPU compute**: Unchanged — Triton kernel and RAUNE inference are identical.

## Backward compatibility

| Scenario | Behavior |
|---|---|
| 8-bit sRGB input (H264) | PyAV decodes to rgb48le (zero-extends 8→16), pipeline processes at 16-bit, encodes to `yuv420p` 8-bit. Correct. |
| 10-bit D-Log M input (HEVC) | Full 10-bit preserved through pipeline. Correct. |
| 10-bit I-Log input (ProRes) | Full 10-bit preserved through pipeline. Correct. |
| `--output-codec h264` | PyAV downconverts 16-bit to 8-bit at encode. Correct — H264 is 8-bit. |
| `--output-codec hevc` | 10-bit output via `yuv420p10le`. Correct. |
| `--output-codec prores_ks` | 10-bit output via `yuv422p10le`. Correct. |
| Legacy pipe mode callers | Will break — pipe mode deleted. Rust CLI already uses single-process mode; no other callers known. |

## Acceptance criteria

- `raune_filter.py` decodes as `rgb48le` and encodes via `rgb48le`.
- `_process_batch` uploads each frame to GPU exactly once (no double upload).
- Output file from a 10-bit HEVC source verified as 10-bit via
  `ffprobe -show_streams` (bits_per_raw_sample=10).
- `run_pipe_mode` function does not exist in the file.
- `dorea <10-bit-input.mp4>` runs on RTX 3060 6 GB without OOM.
- Existing 8-bit input still produces valid output.

## Alternatives rejected

**Approach A (bit-depth fix only, keep double upload):** Smaller diff but leaves
~192 MB redundant PCIe transfer per batch. The GPU cache is 3 extra lines in
the same function we're already modifying.

**Approach C (upgrade Triton kernel to fp32):** fp16's 11-bit mantissa exceeds
10-bit requirements. The speed advantage of fp16 (2× memory bandwidth) is worth
keeping. Precision loss is negligible in the nonlinear OKLab transfer space.

**Keep pipe mode:** Dead code — the Rust CLI exclusively uses single-process
mode (`--input`/`--output`). No known external callers. Deleting while we're
already modifying the file is cleaner than leaving it to rot.
