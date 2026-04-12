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
16-bit, and eliminate the double GPU upload by caching decoded frames as int32
tensors on the GPU (PyTorch lacks native uint16 — see GPU cache section). Also
delete the dead `run_pipe_mode` function in a separate commit.

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

**Note on sRGB assumption:** The OKLab conversion functions (`rgb_to_lab`,
`lab_to_rgb`) and the Triton kernel hardcode the sRGB OETF (pow 2.4
linearization). D-Log M and I-Log footage uses different log curves, so the
linearization is technically incorrect. However, the delta subtraction
largely cancels this: both the original proxy and the RAUNE output go through
the same (wrong) linearization, so the delta captures the relative color
shift correctly. The full-res transfer applies the same wrong linearization
and its inverse, which cancel. This is a pre-existing design choice, not
introduced by this spec. Correct log-to-linear handling is a future
improvement tracked separately.

## Design

### Scope

All changes in one file: `python/dorea_inference/raune_filter.py`. No Rust
changes — the Rust CLI already passes correct codec args and the output stream
is already configured as `yuv422p10le` for ProRes.

### Data flow (after)

```
Decode (PyAV)         GPU upload (once)          Proxy build           RAUNE
rgb48le uint16 → .astype(int32) → .cuda()  →  .float()/65535  →  interpolate  →  model()
                         │                                                          │
                    kept on GPU                                               OKLab delta
                    as int32 cache                                                  │
                         │                                                  interpolate to full-res
                         │
                    .float()/65535  →  Triton transfer(fp32 in)  →  *65535+0.5 int32 → uint16
                                                                          │
                                                              from_ndarray("rgb48le")
                                                                          │
                                                              PyAV encode (yuv422p10le)
                                                                 ↑ real 10-bit
```

**Why int32, not uint16:** PyTorch has no native `torch.uint16` dtype.
`torch.from_numpy(uint16_array)` reinterprets as `torch.int16` (signed),
causing values >32767 to become negative and produce wrong normalized values
after `.float() / 65535.0`. Using `.astype(np.int32)` before
`torch.from_numpy` avoids this. This matches the pattern used by the existing
pipe-mode code at line 581.

**Why `/65535.0` is correct for 10-bit source:** When PyAV/libswscale converts
a 10-bit source to `rgb48le`, it zero-extends via bit-shift replication
(`val << 6 | val >> 4`), producing full 16-bit range values {0, 64, 128, ...,
65535}. Dividing by 65535.0 correctly maps these to [0.0, 1.0]. Do not "fix"
this to `/1023.0` — that would clip all values to [0.0, 0.016].

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

Note: `frame.reformat()` assumes SAR=1:1 (square pixels). All known dorea
input sources (DJI Action 4, Insta360 X5) produce SAR=1:1 footage.

### Change 2: `_process_batch` — GPU frame cache + 16-bit math

Upload each frame once as int32 on GPU, keep for reuse:

```python
def _process_batch(batch_frames_np, model, normalize, fw, fh, pw, ph, transfer_fn, model_dtype):
    """Process a batch of frames on GPU. Returns list of uint16 HWC numpy arrays."""
    n = len(batch_frames_np)
    results = []

    with torch.no_grad():
        # Upload once as int32, keep on GPU for reuse in apply loop.
        # int32 because PyTorch has no uint16 — torch.from_numpy(uint16)
        # reinterprets as int16 (signed), corrupting values > 32767.
        full_gpu_cache = []
        proxy_tensors = []
        for rgb_np in batch_frames_np:
            t_i32 = torch.from_numpy(rgb_np.astype(np.int32)).cuda()  # int32 on GPU
            full_gpu_cache.append(t_i32)
            full_f32 = t_i32.float() / 65535.0                   # (H,W,3) fp32 [0,1]
            full_f32 = full_f32.permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
            proxy_t = F.interpolate(full_f32, size=(ph, pw), mode="bilinear", align_corners=False)
            proxy_norm = (proxy_t - 0.5) / 0.5
            proxy_tensors.append(proxy_norm.squeeze(0))
            del full_f32                                          # free fp32, keep int32

        # RAUNE inference + OKLab delta — carried verbatim from current code.
        # The proxy normalization produces [-1, 1] range regardless of whether
        # the source was 8-bit or 10-bit, so this section is unaffected.
        proxy_batch = torch.stack(proxy_tensors).cuda()
        del proxy_tensors

        if proxy_batch.dtype != model_dtype:
            proxy_batch = proxy_batch.to(model_dtype)

        raune_out = model(proxy_batch)
        raune_out = raune_out.float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)

        rh, rw = raune_out.shape[2], raune_out.shape[3]
        if rh != ph or rw != pw:
            raune_out = F.interpolate(raune_out, size=(ph, pw), mode="bilinear", align_corners=False)

        orig_proxy = (proxy_batch.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        del proxy_batch

        raune_lab = rgb_to_lab(raune_out)
        orig_lab = rgb_to_lab(orig_proxy)
        delta_lab = raune_lab - orig_lab
        del raune_out, orig_proxy, raune_lab, orig_lab

        delta_full = F.interpolate(delta_lab, size=(fh, fw), mode="bilinear", align_corners=False)
        del delta_lab

        # Apply transfer per frame — reuse GPU cache, no re-upload
        for i in range(n):
            full_t = full_gpu_cache[i].float() / 65535.0
            full_t = full_t.permute(2, 0, 1).unsqueeze(0)
            result = transfer_fn(full_t, delta_full[i:i+1])
            del full_t

            # GPU → CPU → uint16 (round-half-up to avoid systematic -0.5 LSB bias)
            result_u16 = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 65535.0 + 0.5
                         ).to(torch.int32).cpu().numpy().astype(np.uint16)
            results.append(result_u16)
            del result

        del full_gpu_cache, delta_full

    return results
```

Key differences from current code:
- `astype(np.int32)` before `torch.from_numpy` to avoid int16 sign corruption
- Single GPU upload per frame (cache reused in apply loop)
- `/ 65535.0` instead of `/ 255.0`
- `+ 0.5` before int32 cast for correct rounding (avoids truncation bias)
- Returns uint16 arrays instead of uint8

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

### Change 4: Delete pipe mode (separate commit)

This is a separate commit from the 10-bit changes to keep rollback clean.

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

**Caller audit:** The Rust CLI invokes `python -m dorea_inference.raune_filter`
with `--input` and `--output` args (see `grading.rs:47-66`), which routes to
`run_single_process`. Pipe mode is entered only when `--input` is omitted. No
Rust code path omits `--input`. No other orchestration scripts call
`raune_filter.py` directly.

### Change 5: Triton wrapper — pass fp32 input to kernel

The Triton kernel itself is unchanged. But `triton_oklab_transfer` currently
casts the input to fp16 *before* passing to the kernel (line 176:
`.squeeze(0).half()`), which the kernel immediately promotes back to fp32
(line 104: `.to(tl.float32)`). This round-trip through fp16 quantizes the
input before the nonlinear `pow(2.4)` and `cbrt` operations, which can
amplify error to ~2 LSBs in dark regions.

Fix: pass fp32 tensors to the kernel, keep fp16 only for the output store:

```python
def triton_oklab_transfer(frame_nchw_f32, delta_nchw_f32):
    """Fused Triton kernel: entire OKLab transfer in one kernel launch."""
    _, C, H, W = frame_nchw_f32.shape
    # Pass fp32 to kernel — avoid precision loss before nonlinear transforms.
    # Kernel output stays fp16 (sufficient for 10-bit).
    frame_flat = frame_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    delta_flat = delta_nchw_f32.squeeze(0).reshape(3, -1).contiguous()   # fp32
    n_pixels = H * W
    out_flat = torch.empty(3, n_pixels, dtype=torch.float16, device=frame_flat.device)

    BLOCK_SIZE = 1024
    grid = ((n_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _oklab_transfer_kernel[grid](
        frame_flat, delta_flat, out_flat,
        n_pixels=n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.reshape(1, 3, H, W).float()
```

The kernel's internal loads (`.to(tl.float32)` at lines 104-106) now receive
fp32 directly instead of fp16-promoted-to-fp32. The output store remains fp16
(line 168-170). The kernel body is unchanged.

The Triton kernel's fp16 output has 11-bit effective mantissa (10-bit stored +
implicit leading 1 for normalized values), which exceeds 10-bit output
requirements.

### What's NOT changing

- Triton kernel `_oklab_transfer_kernel` body — fp16 output stays
- `rgb_to_lab` / `lab_to_rgb` — already fp32
- `pytorch_oklab_transfer` — receives fp32, unchanged
- RAUNE model loading, fp16 conversion, InstanceNorm handling
- 3-thread queue structure, backpressure, error handling
- Rust CLI (`dorea-cli`) — already passes correct codec args
- `dorea-video` crate — unchanged

## VRAM budget (8 frames, 4K 3840x2160, RTX 3060 6 GB)

Individual allocations:

| Allocation | Size | Lifetime |
|---|---|---|
| RAUNE model fp16 | ~100 MB | Entire run |
| int32 GPU cache (8 frames) | 760 MB | Full batch |
| fp32 working copy (1 frame) | 95 MB | Per-frame in proxy loop, freed |
| Proxy batch fp16 (8 frames, 1440p) | 56 MB | Through RAUNE, freed after delta |
| OKLab proxy temporaries (raune_lab, orig_lab, delta_lab) | 3 × 56 MB = 168 MB | During delta computation |
| Delta full-res fp32 (8 frames) | 796 MB | After RAUNE, freed per-frame |
| Triton fp16 intermediates (frame_flat, delta_flat, out_flat) | ~142 MB | Per-frame in apply loop |
| Triton JIT compilation cache | ~50-100 MB | Entire run |

Peak overlap windows (before PyTorch allocator overhead):

| Phase | Concurrent allocations | Total |
|---|---|---|
| Proxy build | model (100) + cache (760) + fp32 working (95) + proxy tensors (56) | ~1.0 GB |
| RAUNE inference | model (100) + cache (760) + proxy batch fp16 (56) | ~0.9 GB |
| OKLab delta compute | model (100) + cache (760) + proxy temps (168) + delta_lab (56) | ~1.1 GB |
| Delta upscale | model (100) + cache (760) + delta_full (796) | ~1.7 GB |
| Apply loop (Triton) | model (100) + cache (760) + delta_full (796) + fp32 working (95) + Triton intermediates (142) | **~1.9 GB** |
| Apply loop (Triton, after first del) | decreases per frame as delta_full slices are consumed | ≤1.9 GB |

**True peak: ~1.9 GB** (before PyTorch caching allocator overhead, typically
1.2-1.5× → ~2.3-2.9 GB). Well within 6 GB.

The int32 cache (760 MB) is larger than the original spec's uint16 estimate
(380 MB) because PyTorch lacks native uint16 and requires int32. This is the
main VRAM cost of the GPU frame cache. Without the cache, the double-upload
approach would use ~0 MB cache but incur ~192 MB redundant PCIe per batch.

## Bit-depth trace (after)

| Stage | Format | Bits | Drop? |
|---|---|---|---|
| Decode | `rgb48le` | 16-bit (10-bit zero-extended via libswscale bit-shift replication) | No |
| numpy → int32 | `.astype(np.int32)` | 32-bit (lossless from uint16) | No |
| GPU upload | `.cuda()` int32 | 32-bit | No |
| Float conversion | `.float() / 65535.0` | fp32 | No |
| Proxy downscale | `F.interpolate` | fp32 | No |
| RAUNE normalize | `(x - 0.5) / 0.5` → fp16 cast | fp16 (11-bit effective mantissa) | No (proxy only) |
| RAUNE inference | `model()` | fp16 | No (proxy only) |
| OKLab delta | `rgb_to_lab` × 2, subtract | fp32 | No |
| Delta upscale | `F.interpolate` | fp32 | No |
| Reuse GPU cache | int32 → `.float() / 65535.0` | fp32 | No |
| Triton transfer | fp32 input → fp32 compute → fp16 output | fp16 output | Negligible (11 > 10 bits) |
| GPU download | `* 65535.0 + 0.5` → int32 → uint16 | 16-bit (correctly rounded) | No |
| Encode handoff | `from_ndarray("rgb48le")` | 16-bit | No |
| ProRes output | `yuv422p10le` | 10-bit | Correct (16→10 at codec level) |

Zero precision drops from 10-bit source to 10-bit output.

## Performance impact

- **PCIe upload**: The single int32 upload is 96 MB per 4K frame (4 bytes ×
  3 channels × 3840 × 2160). Previously: two uint8 uploads = 2 × 24 MB =
  48 MB total. Net increase of ~48 MB upload bandwidth per frame, but in one
  transfer instead of two (fewer kernel launches, less synchronization). The
  upload increase is offset by eliminating the second transfer's latency.
- **PCIe download**: Increases from uint8 (24 MB/frame) to int32 (96 MB/frame)
  — 4× per frame. This is a deliberate tradeoff for 10-bit precision. The
  `.astype(np.uint16)` numpy conversion is CPU-side and negligible.
- **Decode CPU cost**: `rgb48le` conversion is heavier than `rgb24` in
  libswscale (per-pixel byte-swapping and 16-bit scaling vs memcpy-like path).
  The decoder thread may shift closer to being a bottleneck. The existing
  stage timing infrastructure (`decode_busy`, `gpu_busy`, `encode_busy`)
  should be checked after implementation. If decode becomes the bottleneck,
  consider NVDEC for HEVC sources (future work).
- **Queue CPU memory**: `q_decoded` holds up to 2 batches of 8 frames. At
  uint16: `2 × 8 × 3840 × 2160 × 3 × 2 ≈ 794 MB` CPU RAM (was ~397 MB at
  uint8). The queue comment should be updated to reflect this.
- **GPU memory**: +760 MB for int32 cache (was estimated 380 MB for uint16
  before the PyTorch dtype constraint was identified). Peak ~1.9 GB before
  allocator overhead, ~2.3-2.9 GB after. Well within 6 GB budget.
- **GPU compute**: Triton kernel body unchanged. The wrapper saves one fp16
  cast on input (was `.half()`, now stays fp32). RAUNE inference identical.
- **8-bit input overhead**: 8-bit H264 sources now decode to rgb48le (2×
  memory), upload as int32, process at fp32, and downconvert at encode. This
  is a deliberate tradeoff to keep a single code path. Expected throughput
  regression is <5% (dominated by GPU compute, not I/O). If measured
  regression exceeds 10%, consider a conditional path (future work).

## Backward compatibility

| Scenario | Behavior |
|---|---|
| 8-bit sRGB input (H264) | PyAV decodes to rgb48le (zero-extends 8→16), pipeline processes at 16-bit, encodes to `yuv420p` 8-bit. Correct. |
| 10-bit D-Log M input (HEVC) | Full 10-bit preserved through pipeline. Correct. |
| 10-bit I-Log input (ProRes) | Full 10-bit preserved through pipeline. Correct. |
| `--output-codec h264` | PyAV downconverts 16-bit to 8-bit at encode. Correct — H264 is 8-bit. |
| `--output-codec hevc` | 10-bit output via `yuv420p10le`. Correct. |
| `--output-codec prores_ks` | 10-bit output via `yuv422p10le`. Correct. |
| Legacy pipe mode callers | Will break — pipe mode deleted (separate commit). Rust CLI exclusively uses `--input`/`--output` args (verified: `grading.rs:47-66`). No other callers found in repo. |

## Acceptance criteria

- `raune_filter.py` decodes as `rgb48le` and encodes via `rgb48le`.
- `_process_batch` uses `astype(np.int32)` before `torch.from_numpy`, caches
  on GPU, and reuses in the apply loop (no second `torch.from_numpy` call).
- **Pixel-level 10-bit verification:** decode the graded output with
  `ffmpeg -i output.mov -pix_fmt rgb48le -f rawvideo - | python -c "..."`,
  read raw uint16 values, and verify the lower 6 bits are non-zero for a
  10-bit gradient source. A synthetic ramp test pattern (0-1023 across 1024
  pixels) makes this straightforward. Container metadata
  (`ffprobe bits_per_raw_sample=10`) is necessary but not sufficient.
- `run_pipe_mode` function does not exist in the file (separate commit from
  10-bit changes).
- `grep -n "raune_filter.py" repos/dorea/crates/dorea-cli/src/` confirms all
  invocations pass `--input` and `--output` (no pipe-mode callers).
- `dorea <10-bit-input.mp4>` runs on RTX 3060 6 GB without OOM.
- Existing 8-bit input still produces valid output (rgb48le decode zero-extends
  correctly, H264 output downconverts at encode).
- `_process_batch` docstring says "uint16" not "uint8".
- Stage timing output (`decode_busy`, `gpu_busy`, `encode_busy`) is checked
  on a representative clip. No specific threshold required, but logged for
  comparison against pre-change baseline.

## Test plan

No automated tests exist for `raune_filter.py` today. The implementation
should add:

1. **Synthetic round-trip test:** Create a 16×16 uint16 numpy array with known
   values (including >32767 to catch int16 sign corruption), run through
   `_process_batch` with a mocked RAUNE model (identity function), verify
   output dtype is uint16 and values round-trip within ±1 LSB.
2. **8-bit backward compat test:** Same as above but with uint8 input converted
   to uint16 (simulating rgb48le zero-extension of 8-bit source). Verify
   output is valid.
3. **Rounding test:** Verify the `+ 0.5` rounding produces correct
   round-half-up behavior for edge values (0.0, 0.5/65535, 1.0).

These tests mock the RAUNE model to avoid GPU/weight dependencies in CI.
Full end-to-end testing requires the RTX 3060 workstation and real weights.

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
