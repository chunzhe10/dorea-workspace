# TensorRT FP16 Acceleration for RAUNE-Net Inference

**Date:** 2026-04-12
**Issue:** chunzhe10/dorea#72
**Status:** Approved

## Summary

Replace PyTorch's CUDA inference path for RAUNE-Net with a TensorRT FP16 engine to
achieve 2-4x speedup. INT8 is excluded to preserve 10-bit color fidelity (INT8 has
only 256 discrete activation levels vs 1024 needed for 10-bit).

## Context

Current pipeline runs RAUNE-Net at ~7fps (3-thread pipeline) on RTX 3060 6GB using
PyTorch fp16 with manual InstanceNorm fp32 restoration. The GPU thread is the
bottleneck — decode and encode overlap but GPU inference dominates wall time.

## Approach Selection

Four approaches were evaluated:

| Approach | Verdict | Reason |
|---|---|---|
| **A. torch_tensorrt** | Eliminated | Python 3.13 not supported in 2.6.x; InstanceNorm has no TRT converter — 30 graph breaks per forward pass |
| **B. Manual ONNX -> TensorRT** | **Selected** | All RAUNE-Net ops natively supported in TRT 10.x; single unified engine; zero-copy PyTorch tensor integration |
| C. ONNX Runtime + TRT EP | Runner-up | Same TRT engine but adds ORT dependency (~500MB); less control over tensor memory |
| D. torch.compile (inductor) | Complementary | 15-30% speedup as one-liner; used as bonus on PyTorch fallback path |

### Why Approach B Wins

1. **Every RAUNE-Net op is natively supported in TRT 10.x**: Conv2d, InstanceNorm2d,
   ConvTranspose2d, ReflectionPad2d, ReLU, Tanh, Sigmoid, AdaptiveMaxPool2d,
   AdaptiveAvgPool2d (CBAM), Concat, Add, Mul. The entire model compiles into a
   single fused engine with zero fallback partitions.

2. **TRT handles InstanceNorm FP16/FP32 precision internally**: Unlike PyTorch's
   manual `model.half()` + InstanceNorm `.float()` workaround, TRT's native
   InstanceNorm implementation automatically keeps variance computation in FP32
   while running the surrounding ops in FP16.

3. **Zero-copy tensor integration**: TRT's `set_tensor_address()` accepts PyTorch
   tensor `data_ptr()` directly. No memcpy between frameworks.

4. **Python 3.13 + CUDA 12.4 compatible**: `tensorrt-cu12` pip package supports both.

### Why Not torch_tensorrt

Critical finding: `torch_tensorrt` has NO InstanceNorm converter. In a 30-block
ResNet with InstanceNorm in every block, this means 30+ graph breaks where execution
bounces between TRT engines and PyTorch eager mode. The overhead of these transitions
would likely make it *slower* than pure PyTorch. Additionally, torch_tensorrt 2.6.x
does not ship Python 3.13 wheels.

## Architecture

### Change Scope

Surgically scoped — only the RAUNE inference call changes:

```
Current:  PyAV decode -> [PyTorch RAUNE fp16] -> OKLab transfer -> PyAV encode
After:    PyAV decode -> [TRT engine fp16]     -> OKLab transfer -> PyAV encode
```

The surrounding pipeline (PyAV decode/encode, OKLab delta computation, Triton fused
kernel, 3-thread architecture) is completely unchanged.

### New Components

#### 1. ONNX Export Script (`python/dorea_inference/export_onnx.py`)

- Loads RAUNE-Net from weights, exports to ONNX opset 17
- Dynamic axes on H/W dimensions (batch is static per engine build)
- Runs `onnxsim` post-export (required: InstanceNorm bias tensor must be an
  initializer for TRT to parse it)
- Validates with `onnx.checker.check_model()`
- CLI: `python -m dorea_inference.export_onnx --weights <path> --models-dir <path> --output raune_net.onnx`

#### 2. TRT Engine Wrapper (`python/dorea_inference/trt_engine.py`)

`RauneTRTEngine` class:

- `build_engine(onnx_path, engine_path, shapes, fp16)`: Build from ONNX (2-5 min)
- `cache_key(onnx_path, gpu, sm, trt_ver, shapes, fp16)`: SHA-256 for cache invalidation
- `__init__(engine_path)`: Deserialize engine, create execution context + CUDA stream
- `infer(input_batch) -> Tensor`: Zero-copy inference via `data_ptr()`

**Engine caching:**
- Cache dir: `models/raune_net/trt_cache/`
- Key: SHA-256(ONNX hash, TRT version, GPU SM, shape profile, precision)
- Miss: build from ONNX, serialize to disk
- Hit: deserialize (<2s)
- Auto-invalidation when any key component changes

**Shape profile:**
- Optimization profile: min=1x3x270x480, opt=batch x3x proxy_h x proxy_w, max=batch x3x1080x1920
- Covers all proxy resolutions used by the pipeline

#### 3. Integration (`python/dorea_inference/raune_filter.py`)

New CLI flags:
- `--tensorrt`: Opt-in TRT inference (default: off, existing PyTorch path unchanged)
- `--trt-cache-dir`: Engine cache directory (default: `models/raune_net/trt_cache/`)

Changes to `main()`:
- If `--tensorrt`: build/load TRT engine, skip PyTorch model loading
- If not `--tensorrt`: existing path unchanged + `torch.compile(model, mode="default")`

Changes to `_process_batch()`:
- Accept either PyTorch model or TRT engine as the inference callable
- Replace `model(proxy_batch)` with engine-agnostic call

#### 4. Tests (`python/tests/test_trt_engine.py`)

- ONNX export validation
- Numerical accuracy: PyTorch vs TRT output PSNR > 40dB
- Engine cache hit/miss/invalidation
- Performance comparison

### Error Handling

Per project rule (fail-fast enforcement, no silent fallbacks):
- `--tensorrt` specified but tensorrt not installed -> hard error
- Engine build fails -> hard error with diagnostic
- Engine cache invalid -> automatic rebuild (cache management, not fallback)
- TRT inference fails -> hard error
- The opt-in `--tensorrt` flag IS the user's choice — without it, PyTorch runs

### Dependencies

```
pip install onnx onnxsim tensorrt-cu12
```

- `onnx` + `onnxsim`: ~50MB (export-time only)
- `tensorrt-cu12`: ~1GB (runtime)

### Expected Performance

| Metric | PyTorch fp16 (current) | TRT fp16 (expected) |
|---|---|---|
| Inference fps | ~7 | 15-25 |
| VRAM (model + activations) | ~700MB-1GB | ~300-450MB |
| First-run overhead | None | 2-5 min engine build |
| Subsequent-run overhead | None | <2s engine load |

### Files Changed

| File | Change |
|---|---|
| `python/dorea_inference/export_onnx.py` | New |
| `python/dorea_inference/trt_engine.py` | New |
| `python/dorea_inference/raune_filter.py` | Modified |
| `python/tests/test_trt_engine.py` | New |

### Risks

1. **InstanceNorm precision**: TRT's native InstanceNorm keeps variance in fp32
   automatically, but if output diverges from PyTorch baseline, we may need to
   force specific layers to fp32 via layer-level precision flags.
2. **ONNX export warning**: PyTorch emits a training-mode warning for InstanceNorm2d
   even after `model.eval()`. This is cosmetically ugly but functionally correct
   (InstanceNorm always computes per-sample statistics regardless of train/eval).
3. **Engine portability**: TRT engines are tied to GPU architecture. If the GPU
   changes, the engine must be rebuilt. Cache key handles this automatically.
