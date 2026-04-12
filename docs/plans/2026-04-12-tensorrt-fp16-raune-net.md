# TensorRT FP16 RAUNE-Net Acceleration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PyTorch CUDA inference with TensorRT FP16 for 2-4x speedup on RAUNE-Net, opt-in via `--tensorrt` flag.

**Architecture:** Export RAUNE-Net to ONNX, build a TensorRT FP16 engine (cached to disk), and integrate a thin engine wrapper into the existing `raune_filter.py` 3-thread pipeline. The only code path that changes is `model(proxy_batch)` → `trt_engine.infer(proxy_batch)`.

**Tech Stack:** Python 3.13, PyTorch 2.6.0+cu124, TensorRT 10.x (`tensorrt-cu12`), ONNX (`onnx`, `onnxsim`), CUDA 12.4, RTX 3060 (SM 8.6).

**Design Spec:** `docs/decisions/2026-04-12-tensorrt-fp16-raune-net-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `python/dorea_inference/export_onnx.py` | Create | ONNX export + simplification CLI |
| `python/dorea_inference/trt_engine.py` | Create | TRT engine build, cache, inference wrapper |
| `python/dorea_inference/raune_filter.py` | Modify | Add `--tensorrt` / `--trt-cache-dir` flags, TRT inference path |
| `crates/dorea-cli/src/pipeline/mod.rs` | Modify | Add `tensorrt: bool` to PipelineConfig |
| `crates/dorea-cli/src/pipeline/grading.rs` | Modify | Pass `--tensorrt` to Python subprocess |
| `crates/dorea-cli/src/grade.rs` | Modify | Add `--tensorrt` CLI flag |
| `python/tests/test_trt_engine.py` | Create | ONNX export + TRT engine tests |

---

### Task 1: Install Dependencies

**Files:** None (environment setup)

- [ ] **Step 1: Install ONNX and TensorRT packages**

```bash
/opt/dorea-venv/bin/pip install onnx onnxsim tensorrt-cu12
```

Expected: packages install successfully. `tensorrt-cu12` includes `libnvinfer.so.10`, `trtexec`, and Python bindings.

- [ ] **Step 2: Verify installation**

```bash
/opt/dorea-venv/bin/python -c "
import onnx; print(f'onnx {onnx.__version__}')
import onnxsim; print(f'onnxsim {onnxsim.__version__}')
import tensorrt as trt; print(f'tensorrt {trt.__version__}')
"
```

Expected: all three import successfully with version numbers printed.

- [ ] **Step 3: Commit** (no file changes — dependency install only, skip commit)

---

### Task 2: ONNX Export Script

**Files:**
- Create: `python/dorea_inference/export_onnx.py`
- Test: `python/tests/test_trt_engine.py` (partial — ONNX export tests)

- [ ] **Step 1: Write the failing test for ONNX export**

Create `python/tests/test_trt_engine.py`:

```python
"""Tests for TensorRT engine: ONNX export, engine build, inference accuracy."""

import pytest
import torch
import numpy as np
from pathlib import Path

# Skip entire module if tensorrt is not installed
trt = pytest.importorskip("tensorrt")
onnx = pytest.importorskip("onnx")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "raune_net"
WEIGHTS = MODELS_DIR / "weights_95.pth"


def _load_pytorch_model():
    """Load RAUNE-Net in fp32 for reference."""
    import sys
    sys.path.insert(0, str(MODELS_DIR))
    from models.raune_net import RauneNet

    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64)
    state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
class TestOnnxExport:
    def test_export_produces_valid_onnx(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(
            weights=str(WEIGHTS),
            models_dir=str(MODELS_DIR),
            output=str(onnx_path),
        )
        assert onnx_path.exists()
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_export_output_matches_pytorch(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        import onnxruntime as ort

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(
            weights=str(WEIGHTS),
            models_dir=str(MODELS_DIR),
            output=str(onnx_path),
        )

        # PyTorch reference
        pt_model = _load_pytorch_model()
        x = torch.randn(1, 3, 270, 480)
        with torch.no_grad():
            pt_out = pt_model(x).numpy()

        # ONNX Runtime reference
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": x.numpy()})[0]

        # Should be numerically identical (both fp32)
        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_trt_engine.py::TestOnnxExport::test_export_produces_valid_onnx -v 2>&1 | tail -5
```

Expected: FAIL with `ImportError: cannot import name 'export_raune_onnx'`

- [ ] **Step 3: Write the ONNX export module**

Create `python/dorea_inference/export_onnx.py`:

```python
"""Export RAUNE-Net to ONNX with onnxsim simplification.

Usage:
    python -m dorea_inference.export_onnx \
        --weights models/raune_net/weights_95.pth \
        --models-dir models/raune_net \
        --output raune_net.onnx
"""

import sys
import argparse
from pathlib import Path

import torch


def export_raune_onnx(
    weights: str,
    models_dir: str,
    output: str,
    opset: int = 17,
) -> str:
    """Export RAUNE-Net to simplified ONNX.

    Args:
        weights: Path to .pth weights file.
        models_dir: Directory containing models/ package with raune_net.py.
        output: Output .onnx file path.
        opset: ONNX opset version (default 17, TRT 10.x supports 9-20).

    Returns:
        Path to the written ONNX file.
    """
    import onnx
    from onnxsim import simplify

    # Load model
    models_dir = Path(models_dir)
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from models.raune_net import RauneNet

    model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Export in fp32 — TRT does its own fp16 conversion
    dummy = torch.randn(1, 3, 540, 960)

    raw_path = output + ".raw" if not output.endswith(".raw") else output
    torch.onnx.export(
        model,
        dummy,
        raw_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    # Simplify — required for TRT InstanceNorm parsing
    raw_model = onnx.load(raw_path)
    simplified, ok = simplify(raw_model)
    if not ok:
        raise RuntimeError("onnxsim simplification failed — check model for unsupported ops")

    onnx.save(simplified, output)

    # Clean up raw export
    raw = Path(raw_path)
    if raw.exists() and raw_path != output:
        raw.unlink()

    # Validate
    onnx.checker.check_model(onnx.load(output))

    return output


def main():
    parser = argparse.ArgumentParser(description="Export RAUNE-Net to ONNX")
    parser.add_argument("--weights", required=True, help="Path to weights .pth file")
    parser.add_argument("--models-dir", required=True, help="Directory containing models/ package")
    parser.add_argument("--output", required=True, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    path = export_raune_onnx(args.weights, args.models_dir, args.output, args.opset)
    print(f"Exported to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_trt_engine.py::TestOnnxExport -v 2>&1 | tail -10
```

Expected: both tests PASS (test_export_produces_valid_onnx, test_export_output_matches_pytorch)

- [ ] **Step 5: Commit**

```bash
git add python/dorea_inference/export_onnx.py python/tests/test_trt_engine.py
git commit -m "feat: add RAUNE-Net ONNX export with onnxsim simplification"
```

---

### Task 3: TRT Engine Wrapper

**Files:**
- Create: `python/dorea_inference/trt_engine.py`
- Modify: `python/tests/test_trt_engine.py` (add engine tests)

- [ ] **Step 1: Write the failing test for TRT engine build + inference**

Append to `python/tests/test_trt_engine.py`:

```python
@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTRTEngine:
    @pytest.fixture
    def onnx_path(self, tmp_path):
        from dorea_inference.export_onnx import export_raune_onnx
        path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(path))
        return path

    def test_build_engine(self, onnx_path, tmp_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        engine_path = tmp_path / "raune.engine"
        RauneTRTEngine.build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            batch_size=1,
            height=540,
            width=960,
            fp16=True,
        )
        assert engine_path.exists()
        assert engine_path.stat().st_size > 1_000_000  # engine should be >1MB

    def test_inference_matches_pytorch(self, onnx_path, tmp_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        engine_path = tmp_path / "raune.engine"
        RauneTRTEngine.build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            batch_size=1,
            height=270,
            width=480,
            fp16=True,
        )

        engine = RauneTRTEngine(str(engine_path))

        # TRT inference
        x = torch.randn(1, 3, 270, 480, device="cuda", dtype=torch.float16)
        trt_out = engine.infer(x)

        # PyTorch reference (fp16)
        pt_model = _load_pytorch_model().cuda().half()
        with torch.no_grad():
            pt_out = pt_model(x)

        # FP16 TRT vs FP16 PyTorch — allow wider tolerance for kernel fusion differences
        psnr = 10 * torch.log10(4.0 / torch.mean((trt_out.float() - pt_out.float()) ** 2))
        assert psnr > 40, f"PSNR {psnr:.1f}dB < 40dB threshold"

    def test_cache_key_changes_with_gpu(self, onnx_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        key1 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, True)
        key2 = RauneTRTEngine.cache_key(str(onnx_path), (8, 9), "10.9.0", 1, 540, 960, True)
        assert key1 != key2

    def test_cache_key_changes_with_precision(self, onnx_path):
        from dorea_inference.trt_engine import RauneTRTEngine

        key_fp16 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, True)
        key_fp32 = RauneTRTEngine.cache_key(str(onnx_path), (8, 6), "10.9.0", 1, 540, 960, False)
        assert key_fp16 != key_fp32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_trt_engine.py::TestTRTEngine::test_build_engine -v 2>&1 | tail -5
```

Expected: FAIL with `ImportError: cannot import name 'RauneTRTEngine'`

- [ ] **Step 3: Write the TRT engine wrapper**

Create `python/dorea_inference/trt_engine.py`:

```python
"""TensorRT engine wrapper for RAUNE-Net inference.

Handles engine building from ONNX, disk caching with automatic invalidation,
and zero-copy inference with PyTorch CUDA tensors.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _require_tensorrt():
    if trt is None:
        raise RuntimeError(
            "tensorrt is required for --tensorrt mode. "
            "Install with: pip install tensorrt-cu12"
        )


class RauneTRTEngine:
    """TensorRT engine for RAUNE-Net inference."""

    @staticmethod
    def cache_key(
        onnx_path: str,
        compute_cap: tuple[int, int],
        trt_version: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool,
    ) -> str:
        """Deterministic cache key for engine invalidation."""
        onnx_hash = hashlib.md5(Path(onnx_path).read_bytes()).hexdigest()
        sig = json.dumps({
            "onnx": onnx_hash,
            "sm": f"{compute_cap[0]}.{compute_cap[1]}",
            "trt": trt_version,
            "batch": batch_size,
            "h": height,
            "w": width,
            "fp16": fp16,
        }, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    @staticmethod
    def build_engine(
        onnx_path: str,
        engine_path: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool = True,
    ) -> None:
        """Build TRT engine from ONNX and serialize to disk.

        Takes 2-5 minutes on first run. Result is cached via engine_path.
        """
        _require_tensorrt()

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[trt-engine] ONNX parse error: {parser.get_error(i)}",
                          file=sys.stderr)
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, 3, height, width),
            opt=(batch_size, 3, height, width),
            max=(batch_size, 3, height, width),
        )
        config.add_optimization_profile(profile)

        print(f"[trt-engine] Building engine: {batch_size}x3x{height}x{width}, "
              f"fp16={fp16}. This takes 2-5 minutes...",
              file=sys.stderr, flush=True)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

        print(f"[trt-engine] Engine saved to {engine_path} "
              f"({Path(engine_path).stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr, flush=True)

    @classmethod
    def get_or_build(
        cls,
        onnx_path: str,
        cache_dir: str,
        batch_size: int,
        height: int,
        width: int,
        fp16: bool = True,
    ) -> "RauneTRTEngine":
        """Load cached engine or build from ONNX."""
        _require_tensorrt()

        compute_cap = torch.cuda.get_device_capability()
        trt_version = trt.__version__
        key = cls.cache_key(onnx_path, compute_cap, trt_version,
                            batch_size, height, width, fp16)
        engine_path = str(Path(cache_dir) / f"{key}.engine")

        if not Path(engine_path).exists():
            print(f"[trt-engine] Cache miss (key={key}), building engine...",
                  file=sys.stderr, flush=True)
            cls.build_engine(onnx_path, engine_path, batch_size, height, width, fp16)
        else:
            print(f"[trt-engine] Cache hit (key={key}), loading engine...",
                  file=sys.stderr, flush=True)

        return cls(engine_path)

    def __init__(self, engine_path: str):
        """Deserialize engine from disk."""
        _require_tensorrt()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference. Input: (B,3,H,W) fp16 CUDA tensor. Returns same shape."""
        B, C, H, W = input_tensor.shape
        self.context.set_input_shape("input", (B, C, H, W))

        output = torch.empty_like(input_tensor)

        self.context.set_tensor_address("input", input_tensor.data_ptr())
        self.context.set_tensor_address("output", output.data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        return output
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_trt_engine.py::TestTRTEngine -v 2>&1 | tail -15
```

Expected: all 4 tests PASS. The `test_build_engine` and `test_inference_matches_pytorch` tests will take 2-5 minutes each (engine build time).

- [ ] **Step 5: Commit**

```bash
git add python/dorea_inference/trt_engine.py python/tests/test_trt_engine.py
git commit -m "feat: add TensorRT engine wrapper with cache and zero-copy inference"
```

---

### Task 4: Integrate TRT into raune_filter.py

**Files:**
- Modify: `python/dorea_inference/raune_filter.py:554-707` (argparser + main + _process_batch)

- [ ] **Step 1: Add CLI flags to argparser**

In `raune_filter.py`, add after `--output-codec` argument (line 656):

```python
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT FP16 engine instead of PyTorch (requires tensorrt-cu12)")
    parser.add_argument("--trt-cache-dir", default=None,
                        help="TRT engine cache directory (default: <models-dir>/trt_cache)")
    parser.add_argument("--onnx-path", default=None,
                        help="Pre-exported ONNX model path (default: auto-export to <models-dir>/raune_net.onnx)")
```

- [ ] **Step 2: Add TRT model loading path in main()**

Replace the model loading section in `main()` (lines 659-693) with:

```python
    # Load RAUNE model — either TRT engine or PyTorch
    models_dir = Path(args.models_dir)
    if (models_dir / "models" / "raune_net.py").exists():
        raune_dir = models_dir
    elif (models_dir / "models" / "RAUNE-Net" / "models" / "raune_net.py").exists():
        raune_dir = models_dir / "models" / "RAUNE-Net"
    else:
        raune_dir = models_dir
    if str(raune_dir) not in sys.path:
        sys.path.insert(0, str(raune_dir))

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if args.tensorrt:
        from dorea_inference.trt_engine import RauneTRTEngine
        from dorea_inference.export_onnx import export_raune_onnx

        # Resolve ONNX path
        onnx_path = args.onnx_path
        if onnx_path is None:
            onnx_path = str(models_dir / "raune_net.onnx")
            if not Path(onnx_path).exists():
                print(f"[raune-filter] Exporting ONNX to {onnx_path}...",
                      file=sys.stderr, flush=True)
                export_raune_onnx(args.weights, str(models_dir), onnx_path)

        # Resolve cache dir and proxy dimensions
        trt_cache_dir = args.trt_cache_dir or str(models_dir / "trt_cache")

        model = RauneTRTEngine.get_or_build(
            onnx_path=onnx_path,
            cache_dir=trt_cache_dir,
            batch_size=args.batch_size,
            height=args.proxy_height,
            width=args.proxy_width,
            fp16=True,
        )
        model_dtype = torch.float16
        print(f"[raune-filter] Using TensorRT FP16 engine", file=sys.stderr, flush=True)
    else:
        from models.raune_net import RauneNet

        model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).cuda()
        state = torch.load(args.weights, map_location="cuda", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        import torch.nn as nn
        model = model.half()
        instance_norm_count = 0
        for m in model.modules():
            if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                m.float()
                instance_norm_count += 1
        model_dtype = next(model.parameters()).dtype
        print(f"[raune-filter] RAUNE converted to fp16 "
              f"({instance_norm_count} InstanceNorm layers kept in fp32, "
              f"model_dtype={model_dtype})",
              file=sys.stderr, flush=True)
```

- [ ] **Step 3: Modify _process_batch to support TRT engine**

In `_process_batch()` (line 475), the inference call `raune_out = model(proxy_batch)` needs to work with both PyTorch model and TRT engine. The TRT engine's `.infer()` method accepts the same tensor shape — so the change is minimal.

Replace lines 500-503 (`# RAUNE inference` block):

```python
        # RAUNE inference
        if hasattr(model, 'infer'):
            # TRT engine path — input must be fp16 contiguous
            raune_out = model.infer(proxy_batch.half().contiguous())
            raune_out = raune_out.float()
        else:
            # PyTorch model path
            raune_out = model(proxy_batch)
            raune_out = raune_out.float()
        raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

- [ ] **Step 4: Apply same change to run_pipe_mode**

In `run_pipe_mode()` (line 594), replace the inference call:

```python
            if hasattr(model, 'infer'):
                raune_out = model.infer(proxy_batch.half().contiguous()).float()
            else:
                raune_out = model(proxy_batch).float()
            raune_out = ((raune_out + 1.0) / 2.0).clamp(0.0, 1.0)
```

- [ ] **Step 5: Test the Python integration manually**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -c "
from dorea_inference.raune_filter import main
# Verify argparser accepts --tensorrt flag
import argparse
from dorea_inference.raune_filter import main
import sys
sys.argv = ['raune_filter', '--help']
try:
    main()
except SystemExit:
    pass
" 2>&1 | grep -E "tensorrt|trt-cache|onnx-path"
```

Expected: `--tensorrt`, `--trt-cache-dir`, and `--onnx-path` appear in help output.

- [ ] **Step 6: Commit**

```bash
git add python/dorea_inference/raune_filter.py
git commit -m "feat: integrate TensorRT inference path into raune_filter (--tensorrt flag)"
```

---

### Task 5: Rust CLI Passthrough

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs:17-58` (GradeArgs struct)
- Modify: `crates/dorea-cli/src/pipeline/mod.rs:9-20` (PipelineConfig struct)
- Modify: `crates/dorea-cli/src/pipeline/grading.rs:43-57` (subprocess args)

- [ ] **Step 1: Add tensorrt field to PipelineConfig**

In `crates/dorea-cli/src/pipeline/mod.rs`, add after `proxy_h`:

```rust
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output_codec: OutputCodec,
    pub output: PathBuf,
    pub python: PathBuf,
    pub raune_weights: PathBuf,
    pub raune_models_dir: PathBuf,
    pub raune_proxy_size: usize,
    pub batch_size: usize,
    pub proxy_w: usize,
    pub proxy_h: usize,
    pub tensorrt: bool,
}
```

- [ ] **Step 2: Add --tensorrt CLI flag to GradeArgs**

In `crates/dorea-cli/src/grade.rs`, add after `verbose` field:

```rust
    /// Use TensorRT FP16 engine for RAUNE-Net inference (requires tensorrt-cu12 in venv).
    #[arg(long)]
    pub tensorrt: bool,
```

- [ ] **Step 3: Pass tensorrt through PipelineConfig construction**

In `crates/dorea-cli/src/grade.rs`, update the PipelineConfig construction (around line 159):

```rust
    let pipeline_cfg = PipelineConfig {
        input: args.input,
        output_codec,
        output,
        python,
        raune_weights,
        raune_models_dir,
        raune_proxy_size,
        batch_size,
        proxy_w,
        proxy_h,
        tensorrt: args.tensorrt,
    };
```

- [ ] **Step 4: Pass --tensorrt to Python subprocess**

In `crates/dorea-cli/src/pipeline/grading.rs`, add after the existing `.args([...])` block:

```rust
    let mut raune_proc = Command::new(&cfg.python)
        .env("PYTHONPATH", &python_dir)
        .args([
            "-m", "dorea_inference.raune_filter",
            "--weights", raune_weights_str,
            "--models-dir", raune_models_dir_str,
            "--full-width", &info.width.to_string(),
            "--full-height", &info.height.to_string(),
            "--proxy-width", &cfg.proxy_w.to_string(),
            "--proxy-height", &cfg.proxy_h.to_string(),
            "--batch-size", &cfg.batch_size.to_string(),
            "--input", input_str,
            "--output", output_str,
            "--output-codec", pyav_codec,
        ]);

    if cfg.tensorrt {
        raune_proc.args(["--tensorrt"]);
    }

    let mut raune_proc = raune_proc
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn raune_filter.py")?;
```

- [ ] **Step 5: Verify Rust builds**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build 2>&1 | tail -5
```

Expected: build succeeds with no errors.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/pipeline/mod.rs crates/dorea-cli/src/pipeline/grading.rs
git commit -m "feat: add --tensorrt flag to dorea CLI, pass through to raune_filter.py"
```

---

### Task 6: torch.compile Bonus for PyTorch Fallback

**Files:**
- Modify: `python/dorea_inference/raune_filter.py` (main function, PyTorch path)

- [ ] **Step 1: Add torch.compile to PyTorch model path**

In the `else` branch of main() (the PyTorch path), add after the fp16 conversion print statement:

```python
        # torch.compile for ~15-30% speedup on the PyTorch fallback path.
        # Must be called AFTER model.half() + InstanceNorm float() restoration
        # so the compiled graph sees the final dtype layout.
        model = torch.compile(model, mode="default")
        print("[raune-filter] Applied torch.compile (inductor backend)",
              file=sys.stderr, flush=True)
```

- [ ] **Step 2: Verify the PyTorch path still works**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -c "
import torch
import sys
sys.path.insert(0, 'models/raune_net')
from models.raune_net import RauneNet

model = RauneNet(input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64).cuda()
state = torch.load('models/raune_net/weights_95.pth', map_location='cuda', weights_only=True)
model.load_state_dict(state)
model.eval()
model = model.half()

import torch.nn as nn
for m in model.modules():
    if isinstance(m, nn.InstanceNorm2d):
        m.float()

model = torch.compile(model, mode='default')
x = torch.randn(1, 3, 270, 480, device='cuda', dtype=torch.float16)
with torch.no_grad():
    y = model(x)
print(f'torch.compile output: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]')
print('OK')
" 2>&1 | tail -5
```

Expected: prints shape and "OK" (first call triggers ~10-30s compilation).

- [ ] **Step 3: Commit**

```bash
git add python/dorea_inference/raune_filter.py
git commit -m "perf: add torch.compile to PyTorch fallback path for ~15-30% speedup"
```

---

### Task 7: Full Test Suite and Cleanup

**Files:**
- Modify: `python/tests/test_trt_engine.py` (add integration test)

- [ ] **Step 1: Add integration test that exercises the full pipeline**

Append to `python/tests/test_trt_engine.py`:

```python
@pytest.mark.skipif(not WEIGHTS.exists(), reason="RAUNE weights not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIntegration:
    def test_trt_batch_inference(self, tmp_path):
        """Test TRT engine with batch_size > 1, matching pipeline usage."""
        from dorea_inference.export_onnx import export_raune_onnx
        from dorea_inference.trt_engine import RauneTRTEngine

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(onnx_path))

        engine = RauneTRTEngine.get_or_build(
            onnx_path=str(onnx_path),
            cache_dir=str(tmp_path / "cache"),
            batch_size=4,
            height=270,
            width=480,
            fp16=True,
        )

        # Batch of 4 frames
        x = torch.randn(4, 3, 270, 480, device="cuda", dtype=torch.float16)
        out = engine.infer(x)
        assert out.shape == (4, 3, 270, 480)
        assert out.dtype == torch.float16

    def test_cache_hit_loads_fast(self, tmp_path):
        """Second get_or_build should load from cache, not rebuild."""
        import time
        from dorea_inference.export_onnx import export_raune_onnx
        from dorea_inference.trt_engine import RauneTRTEngine

        onnx_path = tmp_path / "raune.onnx"
        export_raune_onnx(str(WEIGHTS), str(MODELS_DIR), str(onnx_path))
        cache_dir = str(tmp_path / "cache")

        # First build (slow)
        RauneTRTEngine.get_or_build(str(onnx_path), cache_dir, 1, 270, 480)

        # Second load (fast — should be <10s)
        t0 = time.time()
        RauneTRTEngine.get_or_build(str(onnx_path), cache_dir, 1, 270, 480)
        elapsed = time.time() - t0
        assert elapsed < 10, f"Cache load took {elapsed:.1f}s — expected <10s"
```

- [ ] **Step 2: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m pytest python/tests/test_trt_engine.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_trt_engine.py
git commit -m "test: add TRT integration tests for batch inference and cache hit"
```
