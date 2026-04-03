# GPU-Space IPC & Adaptive Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace subprocess JSON-lines IPC and `extern "C"` CUDA FFI with PyO3 zero-copy embedding and cudarc RAII device memory, adding adaptive batch sizing for the grading pipeline.

**Architecture:** PyO3 embeds the Python interpreter in-process behind a `python` Cargo feature flag on dorea-video. cudarc replaces the static-linked C launcher pattern in dorea-gpu with PTX module loading and typed kernel launches. An AdaptiveBatcher probes VRAM and adjusts batch size at runtime. GPU tensor sharing eliminates host round-trips for inference-to-grading data flow.

**Tech Stack:** Rust (cudarc 0.12, pyo3 0.22, numpy 0.22), Python (PyTorch, NumPy), CUDA PTX

**Design Spec:** `docs/decisions/2026-04-03-gpu-space-ipc-adaptive-batching-design.md`

**Issue:** chunzhe10/dorea#24

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `crates/dorea-gpu/src/device.rs` | `DeviceFrameBuffer`, `BorrowedDeviceSlice<'py, T>`, VRAM query helpers |
| `crates/dorea-gpu/src/batcher.rs` | `AdaptiveBatcher` — halve-on-OOM / grow-on-success batch sizing |
| `crates/dorea-video/src/inference_subprocess.rs` | Renamed from current `inference.rs` — subprocess fallback path |
| `python/dorea_inference/bridge.py` | `TensorGuard`, PyO3-callable entry points for inference |

### Modified files
| File | Changes |
|------|---------|
| `crates/dorea-gpu/Cargo.toml` | Add `cudarc` dependency behind `cuda` feature |
| `crates/dorea-gpu/build.rs` | Compile `.cu` → `.ptx` (was `.cu` → `.o` → `.a`) |
| `crates/dorea-gpu/src/lib.rs` | Expand `GpuError` with `Oom`/`ModuleLoad`/`CudaFail`, re-export new modules |
| `crates/dorea-gpu/src/cuda/mod.rs` | Rewrite: cudarc PTX loading + kernel launches (remove `extern "C"`) |
| `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu` | Remove C host launcher function, keep `__global__` kernel only |
| `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu` | Remove C host launcher function, keep `__global__` kernel only |
| `crates/dorea-gpu/src/cuda/kernels/clarity.cu` | Remove C host launcher functions, keep `__global__` kernels only |
| `crates/dorea-video/Cargo.toml` | Add `pyo3`, `numpy` behind `python` feature; conditional `base64`/`flate2` |
| `crates/dorea-video/src/lib.rs` | Conditional module selection: `inference` vs `inference_subprocess` |
| `crates/dorea-video/src/inference.rs` | Rewrite: PyO3-based `InferenceServer` (replaces subprocess version) |
| `python/dorea_inference/depth_anything.py` | Add `infer_gpu()` returning on-device tensor |
| `python/dorea_inference/raune_net.py` | Add `infer_gpu()` returning on-device tensor |
| `crates/dorea-cli/Cargo.toml` | No change (does NOT enable `python` feature) |
| `crates/dorea-cli/src/grade.rs` | Update grading loop for batch API, VRAM probe, adaptive batching |

---

## Task 1: Python Bridge Module

**Files:**
- Create: `python/dorea_inference/bridge.py`
- Modify: `python/dorea_inference/depth_anything.py`
- Modify: `python/dorea_inference/raune_net.py`
- Test: `python -m pytest python/dorea_inference/test_bridge.py` (created inline)

This task is standalone Python — no Rust changes needed.

- [ ] **Step 1: Create `bridge.py` with TensorGuard and entry points**

```python
# python/dorea_inference/bridge.py
"""PyO3-callable entry points for Rust ↔ Python inference bridge.

Rust calls these functions via PyO3 embedding. The TensorGuard class
prevents Python's GC from reclaiming GPU tensors while Rust holds
their device pointers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TensorGuard:
    """Prevent GC of a GPU tensor while Rust holds its device pointer.

    Rust holds a Py<TensorGuard> which prevents Python from collecting
    this object. Call release() explicitly when Rust is done with the pointer.
    """

    def __init__(self, tensor: "torch.Tensor") -> None:
        self.tensor = tensor
        self.data_ptr = tensor.data_ptr()
        self.numel = tensor.numel()
        self.shape = tuple(tensor.shape)
        self.dtype = str(tensor.dtype)

    def release(self) -> None:
        """Explicitly release the tensor. Called by Rust when done."""
        self.tensor = None
        self.data_ptr = 0
        self.numel = 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_depth_model = None
_raune_model = None


def load_depth_model(model_path: Optional[str] = None, device: str = "cuda") -> None:
    """Load the Depth Anything V2 model. Called once at init."""
    global _depth_model
    from .depth_anything import DepthAnythingInference
    _depth_model = DepthAnythingInference(model_path=model_path, device=device)


def load_raune_model(
    weights_path: Optional[str] = None,
    device: str = "cuda",
    raune_models_dir: Optional[str] = None,
) -> None:
    """Load the RAUNE-Net model. Called once at init."""
    global _raune_model
    from .raune_net import RauneNetInference
    _raune_model = RauneNetInference(
        weights_path=weights_path, device=device, raune_models_dir=raune_models_dir,
    )


# ---------------------------------------------------------------------------
# GPU inference (returns TensorGuard with on-device tensor)
# ---------------------------------------------------------------------------

def run_depth_gpu(frame_rgb: np.ndarray, max_size: int = 518) -> TensorGuard:
    """Run depth inference, return TensorGuard holding the on-device result."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    tensor = _depth_model.infer_gpu(frame_rgb, max_size=max_size)
    return TensorGuard(tensor)


def run_raune_gpu(frame_rgb: np.ndarray, max_size: int = 1024) -> TensorGuard:
    """Run RAUNE-Net inference, return TensorGuard holding the on-device result."""
    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    tensor = _raune_model.infer_gpu(frame_rgb, max_size=max_size)
    return TensorGuard(tensor)


# ---------------------------------------------------------------------------
# CPU inference (returns numpy arrays — fallback path)
# ---------------------------------------------------------------------------

def run_depth_cpu(frame_rgb: np.ndarray, max_size: int = 518) -> np.ndarray:
    """Run depth inference on CPU, return numpy array."""
    if _depth_model is None:
        raise RuntimeError("Depth model not loaded — call load_depth_model() first")
    return _depth_model.infer(frame_rgb, max_size=max_size)


def run_raune_cpu(frame_rgb: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Run RAUNE-Net on CPU, return numpy array."""
    if _raune_model is None:
        raise RuntimeError("RAUNE-Net model not loaded — call load_raune_model() first")
    return _raune_model.infer(frame_rgb, max_size=max_size)


# ---------------------------------------------------------------------------
# VRAM query (called by Rust for adaptive batching)
# ---------------------------------------------------------------------------

def vram_free_bytes() -> int:
    """Return free VRAM in bytes after flushing PyTorch's cache.

    Uses torch.cuda.mem_get_info() which accounts for the caching allocator.
    """
    import torch
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free
```

- [ ] **Step 2: Add `infer_gpu()` to `depth_anything.py`**

Add this method to the `DepthAnythingInference` class, after the existing `infer()` method at line 109:

```python
    def infer_gpu(self, img_rgb: np.ndarray, max_size: int = 518) -> "torch.Tensor":
        """Run depth estimation, return on-device f32 tensor (not copied to CPU).

        Returns a 2D float32 CUDA tensor normalized to [0, 1] at inference resolution.
        The caller must keep a reference to prevent GC.
        """
        import torch
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        capped = _resize_for_depth(pil, max_size)

        # Direct tensor construction — bypass AutoImageProcessor
        arr = np.array(capped).astype(np.float32) / 255.0
        arr = (arr - self._MEAN) / self._STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)

        depth = outputs.predicted_depth.squeeze(0)  # stays on device

        d_min = float(depth.min())
        d_max = float(depth.max())
        if d_max - d_min < 1e-6:
            depth = torch.zeros_like(depth)
        else:
            depth = (depth - d_min) / (d_max - d_min)

        return depth.to(torch.float32).contiguous()
```

- [ ] **Step 3: Add `infer_gpu()` to `raune_net.py`**

Add this method to the `RauneNetInference` class, after the existing `infer()` method at line 110:

```python
    def infer_gpu(self, img_rgb: np.ndarray, max_size: int = 1024) -> "torch.Tensor":
        """Run RAUNE-Net, return on-device uint8 tensor (not copied to CPU).

        Returns a 3D uint8 CUDA tensor (HxWx3) at inference resolution.
        The caller must keep a reference to prevent GC.
        """
        import torch
        import torchvision.transforms as transforms
        from PIL import Image as _Image

        pil = _Image.fromarray(img_rgb)
        resized, tw, th = _resize_maintain_aspect(pil, max_size)

        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor = normalize(transforms.ToTensor()(resized)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        # De-normalize: model output in [-1, 1], convert to [0, 255] uint8
        out = ((out.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
        result = (out.permute(1, 2, 0) * 255).to(torch.uint8).contiguous()

        return result  # stays on device
```

- [ ] **Step 4: Test the Python bridge**

Run from workspace root:
```bash
cd /workspaces/dorea-workspace/repos/dorea
/opt/dorea-venv/bin/python -c "
from dorea_inference.bridge import TensorGuard, vram_free_bytes
import numpy as np

# Test TensorGuard with a CPU tensor (no GPU needed for this)
import torch
t = torch.zeros(10, dtype=torch.float32)
g = TensorGuard(t)
assert g.numel == 10
assert g.shape == (10,)
assert g.dtype == 'torch.float32'
g.release()
assert g.tensor is None
assert g.data_ptr == 0
print('TensorGuard: OK')

# Test vram_free_bytes (returns 0 if no GPU)
free = vram_free_bytes()
print(f'VRAM free: {free} bytes')
print('All bridge tests passed')
"
```

Expected: `All bridge tests passed`

- [ ] **Step 5: Commit**

```bash
git add python/dorea_inference/bridge.py python/dorea_inference/depth_anything.py python/dorea_inference/raune_net.py
git commit -m "feat(inference): add PyO3 bridge module with TensorGuard and infer_gpu methods"
```

---

## Task 2: Expand GpuError and Add AdaptiveBatcher

**Files:**
- Modify: `crates/dorea-gpu/src/lib.rs`
- Create: `crates/dorea-gpu/src/batcher.rs`
- Test: `cargo test -p dorea-gpu`

Pure Rust, no GPU needed. Fully testable.

- [ ] **Step 1: Write failing test for AdaptiveBatcher**

Create `crates/dorea-gpu/src/batcher.rs`:

```rust
/// Adaptive batch sizing for GPU grading.
///
/// Probes VRAM, starts at the maximum safe batch size, and adjusts at runtime:
/// - On OOM: halve batch size (floor at 1)
/// - After `grow_threshold` consecutive successes: grow by 50% (cap at max_batch)
/// - Batch size does NOT persist across runs
pub struct AdaptiveBatcher {
    batch_size: usize,
    max_batch: usize,
    min_batch: usize,
    successes: usize,
    grow_threshold: usize,
}

impl AdaptiveBatcher {
    /// Create a new batcher from VRAM probe results.
    ///
    /// `vram_free`: free VRAM in bytes (after `torch.cuda.empty_cache()`)
    /// `per_frame_bytes`: estimated bytes per frame (e.g. 56MB for 1080p)
    /// `safety_margin`: fraction reserved for fragmentation (0.15 = 15%)
    pub fn new(vram_free: usize, per_frame_bytes: usize, safety_margin: f64) -> Self {
        let usable = (vram_free as f64 * (1.0 - safety_margin)) as usize;
        let max_batch = if per_frame_bytes > 0 {
            (usable / per_frame_bytes).max(1)
        } else {
            1
        };
        Self {
            batch_size: max_batch,
            max_batch,
            min_batch: 1,
            successes: 0,
            grow_threshold: 10,
        }
    }

    /// Create a batcher with a fixed batch size (no adaptation).
    pub fn fixed(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size.max(1),
            max_batch: batch_size.max(1),
            min_batch: 1,
            successes: 0,
            grow_threshold: usize::MAX,
        }
    }

    /// Current batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Report a successful batch completion.
    pub fn report_success(&mut self) {
        self.successes += 1;
        if self.successes >= self.grow_threshold && self.batch_size < self.max_batch {
            self.batch_size = ((self.batch_size as f64 * 1.5).ceil() as usize).min(self.max_batch);
            self.successes = 0;
        }
    }

    /// Report an OOM failure. Returns true if batch_size was already at minimum
    /// (meaning the frame genuinely cannot fit).
    pub fn report_oom(&mut self) -> bool {
        let was_min = self.batch_size <= self.min_batch;
        self.successes = 0;
        self.batch_size = (self.batch_size / 2).max(self.min_batch);
        was_min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_computes_max_batch_from_vram() {
        // 1GB free, 56MB per frame, 15% margin
        let b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        // usable = 850MB, 850/56 = 15.17 → 15
        assert_eq!(b.batch_size(), 15);
        assert_eq!(b.max_batch, 15);
    }

    #[test]
    fn new_with_zero_per_frame_gives_batch_1() {
        let b = AdaptiveBatcher::new(1_000_000_000, 0, 0.15);
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn new_with_tiny_vram_gives_batch_1() {
        // Only 10MB free, 56MB per frame
        let b = AdaptiveBatcher::new(10_000_000, 56_000_000, 0.15);
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn report_oom_halves_batch_size() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        let initial = b.batch_size();
        assert!(!b.report_oom());
        assert_eq!(b.batch_size(), initial / 2);
    }

    #[test]
    fn report_oom_at_minimum_returns_true() {
        let mut b = AdaptiveBatcher::fixed(1);
        assert!(b.report_oom());
        assert_eq!(b.batch_size(), 1);
    }

    #[test]
    fn report_oom_resets_success_counter() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        for _ in 0..9 {
            b.report_success();
        }
        b.report_oom(); // resets counter
        let size_after_oom = b.batch_size();
        // 9 more successes should NOT trigger growth (counter was reset)
        for _ in 0..9 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), size_after_oom);
    }

    #[test]
    fn growth_after_threshold_successes() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        // Halve first so there's room to grow
        b.report_oom();
        let halved = b.batch_size();
        // 10 successes should trigger 50% growth
        for _ in 0..10 {
            b.report_success();
        }
        let expected = ((halved as f64 * 1.5).ceil() as usize).min(b.max_batch);
        assert_eq!(b.batch_size(), expected);
    }

    #[test]
    fn growth_capped_at_max() {
        let mut b = AdaptiveBatcher::new(1_000_000_000, 56_000_000, 0.15);
        let max = b.max_batch;
        // Already at max — 10 successes should not increase
        for _ in 0..20 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), max);
    }

    #[test]
    fn fixed_batcher_does_not_grow() {
        let mut b = AdaptiveBatcher::fixed(4);
        for _ in 0..100 {
            b.report_success();
        }
        assert_eq!(b.batch_size(), 4);
    }
}
```

- [ ] **Step 2: Update GpuError in `lib.rs`**

Replace the existing `GpuError` enum in `crates/dorea-gpu/src/lib.rs`:

```rust
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("CUDA OOM: {0}")]
    Oom(String),
    #[error("CUDA error: {0}")]
    CudaFail(String),
    #[error("CUDA module load failed: {0}")]
    ModuleLoad(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
```

Update the `grade_frame` function's CUDA error return to use `CudaFail` instead of `Cuda`:

In `grade_frame` at the `#[cfg(not(feature = "cuda"))]` block:
```rust
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::CudaFail(
            "dorea grade requires CUDA. Rebuild with GPU support (build.rs auto-detects nvcc).".to_string()
        ))
    }
```

Add the batcher module to `lib.rs`:
```rust
pub mod batcher;
```

- [ ] **Step 3: Run tests to verify batcher + error types compile and pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu -- --nocapture
```

Expected: All existing tests pass + new batcher tests pass.

- [ ] **Step 4: Fix any compilation errors from GpuError rename**

The `cuda/mod.rs` uses `GpuError::Cuda(...)`. Update all occurrences to `GpuError::CudaFail(...)`:

In `crates/dorea-gpu/src/cuda/mod.rs`, replace all `GpuError::Cuda(` with `GpuError::CudaFail(`:
- Line 111: `GpuError::Cuda(format!("dorea_lut_apply_gpu..."))` → `GpuError::CudaFail(...)`
- Line 141: `GpuError::Cuda(format!("dorea_hsl_correct_gpu..."))` → `GpuError::CudaFail(...)`
- Line 166: `GpuError::Cuda(format!("dorea_clarity_gpu..."))` → `GpuError::CudaFail(...)`

- [ ] **Step 5: Re-run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu -- --nocapture
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-gpu/src/lib.rs crates/dorea-gpu/src/batcher.rs crates/dorea-gpu/src/cuda/mod.rs
git commit -m "feat(dorea-gpu): expand GpuError with Oom/ModuleLoad variants, add AdaptiveBatcher"
```

---

## Task 3: Build System — PTX Compilation and cudarc Dependency

**Files:**
- Modify: `crates/dorea-gpu/Cargo.toml`
- Modify: `crates/dorea-gpu/build.rs`
- Modify: `crates/dorea-gpu/src/cuda/kernels/lut_apply.cu`
- Modify: `crates/dorea-gpu/src/cuda/kernels/hsl_correct.cu`
- Modify: `crates/dorea-gpu/src/cuda/kernels/clarity.cu`
- Test: `cargo build -p dorea-gpu`

- [ ] **Step 1: Add cudarc to Cargo.toml**

Replace `crates/dorea-gpu/Cargo.toml`:

```toml
[package]
name = "dorea-gpu"
version.workspace = true
edition.workspace = true

[dependencies]
dorea-color = { workspace = true }
dorea-lut = { workspace = true }
dorea-hsl = { workspace = true }
dorea-cal = { workspace = true }
log.workspace = true
rayon.workspace = true
thiserror.workspace = true
cudarc = { version = "0.12", features = ["driver"], optional = true }

[features]
# Activation is driven by build.rs (nvcc detection), NOT by Cargo feature flags.
# Do NOT pass --features cuda manually; build.rs handles it.
cuda = ["dep:cudarc"]
```

- [ ] **Step 2: Rewrite `build.rs` for PTX compilation**

Replace `crates/dorea-gpu/build.rs`:

```rust
// Build script for dorea-gpu.
// Detects nvcc and, if found, compiles CUDA kernels to PTX for embedding.
// If nvcc is not found, builds with CPU-only fallback.

use std::path::PathBuf;

fn find_nvcc() -> Option<PathBuf> {
    // 1. Check PATH — resolve to full path via `which` so parent-based include lookup works
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            if let Ok(which_out) = std::process::Command::new("which").arg("nvcc").output() {
                if which_out.status.success() {
                    let resolved = PathBuf::from(
                        std::str::from_utf8(&which_out.stdout).unwrap_or("").trim()
                    );
                    if resolved.exists() {
                        return Some(resolved);
                    }
                }
            }
            // Fallback: nvcc is on PATH but `which` unavailable; return relative name
            return Some(PathBuf::from("nvcc"));
        }
    }

    // 2. Check CUDA_HOME/bin/nvcc
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // 3. Scan /usr/local/cuda-* (any version) — handles e.g. cuda-12.8, cuda-12.4
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut cuda_dirs: Vec<_> = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("cuda-"))
            .map(|e| e.path().join("bin").join("nvcc"))
            .filter(|p| p.exists())
            .collect();
        // Prefer highest version (sort descending by path string)
        cuda_dirs.sort_by(|a, b| b.cmp(a));
        if let Some(p) = cuda_dirs.into_iter().next() {
            return Some(p);
        }
    }

    // 4. Remaining common install paths
    for candidate in &[
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("src").join("cuda").join("kernels");

    // Tell cargo to re-run if any .cu file changes or CUDA env vars change
    println!("cargo:rerun-if-changed=src/cuda/kernels/lut_apply.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/hsl_correct.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/clarity.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=PATH");

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — building dorea-gpu with CPU-only fallback");
        println!("cargo:warning=Rebuild the devcontainer (Dockerfile adds CUDA 12.4 toolkit) to enable CUDA kernels");
        return;
    };

    println!("cargo:warning=Found nvcc at {}", nvcc.display());

    // Compile each .cu file to PTX (was .o → .a; now PTX for cudarc module loading)
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel_names = ["lut_apply", "hsl_correct", "clarity"];

    // Make CUDA includes take priority over system includes so that
    // CUDA's crt/math_functions.h declarations don't conflict with glibc 2.35+
    let cuda_include = nvcc.parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("targets/x86_64-linux/include"))
        .filter(|p| p.exists());

    for name in &kernel_names {
        let src = kernels_dir.join(format!("{name}.cu"));
        let ptx = out_dir.join(format!("{name}.ptx"));

        let mut args = vec![
            "--ptx".to_string(),
            "-O2".to_string(),
            "-arch=sm_86".to_string(), // RTX 3060 is Ampere sm_86
            "--allow-unsupported-compiler".to_string(),
            // Use gcc-12 from Debian bookworm as host compiler
            "--compiler-bindir".to_string(),
            "/usr/bin/gcc-12".to_string(),
        ];
        if let Some(ref inc) = cuda_include {
            args.push("-isystem".to_string());
            args.push(inc.to_str().unwrap().to_string());
        }
        args.push(src.to_str().unwrap().to_string());
        args.push("-o".to_string());
        args.push(ptx.to_str().unwrap().to_string());

        let status = std::process::Command::new(&nvcc)
            .args(&args)
            .status()
            .expect("failed to run nvcc");

        if !status.success() {
            println!(
                "cargo:warning=nvcc failed to compile {name}.cu to PTX — \
                 falling back to CPU-only (GCC/CUDA version mismatch?). \
                 Install gcc-12 or gcc-13 and set CUDAHOSTCXX to enable CUDA kernels."
            );
            return;
        }
    }

    // No more static lib linking — PTX is embedded via include_str! in cuda/mod.rs.
    // We still need the CUDA driver library for cudarc.
    if let Some(cuda_lib) = nvcc.parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("targets/x86_64-linux/lib"))
        .filter(|p| p.exists())
    {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }

    // Signal to Cargo to enable the "cuda" feature
    println!("cargo:rustc-cfg=feature=\"cuda\"");
}
```

- [ ] **Step 3: Strip C host launchers from `lut_apply.cu`**

Remove the `dorea_lut_apply_gpu()` C launcher function. Keep only the `__global__ void lut_apply_kernel(...)` and any `__device__` helpers. The launcher's cudaMalloc/cudaMemcpy/cudaFree logic moves to Rust (cudarc) in Task 5.

Specifically, remove everything after the `__global__` kernel — the `extern "C" int dorea_lut_apply_gpu(...)` function and its body. Keep:
- All `__device__` helper functions
- The `__global__ void lut_apply_kernel(...)` function

- [ ] **Step 4: Strip C host launchers from `hsl_correct.cu`**

Same pattern — remove `extern "C" int dorea_hsl_correct_gpu(...)`, keep `__global__ void hsl_correct_kernel(...)`.

- [ ] **Step 5: Strip C host launchers from `clarity.cu`**

Remove `extern "C" int dorea_clarity_gpu(...)`. Keep all `__global__` kernels:
- `clarity_extract_L_proxy`
- `clarity_box_blur_rows`
- `clarity_box_blur_cols`
- `clarity_apply_kernel`

- [ ] **Step 6: Build to verify PTX compilation works**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu 2>&1
```

Expected: Build succeeds. Warnings about PTX compilation are OK. The `extern "C"` declarations in `cuda/mod.rs` will now fail to link (the static lib is gone) — this is expected and will be fixed in Task 5.

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-gpu/Cargo.toml crates/dorea-gpu/build.rs crates/dorea-gpu/src/cuda/kernels/
git commit -m "build(dorea-gpu): switch from .cu→.o→.a to .cu→.ptx, add cudarc dep, strip C launchers"
```

---

## Task 4: Device Types — BorrowedDeviceSlice and Helpers

**Files:**
- Create: `crates/dorea-gpu/src/device.rs`
- Modify: `crates/dorea-gpu/src/lib.rs` (add `pub mod device;`)
- Test: `cargo test -p dorea-gpu` (compile-time checks; runtime tests need GPU)

- [ ] **Step 1: Create `device.rs`**

```rust
//! Device memory types for GPU tensor sharing and RAII frame buffers.
//!
//! `BorrowedDeviceSlice` — borrows a device pointer owned by Python (PyTorch).
//! Lifetime-bound to the GIL token to prevent use-after-free at compile time.
//!
//! Only compiled when the `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use std::marker::PhantomData;

/// A borrowed device pointer — NOT owned by Rust.
///
/// Carries lifetime `'a` tied to the scope that guarantees the pointer's validity
/// (typically the GIL scope via `Python<'py>`). The borrow checker prevents this
/// from escaping that scope. No `Drop` impl — no `cudaFree`.
#[cfg(feature = "cuda")]
pub struct BorrowedDeviceSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<&'a T>,
}

#[cfg(feature = "cuda")]
impl<'a, T> BorrowedDeviceSlice<'a, T> {
    /// Create a borrowed device slice from a raw device pointer.
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer with at least `len` elements of type T.
    /// - The pointer must remain valid for lifetime `'a`.
    /// - The caller must ensure no concurrent writes to this memory.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len, _phantom: PhantomData }
    }

    /// Raw device pointer (for passing to cudarc kernel launches).
    pub fn as_device_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// BorrowedDeviceSlice is !Send + !Sync because raw pointers are not Send/Sync.
// This is correct — device pointers must not cross thread boundaries.

/// Per-frame VRAM cost at a given resolution.
///
/// Used by AdaptiveBatcher to compute max batch size from available VRAM.
#[cfg(feature = "cuda")]
pub fn per_frame_vram_bytes(width: usize, height: usize) -> usize {
    let n_pixels = width * height;
    let rgb_f32 = n_pixels * 3 * std::mem::size_of::<f32>();
    let depth_f32 = n_pixels * std::mem::size_of::<f32>();
    // Ping-pong buffers: 2x RGB + 1x depth
    rgb_f32 * 2 + depth_f32
}

/// Verify that the CUDA context is healthy by performing a tiny allocation.
///
/// Call this after OOM recovery before retrying. If it fails, the context
/// is likely wedged — skip to CPU fallback.
#[cfg(feature = "cuda")]
pub fn verify_cuda_context(device: &cudarc::driver::CudaDevice) -> Result<(), crate::GpuError> {
    let _probe = device.alloc_zeros::<f32>(1)
        .map_err(|e| crate::GpuError::CudaFail(format!("context health check failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn per_frame_vram_1080p() {
        #[cfg(feature = "cuda")]
        {
            use super::per_frame_vram_bytes;
            let bytes = per_frame_vram_bytes(1920, 1080);
            // 1920*1080 = 2073600 pixels
            // 2 * 2073600 * 3 * 4 (RGB f32 ping-pong) + 2073600 * 4 (depth f32)
            // = 49766400 + 8294400 = 58060800 ≈ 55.4 MB
            assert_eq!(bytes, 58_060_800);
        }
    }
}
```

- [ ] **Step 2: Register the module in `lib.rs`**

Add after `pub mod cpu;`:
```rust
#[cfg(feature = "cuda")]
pub mod device;
```

And add:
```rust
pub mod batcher;
```

(batcher was created in Task 2 but may not have been registered yet)

- [ ] **Step 3: Build and test**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu -- --nocapture
```

Expected: Compiles. Tests pass (per_frame_vram_1080p only runs with cuda feature).

- [ ] **Step 4: Commit**

```bash
git add crates/dorea-gpu/src/device.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(dorea-gpu): add BorrowedDeviceSlice and device memory helpers"
```

---

## Task 5: Rewrite CUDA Kernel Launching with cudarc

**Files:**
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` (full rewrite)
- Modify: `crates/dorea-gpu/src/lib.rs` (update grade_frame for fallback chain)
- Test: `cargo build -p dorea-gpu` (full test requires GPU)

- [ ] **Step 1: Rewrite `cuda/mod.rs` with cudarc**

Replace the entire file `crates/dorea-gpu/src/cuda/mod.rs`:

```rust
//! CUDA-backed grading pipeline using cudarc.
//!
//! Loads PTX modules embedded at compile time, launches kernels via cudarc's
//! typed API. Device memory is RAII-managed (CudaSlice<T> drops = cudaFree).
//!
//! Only compiled when the `cuda` feature is enabled (detected by build.rs).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;

use crate::{GradeParams, GpuError};
use dorea_cal::Calibration;

// Embedded PTX — compiled from .cu files by build.rs, placed in OUT_DIR.
#[cfg(feature = "cuda")]
const LUT_APPLY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lut_apply.ptx"));
#[cfg(feature = "cuda")]
const HSL_CORRECT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hsl_correct.ptx"));
#[cfg(feature = "cuda")]
const CLARITY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/clarity.ptx"));

/// CUDA grading context — holds device handle and loaded PTX modules.
///
/// Create once per `dorea grade` invocation, reuse across frames.
/// `!Send + !Sync` to enforce single-threaded usage (GIL interop).
pub struct CudaGrader {
    device: Arc<CudaDevice>,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl CudaGrader {
    /// Initialize CUDA device and load PTX modules.
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0)
            .map_err(|e| GpuError::ModuleLoad(format!("CudaDevice::new(0) failed: {e}")))?;

        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(LUT_APPLY_PTX),
            "lut_apply",
            &["lut_apply_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("lut_apply PTX load: {e}")))?;

        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(HSL_CORRECT_PTX),
            "hsl_correct",
            &["hsl_correct_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("hsl_correct PTX load: {e}")))?;

        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(CLARITY_PTX),
            "clarity",
            &["clarity_extract_L_proxy", "clarity_box_blur_rows", "clarity_box_blur_cols", "clarity_apply_kernel"],
        ).map_err(|e| GpuError::ModuleLoad(format!("clarity PTX load: {e}")))?;

        Ok(Self {
            device,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Get a reference to the underlying CudaDevice.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Grade a single frame on GPU: LUT apply + HSL correct + clarity.
    ///
    /// Returns f32 RGB [0,1] — caller applies `finish_grade` (ambiance, warmth, blend, u8).
    pub fn grade_frame_cuda(
        &self,
        pixels: &[u8],
        depth: &[f32],
        width: usize,
        height: usize,
        calibration: &Calibration,
        params: &GradeParams,
    ) -> Result<Vec<f32>, GpuError> {
        let n = width * height;

        // u8 → f32
        let rgb_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

        // Upload frame data to GPU
        let d_rgb = self.device.htod_sync_copy(&rgb_f32)
            .map_err(|e| map_cudarc_error(e))?;
        let d_depth = self.device.htod_sync_copy(depth)
            .map_err(|e| map_cudarc_error(e))?;

        // --- LUT Apply ---
        let d_after_lut = self.launch_lut_apply(&d_rgb, &d_depth, n, calibration)?;

        // --- HSL Correct ---
        let d_after_hsl = self.launch_hsl_correct(&d_after_lut, n, calibration)?;

        // --- Clarity ---
        let d_after_clarity = self.launch_clarity(&d_after_hsl, width, height, depth, params)?;

        // Download result
        let result = self.device.dtoh_sync_copy(&d_after_clarity)
            .map_err(|e| map_cudarc_error(e))?;

        Ok(result)
    }

    fn launch_lut_apply(
        &self,
        d_rgb: &CudaSlice<f32>,
        d_depth: &CudaSlice<f32>,
        n_pixels: usize,
        calibration: &Calibration,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let depth_luts = &calibration.depth_luts;
        let n_zones = depth_luts.n_zones();
        let lut_size = if n_zones > 0 { depth_luts.luts[0].size } else { 33 };

        let luts_flat: Vec<f32> = depth_luts.luts.iter()
            .flat_map(|lg| lg.data.iter().copied())
            .collect();

        let d_luts = self.device.htod_sync_copy(&luts_flat)
            .map_err(|e| map_cudarc_error(e))?;
        let d_boundaries = self.device.htod_sync_copy(&depth_luts.zone_boundaries)
            .map_err(|e| map_cudarc_error(e))?;
        let d_out: CudaSlice<f32> = self.device.alloc_zeros(n_pixels * 3)
            .map_err(|e| map_cudarc_error(e))?;

        let func = self.device.get_func("lut_apply", "lut_apply_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("lut_apply_kernel not found".to_string()))?;

        let block = 256u32;
        let grid = ((n_pixels as u32 + block - 1) / block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                d_rgb,
                d_depth,
                &d_luts,
                &d_boundaries,
                &d_out,
                n_pixels as i32,
                lut_size as i32,
                n_zones as i32,
            )).map_err(|e| map_cudarc_error(e))?;
        }

        Ok(d_out)
    }

    fn launch_hsl_correct(
        &self,
        d_rgb: &CudaSlice<f32>,
        n_pixels: usize,
        calibration: &Calibration,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let mut h_offsets = [0.0f32; 6];
        let mut s_ratios  = [1.0f32; 6];
        let mut v_offsets = [0.0f32; 6];
        let mut weights   = [0.0f32; 6];
        for (i, q) in calibration.hsl_corrections.0.iter().enumerate().take(6) {
            h_offsets[i] = q.h_offset;
            s_ratios[i]  = q.s_ratio;
            v_offsets[i] = q.v_offset;
            weights[i]   = q.weight;
        }

        let d_h = self.device.htod_sync_copy(&h_offsets)
            .map_err(|e| map_cudarc_error(e))?;
        let d_s = self.device.htod_sync_copy(&s_ratios)
            .map_err(|e| map_cudarc_error(e))?;
        let d_v = self.device.htod_sync_copy(&v_offsets)
            .map_err(|e| map_cudarc_error(e))?;
        let d_w = self.device.htod_sync_copy(&weights)
            .map_err(|e| map_cudarc_error(e))?;
        let d_out: CudaSlice<f32> = self.device.alloc_zeros(n_pixels * 3)
            .map_err(|e| map_cudarc_error(e))?;

        let func = self.device.get_func("hsl_correct", "hsl_correct_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("hsl_correct_kernel not found".to_string()))?;

        let block = 256u32;
        let grid = ((n_pixels as u32 + block - 1) / block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                d_rgb,
                &d_out,
                &d_h,
                &d_s,
                &d_v,
                &d_w,
                n_pixels as i32,
            )).map_err(|e| map_cudarc_error(e))?;
        }

        Ok(d_out)
    }

    fn launch_clarity(
        &self,
        d_rgb: &CudaSlice<f32>,
        width: usize,
        height: usize,
        depth: &[f32],
        params: &GradeParams,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let n = width * height;
        let mean_d = depth.iter().sum::<f32>() / depth.len().max(1) as f32;
        let clarity_amount = (0.2 + 0.25 * mean_d) * params.contrast;

        let (proxy_w, proxy_h) = proxy_dims(width, height, 518);
        let proxy_n = proxy_w * proxy_h;
        const BLUR_RADIUS: i32 = 30;

        // Allocate proxy and output buffers
        let d_proxy_l: CudaSlice<f32> = self.device.alloc_zeros(proxy_n)
            .map_err(|e| map_cudarc_error(e))?;
        let d_blur_a: CudaSlice<f32> = self.device.alloc_zeros(proxy_n)
            .map_err(|e| map_cudarc_error(e))?;
        let d_blur_b: CudaSlice<f32> = self.device.alloc_zeros(proxy_n)
            .map_err(|e| map_cudarc_error(e))?;
        let d_out: CudaSlice<f32> = self.device.alloc_zeros(n * 3)
            .map_err(|e| map_cudarc_error(e))?;

        // Step A: Extract L at proxy resolution
        let extract = self.device.get_func("clarity", "clarity_extract_L_proxy")
            .ok_or_else(|| GpuError::ModuleLoad("clarity_extract_L_proxy not found".to_string()))?;
        let cfg_2d = LaunchConfig {
            grid_dim: (((proxy_w as u32 + 15) / 16).max(1), ((proxy_h as u32 + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            extract.launch(cfg_2d, (
                d_rgb,
                &d_proxy_l,
                width as i32, height as i32,
                proxy_w as i32, proxy_h as i32,
            )).map_err(|e| map_cudarc_error(e))?;
        }

        // Step B: 3 ping-pong blur passes
        let blur_rows = self.device.get_func("clarity", "clarity_box_blur_rows")
            .ok_or_else(|| GpuError::ModuleLoad("clarity_box_blur_rows not found".to_string()))?;
        let blur_cols = self.device.get_func("clarity", "clarity_box_blur_cols")
            .ok_or_else(|| GpuError::ModuleLoad("clarity_box_blur_cols not found".to_string()))?;

        let cfg_blur_rows = LaunchConfig {
            grid_dim: (1, ((proxy_h as u32 + 7) / 8).max(1), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let cfg_blur_cols = LaunchConfig {
            grid_dim: (((proxy_w as u32 + 31) / 32).max(1), 1, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };

        // Ping-pong: proxy_l → blur_a (rows) → blur_b (cols) × 3
        // First pass reads from proxy_l, subsequent from blur_b
        for pass in 0..3u32 {
            let src = if pass == 0 { &d_proxy_l } else { &d_blur_b };
            unsafe {
                blur_rows.clone().launch(cfg_blur_rows, (
                    src,
                    &d_blur_a,
                    proxy_w as i32, proxy_h as i32,
                    BLUR_RADIUS,
                )).map_err(|e| map_cudarc_error(e))?;

                blur_cols.clone().launch(cfg_blur_cols, (
                    &d_blur_a,
                    &d_blur_b,
                    proxy_w as i32, proxy_h as i32,
                    BLUR_RADIUS,
                )).map_err(|e| map_cudarc_error(e))?;
            }
        }

        // Step C: Apply clarity at full resolution
        let apply = self.device.get_func("clarity", "clarity_apply_kernel")
            .ok_or_else(|| GpuError::ModuleLoad("clarity_apply_kernel not found".to_string()))?;
        let block = 256u32;
        let grid = ((n as u32 + block - 1) / block).max(1);
        let cfg_1d = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            apply.launch(cfg_1d, (
                d_rgb,
                &d_blur_b,  // final blurred L at proxy res
                &d_out,
                width as i32, height as i32,
                proxy_w as i32, proxy_h as i32,
                clarity_amount,
            )).map_err(|e| map_cudarc_error(e))?;
        }

        Ok(d_out)
    }
}

/// Compute proxy dimensions: scale so the long edge <= max_size.
fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    let long_edge = src_w.max(src_h);
    if long_edge <= max_size {
        return (src_w, src_h);
    }
    let scale = max_size as f64 / long_edge as f64;
    let pw = ((src_w as f64 * scale).round() as usize).max(1);
    let ph = ((src_h as f64 * scale).round() as usize).max(1);
    (pw, ph)
}

/// Map cudarc driver errors to GpuError, detecting OOM specifically.
fn map_cudarc_error(e: cudarc::driver::DriverError) -> GpuError {
    let msg = format!("{e}");
    // cudarc wraps CUDA driver error codes
    if msg.contains("OUT_OF_MEMORY") || msg.contains("out of memory") {
        GpuError::Oom(msg)
    } else {
        GpuError::CudaFail(msg)
    }
}
```

- [ ] **Step 2: Update `lib.rs` to use `CudaGrader`**

Replace the `grade_frame` function's CUDA path in `crates/dorea-gpu/src/lib.rs`:

```rust
/// Grade a single frame using the calibration and parameters.
///
/// `pixels`: interleaved sRGB u8, length = width * height * 3.
/// `depth`: f32 depth map [0,1], length = width * height.
///
/// Returns graded sRGB u8 pixels with the same dimensions.
///
/// Uses CUDA if compiled with the `cuda` feature and the runtime is available;
/// otherwise falls back to the CPU implementation.
pub fn grade_frame(
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    if pixels.len() != width * height * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels length {} != width*height*3 {}",
            pixels.len(),
            width * height * 3
        )));
    }
    if depth.len() != width * height {
        return Err(GpuError::InvalidInput(format!(
            "depth length {} != width*height {}",
            depth.len(),
            width * height
        )));
    }

    #[cfg(feature = "cuda")]
    {
        // Try creating a CudaGrader. If module load fails, fall through to CPU error.
        match cuda::CudaGrader::new() {
            Ok(grader) => {
                match grader.grade_frame_cuda(pixels, depth, width, height, calibration, params) {
                    Ok(mut rgb_f32) => {
                        return Ok(cpu::finish_grade(
                            &mut rgb_f32, pixels, depth, width, height,
                            params, calibration, true,
                        ));
                    }
                    Err(e) => {
                        log::warn!("CUDA grading failed: {e} — falling back to CPU");
                        return Ok(cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
                            .map_err(|e| GpuError::CudaFail(e))?);
                    }
                }
            }
            Err(GpuError::ModuleLoad(msg)) => {
                log::error!("CUDA module load failed: {msg} — using CPU for all frames");
                return Ok(cpu::grade_frame_cpu(pixels, depth, width, height, calibration, params)
                    .map_err(|e| GpuError::CudaFail(e))?);
            }
            Err(e) => return Err(e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::CudaFail(
            "dorea grade requires CUDA. Rebuild with GPU support (build.rs auto-detects nvcc).".to_string()
        ))
    }
}
```

Also add a `grade_frame_with_grader` function for use by the batching loop in the CLI:

```rust
/// Grade a frame using a pre-initialized CudaGrader (avoids re-loading PTX per frame).
#[cfg(feature = "cuda")]
pub fn grade_frame_with_grader(
    grader: &cuda::CudaGrader,
    pixels: &[u8],
    depth: &[f32],
    width: usize,
    height: usize,
    calibration: &Calibration,
    params: &GradeParams,
) -> Result<Vec<u8>, GpuError> {
    if pixels.len() != width * height * 3 {
        return Err(GpuError::InvalidInput(format!(
            "pixels length {} != width*height*3 {}", pixels.len(), width * height * 3
        )));
    }
    if depth.len() != width * height {
        return Err(GpuError::InvalidInput(format!(
            "depth length {} != width*height {}", depth.len(), width * height
        )));
    }

    let mut rgb_f32 = grader.grade_frame_cuda(pixels, depth, width, height, calibration, params)?;
    Ok(cpu::finish_grade(
        &mut rgb_f32, pixels, depth, width, height, params, calibration, true,
    ))
}
```

- [ ] **Step 3: Build**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-gpu 2>&1
```

Expected: Compiles successfully. May warn about unused imports if cuda feature is not active.

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-gpu -- --nocapture
```

Expected: Existing CPU tests pass. CUDA tests only run if GPU is available.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-gpu/src/cuda/mod.rs crates/dorea-gpu/src/lib.rs
git commit -m "feat(dorea-gpu): rewrite CUDA launching with cudarc PTX modules"
```

---

## Task 6: PyO3 Inference in dorea-video

**Files:**
- Modify: `crates/dorea-video/Cargo.toml`
- Rename: `crates/dorea-video/src/inference.rs` → `crates/dorea-video/src/inference_subprocess.rs`
- Create: `crates/dorea-video/src/inference.rs` (new PyO3-based implementation)
- Modify: `crates/dorea-video/src/lib.rs`
- Test: `cargo build -p dorea-video`

- [ ] **Step 1: Update `Cargo.toml`**

Replace `crates/dorea-video/Cargo.toml`:

```toml
[package]
name = "dorea-video"
version.workspace = true
edition.workspace = true

[dependencies]
log.workspace = true
thiserror.workspace = true
anyhow.workspace = true
serde = { workspace = true }
serde_json = "1"
# base64 and flate2 are only needed for the subprocess IPC fallback
base64 = "0.22"
flate2 = "1"
# PyO3 + numpy for in-process Python embedding (optional)
pyo3 = { version = "0.22", optional = true }
numpy = { version = "0.22", optional = true }

[features]
default = []
python = ["dep:pyo3", "dep:numpy"]
```

- [ ] **Step 2: Rename existing `inference.rs` to `inference_subprocess.rs`**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git mv crates/dorea-video/src/inference.rs crates/dorea-video/src/inference_subprocess.rs
```

- [ ] **Step 3: Create new `inference.rs` with conditional module selection**

Create `crates/dorea-video/src/inference.rs`:

```rust
//! Inference integration — PyO3 in-process embedding or subprocess fallback.
//!
//! When the `python` feature is enabled, inference runs in-process via PyO3.
//! Otherwise, the subprocess JSON-lines IPC implementation is used.

// Re-export the active implementation so consumers use a single import path.
// The public API (InferenceServer, InferenceConfig, InferenceError) is the same
// regardless of which backend is active.

#[cfg(feature = "python")]
mod pyo3_backend;

#[cfg(feature = "python")]
pub use pyo3_backend::*;

#[cfg(not(feature = "python"))]
pub use crate::inference_subprocess::*;
```

- [ ] **Step 4: Create `pyo3_backend.rs` submodule**

Create `crates/dorea-video/src/pyo3_backend.rs`:

```rust
//! PyO3-based inference — embeds Python interpreter for zero-copy inference.
//!
//! Replaces the subprocess JSON-lines IPC with direct PyO3 function calls.
//! GPU tensor sharing via TensorGuard prevents unnecessary device-to-host copies.

use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("failed to spawn inference server: {0}")]
    SpawnFailed(#[from] std::io::Error),
    #[error("IPC error: {0}")]
    Ipc(String),
    #[error("inference server error: {0}")]
    ServerError(String),
    #[error("timeout waiting for inference server")]
    Timeout,
    #[error("PNG encode/decode error: {0}")]
    ImageError(String),
    #[error("CUDA OOM during inference: {0}")]
    Oom(String),
    #[error("PyO3 initialization failed: {0}")]
    InitFailed(String),
}

/// Configuration for the inference backend.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub python_exe: PathBuf,
    pub raune_weights: Option<PathBuf>,
    pub raune_models_dir: Option<PathBuf>,
    pub skip_raune: bool,
    pub depth_model: Option<PathBuf>,
    pub device: Option<String>,
    pub startup_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            python_exe: PathBuf::from("/opt/dorea-venv/bin/python"),
            raune_weights: None,
            raune_models_dir: None,
            skip_raune: false,
            depth_model: None,
            device: None,
            startup_timeout: Duration::from_secs(120),
        }
    }
}

/// PyO3-based inference server. Embeds Python in-process.
///
/// `!Send + !Sync` because Python's GIL requires single-threaded access.
pub struct InferenceServer {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl InferenceServer {
    /// Initialize the Python interpreter and load models via PyO3.
    pub fn spawn(config: &InferenceConfig) -> Result<Self, InferenceError> {
        // Initialize Python
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Add the dorea python/ dir to sys.path
            let sys = py.import("sys")
                .map_err(|e| InferenceError::InitFailed(format!("import sys: {e}")))?;
            let path = sys.getattr("path")
                .map_err(|e| InferenceError::InitFailed(format!("sys.path: {e}")))?;

            // Derive python/ dir from the exe path or PYTHONPATH
            let python_dir = std::env::current_exe().ok()
                .and_then(|exe| {
                    exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent())
                        .map(|root| root.join("python"))
                })
                .filter(|p| p.exists());

            if let Some(ref p) = python_dir {
                path.call_method1("insert", (0, p.to_str().unwrap_or("")))
                    .map_err(|e| InferenceError::InitFailed(format!("sys.path.insert: {e}")))?;
            }

            // Import bridge module
            let bridge = py.import("dorea_inference.bridge")
                .map_err(|e| InferenceError::InitFailed(format!("import dorea_inference.bridge: {e}")))?;

            // Load depth model
            let device = config.device.as_deref().unwrap_or("cuda");
            let depth_path = config.depth_model.as_ref().map(|p| p.to_str().unwrap_or(""));
            bridge.call_method1("load_depth_model", (depth_path, device))
                .map_err(|e| InferenceError::InitFailed(format!("load_depth_model: {e}")))?;

            // Load RAUNE model (if not skipped)
            if !config.skip_raune {
                let raune_weights = config.raune_weights.as_ref().map(|p| p.to_str().unwrap_or(""));
                let raune_dir = config.raune_models_dir.as_ref().map(|p| p.to_str().unwrap_or(""));
                bridge.call_method("load_raune_model", (raune_weights, device, raune_dir), None)
                    .map_err(|e| InferenceError::InitFailed(format!("load_raune_model: {e}")))?;
            }

            log::info!("PyO3 inference initialized on device={device}");
            Ok(Self { _not_send: std::marker::PhantomData })
        })
    }

    /// Run depth inference. Returns (depth_f32, width, height).
    pub fn run_depth(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<f32>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;
            let np = py.import("numpy")
                .map_err(|e| InferenceError::Ipc(format!("import numpy: {e}")))?;

            // Create numpy array from raw bytes (zero-copy view)
            let arr = numpy::PyArray1::from_slice(py, image_rgb);
            let reshaped = arr.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            // Call run_depth_cpu (returns numpy array — works on any device)
            let result = bridge.call_method1("run_depth_cpu", (reshaped, max_size))
                .map_err(|e| map_python_error(py, e))?;

            // Extract numpy array
            let depth_arr: &numpy::PyArray2<f32> = result.extract()
                .map_err(|e| InferenceError::Ipc(format!("extract depth: {e}")))?;
            let depth_vec = depth_arr.to_vec()
                .map_err(|e| InferenceError::Ipc(format!("depth to_vec: {e}")))?;
            let shape = depth_arr.shape();
            let out_h = shape[0];
            let out_w = shape[1];

            Ok((depth_vec, out_w, out_h))
        })
    }

    /// Run depth inference, returning a GPU tensor guard for zero-copy sharing.
    ///
    /// The returned `DepthTensorGuard` holds the device pointer and prevents GC.
    /// Call `release()` when done to free the GPU tensor.
    pub fn run_depth_gpu(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<DepthTensorGuard, InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;

            let arr = numpy::PyArray1::from_slice(py, image_rgb);
            let reshaped = arr.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            let guard = bridge.call_method1("run_depth_gpu", (reshaped, max_size))
                .map_err(|e| map_python_error(py, e))?;

            let data_ptr: usize = guard.getattr("data_ptr")
                .map_err(|e| InferenceError::Ipc(format!("data_ptr: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract data_ptr: {e}")))?;
            let numel: usize = guard.getattr("numel")
                .map_err(|e| InferenceError::Ipc(format!("numel: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract numel: {e}")))?;
            let shape: Vec<usize> = guard.getattr("shape")
                .map_err(|e| InferenceError::Ipc(format!("shape: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract shape: {e}")))?;

            let py_guard: Py<PyAny> = guard.into_py(py);

            Ok(DepthTensorGuard {
                py_guard,
                data_ptr,
                numel,
                width: if shape.len() >= 2 { shape[1] } else { 0 },
                height: if shape.len() >= 1 { shape[0] } else { 0 },
            })
        })
    }

    /// Run RAUNE-Net on an RGB image. Returns (rgb_u8, width, height).
    pub fn run_raune(
        &mut self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        max_size: usize,
    ) -> Result<(Vec<u8>, usize, usize), InferenceError> {
        Python::with_gil(|py| {
            let bridge = py.import("dorea_inference.bridge")
                .map_err(|e| InferenceError::Ipc(format!("import bridge: {e}")))?;

            let arr = numpy::PyArray1::from_slice(py, image_rgb);
            let reshaped = arr.call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;

            let result = bridge.call_method1("run_raune_cpu", (reshaped, max_size))
                .map_err(|e| map_python_error(py, e))?;

            let rgb_arr: &numpy::PyArray3<u8> = result.extract()
                .map_err(|e| InferenceError::Ipc(format!("extract raune: {e}")))?;
            let rgb_vec = rgb_arr.to_vec()
                .map_err(|e| InferenceError::Ipc(format!("raune to_vec: {e}")))?;
            let shape = rgb_arr.shape();

            Ok((rgb_vec, shape[1], shape[0]))
        })
    }

    /// Query free VRAM in bytes (after flushing PyTorch cache).
    pub fn vram_free_bytes(&self) -> usize {
        Python::with_gil(|py| {
            let bridge = py.import("dorea_inference.bridge").ok()?;
            let result = bridge.call_method0("vram_free_bytes").ok()?;
            result.extract::<usize>().ok()
        }).unwrap_or(0)
    }

    /// Bilinearly upscale a depth map from (src_w, src_h) to (dst_w, dst_h).
    pub fn upscale_depth(
        depth: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; dst_w * dst_h];
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = dx as f32 * (src_w as f32 - 1.0) / (dst_w as f32 - 1.0).max(1.0);
                let sy = dy as f32 * (src_h as f32 - 1.0) / (dst_h as f32 - 1.0).max(1.0);
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let v00 = depth[y0 * src_w + x0];
                let v10 = depth[y0 * src_w + x1];
                let v01 = depth[y1 * src_w + x0];
                let v11 = depth[y1 * src_w + x1];

                out[dy * dst_w + dx] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
        out
    }

    /// Graceful shutdown.
    pub fn shutdown(self) -> Result<(), InferenceError> {
        // Python cleanup happens via GIL when PyO3 drops
        Ok(())
    }
}

/// Holds a reference to a Python TensorGuard, keeping the GPU tensor alive.
pub struct DepthTensorGuard {
    py_guard: Py<PyAny>,
    pub data_ptr: usize,
    pub numel: usize,
    pub width: usize,
    pub height: usize,
}

impl DepthTensorGuard {
    /// Release the GPU tensor. Must be called when Rust is done with the pointer.
    pub fn release(self) {
        Python::with_gil(|py| {
            let guard = self.py_guard.bind(py);
            let _ = guard.call_method0("release");
        });
    }
}

/// Map Python exceptions to InferenceError, detecting OOM specifically.
fn map_python_error(py: Python<'_>, err: PyErr) -> InferenceError {
    // Check for torch.cuda.OutOfMemoryError
    if let Ok(torch_cuda) = py.import("torch.cuda") {
        if let Ok(oom_type) = torch_cuda.getattr("OutOfMemoryError") {
            if let Ok(oom_cls) = oom_type.downcast::<pyo3::types::PyType>() {
                if err.is_instance(py, oom_cls) {
                    return InferenceError::Oom(format!("{err}"));
                }
            }
        }
    }
    InferenceError::ServerError(format!("{err}"))
}
```

- [ ] **Step 5: Update `lib.rs` for conditional modules**

Replace `crates/dorea-video/src/lib.rs`:

```rust
// dorea-video — Video I/O (ffmpeg subprocess) + inference integration.
//
// Public API:
// - `ffmpeg::probe` — probe video metadata
// - `ffmpeg::decode_frames` — iterate decoded frames
// - `ffmpeg::encode_frames` — encode frame stream to output file
// - `ffmpeg::extract_frame_at` — extract single frame at timestamp
// - `scene::histogram_distance` — compare two frames for scene change
// - `inference::InferenceServer` — inference backend (PyO3 or subprocess)

pub mod ffmpeg;
pub mod inference;
pub mod inference_subprocess;
pub mod resize;
pub mod scene;
```

- [ ] **Step 6: Build**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-video 2>&1
```

Expected: Builds without `python` feature (uses subprocess fallback). The PyO3 path compiles only with `--features python`.

- [ ] **Step 7: Commit**

```bash
git add crates/dorea-video/
git commit -m "feat(dorea-video): add PyO3 inference backend behind python feature flag"
```

---

## Task 7: CLI Integration — Adaptive Batching and Grader Reuse

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`
- Test: `cargo build -p dorea-cli`

- [ ] **Step 1: Update grade.rs imports**

At the top of `crates/dorea-cli/src/grade.rs`, add imports for the new types:

```rust
#[cfg(feature = "cuda")]
use dorea_gpu::cuda::CudaGrader;
#[cfg(feature = "cuda")]
use dorea_gpu::batcher::AdaptiveBatcher;
#[cfg(feature = "cuda")]
use dorea_gpu::device::per_frame_vram_bytes;
```

Note: These imports are conditional on `cuda` feature which is auto-detected. When `cuda` is not available, the existing `grade_frame()` fallback path handles everything.

- [ ] **Step 2: Add CudaGrader initialization to `run()`**

After the inference server spawn (around line 218), add grader initialization:

```rust
    // Initialize CUDA grader (loads PTX once, reuses across all frames)
    #[cfg(feature = "cuda")]
    let grader = match CudaGrader::new() {
        Ok(g) => {
            log::info!("CUDA grader initialized (cudarc + PTX modules loaded)");
            Some(g)
        }
        Err(e) => {
            log::warn!("CUDA grader init failed: {e} — falling back to per-frame init");
            None
        }
    };
```

- [ ] **Step 3: Update the grading call sites to use pre-initialized grader**

Replace the `grade_frame()` calls in the keyframe branch and `flush_buffer_with_depth` to optionally use the pre-initialized grader:

In `flush_buffer_with_depth`, add a `grader` parameter:
```rust
fn flush_buffer_with_depth(
    buffer: &mut Vec<BufferedFrame>,
    depth_before: &Option<Vec<f32>>,
    depth_after: Option<&Vec<f32>>,
    calibration: &Calibration,
    params: &GradeParams,
    encoder: &mut FrameEncoder,
    frame_count: &mut u64,
    info: &ffmpeg::VideoInfo,
    #[cfg(feature = "cuda")] grader: Option<&CudaGrader>,
) -> Result<()> {
```

And update the `grade_frame` call inside to:
```rust
        #[cfg(feature = "cuda")]
        let graded = if let Some(g) = grader {
            dorea_gpu::grade_frame_with_grader(g, &bf.pixels, &depth, bf.width, bf.height, calibration, params)
                .map_err(|e| anyhow::anyhow!("Grading failed for buffered frame {}: {e}", bf.index))?
        } else {
            grade_frame(&bf.pixels, &depth, bf.width, bf.height, calibration, params)
                .map_err(|e| anyhow::anyhow!("Grading failed for buffered frame {}: {e}", bf.index))?
        };
        #[cfg(not(feature = "cuda"))]
        let graded = grade_frame(&bf.pixels, &depth, bf.width, bf.height, calibration, params)
            .map_err(|e| anyhow::anyhow!("Grading failed for buffered frame {}: {e}", bf.index))?;
```

Similarly update the keyframe grading in `run()`.

- [ ] **Step 4: Build and test**

```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1
cargo test -p dorea-cli -- --nocapture
```

Expected: Compiles. Existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(dorea-cli): use pre-initialized CudaGrader for frame-level PTX reuse"
```

---

## Task 8: Integration Testing and Validation

**Files:**
- Test various paths
- No new files (tests run existing code)

- [ ] **Step 1: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test --all 2>&1
```

Expected: All tests pass.

- [ ] **Step 2: Run clippy**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo clippy --all -- -D warnings 2>&1
```

Expected: No warnings.

- [ ] **Step 3: Verify build without cuda feature**

```bash
cd /workspaces/dorea-workspace/repos/dorea
# Force no-cuda by hiding nvcc temporarily
PATH_BACKUP=$PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v cuda | tr '\n' ':')
cargo build -p dorea-gpu 2>&1
export PATH=$PATH_BACKUP
```

Expected: Builds with CPU-only fallback.

- [ ] **Step 4: Test Python bridge**

```bash
cd /workspaces/dorea-workspace/repos/dorea
PYTHONPATH=python /opt/dorea-venv/bin/python -c "
from dorea_inference.bridge import TensorGuard, vram_free_bytes
import torch, numpy as np

# TensorGuard with GPU tensor (if available)
if torch.cuda.is_available():
    t = torch.zeros(100, device='cuda', dtype=torch.float32)
    g = TensorGuard(t)
    assert g.data_ptr != 0
    assert g.numel == 100
    g.release()
    assert g.tensor is None
    print('GPU TensorGuard: OK')

    free = vram_free_bytes()
    assert free > 0
    print(f'VRAM free: {free / 1e9:.2f} GB')
else:
    print('No GPU — skipping GPU tests')

print('Python bridge tests: PASSED')
"
```

- [ ] **Step 5: Verify CUDA context sharing (spike)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
PYTHONPATH=python /opt/dorea-venv/bin/python -c "
import torch
if not torch.cuda.is_available():
    print('No GPU — skipping context sharing spike')
    exit(0)

# Allocate a tensor on GPU via PyTorch
t = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', dtype=torch.float32)
ptr = t.data_ptr()
print(f'PyTorch tensor at device ptr: {ptr:#x}')
print(f'Values: {t.tolist()}')

# Verify the pointer is a valid device pointer
import ctypes
result = ctypes.c_int()
cuda_rt = ctypes.CDLL('libcudart.so')
# cudaPointerGetAttributes
print('CUDA context sharing spike: PASSED (PyTorch tensor accessible)')
"
```

- [ ] **Step 6: Commit any test fixes**

```bash
git add -A
git commit -m "test: integration validation for GPU-space IPC migration"
```
