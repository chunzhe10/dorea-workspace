# GPU-Space IPC & Adaptive Batching Design

**Date:** 2026-04-03
**Status:** Proposed
**Scope:** dorea-gpu, dorea-video, python/dorea_inference

## Problem

Dorea v2's current architecture has two performance bottlenecks in the Rust-Python
video grading pipeline:

1. **IPC overhead**: The Rust `dorea-video` crate communicates with the Python
   inference subprocess via JSON lines over stdin/stdout, with base64-encoded image
   data. Each inference call encodes ~24MB of pixel data to base64 (33% size
   overhead), serializes to JSON, pipes through stdin, and reverses the process on
   return. This replaces what could be a zero-copy pointer pass.

2. **Unnecessary host-device round-trips**: The CUDA grading pipeline in `dorea-gpu`
   runs three sequential kernels (LUT apply, HSL correct, clarity), each with its own
   host launcher that performs `cudaMalloc` x5, `cudaMemcpy` H-to-D x4, kernel launch,
   `cudaMemcpy` D-to-H x1, `cudaFree` x5. Per frame at 1080p, this produces ~144MB
   of PCIe transfers when only ~48MB is necessary (one upload, one download).

   Additionally, when inference (Depth Anything) produces a depth map on the GPU
   and the grading pipeline needs it, the current flow is:
   GPU (PyTorch) -> CPU (Python) -> base64 -> pipe -> base64 decode (Rust) -> CPU
   -> GPU (CUDA kernel). The depth map never needed to leave the GPU.

## Solution

Two complementary changes:

1. **PyO3 embedding** replaces subprocess IPC. Rust embeds the Python interpreter
   via PyO3 and calls inference functions directly, passing NumPy arrays by
   zero-copy pointer sharing (rust-numpy). Python exceptions (including
   `torch.cuda.OutOfMemoryError`) become Rust `Result::Err`, not dead pipes.

2. **cudarc migration** replaces the `extern "C"` FFI to CUDA host launchers.
   The cudarc crate provides RAII device memory (`CudaSlice<T>`), PTX module
   loading, and typed kernel launches. This enables:
   - Keeping frame data on GPU across the three-kernel grading chain
   - Sharing device pointers between PyTorch and Rust CUDA kernels (same primary
     CUDA context within one process)
   - Adaptive batch sizing with OOM recovery

## Crate Architecture

### dorea-gpu changes

```
crates/dorea-gpu/
  Cargo.toml          # + cudarc dependency (behind "cuda" feature)
  build.rs            # .cu -> .ptx (was .cu -> .o -> .a), embed via include_bytes!
  src/
    lib.rs            # grade_frame() public API unchanged
    cpu.rs            # unchanged
    device.rs         # NEW: DeviceFrameBuffer, BorrowedDeviceSlice, helpers
    cuda/
      mod.rs          # REWRITTEN: cudarc module loading + kernel launch
      kernels/
        lut_apply.cu    # delete host launcher, keep __global__ kernel only
        hsl_correct.cu  # same
        clarity.cu      # same
```

**build.rs change**: nvcc flag changes from `-c` (compile to .o) to `--ptx`
(compile to .ptx). The `.ptx` files are placed in `OUT_DIR` and **embedded into
the binary** via `include_str!(concat!(env!("OUT_DIR"), "/lut_apply.ptx"))` in
`cuda/mod.rs`. This eliminates loose-file deployment issues — no runtime path
resolution needed. The static archive (`libdorea_cuda_kernels.a`) and `extern "C"`
declarations are removed.

**PTX cold-start**: cudarc JIT-compiles embedded PTX on first use (~50-200ms per
module). This is a one-time cost per `dorea grade` invocation, acceptable for a
batch pipeline. If profiling shows this matters, switch to fatbin embedding
(`--fatbin` flag) which includes pre-compiled SASS and skips JIT.

**Cargo.toml additions**:
```toml
[dependencies]
cudarc = { version = "0.12", features = ["driver"], optional = true }

[features]
cuda = ["dep:cudarc"]
```

The `cuda` feature is retained for conditional compilation. `build.rs` still
auto-detects nvcc and sets `rustc-cfg=feature="cuda"`.

### dorea-video changes

```
crates/dorea-video/
  Cargo.toml          # + pyo3, numpy behind "python" feature; - base64, flate2
  src/
    inference.rs      # REWRITTEN: PyO3 embedded calls + device tensor bridge
    inference_subprocess.rs  # KEPT: renamed from old inference.rs, fallback path
    lib.rs            # minor: re-export new inference types
    ffmpeg.rs         # unchanged
    scene.rs          # unchanged
    resize.rs         # unchanged
```

**Cargo.toml additions**:
```toml
[dependencies]
pyo3 = { version = "0.22", optional = true }
numpy = { version = "0.22", optional = true }

[features]
python = ["dep:pyo3", "dep:numpy"]
```

**IMPORTANT**: PyO3 is behind the `python` feature, NOT default. This prevents
`dorea-cli` from inheriting a libpython link dependency. The `auto-initialize`
feature is NOT used — the Python interpreter is initialized explicitly in
`inference.rs` so that library consumers control when/whether Python starts.

**Build-time requirements** when `python` feature is enabled:
- `python3-dev` headers (matching the target Python version)
- `numpy` installed in the target Python environment (for C header FFI)
- `PYO3_PYTHON` env var pointing to the correct Python binary

When `python` feature is disabled, `inference_subprocess.rs` (the current
JSON-lines subprocess approach) remains available as fallback. `base64` and
`flate2` dependencies are retained only when `python` feature is off.

**Removed dependencies** (when `python` feature is on): `base64`, `flate2`. The
manual PNG encoder/decoder (~200 lines) is deleted.

### Python changes

```
python/dorea_inference/
  server.py           # KEPT for standalone testing / fallback
  depth_anything.py   # + infer_gpu() returning on-device tensor
  raune_net.py        # + infer_gpu() returning on-device tensor
  bridge.py           # NEW: PyO3-callable entry points, tensor guard
```

`bridge.py` provides the functions that Rust calls via PyO3:

```python
class TensorGuard:
    """Prevent GC of a GPU tensor while Rust holds its device pointer.
    Explicitly prevents accidental release via reference counting."""
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.data_ptr = tensor.data_ptr()
        self.numel = tensor.numel()
        self.shape = tuple(tensor.shape)
        self.dtype = str(tensor.dtype)

    def release(self) -> None:
        """Explicitly release the tensor. Called by Rust when done."""
        self.tensor = None

def load_depth_model(model_path: str, device: str) -> DepthAnythingInference:
    ...

def run_depth_gpu(model, frame_rgb: np.ndarray, max_size: int) -> TensorGuard:
    """Returns a TensorGuard holding the on-device result tensor."""
    ...

def run_depth_cpu(model, frame_rgb: np.ndarray, max_size: int) -> np.ndarray:
    """CPU fallback — returns numpy array."""
    ...
```

The `TensorGuard` class replaces the bare dict pattern. It holds a strong
reference to the PyTorch tensor, preventing GC. Rust holds a `Py<TensorGuard>`
which prevents Python from collecting the guard. The `release()` method is called
explicitly by Rust when grading completes, dropping the tensor reference.

The existing `server.py` remains for:
- Standalone testing without Rust (`python -m dorea_inference.server`)
- CI environments without GPU / PyO3 build dependencies
- Debugging inference in isolation

## Threading Model

**The grading pipeline is single-threaded.** `Python::with_gil` acquires the GIL,
which serializes all Python calls to one thread at a time. Since the grading hot
loop interleaves PyO3 inference calls with CUDA kernel launches, the entire
`grade_video` function must run on a single thread.

This is acceptable because:
- The pipeline is already sequential (6GB VRAM, one model at a time)
- GPU kernels release the GIL implicitly (the GPU does the work, the CPU thread
  just dispatches and waits)
- Parallelism comes from GPU occupancy (batching), not CPU threads

**Do NOT call `grade_frame` from rayon or tokio.** GIL contention would serialize
the threads anyway, and holding a mutex while calling `with_gil` risks deadlock
if Python callbacks re-enter Rust.

The `InferenceServer` (PyO3-based) struct should be `!Send + !Sync` to prevent
accidental use from other threads. Use `PhantomData<*const ()>` to opt out.

## Data Flow

### Calibration pipeline (keyframes only, runs once per dive)

```
ffmpeg decode keyframe -> CPU Vec<u8> RGB
  |
  +-> PyO3: run_depth_gpu(model, numpy_array, max_size)
  |     Python: numpy -> tensor -> GPU inference -> tensor stays on device
  |     Returns: TensorGuard with device pointer + dims
  |     Rust: wrap as BorrowedDeviceSlice<'py, f32>
  |     Then: cuMemcpy D-to-H -> Vec<f32> (LUT building needs host data)
  |
  +-> PyO3: run_raune_gpu(model, numpy_array, max_size)
  |     Same pattern -> BorrowedDeviceSlice<'py, u8>
  |     Then: cuMemcpy D-to-H -> Vec<u8> (LUT building needs host data)
  |
  +-> LUT building on CPU (dorea-lut, dorea-hsl) — unchanged
```

Calibration benefits from eliminating base64 encoding but not from GPU-space
sharing, since LUT building algorithms require host memory.

### Grading pipeline (per frame, hot path)

```
ffmpeg decode frame -> CPU Vec<u8> RGB
  |
  v  upload once
CudaSlice<f32> d_rgb       <- u8-to-f32 + htod copy (cudarc)
CudaSlice<f32> d_depth     <- from PyO3 inference (BorrowedDeviceSlice, zero-copy)
                               OR from precomputed host depth (htod copy)
  |
  v  stays on GPU (no host round-trips)
  |
  +- lut_apply_kernel(d_rgb, d_depth, d_luts, ...) -> d_after_lut
  |
  +- hsl_correct_kernel(d_after_lut, ...) -> d_after_hsl
  |
  +- clarity_kernel(d_after_hsl, ...) -> d_after_clarity
  |
  v  download once
CudaSlice.dtoh_sync_copy() -> Vec<f32>
  |
  v
cpu::finish_grade() (ambiance, warmth, blend, u8 conversion)
  |
  v
ffmpeg encode frame <- CPU Vec<u8> RGB
```

**Per frame at 1080p (1920x1080):**

| Metric | Before (extern C) | After (cudarc) |
|---|---|---|
| cudaMalloc calls | 15 | 3 (+ cached calibration) |
| cudaMemcpy H-to-D | 12 | 1-2 |
| cudaMemcpy D-to-H | 3 | 1 |
| cudaFree calls | 15 | 0 (RAII Drop) |
| PCIe transfer | ~144MB | ~48MB (or ~24MB if depth from GPU) |

### GPU tensor bridge (inference to grading, zero-copy path)

When Depth Anything inference feeds directly into grading:

```rust
Python::with_gil(|py| {
    let guard: &PyAny = call_depth_inference(py, frame)?;
    let data_ptr: usize = guard.getattr("data_ptr")?.extract()?;
    let numel: usize = guard.getattr("numel")?.extract()?;

    // Lifetime-bound to 'py — cannot escape this closure
    let borrowed = unsafe {
        BorrowedDeviceSlice::from_raw(py, data_ptr as *mut f32, numel)
    };

    // Grading uses borrowed depth — completes within GIL scope
    let graded = grade_with_device_depth(device, &d_rgb, &borrowed, cal, params)?;

    // Explicitly release the Python tensor
    guard.call_method0("release")?;

    // borrowed drops here — no cudaFree (not owned)
    // guard drops here — Python GC reclaims TensorGuard
    Ok(graded)
})
```

The `'py` lifetime on `BorrowedDeviceSlice` prevents it from escaping the GIL
scope at compile time. See "Borrowed device pointer safety" below for the type
design.

## Adaptive Batching

### Motivation

When grading a full video (`dorea grade`), processing frames one at a time
under-utilizes GPU parallelism. Batching N frames amortizes kernel dispatch
overhead (~5-10us per launch x 3 kernels per frame). The adaptive layer
automatically finds the largest batch size that fits in available VRAM.

Note: batching does NOT improve memory coalescing unless frames are interleaved
in memory (struct-of-arrays layout). The current layout is array-of-structs
(contiguous per frame), so the benefit is purely dispatch amortization and
reduced synchronization points.

### VRAM budget

Per frame (1080p, ping-pong buffer strategy):
- Buffer A (input/output ping-pong): 24MB
- Buffer B (input/output ping-pong): 24MB
- Depth map: 8MB
- Total per frame: **56MB**

Calibration data (uploaded once): ~2.1MB (5 zones x 33^3 x 3 x f32)

Available headroom:

| Scenario | VRAM free | Max frames (1080p) | Max frames (4K) |
|---|---|---|---|
| Grading only | ~5.8GB | ~100 | ~25 |
| Depth Anything loaded (1.5GB) | ~4.3GB | ~73 | ~18 |
| SAM2 loaded (3GB) | ~2.8GB | ~46 | ~11 |

### VRAM probing

**Do NOT use raw `cuMemGetInfo` for the initial probe.** PyTorch's caching
allocator holds freed blocks in its internal pool; the driver API reports these
as allocated, undercounting available memory by potentially gigabytes.

Correct probe sequence:
1. `torch.cuda.empty_cache()` — release PyTorch's cached blocks back to the driver
2. `torch.cuda.mem_get_info()` — returns `(free, total)` with allocator awareness
3. Apply 15% safety margin for fragmentation and dual-allocator overhead
4. Divide by per-frame cost

This probe runs via PyO3 at the start of `grade_video`, before the batch loop.

### Dual allocator fragmentation

cudarc uses `cuMemAlloc` (driver API). PyTorch uses `cudaMalloc` (runtime API)
with its caching allocator on top. Both carve from the same device memory but
neither knows about the other's pools. On a 6GB card, a cached PyTorch block
sitting in the middle of the address space can prevent cudarc from allocating a
contiguous batch buffer even when total free memory suffices.

Mitigations:
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- The 15% safety margin accounts for some fragmentation
- The adaptive batcher handles the rest: if allocation fails, it halves and retries
- Call `torch.cuda.empty_cache()` before cudarc allocations when transitioning
  from inference to grading

### Algorithm

```rust
pub struct AdaptiveBatcher {
    batch_size: usize,
    max_batch: usize,
    min_batch: usize,       // always 1
    successes: usize,
    grow_threshold: usize,  // default 10, tunable
}
```

**Initialization**: VRAM probe (see above), divide by per-frame cost.

**On success**: increment success counter. After `grow_threshold` (default: 10)
consecutive successes, grow batch size by 50% (capped at `max_batch`).
Conservative ramp-up avoids oscillation.

**On OOM**: reset success counter, halve batch size (floor at `min_batch=1`). If
batch_size=1 still OOMs, propagate the error — the frame genuinely cannot fit.

**On non-OOM CUDA error**: do not adjust batch size. Fall through to single-frame
retry, then CPU fallback.

**Batch size does NOT persist across runs.** Each `dorea grade` invocation probes
fresh because VRAM state depends on what else is loaded at the time.

**External VRAM consumers**: if another process grabs VRAM mid-run, the batcher
discovers this via OOM and halves. Recovery is slow (10 successes to grow back by
50%). This is acceptable for a batch pipeline — correctness over throughput. If
the external consumer is periodic and causes oscillation, the batcher will
settle at the minimum stable batch size. No attempt is made to detect or
coordinate with external consumers.

### Batch grading flow

```rust
while let Some(chunk) = frame_iter.take_up_to(batcher.batch_size) {
    loop {
        match grade_batch(device, &chunk[..batcher.batch_size.min(chunk.len())], ...) {
            Ok(results) => {
                batcher.report_success();
                encode_results(results);
                break;
            }
            Err(GradeError::Oom) => {
                let was_minimum = batcher.batch_size <= batcher.min_batch;
                batcher.report_oom();
                if was_minimum {
                    // Already at batch_size=1 and still OOM — cannot fit
                    return Err(GradeError::Oom);
                }
                // retry same chunk with smaller batch
            }
            Err(e) => return Err(e),
        }
    }
}
```

## Error Handling

### Error types

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

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("CUDA OOM during inference: {0}")]
    Oom(String),

    #[error("Python exception: {0}")]
    PythonError(String),

    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("PyO3 initialization failed: {0}")]
    InitFailed(String),
}
```

### CUDA error mapping

cudarc returns `cudarc::driver::DriverError`. Use structured enum matching, not
string matching:

```rust
fn from_cudarc(e: DriverError) -> GpuError {
    match e {
        DriverError::Cuda(CudaError::OutOfMemory) => {
            GpuError::Oom(format!("{e}"))
        }
        _ => GpuError::CudaFail(format!("{e}")),
    }
}
```

### Python exception handling

PyO3 converts Python exceptions to `PyErr`. Use `isinstance` checking, not
string matching on type names (PyTorch renamed the OOM class in 2.0):

```rust
fn is_torch_oom(py: Python<'_>, err: &PyErr) -> bool {
    if let Ok(torch_cuda) = py.import("torch.cuda") {
        if let Ok(oom_type) = torch_cuda.getattr("OutOfMemoryError") {
            return err.is_instance(py, oom_type.downcast().unwrap_or_else(|_| return false));
        }
    }
    false
}
```

Key mapping:
- `torch.cuda.OutOfMemoryError` -> `InferenceError::Oom` (recoverable,
  call `torch.cuda.empty_cache()` before retry)
- Any other exception -> `InferenceError::PythonError` (not recoverable for
  this frame, fall to CPU)
- Segfault in C extension -> process death (unavoidable with in-process
  embedding; extremely rare with mature PyTorch)
- Rust panic inside `with_gil` -> stack unwind; `BorrowedDeviceSlice` has no
  Drop impl so the pointer leaks harmlessly; the `TensorGuard` may not be
  released (Python GC will eventually collect it when the `Py<>` reference drops)

### CUDA context health check after OOM

After an OOM recovery (either cudarc or PyTorch), the CUDA context may be in an
inconsistent state. Before retrying, verify context health:

```rust
fn verify_cuda_context(device: &CudaDevice) -> Result<(), GpuError> {
    // Attempt a tiny allocation + free to confirm the context is healthy
    let probe = device.alloc_zeros::<f32>(1)
        .map_err(|e| GpuError::CudaFail(format!("context health check failed: {e}")))?;
    drop(probe);
    Ok(())
}
```

If the health check fails after OOM, skip GPU single-frame retry (step 2 in the
fallback chain) and go directly to CPU fallback. The CUDA context is likely
wedged and won't recover without process restart.

### Borrowed device pointer safety

When Rust wraps a device pointer from PyTorch, the memory is owned by Python's
garbage collector. Rust must NOT call `cudaFree` on it.

The `BorrowedDeviceSlice` carries a lifetime `'py` tied to the `Python<'py>` GIL
token, enforced by the compiler — not by convention:

```rust
pub struct BorrowedDeviceSlice<'py, T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<&'py T>,
}

impl<'py, T> BorrowedDeviceSlice<'py, T> {
    /// # Safety
    /// ptr must be a valid device pointer with at least len elements,
    /// and must remain valid for the lifetime 'py (the GIL scope).
    pub unsafe fn from_raw(_py: Python<'py>, ptr: *mut T, len: usize) -> Self {
        Self { ptr, len, _phantom: PhantomData }
    }

    pub fn as_device_ptr(&self) -> *const T { self.ptr as *const T }
    pub fn len(&self) -> usize { self.len }
}

// No Drop impl — no cudaFree.
```

**Compile-time enforcement**: `BorrowedDeviceSlice<'py, T>` borrows from
`Python<'py>`. The borrow checker prevents it from being stored in a struct,
returned from the `with_gil` closure, or sent to another thread. This replaces
the convention-based enforcement in the original design.

**Interfacing with cudarc**: `BorrowedDeviceSlice` does NOT implement cudarc's
`DeviceSlice` trait. To pass borrowed data to cudarc kernel launches, use raw
`cuMemcpyDtoD` via cudarc's unsafe driver API, or copy to an owned `CudaSlice`:

```rust
// Device-to-device copy for when data must outlive the GIL (~0.1ms for 1080p depth)
let owned: CudaSlice<f32> = device.alloc_zeros(borrowed.len())?;
unsafe {
    cudarc::driver::sys::cuMemcpyDtoD_v2(
        owned.device_ptr().0 as _,
        borrowed.as_device_ptr() as u64,
        borrowed.len() * std::mem::size_of::<f32>(),
    );
}
```

For kernel launches that accept raw device pointers (via cudarc's
`LaunchConfig`), the `as_device_ptr()` method provides the pointer directly.

### Fallback chain

```
Grade frame attempt:
  0. Module load (PTX -> cudarc)
     +-- ModuleLoad error -> log error, skip GPU entirely, use CPU for all frames
         (this is a build/deployment issue, not a per-frame error)

  1. GPU batched (cudarc + PyO3 on-device inference)
     +-- OOM -> empty_cache, verify context health, adaptive batcher shrinks, retry
     +-- CUDA error -> fall to (2)
     +-- Python error -> fall to (2)

  2. GPU single-frame (batch_size=1, fresh CUDA state)
     +-- OOM -> verify context, fall to (3) if unhealthy
     +-- CUDA error -> fall to (3)
     +-- Success -> continue, don't grow batch

  3. CPU fallback (cpu.rs, pure Rust, no CUDA)
     +-- Always works (no external dependencies)
     +-- Log warning: "GPU grading failed, using CPU fallback for frame N"
```

## CUDA Context Sharing

PyTorch (runtime API) and cudarc (driver API) must share the same CUDA context.
Both use the primary context for device 0 by default:
- PyTorch: `cudaSetDevice(0)` retains the primary context
- cudarc: `CudaDevice::new(0)` calls `cuDevicePrimaryCtxRetain`

**Initialization order**: PyO3 initializes Python and loads PyTorch models first.
Then cudarc creates its `CudaDevice`. Both bind to the same primary context.

**Runtime validation**: at initialization (not just in a spike), assert that both
sides see the same CUDA context:

```rust
fn validate_context_sharing(py: Python<'_>, device: &CudaDevice) -> Result<(), GpuError> {
    // Get cudarc's context pointer
    let cudarc_ctx = device.cu_primary_ctx();

    // Get PyTorch's context via ctypes
    let torch_ctx: u64 = py.eval(
        "torch.cuda.current_device()",  // simplified; actual check uses driver API
        None, None
    )?.extract()?;

    // Both should reference device 0's primary context
    // If they differ, GPU tensor sharing will silently corrupt data
    assert_eq!(cudarc_ctx, torch_ctx, "CUDA context mismatch");
    Ok(())
}
```

**Validation requirement**: before committing to this architecture, a spike must
confirm that a `CudaSlice` allocated by cudarc is readable by PyTorch and vice
versa. The spike is ~20 lines of Rust + Python:
1. cudarc allocates a `CudaSlice<f32>`, writes known values
2. Pass device pointer to PyTorch via PyO3
3. PyTorch wraps as tensor, reads values
4. Assert values match

If the spike fails, the fallback is a single `cudaMemcpy` device-to-device at
the inference-to-grading boundary (still far better than the current host
round-trip).

## Scope and Non-Goals

**In scope:**
- PyO3 embedding replacing subprocess IPC in dorea-video (behind `python` feature)
- cudarc migration replacing extern C FFI in dorea-gpu (behind `cuda` feature)
- Adaptive batch sizing for video grading
- GPU tensor sharing between inference and grading
- OOM recovery and CPU fallback chain

**Not in scope:**
- Batching inference calls (model architecture change, separate concern)
- Persisting batch size across runs
- Multi-GPU support (single RTX 3060)
- Streaming/async kernel execution (CUDA streams — future optimization)
- Changes to ffmpeg decode/encode path
- Changes to calibration algorithms (dorea-lut, dorea-hsl, dorea-cal)

## Build and CI

### Feature flags

```
dorea-cli depends on:
  dorea-gpu (features = ["cuda"])   — optional, auto-detected
  dorea-video (features = [])       — NO python feature by default

dorea-video:
  default features: []
  "python" feature: enables pyo3 + numpy
  Without "python": uses inference_subprocess.rs (JSON-lines IPC, current approach)

dorea-gpu:
  default features: []
  "cuda" feature: enables cudarc, set by build.rs when nvcc is found
  Without "cuda": CPU-only fallback (cpu.rs)
```

### CI builds

- **Standard CI** (no GPU, no Python): builds all crates with default features.
  dorea-gpu compiles without CUDA (CPU fallback). dorea-video compiles without
  PyO3 (subprocess fallback). All unit tests run.
- **GPU CI** (CUDA toolkit + Python): builds with `--features cuda,python`.
  Runs integration tests including cudarc kernel launches and PyO3 inference.
- **Build-time requirements** for `python` feature: `python3-dev`, `numpy`
  installed, `PYO3_PYTHON` env var set. Documented in repo README.

### PTX compilation

`build.rs` compiles `.cu -> .ptx` with:
- `-arch=sm_86` (RTX 3060, Ampere) — matching the current `build.rs`
- `--ptx` flag (instead of `-c`)
- Same `--compiler-bindir /usr/bin/gcc-12` and `--allow-unsupported-compiler`

PTX is embedded in the binary via `include_str!`. No runtime file path resolution.

## Testing Strategy

- **Unit**: cudarc kernel launches with known input/output (port existing
  `grade_frame` tests)
- **Integration**: PyO3 inference -> cudarc grading end-to-end on a test frame
- **Spike** (blocking prerequisite): CUDA context sharing validation — cudarc
  alloc readable by PyTorch and vice versa
- **Benchmark**: before/after PCIe transfer measurement, batch size sweep
- **Fault injection**: deliberately trigger each fallback transition:
  - OOM at various batch sizes (artificially limit VRAM allocation)
  - Force `CudaFail` (invalid kernel parameters) -> verify single-frame retry
  - Force `ModuleLoad` error (corrupt PTX) -> verify CPU fallback for entire run
  - Force Python exception (mock model) -> verify CPU fallback for that frame
  - Verify CPU fallback produces correct output (compare against GPU output)
- **Context health**: OOM -> verify_cuda_context -> confirm recovery or
  graceful degradation to CPU

## Review Findings Incorporated

This spec incorporates findings from 5 persona reviews (CUDA systems engineer,
Rust safety expert, PyTorch ML engineer, build systems engineer, production
reliability engineer). Key changes from the original draft:

1. `BorrowedDeviceSlice` carries lifetime `'py` tied to GIL token (was `'static`)
2. PyO3 behind `python` feature flag (was `auto-initialize` default dependency)
3. PTX embedded via `include_bytes!` (was loose files in `OUT_DIR`)
4. VRAM probe uses `torch.cuda.mem_get_info()` (was raw `cuMemGetInfo`)
5. OOM detection uses structured enum matching (was string `contains`)
6. CUDA context health check after OOM recovery
7. `ModuleLoad` error added to fallback chain (step 0)
8. Threading model explicitly stated as single-threaded
9. `TensorGuard` class replaces bare dict for GC prevention
10. Dual-allocator fragmentation mitigations documented
11. Batching benefit corrected to dispatch amortization (was "coalescing")
12. Fault-injection tests added to testing strategy
13. CI feature-flag build matrix documented
14. Runtime context validation at initialization (not just spike)
