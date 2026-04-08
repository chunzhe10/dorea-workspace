# PCIe Barrier Reduction Implementation Plan

**Issue:** chunzhe10/dorea#31
**Design:** `docs/decisions/2026-04-08-pcie-barrier-reduction-design.md`
**File:** `crates/dorea-gpu/src/cuda/mod.rs` (all changes)

---

## Task 1: CudaGrader — cache constant data + pre-allocate FrameBuffers

**Goal:** Reduce CudaGrader from 6 per-frame CUDA operations to 3 barriers.

### Step 1.1: Add FrameBuffers struct

```rust
struct FrameBuffers {
    width: usize,
    height: usize,
    d_pixels_in: CudaSlice<u8>,   // n * 3
    d_depth: CudaSlice<f32>,       // n
    d_pixels_out: CudaSlice<u8>,   // n * 3
}
```

### Step 1.2: Add fields to CudaGrader

- `d_textures: CudaSlice<u64>` — constant, uploaded once in `new()`
- `d_boundaries: CudaSlice<f32>` — constant, uploaded once in `new()`
- `frame_bufs: RefCell<Option<FrameBuffers>>` — resolution-keyed

### Step 1.3: Update CudaGrader::new()

After `CombinedLut::build()`:
```rust
let d_textures = device.htod_sync_copy(&combined_lut.textures).map_err(map_cudarc_error)?;
let d_boundaries = device.htod_sync_copy(&combined_lut.zone_boundaries).map_err(map_cudarc_error)?;
```

### Step 1.4: Refactor grade_frame_cuda()

1. Ensure FrameBuffers exist at correct resolution (reallocate if mismatch)
2. Use `htod_sync_copy_into` for pixels and depth
3. Use `self.d_textures` and `self.d_boundaries` directly (no per-frame upload)
4. Reuse `d_pixels_out` (kernel overwrites all elements)
5. `dtoh_sync_copy` from pre-allocated output (host Vec alloc is negligible)

---

## Task 2: AdaptiveGrader — cache texture data + pre-allocate FrameBuffers

**Goal:** Reduce AdaptiveGrader from 8 per-frame CUDA operations to 3 barriers.

### Step 2.1: Add cached device slices to AdaptiveGrader

- `d_textures_a: CudaSlice<u64>` — texture handles for active set (constant after build)
- `d_textures_b: CudaSlice<u64>` — texture handles for inactive set (constant after build)
- `d_bounds_a: RefCell<CudaSlice<f32>>` — boundaries for active set (updated on swap/rebuild)
- `d_bounds_b: RefCell<CudaSlice<f32>>` — boundaries for inactive set (updated on swap/rebuild)
- `frame_bufs: RefCell<Option<FrameBuffers>>` — resolution-keyed

Note: texture handles (CUtexObject) are allocated in TextureSet::allocate and never
change — only the 3D array content is updated via cuMemcpy3D. So d_textures_a/b are
truly constant. Boundaries change per keyframe.

### Step 2.2: Update AdaptiveGrader::new()

Upload texture handles and initial boundaries for both sets.

### Step 2.3: Update prepare_keyframe() and swap_textures()

Re-upload boundary device slices when they change (once per keyframe, not per frame).

### Step 2.4: Refactor grade_frame_blended()

Same pattern as Task 1.4 — use cached data, pre-allocated FrameBuffers.

---

## Task 3: Verify

- Run existing test suite: `cargo test -p dorea-gpu --test-threads=1`
- All 9 existing tests must pass unchanged
- Commit
