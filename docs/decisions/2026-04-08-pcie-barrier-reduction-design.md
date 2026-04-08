# PCIe Barrier Reduction Design (Updated for CombinedLut Architecture)

**Date:** 2026-04-08
**Status:** Approved
**Issue:** chunzhe10/dorea#31
**Supersedes:** `docs/plans/2026-04-03-pcie-sync-barrier-reduction.md` (stale — written for pre-CombinedLut architecture)

## Problem

`CudaGrader::grade_frame_cuda` and `AdaptiveGrader::grade_frame_blended` issue
multiple allocating CUDA memory operations per frame. In `CudaGrader`, 4 of the
6 per-frame operations upload constant data or allocate buffers that could be
reused. `AdaptiveGrader` has 6 eliminable operations out of 8.

Each `htod_sync_copy` calls `cudaMalloc` + `cuMemcpyHtoD` + `stream.synchronize()`.
The `cudaMalloc` alone costs ~3-8ms on RTX 3060, dwarfing the actual kernel compute.

## Current Per-Frame Operations

### CudaGrader (6 per frame)

| # | Operation | Size (1080p) | Needed every frame? |
|---|-----------|-------------|---------------------|
| 1 | `htod_sync_copy(pixels)` u8 | 6.2 MB | Yes — new pixels |
| 2 | `htod_sync_copy(depth)` f32 | 8.3 MB | Yes — new depth |
| 3 | `htod_sync_copy(textures)` u64 | ~64 B | **No — constant** |
| 4 | `htod_sync_copy(boundaries)` f32 | ~24 B | **No — constant** |
| 5 | `alloc_zeros(n*3)` output u8 | 6.2 MB | **No — reusable** |
| 6 | `dtoh_sync_copy(output)` → Vec | 6.2 MB | Yes — but alloc reusable |

### AdaptiveGrader (8 per frame)

Same as CudaGrader #1-2, plus 4 texture/boundary uploads (stable between
`swap_textures()` calls), output alloc, and dtoh.

## Solution

### 1. Cache constant device data

Upload texture handles (`CudaSlice<u64>`) and zone boundaries (`CudaSlice<f32>`)
once at construction time. Store as persistent fields on the grader structs.

For `AdaptiveGrader`, re-upload cached texture/boundary slices only when
`prepare_keyframe()` or `swap_textures()` is called, not on every frame.

### 2. Pre-allocate resolution-dependent buffers

New struct `FrameBuffers` keyed by `(width, height)`:

```rust
struct FrameBuffers {
    width: usize,
    height: usize,
    d_pixels_in: CudaSlice<u8>,   // n * 3
    d_depth: CudaSlice<f32>,       // n
    d_pixels_out: CudaSlice<u8>,   // n * 3
    h_result: Vec<u8>,             // n * 3 (host-side)
}
```

Stored as `RefCell<Option<FrameBuffers>>` on each grader. On resolution change,
drop and reallocate. In steady state (constant resolution), zero allocations.

### 3. Replace allocating copies with copy-into

- `htod_sync_copy(src)` → `htod_sync_copy_into(src, &mut existing_slice)`
- `alloc_zeros` → reuse existing slice (kernel overwrites all elements)
- `dtoh_sync_copy` → copy into pre-allocated `h_result` Vec

## After Optimization

| Grader | Before | After (steady state) |
|--------|--------|---------------------|
| CudaGrader | 4 htod + 1 alloc + 1 dtoh = 6 ops | 2 htod_into + 1 dtoh = 3 barriers |
| AdaptiveGrader | 6 htod + 1 alloc + 1 dtoh = 8 ops | 2 htod_into + 1 dtoh = 3 barriers |

## Files Changed

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cuda/mod.rs` | Add `FrameBuffers`, refactor `CudaGrader` and `AdaptiveGrader` |

No kernel changes. No public API changes. No new unsafe code.

## Why the Original Plan Is Stale

The plan at `docs/plans/2026-04-03-pcie-sync-barrier-reduction.md` was written
for an architecture with separate LUT/HSL/clarity kernels and `CalibrationBuffers`
holding `d_luts`, `d_h_offsets`, etc. The codebase now uses `CombinedLut` which
bakes calibration into CUDA 3D textures at construction. There are no separate
calibration uploads to skip, no HSL params to pack, and no u8→f32 host conversion
(the kernel takes u8 directly).

The optimization goal is the same (eliminate unnecessary per-frame PCIe barriers),
but the implementation is different.
