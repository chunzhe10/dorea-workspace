# Inference + Grading Performance Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three compounding bottlenecks that cause `dorea grade` to process a 14-second 4K clip in ~52 hours instead of minutes.

**Architecture:** Two independent fixes targeting separate parts of the pipeline: (1) replace the O(N×radius) naive box blur in `cpu.rs` with a sliding-window O(N) implementation + transpose-for-columns to avoid cache-miss thrash; (2) downscale frames to proxy resolution in Rust before sending to the Python depth inference server, eliminating 24MB/frame uncompressed pipe I/O.

**Tech Stack:** Rust, `dorea-gpu` crate (cpu.rs), `dorea-video` crate (new resize.rs), `dorea-cli` crate (grade.rs). All inline unit tests (`#[cfg(test)]` blocks), no separate test files. Build with `cargo build` inside `repos/dorea/`.

---

## Root Cause Summary

### Bug 1 — O(N×radius) box blur (primary bottleneck, ~9 s/frame)

`crates/dorea-gpu/src/cpu.rs:215-243` — `box_blur_rows` and `box_blur_cols` recompute the window sum from scratch for every pixel:

```rust
for k in lo..=hi {       // ← 181 iterations per pixel at radius=90
    s += src[base + k];
}
```

For 4K (3840×2160 = 8.3M pixels), radius=90, 3 passes × 2 directions:
`8.3M × 181 × 6 = 9 billion inner-loop iterations per frame`.

`box_blur_cols` also accesses memory with a 15 KB stride (column stride = `width × 4 bytes = 15 KB`), defeating all hardware prefetch and causing ~2.5 s of cache misses on top.

### Bug 2 — Full 4K frame serialized for Python depth inference (~5–10 s/frame)

`crates/dorea-cli/src/grade.rs:142` sends `frame.pixels` (3840×2160×3 = 24 MB raw) to the Python inference server, which immediately resizes it to 518 px. Per-frame cost on Rust side:

- `encode_png_bytes` (store mode): copy 24 MB + Adler-32 over 24 M bytes
- `B64.encode`: 24 MB → 32 MB
- 32 MB JSON write to pipe

Python receives 32 MB, decodes, resizes 4K→518 px, runs DepthAnything (~100 ms GPU), returns ~200 KB. The 32 MB encode/transmit dominates.

---

## File Map

| File | Change |
|------|--------|
| `crates/dorea-gpu/src/cpu.rs` | Replace `box_blur_rows`, `box_blur_cols`, `three_pass_box_blur` |
| `crates/dorea-video/src/resize.rs` | **New file** — `resize_rgb_bilinear`, `proxy_dims` |
| `crates/dorea-video/src/lib.rs` | Add `pub mod resize;` |
| `crates/dorea-cli/src/grade.rs` | Downscale to proxy before `run_depth` call |

---

## Task 1: Fix O(N×radius) box blur with sliding window + transpose

**Files:** Modify `crates/dorea-gpu/src/cpu.rs`

The replacement is two functions:
- `box_blur_rows_sliding` — O(W) per row using a running sum
- `box_blur_cols_via_transpose` — transposes the array, calls `box_blur_rows_sliding`, transposes back (fixes cache-miss problem for column access)

- [ ] **Step 1: Write failing tests for `box_blur_rows_sliding`**

Add inside the existing `#[cfg(test)] mod tests` block at the bottom of `crates/dorea-gpu/src/cpu.rs`:

```rust
#[test]
fn box_blur_rows_sliding_matches_naive_small() {
    // 5-wide, 2-tall, radius 1
    let src: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let mut dst_sliding = vec![0.0f32; 10];
    let mut dst_naive   = vec![0.0f32; 10];
    box_blur_rows_sliding(&src, &mut dst_sliding, 5, 2, 1);
    box_blur_rows(&src, &mut dst_naive, 5, 2, 1);
    for (a, b) in dst_sliding.iter().zip(dst_naive.iter()) {
        assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
    }
}

#[test]
fn box_blur_rows_sliding_radius_zero() {
    let src: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut dst = vec![0.0f32; 3];
    box_blur_rows_sliding(&src, &mut dst, 3, 1, 0);
    assert_eq!(dst, src, "radius=0 should be identity");
}

#[test]
fn box_blur_rows_sliding_radius_exceeds_width() {
    // radius > width: all pixels see the full row — each output = mean of whole row
    let src: Vec<f32> = vec![1.0, 2.0, 3.0];
    let expected_mean = 2.0f32;
    let mut dst = vec![0.0f32; 3];
    box_blur_rows_sliding(&src, &mut dst, 3, 1, 100);
    for v in &dst {
        assert!((v - expected_mean).abs() < 1e-5, "expected {expected_mean}, got {v}");
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail (function not defined yet)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu box_blur_rows_sliding 2>&1 | tail -5
```
Expected: compile error — `box_blur_rows_sliding` not found.

- [ ] **Step 3: Add `box_blur_rows_sliding` to `cpu.rs`**

Add this function immediately after the existing `box_blur_rows` function (after line 228):

```rust
/// Sliding-window box blur over rows. O(W) per row — replaces the naive O(W×radius) version.
fn box_blur_rows_sliding(src: &[f32], dst: &mut [f32], width: usize, height: usize, radius: usize) {
    let r = radius as isize;
    for row in 0..height {
        let base = row * width;
        let mut sum = 0.0f32;
        let mut count = 0usize;

        // Seed window for col=0: indices [0..min(r, W-1)]
        let init_hi = r.min(width as isize - 1) as usize;
        for k in 0..=init_hi {
            sum += src[base + k];
            count += 1;
        }

        for col in 0..width {
            dst[base + col] = sum / count as f32;

            // Expand right edge for next column
            let add = col as isize + r + 1;
            if add < width as isize {
                sum += src[base + add as usize];
                count += 1;
            }
            // Shrink left edge for next column
            let rem = col as isize - r;
            if rem >= 0 {
                sum -= src[base + rem as usize];
                count -= 1;
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to confirm `box_blur_rows_sliding` passes**

```bash
cargo test -p dorea-gpu box_blur_rows_sliding 2>&1 | tail -10
```
Expected: `3 tests passed`.

- [ ] **Step 5: Write failing tests for `box_blur_cols_via_transpose`**

Add to the `tests` block:

```rust
#[test]
fn box_blur_cols_via_transpose_matches_naive_small() {
    // 3-wide, 5-tall, radius 1
    let src: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0,
    ];
    let mut dst_transpose = vec![0.0f32; 15];
    let mut dst_naive     = vec![0.0f32; 15];
    box_blur_cols_via_transpose(&src, &mut dst_transpose, 3, 5, 1);
    box_blur_cols(&src, &mut dst_naive, 3, 5, 1);
    for (i, (a, b)) in dst_transpose.iter().zip(dst_naive.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: {a} vs {b}");
    }
}
```

- [ ] **Step 6: Run to confirm failure**

```bash
cargo test -p dorea-gpu box_blur_cols_via_transpose 2>&1 | tail -5
```
Expected: compile error.

- [ ] **Step 7: Add `box_blur_cols_via_transpose` to `cpu.rs`**

Add immediately after `box_blur_cols` (after line 243):

```rust
/// Column box blur using transpose → row blur → transpose back.
/// This avoids the strided-memory cache thrash of direct column access.
fn box_blur_cols_via_transpose(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    // Transpose src (WxH layout) → transposed (HxW layout: height rows of width columns)
    // In the transposed layout, "rows" are original columns → sequential access.
    let mut transposed = vec![0.0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            transposed[col * height + row] = src[row * width + col];
        }
    }

    // Row-blur on transposed array: dimensions are (width rows) × (height cols)
    let mut blurred_t = vec![0.0f32; width * height];
    box_blur_rows_sliding(&transposed, &mut blurred_t, height, width, radius);

    // Transpose back
    for row in 0..height {
        for col in 0..width {
            dst[row * width + col] = blurred_t[col * height + row];
        }
    }
}
```

- [ ] **Step 8: Run tests to confirm both col tests pass**

```bash
cargo test -p dorea-gpu box_blur 2>&1 | tail -10
```
Expected: all 4 box_blur tests pass.

- [ ] **Step 9: Update `three_pass_box_blur` to use the new functions**

In `cpu.rs`, replace the body of `three_pass_box_blur` (lines 207-212):

```rust
fn three_pass_box_blur(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut buf_a = input.to_vec();
    let mut buf_b = vec![0.0f32; input.len()];
    for _ in 0..3 {
        box_blur_rows_sliding(&buf_a, &mut buf_b, width, height, radius);
        box_blur_cols_via_transpose(&buf_b, &mut buf_a, width, height, radius);
    }
    buf_a
}
```

- [ ] **Step 10: Run all dorea-gpu tests**

```bash
cargo test -p dorea-gpu 2>&1 | tail -15
```
Expected: all tests pass (including existing `depth_aware_ambiance_deterministic`, `finish_grade_roundtrip`, etc.).

- [ ] **Step 11: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/src/cpu.rs
git commit -m "perf(dorea-gpu): sliding-window box blur + transpose-cols — O(N) vs O(N×radius)"
```

---

## Task 2: Add proxy resize utilities to `dorea-video`

**Files:** Create `crates/dorea-video/src/resize.rs`, modify `crates/dorea-video/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/dorea-video/src/resize.rs` with just the tests first:

```rust
// RGB frame resize utilities used by the grading pipeline.

/// Compute `(proxy_w, proxy_h)` scaled so the long edge ≤ `max_size`.
/// Returns the original dimensions unchanged if they are already within bounds.
pub fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    todo!()
}

/// Bilinearly downsample an RGB24 frame.
///
/// `src` is interleaved RGB u8, length = `src_w * src_h * 3`.
/// Returns interleaved RGB u8, length = `dst_w * dst_h * 3`.
pub fn resize_rgb_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_dims_no_change_when_within_bounds() {
        assert_eq!(proxy_dims(518, 292, 518), (518, 292));
        assert_eq!(proxy_dims(100, 50, 1000), (100, 50));
    }

    #[test]
    fn proxy_dims_scales_down_landscape() {
        // 3840×2160 with max_size=518: long edge (3840) → 518
        let (pw, ph) = proxy_dims(3840, 2160, 518);
        assert!(pw.max(ph) <= 518, "long edge {pw}×{ph} > 518");
        // Aspect ratio preserved to within 1 pixel rounding
        let ratio_orig = 3840.0f64 / 2160.0;
        let ratio_proxy = pw as f64 / ph as f64;
        assert!((ratio_proxy - ratio_orig).abs() < 0.1, "aspect ratio {ratio_proxy} vs {ratio_orig}");
    }

    #[test]
    fn proxy_dims_scales_down_portrait() {
        let (pw, ph) = proxy_dims(1080, 1920, 518);
        assert!(pw.max(ph) <= 518);
    }

    #[test]
    fn resize_rgb_bilinear_dimensions() {
        let src = vec![128u8; 4 * 4 * 3];
        let dst = resize_rgb_bilinear(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
    }

    #[test]
    fn resize_rgb_bilinear_solid_color_preserved() {
        // A solid-color image should stay the same color after resizing
        let src = vec![200u8, 100u8, 50u8].repeat(16); // 4×4 all same color
        let dst = resize_rgb_bilinear(&src, 4, 4, 2, 2);
        for chunk in dst.chunks_exact(3) {
            assert_eq!(chunk[0], 200, "R channel changed");
            assert_eq!(chunk[1], 100, "G channel changed");
            assert_eq!(chunk[2], 50,  "B channel changed");
        }
    }

    #[test]
    fn resize_rgb_bilinear_identity_same_size() {
        let src: Vec<u8> = (0u8..48).collect(); // 4×4×3
        let dst = resize_rgb_bilinear(&src, 4, 4, 4, 4);
        assert_eq!(dst, src, "same-size resize should be identity");
    }
}
```

- [ ] **Step 2: Add `pub mod resize;` to `lib.rs`**

In `crates/dorea-video/src/lib.rs`, add after the existing `pub mod scene;` line:

```rust
pub mod resize;
```

- [ ] **Step 3: Run to confirm tests fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video resize 2>&1 | tail -10
```
Expected: compile error or panics from `todo!()`.

- [ ] **Step 4: Implement `proxy_dims` and `resize_rgb_bilinear`**

Replace the `todo!()` bodies in `resize.rs`:

```rust
pub fn proxy_dims(src_w: usize, src_h: usize, max_size: usize) -> (usize, usize) {
    let long_edge = src_w.max(src_h);
    if long_edge <= max_size {
        return (src_w, src_h);
    }
    let scale = max_size as f64 / long_edge as f64;
    let pw = ((src_w as f64 * scale).round() as usize).max(1);
    let ph = ((src_h as f64 * scale).round() as usize).max(1);
    (pw, ph)
}

pub fn resize_rgb_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    assert_eq!(src.len(), src_w * src_h * 3, "src length mismatch");
    let mut out = vec![0u8; dst_w * dst_h * 3];
    let sw = (src_w as f32 - 1.0).max(0.0);
    let sh = (src_h as f32 - 1.0).max(0.0);
    let dw = (dst_w as f32 - 1.0).max(1.0);
    let dh = (dst_h as f32 - 1.0).max(1.0);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * sw / dw;
            let sy = dy as f32 * sh / dh;
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let out_base = (dy * dst_w + dx) * 3;
            for c in 0..3 {
                let v00 = src[(y0 * src_w + x0) * 3 + c] as f32;
                let v10 = src[(y0 * src_w + x1) * 3 + c] as f32;
                let v01 = src[(y1 * src_w + x0) * 3 + c] as f32;
                let v11 = src[(y1 * src_w + x1) * 3 + c] as f32;
                let v = v00 * (1.0 - fx) * (1.0 - fy)
                      + v10 * fx * (1.0 - fy)
                      + v01 * (1.0 - fx) * fy
                      + v11 * fx * fy;
                out[out_base + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}
```

- [ ] **Step 5: Run tests to confirm all pass**

```bash
cargo test -p dorea-video resize 2>&1 | tail -15
```
Expected: `5 tests passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/dorea-video/src/resize.rs crates/dorea-video/src/lib.rs
git commit -m "feat(dorea-video): add resize_rgb_bilinear + proxy_dims utilities"
```

---

## Task 3: Wire proxy downscale into `grade.rs`

**Files:** Modify `crates/dorea-cli/src/grade.rs`

The grading loop currently sends `frame.pixels` (full 4K, 24 MB) to Python. Replace with a proxy-sized copy.

- [ ] **Step 1: Add the proxy downscale in `grade.rs`**

In `grade.rs`, locate the depth inference call at line 141–154. Replace the entire block from `// Run depth inference at proxy resolution` to the closing `.unwrap_or_else(...)` with:

```rust
// Downscale to proxy resolution before sending to inference.
// Avoids serializing 24 MB/frame over the pipe when Python resizes to 518 px anyway.
let (proxy_w, proxy_h) =
    dorea_video::resize::proxy_dims(frame.width, frame.height, args.proxy_size);
let proxy_pixels = if proxy_w != frame.width || proxy_h != frame.height {
    dorea_video::resize::resize_rgb_bilinear(
        &frame.pixels,
        frame.width,
        frame.height,
        proxy_w,
        proxy_h,
    )
} else {
    frame.pixels.clone()
};

// Run depth inference at proxy resolution
let (depth_proxy, dw, dh) = inf_server
    .run_depth(
        &frame.index.to_string(),
        &proxy_pixels,
        proxy_w,
        proxy_h,
        args.proxy_size,
    )
    .unwrap_or_else(|e| {
        log::warn!("Depth inference failed for frame {}: {e} — using uniform depth", frame.index);
        let n = frame.width * frame.height;
        (vec![0.5f32; n], frame.width, frame.height)
    });
```

- [ ] **Step 2: Add `dorea-video` import at the top of the function (it's already a dep, just use the path)**

Verify `dorea-video` is already in `crates/dorea-cli/Cargo.toml` as a dependency — it should be since `grade.rs` already uses `dorea_video::ffmpeg` and `dorea_video::inference`. No Cargo.toml change needed.

- [ ] **Step 3: Build to confirm it compiles**

```bash
cargo build -p dorea-cli 2>&1 | tail -15
```
Expected: `Finished dev profile`.

- [ ] **Step 4: Run all workspace tests**

```bash
cargo test --workspace 2>&1 | tail -20
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dorea-cli/src/grade.rs
git commit -m "perf(dorea-cli): downscale frames to proxy res before depth inference — 160x pipe I/O reduction"
```

---

## Task 4: End-to-end smoke test on the real video

- [ ] **Step 1: Run grade on the test clip with timing**

```bash
cd /workspaces/dorea-workspace/repos/dorea
time cargo run --release --bin dorea -- grade \
  --input /workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D.MP4 \
  --output /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_graded_v2.mp4 \
  --raune-weights /workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth \
  --raune-models-dir /workspaces/dorea-workspace/working/sea_thru_poc \
  --depth-model /workspaces/dorea-workspace/models/depth_anything_v2_small \
  --verbose 2>&1
```

Note: `--release` matters here — debug build is ~5–10× slower for the Rust grading code.

Expected: completes in under 30 minutes. Progress lines should appear every 100 frames:
```
[INFO  dorea_cli::grade] Progress: 100/1671 frames (6.0%)
[INFO  dorea_cli::grade] Progress: 200/1671 frames (12.0%)
...
[INFO  dorea_cli::grade] Done. Graded 1671 frames → .../DJI_20251101111428_0055_D_graded_v2.mp4
```

- [ ] **Step 2: Verify output is a valid video**

```bash
ffprobe -v quiet -print_format json -show_streams \
  /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_graded_v2.mp4 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d.get('streams', []):
    if s.get('codec_type') == 'video':
        print(f'frames: {s.get(\"nb_frames\", \"?\")}')
        print(f'duration: {s.get(\"duration\", \"?\")}s')
        print(f'codec: {s.get(\"codec_name\", \"?\")}')
"
```

Expected:
```
frames: 1671
duration: ~13.9s
codec: h264
```

- [ ] **Step 3: Check output file size is reasonable**

```bash
ls -lh /workspaces/dorea-workspace/working/DJI_20251101111428_0055_D_graded_v2.mp4
```

Expected: 20–300 MB (comparable to the 218 MB source). If < 1 MB, the encoder didn't receive frames.

- [ ] **Step 4: Record the total wall time and commit the result to corvia**

After a successful run, note the actual wall time. Then record the fix:

```bash
# Use the corvia MCP tool (from Claude Code) to write a decision record
# corvia_write: scope_id="dorea", source_origin="repo:dorea"
# content: "Fixed two performance bugs in dorea grade:
#   1. Box blur sliding window: O(N×radius) → O(N), ~180x speedup on clarity pass
#   2. Proxy downscale before depth inference: 24 MB/frame → ~200 KB/frame
#   Measured wall time for DJI_20251101111428_0055_D.MP4 (4K, 14s, 1671 frames): <actual time>"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Bug 1 (O(N×radius) blur) → Task 1 (sliding window + transpose)
- [x] Bug 2 (full-frame pipe I/O) → Task 2 (proxy_dims + resize) + Task 3 (wired in grade.rs)
- [x] Cache miss for column blur → Task 1 (transpose approach)
- [x] End-to-end verification → Task 4

**Placeholder scan:** No TBDs. All code blocks are complete and compilable.

**Type consistency:**
- `proxy_dims` returns `(usize, usize)` — used correctly in Task 3
- `resize_rgb_bilinear` takes `&[u8]`, returns `Vec<u8>` — matches `frame.pixels` type and `run_depth` signature
- `box_blur_rows_sliding` and `box_blur_cols_via_transpose` have identical signatures to the functions they replace — `three_pass_box_blur` call sites are unchanged
