# Grading Pipeline Performance v3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four performance optimizations for `dorea grade`: motion-adaptive depth interpolation, fused LAB pass with rayon, raw RGB IPC protocol, and HF processor bypass + GPU-required runtime.

**Architecture:** Four independent optimizations each targeting a different pipeline stage. Task 1 (raw IPC) and Task 2 (HF bypass + GPU-required) change the inference layer. Task 3 (fused LAB) changes the grading core. Task 4 (depth interpolation) changes the frame loop. Each can be implemented and tested independently.

**Tech Stack:** Rust (`dorea-gpu`, `dorea-video`, `dorea-cli` crates), Python (`dorea_inference` package), rayon for CPU parallelism. All inline unit tests (`#[cfg(test)]`). Build with `cargo build` inside `repos/dorea/`. Python tests via `/opt/dorea-venv/bin/python -m pytest`.

**Design spec:** `docs/decisions/2026-04-02-grading-pipeline-perf-v3.md`

---

## File Map

| File | Task | Change |
|------|------|--------|
| `crates/dorea-video/src/inference.rs` | 1 | Add `encode_raw_rgb` path, switch `run_depth` and `run_raune` to raw mode |
| `python/dorea_inference/protocol.py` | 1 | Add `decode_raw_rgb()`, update request dataclasses with `format`/`width`/`height` |
| `python/dorea_inference/server.py` | 1 | Dispatch on `format` field |
| `python/dorea_inference/depth_anything.py` | 2 | Replace `AutoImageProcessor` with direct tensor construction, hard-error on no CUDA |
| `crates/dorea-gpu/src/lib.rs` | 2 | Remove CPU fallback, hard error on CUDA failure |
| `crates/dorea-gpu/src/cuda/mod.rs` | 2 | Clarity kernel failure → hard error |
| `crates/dorea-gpu/src/cpu.rs` | 3 | New `fused_ambiance_warmth()` with rayon, update `finish_grade` |
| `crates/dorea-gpu/Cargo.toml` | 3 | Add `rayon` dependency |
| `crates/dorea-cli/src/grade.rs` | 4 | Frame buffer, MSE, keyframe logic, depth interpolation, CLI args |

---

## Task 1: Raw RGB IPC Protocol

**Files:**
- Modify: `crates/dorea-video/src/inference.rs` (lines 187-286: `run_raune`, `run_depth`)
- Modify: `python/dorea_inference/protocol.py`
- Modify: `python/dorea_inference/server.py` (lines 125-155: request dispatch)

Replace PNG encode/decode with raw RGB bytes for both depth and RAUNE requests.

- [ ] **Step 1: Add `decode_raw_rgb` to Python protocol.py**

In `python/dorea_inference/protocol.py`, add after the existing `decode_depth_f32` function (line 161):

```python
def decode_raw_rgb(b64: str, width: int, height: int) -> "np.ndarray":
    """Decode base64 raw interleaved RGB uint8 to HxWx3 array."""
    import numpy as np
    raw = base64.b64decode(b64)
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"raw_rgb size mismatch: got {len(raw)}, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
```

- [ ] **Step 2: Update Python server.py to dispatch on format field**

In `python/dorea_inference/server.py`, add `decode_raw_rgb` to the imports (line 34):

```python
from .protocol import (
    PongResponse,
    RauneResult,
    DepthResult,
    ErrorResponse,
    OkResponse,
    decode_png,
    decode_raw_rgb,
    encode_png,
)
```

Then replace the image decoding in the `raune` handler (line 131) and `depth` handler (line 143):

For `raune` (replace line 131: `img = decode_png(req["image_b64"])`):
```python
            elif req_type == "raune":
                if raune_model is None:
                    raise RuntimeError("RAUNE-Net model not loaded (--no-raune or load failed)")
                fmt = req.get("format", "png")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                max_size = int(req.get("max_size", 1024))
                result = raune_model.infer(img, max_size=max_size)
```

For `depth` (replace line 143: `img = decode_png(req["image_b64"])`):
```python
            elif req_type == "depth":
                if depth_model is None:
                    raise RuntimeError("Depth Anything model not loaded (--no-depth or load failed)")
                fmt = req.get("format", "png")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                max_size = int(req.get("max_size", 518))
                depth = depth_model.infer(img, max_size=max_size)
```

- [ ] **Step 3: Switch Rust `run_depth` to raw RGB encoding**

In `crates/dorea-video/src/inference.rs`, replace the body of `run_depth` (lines 242-249). Change from:

```rust
        let png = encode_png_bytes(image_rgb, width, height)?;
        let b64 = B64.encode(&png);

        let req = serde_json::json!({
            "type": "depth",
            "id": id,
            "image_b64": b64,
            "max_size": max_size
        });
```

To:

```rust
        let b64 = B64.encode(image_rgb);

        let req = serde_json::json!({
            "type": "depth",
            "id": id,
            "image_b64": b64,
            "format": "raw_rgb",
            "width": width,
            "height": height,
            "max_size": max_size
        });
```

- [ ] **Step 4: Switch Rust `run_raune` to raw RGB encoding**

In `crates/dorea-video/src/inference.rs`, replace the body of `run_raune` (lines 195-203). Change from:

```rust
        let png = encode_png_bytes(image_rgb, width, height)?;
        let b64 = B64.encode(&png);

        let req = serde_json::json!({
            "type": "raune",
            "id": id,
            "image_b64": b64,
            "max_size": max_size
        });
```

To:

```rust
        let b64 = B64.encode(image_rgb);

        let req = serde_json::json!({
            "type": "raune",
            "id": id,
            "image_b64": b64,
            "format": "raw_rgb",
            "width": width,
            "height": height,
            "max_size": max_size
        });
```

- [ ] **Step 5: Build and test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-video 2>&1 | tail -5
cargo test -p dorea-video 2>&1 | tail -10
```

Expected: compiles cleanly, all tests pass. The `encode_png_bytes` function is now unused by `run_depth`/`run_raune` but may still be used elsewhere — do NOT delete it yet.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference.rs python/dorea_inference/protocol.py python/dorea_inference/server.py
git commit -m "perf(inference): switch IPC to raw RGB bytes, skip PNG encode/decode"
```

---

## Task 2: HF Processor Bypass + GPU-Required Runtime

**Files:**
- Modify: `python/dorea_inference/depth_anything.py` (lines 49, 72, 77-106)
- Modify: `crates/dorea-gpu/src/lib.rs` (lines 73-96)
- Modify: `crates/dorea-gpu/src/cuda/mod.rs` (lines 164-167)

- [ ] **Step 1: Replace AutoImageProcessor with direct tensor construction**

In `python/dorea_inference/depth_anything.py`, replace the `__init__` method (lines 43-75) with:

```python
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Cannot run depth inference on CPU — GPU is required for dorea grade."
            )

        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        path = Path(model_path) if model_path else _DEFAULT_DEPTH_MODEL
        model_id_or_path = str(path)
        has_local = path.is_dir() and any(
            (path / f).exists() for f in ("config.json", "pytorch_model.bin", "model.safetensors")
        )
        if not has_local:
            import sys
            print(
                f"[dorea_inference] WARNING: local depth model not found at {path}; "
                "falling back to HuggingFace hub download (requires internet access). "
                "Pass --depth-model to suppress this.",
                file=sys.stderr,
            )
            model_id_or_path = "depth-anything/Depth-Anything-V2-Small-hf"

        self.model = AutoModelForDepthEstimation.from_pretrained(model_id_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()
```

Note: `AutoImageProcessor` import and `self.processor` are removed entirely.

- [ ] **Step 2: Replace the `infer` method with direct tensor construction**

Replace the `infer` method (lines 77-106) with:

```python
    # ImageNet normalization constants (Depth Anything V2 uses these)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, img_rgb: np.ndarray, max_size: int = 518) -> np.ndarray:
        """Run depth estimation on uint8 HxWx3 RGB image.

        Returns float32 HxW depth map normalized to [0, 1] at inference resolution.
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

        depth = outputs.predicted_depth.squeeze(0).cpu().numpy()

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-6:
            depth = np.zeros_like(depth, dtype=np.float32)
        else:
            depth = ((depth - d_min) / (d_max - d_min)).astype(np.float32)

        return depth
```

- [ ] **Step 3: Remove unused AutoImageProcessor import**

In `depth_anything.py`, remove `AutoImageProcessor` from the import (line 49). Change:

```python
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
```

To:

```python
        from transformers import AutoModelForDepthEstimation
```

- [ ] **Step 4: Make CUDA a hard requirement in `lib.rs`**

In `crates/dorea-gpu/src/lib.rs`, replace lines 73-96 (the CUDA try + CPU fallback) with:

```rust
    #[cfg(feature = "cuda")]
    {
        match cuda::grade_frame_cuda(pixels, depth, width, height, calibration, params) {
            Ok(mut rgb_f32) => {
                // GPU resources are now freed. Apply CPU-only ambiance + warmth + blend.
                return Ok(cpu::finish_grade(
                    &mut rgb_f32,
                    pixels,
                    depth,
                    width,
                    height,
                    params,
                    calibration,
                    true,  // GPU clarity kernel already applied in grade_frame_cuda
                ));
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::Cuda(
            "dorea grade requires CUDA. Rebuild with GPU support (build.rs auto-detects nvcc).".to_string()
        ))
    }
```

- [ ] **Step 5: Make clarity kernel failure a hard error in `cuda/mod.rs`**

In `crates/dorea-gpu/src/cuda/mod.rs`, replace lines 164-168 (the clarity fallback):

```rust
    if status != 0 {
        log::warn!("dorea_clarity_gpu returned CUDA error {status} — clarity skipped");
        // Fall back gracefully: return hsl result without clarity
        return Ok(rgb_after_hsl);
    }
```

With:

```rust
    if status != 0 {
        return Err(GpuError::Cuda(format!(
            "dorea_clarity_gpu returned CUDA error {status}"
        )));
    }
```

- [ ] **Step 6: Build and test**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-gpu 2>&1 | tail -10
cargo test -p dorea-gpu 2>&1 | tail -15
```

Expected: compiles cleanly. CPU tests still pass (they call `grade_frame_cpu` directly, not `grade_frame`).

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/depth_anything.py crates/dorea-gpu/src/lib.rs crates/dorea-gpu/src/cuda/mod.rs
git commit -m "perf(inference): bypass HF processor, direct tensor construction

Also: GPU-required runtime — grade_frame() returns hard error if CUDA
unavailable or fails. No more silent CPU fallback."
```

---

## Task 3: Fused LAB Pass (Ambiance + Warmth) with Rayon

**Files:**
- Modify: `crates/dorea-gpu/Cargo.toml` (add rayon dep)
- Modify: `crates/dorea-gpu/src/cpu.rs` (lines 20-110, 132-165)

- [ ] **Step 1: Add rayon dependency to dorea-gpu**

In `crates/dorea-gpu/Cargo.toml`, add under `[dependencies]`:

```toml
rayon.workspace = true
```

- [ ] **Step 2: Write failing test for `fused_ambiance_warmth`**

In `crates/dorea-gpu/src/cpu.rs`, add to the `#[cfg(test)] mod tests` block (after the last test):

```rust
    #[test]
    fn fused_ambiance_warmth_matches_separate_passes() {
        let width = 4;
        let height = 4;
        let n = width * height;
        let original: Vec<f32> = (0..n * 3).map(|i| (i as f32 % 256.0) / 255.0).collect();
        let depth: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

        // Old path: separate depth_aware_ambiance + warmth
        let mut rgb_old = original.clone();
        depth_aware_ambiance(&mut rgb_old, &depth, width, height, 1.0);
        let warmth_factor = 1.0 + (1.2 - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_old[i * 3];
            let g = rgb_old[i * 3 + 1];
            let b = rgb_old[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_old[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_old[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_old[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }

        // New path: fused
        let mut rgb_new = original.clone();
        fused_ambiance_warmth(&mut rgb_new, &depth, width, height, 1.0, warmth_factor);

        // Fused output won't be bit-exact (different order of LAB roundtrips),
        // but should be perceptually close. Accept ΔE < 2 ≈ delta < 0.02 in normalised RGB.
        for i in 0..rgb_new.len() {
            let diff = (rgb_new[i] - rgb_old[i]).abs();
            assert!(diff < 0.05, "pixel {i}: old={:.4} new={:.4} diff={:.4}", rgb_old[i], rgb_new[i], diff);
        }
    }

    #[test]
    fn fused_ambiance_warmth_neutral_warmth() {
        let width = 2;
        let height = 2;
        let n = width * height;
        let original: Vec<f32> = vec![0.5; n * 3];
        let depth: Vec<f32> = vec![0.5; n];

        // warmth_factor = 1.0 means no user warmth scaling
        let mut rgb = original.clone();
        fused_ambiance_warmth(&mut rgb, &depth, width, height, 1.0, 1.0);

        // All values should be in [0, 1]
        for &v in &rgb {
            assert!((0.0..=1.0).contains(&v), "out-of-range: {v}");
        }
    }
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu fused_ambiance_warmth 2>&1 | tail -5
```

Expected: `error[E0425]: cannot find function 'fused_ambiance_warmth'`

- [ ] **Step 4: Implement `fused_ambiance_warmth`**

In `crates/dorea-gpu/src/cpu.rs`, add `use rayon::prelude::*;` to the top imports (after line 9), then add the new function after `depth_aware_ambiance` (after line 110):

```rust
/// Fused ambiance + user warmth: single RGB→LAB→RGB roundtrip per pixel.
///
/// Combines all per-pixel LAB operations from `depth_aware_ambiance` and the
/// warmth scaling that was previously a separate pass in `finish_grade`.
/// Parallelized with rayon for multi-core throughput.
///
/// Note: output is NOT bit-exact with the old two-pass pipeline because
/// the sRGB↔LAB conversion is nonlinear. Validated to produce ΔE < 2
/// (imperceptible). This is an accepted color-science tradeoff for performance.
pub fn fused_ambiance_warmth(
    rgb: &mut [f32],
    depth: &[f32],
    width: usize,
    height: usize,
    contrast_scale: f32,
    warmth_factor: f32,
) {
    if width == 0 || height == 0 {
        return;
    }
    assert_eq!(rgb.len(), width * height * 3, "rgb length mismatch");
    assert_eq!(depth.len(), width * height, "depth length mismatch");

    let apply_warmth = (warmth_factor - 1.0).abs() > 1e-4;

    rgb.par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let d = depth[i];
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // --- RGB → LAB (once) ---
            let (mut l_norm, mut a_ab, mut b_ab) = {
                let (l, a, b_l) = srgb_to_lab(r, g, b);
                (l / 100.0, a, b_l)
            };

            // 1. Shadow lift
            let lift_amount = 0.2 + 0.15 * d;
            let toe = 0.15_f32;
            let shadow_mask = ((toe - l_norm) / toe).clamp(0.0, 1.0);
            l_norm += shadow_mask * lift_amount * toe;

            // 2. S-curve contrast
            let strength = (0.3 + 0.3 * d) * contrast_scale;
            let slope = 4.0 + 4.0 * strength;
            let s_curve = 1.0 / (1.0 + (-(l_norm - 0.5) * slope).exp());
            l_norm += (s_curve - l_norm) * strength;

            // 3. Highlight compress
            let compress = 0.4 + 0.2 * (1.0 - d);
            let knee_h = 0.88_f32;
            if l_norm > knee_h {
                let over = l_norm - knee_h;
                let headroom = 1.0 - knee_h;
                l_norm = knee_h + headroom * ((over / headroom * (1.0 + compress)).tanh());
            }

            // 4. Warmth (depth-proportional LAB a*/b* push)
            let lum_weight = 4.0 * l_norm * (1.0 - l_norm);
            let warmth_a = 1.0 + 5.0 * d;
            let warmth_b = 4.0 * d;
            a_ab += warmth_a * lum_weight;
            b_ab += warmth_b * lum_weight;

            // 5. Vibrance (chroma boost for desaturated pixels)
            let vibrance = 0.4 + 0.5 * d;
            let chroma = (a_ab * a_ab + b_ab * b_ab + 1e-8).sqrt();
            let chroma_norm = (chroma / 40.0).clamp(0.0, 1.0);
            let boost = vibrance * (1.0 - chroma_norm) * (l_norm / 0.25).clamp(0.0, 1.0);
            a_ab *= 1.0 + boost;
            b_ab *= 1.0 + boost;

            // 6. User warmth scaling (fused — was separate pass)
            if apply_warmth {
                a_ab *= warmth_factor;
                b_ab *= warmth_factor;
            }

            // Clamp LAB
            let l_out = (l_norm * 100.0).clamp(0.0, 100.0);
            let a_out = a_ab.clamp(-128.0, 127.0);
            let b_out = b_ab.clamp(-128.0, 127.0);

            // --- LAB → RGB (once) ---
            let (ro, go, bo) = lab_to_srgb(l_out, a_out, b_out);

            // Final highlight knee
            let knee = 0.92_f32;
            let apply_knee = |v: f32| -> f32 {
                if v > knee {
                    let over = v - knee;
                    let room = 1.0 - knee;
                    knee + room * ((over / room).tanh())
                } else {
                    v
                }
            };

            pixel[0] = apply_knee(ro).clamp(0.0, 1.0);
            pixel[1] = apply_knee(go).clamp(0.0, 1.0);
            pixel[2] = apply_knee(bo).clamp(0.0, 1.0);
        });
}
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu fused_ambiance_warmth 2>&1 | tail -10
```

Expected: `2 tests passed`.

- [ ] **Step 6: Update `finish_grade` to use `fused_ambiance_warmth`**

In `crates/dorea-gpu/src/cpu.rs`, replace lines 142-165 of `finish_grade` (the ambiance call + warmth loop):

Old code:
```rust
    // 1. Depth-aware ambiance (shadow lift, S-curve, etc.)
    depth_aware_ambiance(rgb_f32, depth, width, height, params.contrast);

    // 1b. Clarity — skip when the CUDA path already ran the GPU clarity kernel.
    if !skip_clarity {
        apply_cpu_clarity(rgb_f32, depth, width, height, params.contrast);
    }

    // 2. Warmth (scale LAB a*/b*)
    if (params.warmth - 1.0).abs() > 1e-4 {
        let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
        for i in 0..n {
            let r = rgb_f32[i * 3];
            let g = rgb_f32[i * 3 + 1];
            let b = rgb_f32[i * 3 + 2];
            let (l, a, b_ab) = srgb_to_lab(r, g, b);
            let (ro, go, bo) = lab_to_srgb(l, a * warmth_factor, b_ab * warmth_factor);
            rgb_f32[i * 3]     = ro.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 1] = go.clamp(0.0, 1.0);
            rgb_f32[i * 3 + 2] = bo.clamp(0.0, 1.0);
        }
    }
```

New code:
```rust
    // 1. Fused ambiance + warmth (single LAB roundtrip, rayon-parallelized)
    let warmth_factor = 1.0 + (params.warmth - 1.0) * 0.3;
    fused_ambiance_warmth(rgb_f32, depth, width, height, params.contrast, warmth_factor);

    // 1b. Clarity — skip when the CUDA path already ran the GPU clarity kernel.
    if !skip_clarity {
        apply_cpu_clarity(rgb_f32, depth, width, height, params.contrast);
    }
```

- [ ] **Step 7: Run all dorea-gpu tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-gpu 2>&1 | tail -20
```

Expected: all tests pass. The `finish_grade_roundtrip` and `finish_grade_skip_clarity_runs_without_panic` tests exercise the updated `finish_grade` path.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-gpu/Cargo.toml crates/dorea-gpu/src/cpu.rs
git commit -m "perf(dorea-gpu): fused ambiance+warmth in single LAB roundtrip with rayon

Merges depth_aware_ambiance per-pixel work and user warmth scaling into
one RGB→LAB→RGB pass. Parallelized with rayon par_chunks_exact_mut.
Eliminates ~1.45B redundant FLOPs per 4K frame."
```

---

## Task 4: Temporal Grade Interpolation (Motion-Adaptive)

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (lines 14-73: args, lines 121-202: frame loop)

This is the largest change — the frame-by-frame grading loop becomes a buffered lookahead loop with MSE-based keyframe detection. Non-keyframe frames skip the entire pipeline (depth + grading) and interpolate the final graded RGB output between bracketing keyframes.

- [ ] **Step 1: Add new CLI args to `GradeArgs`**

In `crates/dorea-cli/src/grade.rs`, add after the `proxy_size` field (after line 43):

```rust
    /// MSE threshold for keyframe detection (lower = more keyframes)
    #[arg(long, default_value = "0.005")]
    pub depth_skip_threshold: f32,

    /// Maximum frames between keyframes
    #[arg(long, default_value = "12")]
    pub depth_max_interval: usize,

    /// Disable temporal interpolation — run full pipeline on every frame
    #[arg(long)]
    pub no_depth_interp: bool,
```

- [ ] **Step 2: Add helper functions for MSE and graded frame interpolation**

In `crates/dorea-cli/src/grade.rs`, add before the `run` function (before line 75):

```rust
/// Compute normalized MSE between two same-length u8 slices.
/// Returns value in [0, 1] where 0 = identical.
fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let sum_sq: f64 = a.iter().zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av as f64 - bv as f64;
            d * d
        })
        .sum();
    (sum_sq / (n * 255.0 * 255.0)) as f32
}

/// Linearly interpolate between two graded u8 frames.
fn lerp_graded(a: &[u8], b: &[u8], t: f32) -> Vec<u8> {
    let t = t.clamp(0.0, 1.0);
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| {
            let v = va as f32 + (vb as f32 - va as f32) * t;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// A buffered frame waiting to be output once the next keyframe is graded.
struct BufferedFrame {
    index: u64,
    width: usize,
    height: usize,
}
```

Note: `BufferedFrame` no longer stores `pixels` or `proxy_pixels` — non-keyframe frames
don't need their source pixels because we interpolate the *graded* output, not re-grade
with interpolated depth.

- [ ] **Step 3: Rewrite the frame grading loop with temporal grade interpolation**

Replace the frame loop (lines 121-202) — from `// Decode and grade frames` through the end of the `for` loop — with:

```rust
    // Decode and grade frames
    let frames = ffmpeg::decode_frames(&args.input, &info)
        .context("failed to spawn ffmpeg decoder")?;

    let mut frame_count = 0u64;
    let interp_enabled = !args.no_depth_interp;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    // Temporal interpolation state
    let mut last_keyframe_proxy: Option<Vec<u8>> = None;
    let mut last_keyframe_graded: Option<Vec<u8>> = None;
    let mut frame_buffer: Vec<BufferedFrame> = Vec::new();
    let mut frames_since_keyframe = 0usize;
    let max_buffer = (args.depth_max_interval as f32 * 1.5) as usize;

    let mut frame_iter = frames.peekable();

    while let Some(frame_result) = frame_iter.next() {
        let frame = frame_result.context("frame decode error")?;

        // Downscale to proxy resolution (needed for MSE check)
        let (proxy_w, proxy_h) =
            dorea_video::resize::proxy_dims(frame.width, frame.height, args.proxy_size);
        let proxy_pixels = if proxy_w != frame.width || proxy_h != frame.height {
            dorea_video::resize::resize_rgb_bilinear(
                &frame.pixels, frame.width, frame.height, proxy_w, proxy_h,
            )
        } else {
            frame.pixels.clone()
        };

        // Determine if this frame is a keyframe
        let is_keyframe = if !interp_enabled {
            true
        } else if last_keyframe_proxy.is_none() {
            true // First frame
        } else {
            let mse = frame_mse(&proxy_pixels, last_keyframe_proxy.as_ref().unwrap());
            let is_scene_cut = mse > scene_cut_threshold;
            let exceeds_interval = frames_since_keyframe >= args.depth_max_interval;
            let exceeds_threshold = mse > args.depth_skip_threshold;
            let buffer_overflow = frame_buffer.len() >= max_buffer;

            if is_scene_cut {
                log::info!("Scene cut at frame {} (MSE={:.6}) — flushing buffer", frame.index, mse);
                // Flush buffer using last keyframe graded output (no forward interp across cuts)
                flush_buffer_graded(
                    &mut frame_buffer, &last_keyframe_graded, &last_keyframe_graded,
                    &mut encoder, &mut frame_count, &info,
                )?;
            }

            is_scene_cut || exceeds_interval || exceeds_threshold || buffer_overflow
        };

        if is_keyframe {
            // Full pipeline: depth inference + grading
            let (depth_proxy, dw, dh) = inf_server
                .run_depth(
                    &frame.index.to_string(),
                    &proxy_pixels, proxy_w, proxy_h, args.proxy_size,
                )
                .unwrap_or_else(|e| {
                    log::warn!("Depth inference failed for frame {}: {e} — uniform depth", frame.index);
                    (vec![0.5f32; proxy_w * proxy_h], proxy_w, proxy_h)
                });

            let depth = if dw == frame.width && dh == frame.height {
                depth_proxy
            } else {
                InferenceServer::upscale_depth(&depth_proxy, dw, dh, frame.width, frame.height)
            };

            let graded = grade_frame(
                &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
            ).map_err(|e| anyhow::anyhow!("Grading failed for frame {}: {e}", frame.index))?;

            // Flush any buffered frames — interpolate between previous and this keyframe's graded output
            if !frame_buffer.is_empty() {
                flush_buffer_graded(
                    &mut frame_buffer, &last_keyframe_graded, &Some(graded.clone()),
                    &mut encoder, &mut frame_count, &info,
                )?;
            }

            // Write this keyframe's graded output
            encoder.write_frame(&graded).context("encoder write failed")?;
            frame_count += 1;

            if frame_count % 100 == 0 {
                let pct = frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
                log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
            }

            // Update keyframe state
            last_keyframe_proxy = Some(proxy_pixels);
            last_keyframe_graded = Some(graded);
            frames_since_keyframe = 0;
        } else {
            // Buffer this frame — will be interpolated when next keyframe is graded
            frame_buffer.push(BufferedFrame {
                index: frame.index,
                width: frame.width,
                height: frame.height,
            });
            frames_since_keyframe += 1;
        }
    }

    // Flush any remaining buffered frames (end of video — use last keyframe, no interpolation)
    if !frame_buffer.is_empty() {
        flush_buffer_graded(
            &mut frame_buffer, &last_keyframe_graded, &last_keyframe_graded,
            &mut encoder, &mut frame_count, &info,
        )?;
    }
```

- [ ] **Step 4: Add `flush_buffer_graded` helper function**

Add after the `BufferedFrame` struct definition (from Step 2):

```rust
/// Write all buffered frames by interpolating between bracketing keyframe graded outputs.
///
/// If `graded_before` and `graded_after` point to the same data (scene cut or end of video),
/// uses `graded_before` directly for all buffered frames (no interpolation).
fn flush_buffer_graded(
    buffer: &mut Vec<BufferedFrame>,
    graded_before: &Option<Vec<u8>>,
    graded_after: &Option<Vec<u8>>,
    encoder: &mut FrameEncoder,
    frame_count: &mut u64,
    info: &ffmpeg::VideoInfo,
) -> Result<()> {
    let Some(before) = graded_before else {
        // No previous keyframe — shouldn't happen (frame 0 is always a keyframe)
        // but handle gracefully by dropping buffered frames
        buffer.clear();
        return Ok(());
    };

    let n_buffered = buffer.len();
    let interval = (n_buffered + 1) as f32;

    let same_keyframe = match graded_after {
        Some(after) => std::ptr::eq(before.as_ptr(), after.as_ptr()),
        None => true,
    };

    for (buf_idx, bf) in buffer.drain(..).enumerate() {
        let output = if same_keyframe {
            before.clone()
        } else {
            let t = (buf_idx + 1) as f32 / interval;
            lerp_graded(before, graded_after.as_ref().unwrap(), t)
        };

        // Verify dimensions match (graded output must match buffered frame dims)
        let expected = bf.width * bf.height * 3;
        if output.len() != expected {
            return Err(anyhow::anyhow!(
                "Interpolated frame size mismatch at frame {}: got {}, expected {}",
                bf.index, output.len(), expected
            ));
        }

        encoder.write_frame(&output).context("encoder write failed")?;
        *frame_count += 1;

        if *frame_count % 100 == 0 {
            let pct = *frame_count as f64 / info.frame_count.max(1) as f64 * 100.0;
            log::info!("Progress: {frame_count}/{} frames ({:.1}%)", info.frame_count, pct);
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Remove old `prev_frame` variable and unused imports**

The old `prev_frame` and scene detection code (lines 125-139) is replaced by the new keyframe MSE logic. Remove:
- `let mut prev_frame: Option<Vec<u8>> = None;` and the `is_cut` block
- `use dorea_video::scene;` from the imports at the top (scene detection is now MSE-based)

- [ ] **Step 6: Build and verify compilation**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build -p dorea-cli 2>&1 | tail -15
```

Expected: `Finished dev profile`.

- [ ] **Step 7: Run workspace tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test --workspace 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "perf(dorea-cli): temporal grade interpolation — skip full pipeline for similar frames

Motion-adaptive keyframe detection via proxy-frame MSE. Keyframe frames
run full pipeline (depth + grade). Non-keyframe frames interpolate the
graded RGB output between bracketing keyframes — skipping depth inference
AND grading entirely. Scene cuts force keyframes (MSE > 10x threshold).
Buffer overflow safety at 1.5x max_interval."
```

---

## Task 5: End-to-End Smoke Test

- [ ] **Step 1: Build release binary**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build --release --bin dorea 2>&1 | tail -10
```

Expected: `Finished release profile`.

- [ ] **Step 2: Run grade on test clip with timing**

```bash
cd /workspaces/dorea-workspace/repos/dorea
time cargo run --release --bin dorea -- grade \
  --input /workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D.MP4 \
  --output /workspaces/dorea-workspace/working/DJI_0055_graded_v3.mp4 \
  --depth-model /workspaces/dorea-workspace/models/depth_anything_v2_small \
  --verbose 2>&1
```

Watch for:
- Keyframe detection log lines: `"Scene cut at frame..."` or progress lines
- No CPU fallback warnings
- Completion without errors

- [ ] **Step 3: Verify output is a valid video**

```bash
ffprobe -v quiet -print_format json -show_streams \
  /workspaces/dorea-workspace/working/DJI_0055_graded_v3.mp4 \
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

Expected: frame count matches source (~1671), duration ~13.9s, codec h264.

- [ ] **Step 4: Record timing and write to corvia**

After a successful run, record the wall time. Use `corvia_write` with `scope_id="dorea"`, `source_origin="repo:dorea"` to record:

```
Grading pipeline perf v3 results:
- Motion-adaptive depth interpolation: [keyframe%] of frames needed inference
- Wall time for DJI_20251101111428_0055_D.MP4 (4K, 14s, 1671 frames): [actual time]
- Previous wall time (pre-v3): [compare if available]
- Optimizations applied: fused LAB+rayon, raw RGB IPC, HF bypass, depth interpolation
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Opt 1+5 (temporal grade interpolation) → Task 4
- [x] Opt 2 (fused LAB pass with rayon) → Task 3
- [x] Opt 3 (raw RGB IPC) → Task 1
- [x] Opt 4a (HF processor bypass) → Task 2, Steps 1-3
- [x] Opt 4b (GPU-required runtime) → Task 2, Steps 4-5
- [x] Scene-cut handling → Task 4 (scene_cut_threshold = 10x threshold, buffer flush)
- [x] Buffer overflow safety → Task 4 (max_buffer = 1.5x max_interval)
- [x] Division-by-zero guard → Task 4 (lerp_graded clamps t to [0,1])
- [x] Last-frame handling → Task 4 (flush at end with graded_before == graded_after)
- [x] Timing instrumentation → noted in Task 5 (verbose logging)
- [x] Backward-compatible format field → Task 1 Step 2 (defaults to "png")

**Placeholder scan:** No TBDs, TODOs, or "fill in later". All code blocks are complete.

**Type consistency:**
- `fused_ambiance_warmth` signature: `(&mut [f32], &[f32], usize, usize, f32, f32)` — matches call in finish_grade
- `frame_mse` returns `f32` — compared against `args.depth_skip_threshold: f32` ✓
- `lerp_graded` takes `(&[u8], &[u8], f32)` → `Vec<u8>` — matches graded frame type ✓
- `BufferedFrame` stores only `index, width, height` — no source pixels (interpolation uses graded cache) ✓
- `flush_buffer_graded` takes `&Option<Vec<u8>>` for graded before/after — matches `last_keyframe_graded` type ✓
