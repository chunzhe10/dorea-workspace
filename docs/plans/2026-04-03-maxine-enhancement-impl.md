# Maxine Post-Grading Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional NVIDIA Maxine VFX SDK post-grading enhancement (in-resolution oversample) to the dorea pipeline.

**Architecture:** Extend the existing `dorea_inference` Python subprocess with an `enhance` message type. The Rust side calls `enhance()` sequentially after `grade_frame()` in the single-threaded frame loop. Maxine models (VideoSuperRes + ArtifactReduction) share the same Python process and CUDA context as depth inference. A `DOREA_MAXINE_MOCK=1` env var enables CI testing without the proprietary SDK.

**Tech Stack:** Rust (clap CLI, serde_json IPC, base64), Python (nvvfx, torch, cv2, numpy), NVIDIA Maxine VFX SDK

**Spec:** `docs/decisions/2026-04-02-maxine-post-grading-enhancement.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `python/dorea_inference/protocol.py` | Modify | Add `EnhanceResult` dataclass + `encode_raw_rgb()` helper |
| `python/dorea_inference/maxine_enhancer.py` | Create | `MaxineEnhancer` class — nvvfx wrapper with mock mode |
| `python/dorea_inference/server.py` | Modify | Add `--maxine` flag, `enhance` request handler |
| `crates/dorea-video/src/inference.rs` | Modify | Add `maxine` fields to `InferenceConfig`, `enhance()` IPC method |
| `crates/dorea-cli/src/grade.rs` | Modify | Add `--maxine` CLI args, call `enhance()` in keyframe path |
| `docs/guides/maxine-setup.md` | Create | SDK installation guide |

---

### Task 1: Python Protocol Extension

Add `EnhanceResult` response type and `encode_raw_rgb()` helper to the IPC protocol.

**Files:**
- Modify: `python/dorea_inference/protocol.py`

- [ ] **Step 1: Add `encode_raw_rgb()` helper**

Add after the existing `decode_raw_rgb()` function (line ~163) in `protocol.py`:

```python
def encode_raw_rgb(img: "np.ndarray") -> str:
    """Encode HxWx3 RGB uint8 array to base64 raw interleaved bytes."""
    import numpy as np
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {arr.shape}")
    return base64.b64encode(arr.tobytes()).decode("ascii")
```

- [ ] **Step 2: Add `EnhanceResult` dataclass**

Add after the existing `DepthResult` class (line ~113) in `protocol.py`:

```python
@dataclass
class EnhanceResult:
    """Enhanced RGB frame returned from Maxine pipeline."""
    id: Optional[str]
    image_b64: str   # base64-encoded raw RGB u8 bytes
    width: int
    height: int
    type: str = "enhance_result"

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id,
                "image_b64": self.image_b64,
                "width": self.width, "height": self.height}

    @staticmethod
    def from_array(id: Optional[str], img: "np.ndarray") -> "EnhanceResult":
        """Encode an HxWx3 uint8 RGB array as EnhanceResult."""
        return EnhanceResult(
            id=id,
            image_b64=encode_raw_rgb(img),
            width=img.shape[1],
            height=img.shape[0],
        )
```

- [ ] **Step 3: Verify imports**

Ensure `Optional` is imported at the top of `protocol.py` (it should already be from the existing `ErrorResponse` class).

- [ ] **Step 4: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/protocol.py
git commit -m "feat(inference): add EnhanceResult protocol type and encode_raw_rgb helper"
```

---

### Task 2: Python MaxineEnhancer Class

Create the `MaxineEnhancer` class that wraps nvvfx with CUDA context sharing and mock mode.

**Files:**
- Create: `python/dorea_inference/maxine_enhancer.py`

- [ ] **Step 1: Create `maxine_enhancer.py` with mock and real modes**

```python
"""NVIDIA Maxine VFX SDK wrapper for AI enhancement (super-resolution + artifact reduction).

Requires the nvvfx package (NVIDIA Maxine Video Effects SDK Python bindings).
Install separately from NGC — not bundled with dorea. See docs/guides/maxine-setup.md.

Set DOREA_MAXINE_MOCK=1 to enable mock mode for CI testing without the SDK.
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("dorea-inference")

_MOCK_MODE = os.environ.get("DOREA_MAXINE_MOCK", "") == "1"

# Attempt nvvfx import (unless mock mode)
_nvvfx = None
if not _MOCK_MODE:
    try:
        import nvvfx as _nvvfx
    except ImportError:
        _nvvfx = None


class MaxineEnhancer:
    """AI enhancement via Maxine VideoSuperRes + ArtifactReduction.

    If DOREA_MAXINE_MOCK=1 is set, all enhance() calls return the input unchanged.
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        self.upscale_factor = upscale_factor
        self._mock = _MOCK_MODE
        self._passthrough_count = 0
        self._total_count = 0
        self._sr_effect = None
        self._ar_effect = None

        if self._mock:
            log.info("Maxine mock mode enabled (DOREA_MAXINE_MOCK=1)")
            return

        if _nvvfx is None:
            raise RuntimeError(
                "Maxine SDK not found. Install from NGC (nvidia-maxine-vfx-sdk): "
                "see docs/guides/maxine-setup.md, or unset --maxine"
            )

        log.info("Loading Maxine VideoSuperRes (upscale_factor=%d)", upscale_factor)
        # VideoSuperRes and ArtifactReduction are initialized lazily on first
        # enhance() call because we need the input dimensions to set output size.

    def _init_effects(self, width: int, height: int) -> None:
        """Lazily initialize Maxine effects with known input dimensions."""
        import torch

        out_w = width * self.upscale_factor
        out_h = height * self.upscale_factor

        # Super-resolution effect
        self._sr_effect = _nvvfx.VideoSuperRes(
            output_width=out_w,
            output_height=out_h,
        )
        # Share PyTorch's CUDA stream to avoid duplicate CUDA contexts
        stream = torch.cuda.current_stream()
        self._sr_effect.set_cuda_stream(stream.cuda_stream)
        self._sr_effect.load()

        # Artifact reduction effect (same resolution, shares CUDA stream)
        self._ar_effect = _nvvfx.ArtifactReduction()
        self._ar_effect.set_cuda_stream(stream.cuda_stream)
        self._ar_effect.load()

        log.info(
            "Maxine effects initialized: SR %dx%d→%dx%d, AR %dx%d",
            width, height, out_w, out_h, width, height,
        )

    def enhance(
        self,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        artifact_reduce: bool = True,
        upscale_factor: int = 2,
    ) -> np.ndarray:
        """Enhance a single RGB u8 frame. Returns RGB u8 at original resolution.

        On any failure, logs the error and returns the original frame unchanged.
        """
        self._total_count += 1

        if self._mock:
            return rgb_u8

        try:
            return self._enhance_impl(rgb_u8, width, height, artifact_reduce, upscale_factor)
        except Exception as e:
            self._passthrough_count += 1
            log.warning("Maxine enhance failed (frame passthrough): %s", e)
            return rgb_u8

    def _enhance_impl(
        self,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        artifact_reduce: bool,
        upscale_factor: int,
    ) -> np.ndarray:
        """Internal enhancement — exceptions propagate to enhance() for catch."""
        import torch

        # Lazy init on first call (needs dimensions)
        if self._sr_effect is None:
            self._init_effects(width, height)

        # RGB → BGRA (Maxine expects BGRA interleaved uint8)
        bgr = cv2.cvtColor(rgb_u8.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

        # Upload to CUDA tensor via DLPack
        tensor = torch.from_numpy(bgra).cuda()

        # 1. Artifact reduction at original resolution (if enabled and ≤1080p)
        if artifact_reduce:
            if height <= 1080 and width <= 1920:
                tensor = self._ar_effect.run(tensor)
            elif self._total_count == 1:
                log.warning(
                    "Artifact reduction skipped: input %dx%d exceeds 1080p limit",
                    width, height,
                )

        # 2. Super-resolution → upscaled intermediate
        tensor = self._sr_effect.run(tensor)

        # 3. Download to CPU and downsample back to original resolution
        upscaled_bgra = tensor.cpu().numpy()
        downscaled_bgra = cv2.resize(
            upscaled_bgra, (width, height), interpolation=cv2.INTER_AREA,
        )

        # BGRA → RGB
        downscaled_bgr = cv2.cvtColor(downscaled_bgra, cv2.COLOR_BGRA2BGR)
        result_rgb = cv2.cvtColor(downscaled_bgr, cv2.COLOR_BGR2RGB)

        return result_rgb

    def stats(self) -> str:
        """Return summary string for shutdown logging."""
        if self._total_count == 0:
            return "Maxine: no frames processed"
        return (
            f"Maxine: enhanced {self._total_count} keyframes"
            + (f", {self._passthrough_count} passthrough failures"
               if self._passthrough_count > 0 else "")
        )
```

- [ ] **Step 2: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/maxine_enhancer.py
git commit -m "feat(inference): add MaxineEnhancer class with nvvfx wrapper and mock mode"
```

---

### Task 3: Python Server Enhancement Handler

Extend `server.py` to accept `--maxine` flag and handle `enhance` requests.

**Files:**
- Modify: `python/dorea_inference/server.py`

- [ ] **Step 1: Add `--maxine` and `--maxine-upscale-factor` args to `_parse_args()`**

In `server.py`, find the `_parse_args()` function (line ~39) and add after the existing `--no-depth` argument:

```python
    p.add_argument("--maxine", action="store_true",
                   help="Enable Maxine enhancement (requires nvvfx SDK)")
    p.add_argument("--maxine-upscale-factor", type=int, default=2,
                   help="Maxine super-resolution upscale factor (default: 2)")
```

- [ ] **Step 2: Add Maxine model loading in `main()`**

After the existing depth model loading block (line ~95), add:

```python
    maxine_enhancer = None
    if args.maxine:
        from .maxine_enhancer import MaxineEnhancer
        maxine_enhancer = MaxineEnhancer(upscale_factor=args.maxine_upscale_factor)
        print(f"[dorea-inference] Maxine enhancer loaded (upscale_factor={args.maxine_upscale_factor})",
              file=sys.stderr, flush=True)
```

Note: `MaxineEnhancer.__init__()` raises `RuntimeError` if nvvfx is unavailable and `DOREA_MAXINE_MOCK` is not set. This is the "hard error at startup" behavior — the server process exits with a traceback before the IPC loop starts.

- [ ] **Step 3: Add `enhance` request handler in the request loop**

In the request dispatch chain (after the `elif req_type == "depth":` block, before the `elif req_type == "shutdown":` block), add:

```python
        elif req_type == "enhance":
            if maxine_enhancer is None:
                raise RuntimeError("Maxine enhancer not loaded (missing --maxine flag)")
            fmt = req.get("format", "raw_rgb")
            if fmt == "raw_rgb":
                img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
            else:
                img = decode_png(req["image_b64"])
            enhanced = maxine_enhancer.enhance(
                img,
                width=int(req["width"]),
                height=int(req["height"]),
                artifact_reduce=req.get("artifact_reduce", True),
                upscale_factor=int(req.get("upscale_factor", 2)),
            )
            resp = EnhanceResult.from_array(req_id, enhanced)
```

- [ ] **Step 4: Add EnhanceResult import**

At the top of `server.py`, update the protocol import (line ~10):

```python
from .protocol import (
    PongResponse, RauneResult, DepthResult, EnhanceResult,
    ErrorResponse, OkResponse,
    decode_png, decode_raw_rgb, encode_raw_rgb,
)
```

- [ ] **Step 5: Add Maxine stats to shutdown handler**

In the `elif req_type == "shutdown":` block, before `break`, add:

```python
            if maxine_enhancer is not None:
                print(f"[dorea-inference] {maxine_enhancer.stats()}", file=sys.stderr, flush=True)
```

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/server.py
git commit -m "feat(inference): add enhance request handler and --maxine flag to server"
```

---

### Task 4: Rust InferenceConfig Extension

Add Maxine-related fields to `InferenceConfig` and pass them to the Python subprocess.

**Files:**
- Modify: `crates/dorea-video/src/inference.rs`

- [ ] **Step 1: Write failing test for Maxine CLI args in spawn command**

Add to the existing `#[cfg(test)]` module at the bottom of `inference.rs`:

```rust
#[test]
fn spawn_command_includes_maxine_flags() {
    let config = InferenceConfig {
        python_exe: PathBuf::from("/usr/bin/python3"),
        raune_weights: None,
        raune_models_dir: None,
        skip_raune: true,
        depth_model: None,
        device: None,
        startup_timeout: Duration::from_secs(10),
        maxine: true,
        maxine_upscale_factor: 2,
    };
    let args = config.build_args();
    assert!(args.contains(&"--maxine".to_string()));
    assert!(args.contains(&"--maxine-upscale-factor".to_string()));
    assert!(args.contains(&"2".to_string()));
}

#[test]
fn spawn_command_omits_maxine_when_disabled() {
    let config = InferenceConfig {
        python_exe: PathBuf::from("/usr/bin/python3"),
        raune_weights: None,
        raune_models_dir: None,
        skip_raune: true,
        depth_model: None,
        device: None,
        startup_timeout: Duration::from_secs(10),
        maxine: false,
        maxine_upscale_factor: 2,
    };
    let args = config.build_args();
    assert!(!args.contains(&"--maxine".to_string()));
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video spawn_command_includes_maxine_flags 2>&1 | tail -20
```

Expected: FAIL — `maxine` field does not exist on `InferenceConfig`, `build_args()` method does not exist.

- [ ] **Step 3: Add `maxine` fields to `InferenceConfig`**

In `inference.rs`, add to the `InferenceConfig` struct (after line ~62):

```rust
    /// Enable Maxine enhancement in the inference subprocess.
    pub maxine: bool,
    /// Maxine super-resolution upscale factor (default 2).
    pub maxine_upscale_factor: u32,
```

- [ ] **Step 4: Add `build_args()` method to `InferenceConfig`**

Add a method to `InferenceConfig` that builds the CLI args vector. This extracts the arg-building logic that will also be used by `spawn()`:

```rust
impl InferenceConfig {
    /// Build CLI argument list for the Python inference server.
    pub fn build_args(&self) -> Vec<String> {
        let mut args = vec!["-m".to_string(), "dorea_inference.server".to_string()];

        if self.skip_raune {
            args.push("--no-raune".to_string());
        } else if let Some(p) = &self.raune_weights {
            args.push("--raune-weights".to_string());
            args.push(p.display().to_string());
        }

        if let Some(p) = &self.raune_models_dir {
            args.push("--raune-models-dir".to_string());
            args.push(p.display().to_string());
        }

        if let Some(p) = &self.depth_model {
            args.push("--depth-model".to_string());
            args.push(p.display().to_string());
        }

        if let Some(d) = &self.device {
            args.push("--device".to_string());
            args.push(d.clone());
        }

        if self.maxine {
            args.push("--maxine".to_string());
            args.push("--maxine-upscale-factor".to_string());
            args.push(self.maxine_upscale_factor.to_string());
        }

        args
    }
}
```

- [ ] **Step 5: Update `spawn()` to use `build_args()`**

Replace the arg-building section in `spawn()` with:

```rust
    let mut cmd = Command::new(&config.python_exe);
    cmd.args(config.build_args());
```

- [ ] **Step 6: Update all `InferenceConfig` construction sites to include new fields**

Search for `InferenceConfig {` in grade.rs and calibrate.rs. Add `maxine: false, maxine_upscale_factor: 2,` to each. There are two: one for calibration (line ~209 in grade.rs) and one for grading (line ~216 in grade.rs). The calibration server always gets `maxine: false`.

- [ ] **Step 7: Run test to verify it passes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video spawn_command 2>&1 | tail -20
```

Expected: both `spawn_command_includes_maxine_flags` and `spawn_command_omits_maxine_when_disabled` PASS.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference.rs crates/dorea-cli/src/grade.rs crates/dorea-cli/src/calibrate.rs
git commit -m "feat(inference): add maxine fields to InferenceConfig and build_args() method"
```

---

### Task 5: Rust `enhance()` IPC Method

Add the `enhance()` method to `InferenceServer` that sends/receives enhance requests over JSON-lines IPC.

**Files:**
- Modify: `crates/dorea-video/src/inference.rs`

- [ ] **Step 1: Write failing test for enhance IPC round-trip**

Add to the `#[cfg(test)]` module in `inference.rs`:

```rust
#[test]
fn enhance_parses_valid_response() {
    // Simulate a valid enhance_result JSON response
    let width = 4usize;
    let height = 2usize;
    let pixels: Vec<u8> = vec![128; width * height * 3];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&pixels);

    let resp_json = serde_json::json!({
        "type": "enhance_result",
        "id": "test_001",
        "image_b64": b64,
        "width": width,
        "height": height,
    });

    let (result_pixels, rw, rh) = parse_enhance_response(
        &resp_json.to_string(), "test_001", width, height,
    ).unwrap();

    assert_eq!(rw, width);
    assert_eq!(rh, height);
    assert_eq!(result_pixels.len(), width * height * 3);
    assert_eq!(result_pixels[0], 128);
}

#[test]
fn enhance_rejects_dimension_mismatch() {
    let width = 4usize;
    let height = 2usize;
    let pixels: Vec<u8> = vec![128; width * height * 3];
    let b64 = base64::engine::general_purpose::STANDARD.encode(&pixels);

    let resp_json = serde_json::json!({
        "type": "enhance_result",
        "id": "test_001",
        "image_b64": b64,
        "width": 8,  // mismatch
        "height": 4,  // mismatch
    });

    let result = parse_enhance_response(
        &resp_json.to_string(), "test_001", width, height,
    );
    assert!(result.is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video enhance_parses 2>&1 | tail -20
```

Expected: FAIL — `parse_enhance_response` does not exist.

- [ ] **Step 3: Implement `parse_enhance_response()` helper**

Add as a free function in `inference.rs` (before the `impl InferenceServer` block):

```rust
/// Parse an enhance_result JSON response, validating dimensions match the request.
fn parse_enhance_response(
    resp: &str,
    expected_id: &str,
    expected_w: usize,
    expected_h: usize,
) -> Result<(Vec<u8>, usize, usize), InferenceError> {
    let v: serde_json::Value = serde_json::from_str(resp)
        .map_err(|e| InferenceError::Ipc(format!("enhance response parse: {e}")))?;

    if v["type"].as_str() == Some("error") {
        let msg = v["message"].as_str().unwrap_or("unknown error");
        return Err(InferenceError::ServerError(msg.to_string()));
    }
    if v["type"].as_str() != Some("enhance_result") {
        return Err(InferenceError::Ipc(format!(
            "unexpected response type for enhance: {resp}"
        )));
    }

    let w = v["width"].as_u64().unwrap_or(0) as usize;
    let h = v["height"].as_u64().unwrap_or(0) as usize;

    if w != expected_w || h != expected_h {
        return Err(InferenceError::Ipc(format!(
            "enhance dimension mismatch: expected {expected_w}x{expected_h}, got {w}x{h}"
        )));
    }

    let b64_out = v["image_b64"]
        .as_str()
        .ok_or_else(|| InferenceError::Ipc("missing image_b64".to_string()))?;

    let raw = B64
        .decode(b64_out)
        .map_err(|e| InferenceError::Ipc(format!("base64 decode: {e}")))?;

    if raw.len() != w * h * 3 {
        return Err(InferenceError::Ipc(format!(
            "enhance buffer size mismatch: got {}, expected {}",
            raw.len(),
            w * h * 3,
        )));
    }

    Ok((raw, w, h))
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video enhance_ 2>&1 | tail -20
```

Expected: both `enhance_parses_valid_response` and `enhance_rejects_dimension_mismatch` PASS.

- [ ] **Step 5: Add `enhance()` method to `InferenceServer`**

Add to the `impl InferenceServer` block, following the pattern of `run_depth()`:

```rust
    /// Send a graded RGB u8 frame for Maxine enhancement.
    /// Returns enhanced RGB u8 at the same resolution as input.
    pub fn enhance(
        &mut self,
        id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        artifact_reduce: bool,
        upscale_factor: u32,
    ) -> Result<Vec<u8>, InferenceError> {
        let b64 = B64.encode(image_rgb);

        let req = serde_json::json!({
            "type": "enhance",
            "id": id,
            "format": "raw_rgb",
            "image_b64": b64,
            "width": width,
            "height": height,
            "artifact_reduce": artifact_reduce,
            "upscale_factor": upscale_factor,
        });

        self.send_line(&req.to_string())?;
        let resp = self.recv_line()?;

        let (pixels, _w, _h) = parse_enhance_response(&resp, id, width, height)?;
        Ok(pixels)
    }
```

- [ ] **Step 6: Verify full test suite passes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video 2>&1 | tail -20
```

Expected: all existing tests + new enhance tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference.rs
git commit -m "feat(inference): add enhance() IPC method with dimension validation"
```

---

### Task 6: Rust Grade Pipeline Integration

Add `--maxine` CLI args to `GradeArgs` and wire `enhance()` into the keyframe path.

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

- [ ] **Step 1: Add Maxine CLI args to `GradeArgs`**

Add after the existing `pub verbose: bool` field (line ~84):

```rust
    /// Enable Maxine AI enhancement (requires nvvfx SDK on the inference server)
    #[arg(long)]
    pub maxine: bool,

    /// Disable Maxine AI enhancement (overrides --maxine)
    #[arg(long, conflicts_with = "maxine")]
    pub no_maxine: bool,

    /// Enable Maxine artifact reduction before upscale [default: true]
    #[arg(long, default_value = "true")]
    pub maxine_artifact_reduction: bool,

    /// Maxine super-resolution upscale factor [default: 2]
    #[arg(long, default_value = "2")]
    pub maxine_upscale_factor: u32,
```

- [ ] **Step 2: Add startup validation for upscale factor**

In `run()`, after args validation (line ~177), add:

```rust
    let maxine_enabled = args.maxine && !args.no_maxine;

    if maxine_enabled {
        let valid_factors = [2, 3, 4]; // 1.33 and 1.5 deferred — integer only for v1
        if !valid_factors.contains(&args.maxine_upscale_factor) {
            anyhow::bail!(
                "unsupported maxine_upscale_factor: {}. Supported: {:?}",
                args.maxine_upscale_factor, valid_factors,
            );
        }
        log::info!(
            "Maxine enhancement enabled: upscale_factor={}, artifact_reduction={}",
            args.maxine_upscale_factor, args.maxine_artifact_reduction,
        );
    }
```

- [ ] **Step 3: Pass `--maxine` to the grading inference server only**

Find the `InferenceConfig` construction for the grading server (line ~216) and set:

```rust
        maxine: maxine_enabled,
        maxine_upscale_factor: args.maxine_upscale_factor,
```

Ensure the calibration server (line ~209) keeps `maxine: false`.

- [ ] **Step 4: Insert enhance() call in the keyframe path**

Find the keyframe path where `grade_frame()` is called (line ~295) and `encoder.write_frame()` is called (line ~308). Modify to:

```rust
    let graded = grade_frame(
        &frame.pixels, &depth, frame.width, frame.height, &calibration, &params,
    ).map_err(|e| anyhow::anyhow!("Grading failed for frame {}: {e}", frame.index))?;

    // Maxine enhancement (optional, keyframes only)
    let final_frame = if maxine_enabled {
        match inf_server.enhance(
            &frame.index.to_string(),
            &graded,
            frame.width,
            frame.height,
            args.maxine_artifact_reduction,
            args.maxine_upscale_factor,
        ) {
            Ok(enhanced) => enhanced,
            Err(e) => {
                log::warn!(
                    "Maxine enhance failed for frame {} (using unenhanced): {e}",
                    frame.index,
                );
                graded
            }
        }
    } else {
        graded
    };
```

- [ ] **Step 5: Update all downstream references from `graded` to `final_frame`**

Replace `&graded` with `&final_frame` in:
1. The `flush_buffer_graded()` call: `Some(&final_frame)` instead of `Some(&graded)`
2. The `encoder.write_frame(&final_frame)` call
3. The `last_keyframe_graded = Some(final_frame)` assignment

This ensures temporal interpolation uses the enhanced keyframe outputs.

- [ ] **Step 6: Verify compilation**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build 2>&1 | tail -20
```

Expected: clean build, no errors.

- [ ] **Step 7: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): wire Maxine enhancement into keyframe path with --maxine CLI flag"
```

---

### Task 7: Maxine Setup Guide

Create the user-facing documentation for installing the Maxine SDK.

**Files:**
- Create: `docs/guides/maxine-setup.md`

- [ ] **Step 1: Write the setup guide**

```markdown
# NVIDIA Maxine VFX SDK Setup

Optional dependency for AI-enhanced post-grading (in-resolution oversample).

## Prerequisites

- NVIDIA GPU: Turing, Ampere, Ada, or Blackwell with Tensor Cores (e.g. RTX 3060)
- Linux driver: 570.190+, 580.82+, or 590.44+
- Python 3.10+
- dorea inference venv active (`/opt/dorea-venv`)

## Installation

1. Download the Maxine Video Effects SDK from
   [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/maxine_linux_vfx_sdk_ga)

2. Install the Python bindings:

   ```bash
   source /opt/dorea-venv/bin/activate
   pip install nvvfx
   ```

3. Verify:

   ```bash
   python -c "import nvvfx; print('nvvfx OK')"
   ```

## Usage

Enable Maxine enhancement when grading:

```bash
dorea grade --input footage.mp4 --maxine
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--maxine` | off | Enable Maxine enhancement |
| `--no-maxine` | - | Explicitly disable (overrides --maxine) |
| `--maxine-upscale-factor` | 2 | Super-resolution factor (2, 3, 4) |
| `--maxine-artifact-reduction` | true | Clean compression artifacts before upscale |

## What it does

Maxine enhancement runs after color grading on keyframes only:

1. Artifact reduction at original resolution (if enabled, ≤1080p only)
2. AI super-resolution to 2x resolution (e.g. 1080p → 4K)
3. Area downsample back to original resolution

Result: cleaner, sharper footage with reduced compression artifacts. Non-keyframe
frames are temporally interpolated from enhanced keyframe outputs.

## Known limitations

- **8-bit bottleneck**: Maxine accepts only uint8. The 10-bit pipeline's f32 precision
  is quantized to u8 for enhancement. This trades bit-depth for spatial quality.
- **VRAM**: Peak ~4.9-5.7GB with Maxine enabled on RTX 3060 6GB. If OOM occurs,
  try `--maxine-artifact-reduction false` or `--no-maxine`.
- **Artifact reduction** is trained for H.264; effectiveness on H.265 source is
  unvalidated but expected to help with common artifacts.

## CI / Testing

Set `DOREA_MAXINE_MOCK=1` to test the enhancement pipeline without the SDK:

```bash
DOREA_MAXINE_MOCK=1 cargo test
```
```

- [ ] **Step 2: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add docs/guides/maxine-setup.md
git commit -m "docs: add Maxine VFX SDK setup guide"
```

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] `cargo build` succeeds with no warnings
- [ ] `cargo test` passes all tests (existing + new)
- [ ] `dorea grade --help` shows `--maxine` and related flags
- [ ] `dorea grade --input test.mp4 --maxine --maxine-upscale-factor 5` fails with "unsupported" error
- [ ] With `DOREA_MAXINE_MOCK=1`, the Python server starts with `--maxine` and processes enhance requests (returns input unchanged)
- [ ] Without `--maxine`, the pipeline produces byte-identical output to before this change
