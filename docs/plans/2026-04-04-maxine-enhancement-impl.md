# Maxine Pre-Processing Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVIDIA Maxine VFX SDK as first-stage preprocessor — runs on every frame at full resolution in Pass 1, writes a lossless ffv1/mkv temp file, and Pass 2 decodes from that file instead of re-running Maxine.

**Architecture:** One inference server (Maxine + RAUNE + Depth) spawned before Pass 1, shut down after calibration. Pass 1 decodes full-res, calls `enhance()` IPC per frame, writes to temp encoder, downscales to proxy for MSE/keyframe detection. Pass 2 decodes temp file and grades Maxine-enhanced frames.

**Tech Stack:** Rust (clap, serde_json, base64), Python (nvvfx, torch, cv2, numpy), ffmpeg (ffv1 lossless codec)

**Spec:** `docs/decisions/2026-04-04-maxine-preprocessor-design.md`

---

## File Map

| File | Action | Task |
|------|--------|------|
| `python/dorea_inference/protocol.py` | Modify — add `EnhanceResult` + `encode_raw_rgb` | 1 |
| `python/dorea_inference/maxine_enhancer.py` | Create — MaxineEnhancer class | 2 |
| `python/dorea_inference/server.py` | Modify — `--maxine` flag + enhance handler | 3 |
| `python/tests/test_maxine_protocol.py` | Create — protocol tests | 1 |
| `python/tests/test_maxine_server.py` | Create — server mock-mode tests | 3 |
| `crates/dorea-video/src/inference_subprocess.rs` | Modify — InferenceConfig fields, build_args(), enhance() IPC | 4, 5 |
| `crates/dorea-video/src/ffmpeg.rs` | Modify — FrameEncoder::new_lossless_temp() | 6 |
| `crates/dorea-cli/src/grade.rs` | Modify — major restructure for Maxine integration | 7 |
| `crates/dorea-cli/src/calibrate.rs` | Modify — add maxine fields to InferenceConfig literal | 4 |
| `crates/dorea-cli/src/preview.rs` | Modify — add maxine fields to InferenceConfig literal | 4 |
| `docs/guides/maxine-setup.md` | Create — SDK install guide | 8 |

---

### Task 1: Python Protocol Extension (issue #16)

**Files:**
- Modify: `python/dorea_inference/protocol.py`
- Create: `python/tests/test_maxine_protocol.py`

- [ ] **Step 1: Write the failing protocol tests**

Create `python/tests/test_maxine_protocol.py`:

```python
"""Tests for EnhanceResult protocol type and encode_raw_rgb helper."""
import base64
import numpy as np
import pytest

from dorea_inference.protocol import encode_raw_rgb, decode_raw_rgb, EnhanceResult


def test_encode_raw_rgb_roundtrip():
    """encode_raw_rgb output should be decodable by decode_raw_rgb."""
    img = np.array([[[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    h, w = img.shape[:2]
    b64 = encode_raw_rgb(img)
    recovered = decode_raw_rgb(b64, w, h)
    np.testing.assert_array_equal(recovered, img)


def test_encode_raw_rgb_shape_check():
    """encode_raw_rgb must reject non-HxWx3 arrays."""
    bad = np.zeros((4, 4), dtype=np.uint8)  # 2D, not 3D
    with pytest.raises(ValueError, match="HxWx3"):
        encode_raw_rgb(bad)


def test_enhance_result_from_array_roundtrip():
    """EnhanceResult.from_array should encode and store correct dimensions."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, 0] = [255, 128, 64]
    result = EnhanceResult.from_array("frame_001", img)
    assert result.width == 20
    assert result.height == 10
    assert result.id == "frame_001"
    assert result.type == "enhance_result"
    # Decode the b64 to verify pixel data
    raw = base64.b64decode(result.image_b64)
    assert len(raw) == 10 * 20 * 3
    assert raw[0] == 255
    assert raw[1] == 128
    assert raw[2] == 64


def test_enhance_result_to_dict_fields():
    """to_dict must include all required fields."""
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    d = EnhanceResult.from_array("x", img).to_dict()
    assert d["type"] == "enhance_result"
    assert d["id"] == "x"
    assert d["width"] == 6
    assert d["height"] == 4
    assert "image_b64" in d
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
python -m pytest python/tests/test_maxine_protocol.py -v 2>&1 | tail -20
```

Expected: ImportError — `encode_raw_rgb` and `EnhanceResult` do not exist yet.

- [ ] **Step 3: Add `encode_raw_rgb` to protocol.py**

In `python/dorea_inference/protocol.py`, after the `decode_raw_rgb` function (after line 183), add:

```python
def encode_raw_rgb(img: "np.ndarray") -> str:
    """Encode HxWx3 RGB uint8 array to base64 raw interleaved bytes."""
    import numpy as np
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {arr.shape}")
    return base64.b64encode(arr.tobytes()).decode("ascii")
```

- [ ] **Step 4: Add `EnhanceResult` dataclass to protocol.py**

After the `DepthResult` class (after line 113, before `DepthBatchResult`), add:

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

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
python -m pytest python/tests/test_maxine_protocol.py -v 2>&1 | tail -20
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/protocol.py python/tests/test_maxine_protocol.py
git commit -m "feat(inference): add EnhanceResult protocol type and encode_raw_rgb helper"
```

---

### Task 2: Python MaxineEnhancer Class (issue #17)

**Files:**
- Create: `python/dorea_inference/maxine_enhancer.py`

- [ ] **Step 1: Create `maxine_enhancer.py`**

Create `python/dorea_inference/maxine_enhancer.py` with this content:

```python
"""NVIDIA Maxine VFX SDK wrapper for AI enhancement (super-resolution + artifact reduction).

Requires the nvvfx package (NVIDIA Maxine Video Effects SDK Python bindings).
Install separately from NGC — not bundled with dorea. See docs/guides/maxine-setup.md.

Set DOREA_MAXINE_MOCK=1 to enable mock mode for CI testing without the SDK.
"""
from __future__ import annotations

import logging
import os
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

        log.info("Maxine VideoSuperRes initialized (upscale_factor=%d)", upscale_factor)
        # Effects are initialized lazily on first enhance() call because we need
        # the input dimensions to configure the output size.

    def _init_effects(self, width: int, height: int) -> None:
        """Lazily initialize Maxine effects with known input dimensions."""
        import torch

        out_w = width * self.upscale_factor
        out_h = height * self.upscale_factor

        self._sr_effect = _nvvfx.VideoSuperRes(
            output_width=out_w,
            output_height=out_h,
        )
        stream = torch.cuda.current_stream()
        self._sr_effect.set_cuda_stream(stream.cuda_stream)
        self._sr_effect.load()

        self._ar_effect = _nvvfx.ArtifactReduction()
        self._ar_effect.set_cuda_stream(stream.cuda_stream)
        self._ar_effect.load()

        log.info(
            "Maxine effects loaded: SR %dx%d→%dx%d, AR %dx%d",
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

        if self._sr_effect is None:
            self._init_effects(width, height)

        # RGB → BGRA (Maxine expects BGRA interleaved uint8)
        bgr = cv2.cvtColor(rgb_u8.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

        tensor = torch.from_numpy(bgra).cuda()

        # 1. Artifact reduction at original resolution (≤1080p only)
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

        # 3. Download and downsample back to original resolution
        upscaled_bgra = tensor.cpu().numpy()
        downscaled_bgra = cv2.resize(
            upscaled_bgra, (width, height), interpolation=cv2.INTER_AREA,
        )

        # BGRA → RGB
        downscaled_bgr = cv2.cvtColor(downscaled_bgra, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(downscaled_bgr, cv2.COLOR_BGR2RGB)

    def stats(self) -> str:
        """Return summary string for shutdown logging."""
        if self._total_count == 0:
            return "Maxine: no frames processed"
        return (
            f"Maxine: enhanced {self._total_count} frames"
            + (f", {self._passthrough_count} passthrough failures"
               if self._passthrough_count > 0 else "")
        )
```

- [ ] **Step 2: Verify mock mode works (no nvvfx SDK needed)**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -c "
import numpy as np
from dorea_inference.maxine_enhancer import MaxineEnhancer
m = MaxineEnhancer(upscale_factor=2)
img = np.zeros((4, 6, 3), dtype=np.uint8)
out = m.enhance(img, 6, 4)
assert out.shape == (4, 6, 3), f'shape mismatch: {out.shape}'
print('mock mode OK:', m.stats())
"
```

Expected: `mock mode OK: Maxine: enhanced 1 frames`

- [ ] **Step 3: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/maxine_enhancer.py
git commit -m "feat(inference): add MaxineEnhancer class with nvvfx wrapper and mock mode"
```

---

### Task 3: Python Server Enhancement Handler (issue #18)

**Files:**
- Modify: `python/dorea_inference/server.py`
- Create: `python/tests/test_maxine_server.py`

- [ ] **Step 1: Write the failing server tests**

Create `python/tests/test_maxine_server.py`:

```python
"""Tests for enhance request handler in mock mode (DOREA_MAXINE_MOCK=1)."""
import base64
import json
import os
import sys
import io
import numpy as np
import pytest

# Force mock mode before any imports
os.environ["DOREA_MAXINE_MOCK"] = "1"


def _make_enhance_req(width: int, height: int, req_id: str = "f001") -> dict:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    b64 = base64.b64encode(pixels.tobytes()).decode("ascii")
    return {
        "type": "enhance",
        "id": req_id,
        "format": "raw_rgb",
        "image_b64": b64,
        "width": width,
        "height": height,
        "artifact_reduce": True,
        "upscale_factor": 2,
    }


def _run_server_with_reqs(requests: list, extra_argv: list = None) -> list:
    """Run main() with a sequence of requests, return parsed responses."""
    from dorea_inference.server import main

    lines = [json.dumps(r) for r in requests] + ['{"type": "shutdown"}']
    stdin_data = "\n".join(lines) + "\n"

    captured = io.StringIO()
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = captured

    argv = ["--no-raune", "--no-depth"] + (extra_argv or [])
    try:
        main(argv=argv)
    except SystemExit:
        pass
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout

    output = captured.getvalue().strip()
    return [json.loads(line) for line in output.split("\n") if line.strip()]


def test_enhance_handler_returns_enhance_result():
    req = _make_enhance_req(4, 6)
    responses = _run_server_with_reqs([req], extra_argv=["--maxine"])
    # First response is the enhance_result; second is ok (shutdown)
    enhance_resp = responses[0]
    assert enhance_resp["type"] == "enhance_result"
    assert enhance_resp["id"] == "f001"
    assert enhance_resp["width"] == 4
    assert enhance_resp["height"] == 6
    raw = base64.b64decode(enhance_resp["image_b64"])
    assert len(raw) == 4 * 6 * 3


def test_enhance_handler_without_maxine_flag_errors():
    """enhance request without --maxine should return error (no enhancer loaded)."""
    req = _make_enhance_req(4, 6)
    responses = _run_server_with_reqs([req])  # no --maxine
    assert responses[0]["type"] == "error"
    assert "not loaded" in responses[0]["message"].lower() or "maxine" in responses[0]["message"].lower()


def test_enhance_handler_passthrough_preserves_dimensions():
    """Mock mode returns same dimensions as input."""
    req = _make_enhance_req(16, 9)
    responses = _run_server_with_reqs([req], extra_argv=["--maxine"])
    r = responses[0]
    assert r["width"] == 16
    assert r["height"] == 9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_maxine_server.py -v 2>&1 | tail -20
```

Expected: FAIL — `--maxine` arg not recognized, `enhance` handler not found.

- [ ] **Step 3: Update imports in server.py**

In `python/dorea_inference/server.py`, replace the protocol import block (lines 27-38) with:

```python
from .protocol import (
    PongResponse,
    RauneResult,
    DepthResult,
    DepthBatchResult,
    RauneDepthBatchResult,
    EnhanceResult,
    ErrorResponse,
    OkResponse,
    decode_png,
    decode_raw_rgb,
    encode_raw_rgb,
    encode_png,
)
```

- [ ] **Step 4: Add `--maxine` args to `_parse_args()`**

In `_parse_args()`, after the `--no-depth` argument (after line 51), add:

```python
    p.add_argument("--maxine", action="store_true",
                   help="Enable Maxine enhancement (requires nvvfx SDK or DOREA_MAXINE_MOCK=1)")
    p.add_argument("--maxine-upscale-factor", type=int, default=2,
                   help="Maxine super-resolution upscale factor (default: 2)")
    p.add_argument("--no-maxine-artifact-reduction", action="store_true",
                   help="Disable artifact reduction before upscale")
```

- [ ] **Step 5: Add Maxine model loading in `main()`**

In `main()`, after the depth model loading block (after line 97, before `print("[dorea-inference] ready"`), add:

```python
    maxine_enhancer = None
    if args.maxine:
        from .maxine_enhancer import MaxineEnhancer
        maxine_enhancer = MaxineEnhancer(upscale_factor=args.maxine_upscale_factor)
        print(
            f"[dorea-inference] Maxine enhancer loaded "
            f"(upscale_factor={args.maxine_upscale_factor}, "
            f"artifact_reduction={not args.no_maxine_artifact_reduction})",
            file=sys.stderr, flush=True,
        )
```

- [ ] **Step 6: Add `enhance` request handler**

In the request dispatch chain, before the `elif req_type == "shutdown":` block (before line 218), add:

```python
            elif req_type == "enhance":
                if maxine_enhancer is None:
                    raise RuntimeError(
                        "Maxine enhancer not loaded — pass --maxine to the server"
                    )
                fmt = req.get("format", "raw_rgb")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                enhanced = maxine_enhancer.enhance(
                    img,
                    width=int(req["width"]),
                    height=int(req["height"]),
                    artifact_reduce=not req.get("no_artifact_reduce", False),
                    upscale_factor=int(req.get("upscale_factor", 2)),
                )
                resp = EnhanceResult.from_array(req_id, enhanced)
```

- [ ] **Step 7: Add Maxine stats to shutdown handler**

In the `elif req_type == "shutdown":` block (line 218), before `break`, add:

```python
                if maxine_enhancer is not None:
                    print(
                        f"[dorea-inference] {maxine_enhancer.stats()}",
                        file=sys.stderr, flush=True,
                    )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_maxine_server.py -v 2>&1 | tail -20
```

Expected: All 3 tests PASS.

- [ ] **Step 9: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/server.py python/tests/test_maxine_server.py
git commit -m "feat(inference): add enhance request handler and --maxine flag to server"
```

---

### Task 4: Rust InferenceConfig + build_args() (issue #19)

**Files:**
- Modify: `crates/dorea-video/src/inference_subprocess.rs`
- Modify: `crates/dorea-cli/src/grade.rs`
- Modify: `crates/dorea-cli/src/calibrate.rs`
- Modify: `crates/dorea-cli/src/preview.rs`

- [ ] **Step 1: Write failing tests for build_args()**

Add to the `#[cfg(test)]` module at the bottom of `crates/dorea-video/src/inference_subprocess.rs`:

```rust
#[test]
fn spawn_command_includes_maxine_flags() {
    let config = InferenceConfig {
        maxine: true,
        maxine_upscale_factor: 2,
        ..InferenceConfig::default()
    };
    let args = config.build_args();
    assert!(args.contains(&"--maxine".to_string()), "missing --maxine");
    assert!(args.contains(&"--maxine-upscale-factor".to_string()), "missing --maxine-upscale-factor");
    assert!(args.contains(&"2".to_string()), "missing upscale factor value");
}

#[test]
fn spawn_command_omits_maxine_when_disabled() {
    let config = InferenceConfig {
        maxine: false,
        ..InferenceConfig::default()
    };
    let args = config.build_args();
    assert!(!args.contains(&"--maxine".to_string()), "--maxine should be absent");
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video spawn_command 2>&1 | tail -20
```

Expected: FAIL — `maxine` field and `build_args` method do not exist.

- [ ] **Step 3: Add `maxine` fields to `InferenceConfig`**

In `crates/dorea-video/src/inference_subprocess.rs`, add to the `InferenceConfig` struct (after the `startup_timeout` field, around line 47):

```rust
    /// Enable Maxine enhancement in the inference subprocess.
    pub maxine: bool,
    /// Maxine super-resolution upscale factor (default 2).
    pub maxine_upscale_factor: u32,
```

- [ ] **Step 4: Update `Default` impl for `InferenceConfig`**

In the `impl Default for InferenceConfig` block (around line 50), add the new fields:

```rust
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
            maxine: false,
            maxine_upscale_factor: 2,
        }
    }
}
```

- [ ] **Step 5: Add `build_args()` method to `InferenceConfig`**

Add a new `impl InferenceConfig` block immediately after the `Default` impl:

```rust
impl InferenceConfig {
    /// Build the CLI argument list for the Python inference server.
    pub fn build_args(&self) -> Vec<String> {
        let mut args = vec!["-m".to_string(), "dorea_inference.server".to_string()];

        if self.skip_raune {
            args.push("--no-raune".to_string());
        } else {
            if let Some(p) = &self.raune_weights {
                args.push("--raune-weights".to_string());
                args.push(p.to_str().unwrap_or("").to_string());
            }
        }

        if let Some(p) = &self.raune_models_dir {
            args.push("--raune-models-dir".to_string());
            args.push(p.to_str().unwrap_or("").to_string());
        }

        if let Some(p) = &self.depth_model {
            args.push("--depth-model".to_string());
            args.push(p.to_str().unwrap_or("").to_string());
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

- [ ] **Step 6: Update `spawn()` to use `build_args()`**

In `InferenceServer::spawn()`, replace the manual arg-building block (lines 102–124) with:

```rust
        let mut cmd = Command::new(&config.python_exe);
        cmd.args(config.build_args());
```

Remove the old individual `cmd.arg(...)` calls that built `--no-raune`, `--raune-weights`, `--raune-models-dir`, `--depth-model`, `--device`. Keep everything after line 124 (PYTHONPATH, stdio, child spawn, ping/pong) unchanged.

- [ ] **Step 7: Update InferenceConfig literals in grade.rs**

In `crates/dorea-cli/src/grade.rs`, find the two `InferenceConfig { ... }` literals and add the new fields with `false` defaults:

**At line 236** (pre-computed calibration path), the literal uses `..build_inference_config(&args)` so it inherits the new fields automatically — no change needed here as long as `build_inference_config()` sets them.

**At line 704** (`build_inference_config()` function), add after `startup_timeout`:

```rust
fn build_inference_config(args: &GradeArgs) -> InferenceConfig {
    InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        skip_raune: false,
        depth_model: args.depth_model.clone(),
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
        maxine: false,            // overridden in Task 7 when --maxine is added
        maxine_upscale_factor: 2,
    }
}
```

- [ ] **Step 8: Update InferenceConfig literal in calibrate.rs**

In `crates/dorea-cli/src/calibrate.rs` at line 188, add the new fields:

```rust
        let cfg = InferenceConfig {
            python_exe: args.python.clone(),
            raune_weights: args.raune_weights.clone(),
            raune_models_dir: args.raune_models_dir.clone(),
            skip_raune: false,
            depth_model: args.depth_model.clone(),
            device: if args.cpu_only { Some("cpu".to_string()) } else { None },
            startup_timeout: Duration::from_secs(180),
            maxine: false,
            maxine_upscale_factor: 2,
        };
```

- [ ] **Step 9: Update InferenceConfig literal in preview.rs**

In `crates/dorea-cli/src/preview.rs` at line 96, add the new fields:

```rust
    let inf_cfg = InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        skip_raune: false,
        depth_model: args.depth_model.clone(),
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
        maxine: false,
        maxine_upscale_factor: 2,
    };
```

- [ ] **Step 10: Build to verify**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build 2>&1 | tail -20
```

Expected: clean build, no errors.

- [ ] **Step 11: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video spawn_command 2>&1 | tail -20
```

Expected: both `spawn_command_includes_maxine_flags` and `spawn_command_omits_maxine_when_disabled` PASS.

- [ ] **Step 12: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference_subprocess.rs \
        crates/dorea-cli/src/grade.rs \
        crates/dorea-cli/src/calibrate.rs \
        crates/dorea-cli/src/preview.rs
git commit -m "feat(inference): add maxine fields to InferenceConfig and build_args() method"
```

---

### Task 5: Rust `enhance()` IPC Method (issue #20)

**Files:**
- Modify: `crates/dorea-video/src/inference_subprocess.rs`

- [ ] **Step 1: Write failing tests**

Add to the `#[cfg(test)]` module in `inference_subprocess.rs`:

```rust
#[test]
fn enhance_parses_valid_response() {
    let width = 4usize;
    let height = 2usize;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];
    let b64 = B64.encode(&pixels);
    let resp = serde_json::json!({
        "type": "enhance_result",
        "id": "test_001",
        "image_b64": b64,
        "width": width,
        "height": height,
    });
    let (result_pixels, rw, rh) =
        parse_enhance_response(&resp.to_string(), "test_001", width, height).unwrap();
    assert_eq!(rw, width);
    assert_eq!(rh, height);
    assert_eq!(result_pixels.len(), width * height * 3);
    assert_eq!(result_pixels[0], 128);
}

#[test]
fn enhance_rejects_dimension_mismatch() {
    let pixels: Vec<u8> = vec![0u8; 4 * 2 * 3];
    let b64 = B64.encode(&pixels);
    let resp = serde_json::json!({
        "type": "enhance_result",
        "id": "test_001",
        "image_b64": b64,
        "width": 8,   // mismatch: sent 4
        "height": 4,  // mismatch: sent 2
    });
    let result = parse_enhance_response(&resp.to_string(), "test_001", 4, 2);
    assert!(result.is_err(), "should reject dimension mismatch");
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video enhance_ 2>&1 | tail -20
```

Expected: FAIL — `parse_enhance_response` does not exist.

- [ ] **Step 3: Add `parse_enhance_response()` free function**

Add before the `impl InferenceServer` block in `inference_subprocess.rs`:

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

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video enhance_ 2>&1 | tail -20
```

Expected: both tests PASS.

- [ ] **Step 5: Add `enhance()` method to `InferenceServer`**

Add to `impl InferenceServer` (after `run_depth` or `run_raune_depth_batch`):

```rust
    /// Send an RGB u8 frame to Maxine for enhancement.
    /// Returns enhanced RGB u8 at the same resolution as input.
    ///
    /// On success returns the enhanced pixel buffer.
    /// On server-side failure the Python side sends back the original frame
    /// (passthrough), so this method never errors on Maxine inference failure —
    /// only on IPC/protocol errors.
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
            "no_artifact_reduce": !artifact_reduce,
            "upscale_factor": upscale_factor,
        });

        self.send_line(&req.to_string())?;
        let resp = self.recv_line()?;
        let (pixels, _w, _h) = parse_enhance_response(&resp, id, width, height)?;
        Ok(pixels)
    }
```

- [ ] **Step 6: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video 2>&1 | tail -20
```

Expected: all tests PASS (existing + new).

- [ ] **Step 7: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference_subprocess.rs
git commit -m "feat(inference): add enhance() IPC method with dimension validation"
```

---

### Task 6: FrameEncoder Lossless Temp Mode (supports grade.rs)

**Files:**
- Modify: `crates/dorea-video/src/ffmpeg.rs`

- [ ] **Step 1: Write failing test**

Add to the `#[cfg(test)]` section at the bottom of `ffmpeg.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn lossless_temp_encoder_creates_decodable_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("dorea_test_ffv1_{}.mkv", std::process::id()));
        let width = 8usize;
        let height = 4usize;
        let fps = 30.0f64;

        // Write 3 frames
        let mut enc = FrameEncoder::new_lossless_temp(&path, width, height, fps)
            .expect("failed to create lossless encoder");
        let frame: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();
        for _ in 0..3 {
            enc.write_frame(&frame).expect("write_frame failed");
        }
        enc.finish().expect("finish failed");

        // File must exist and be non-empty
        assert!(path.exists(), "output file does not exist");
        assert!(path.metadata().unwrap().len() > 0, "output file is empty");

        // Must be decodable as a video
        let probe_result = probe(&path);
        assert!(probe_result.is_ok(), "ffprobe failed: {:?}", probe_result);
        let info = probe_result.unwrap();
        assert_eq!(info.width, width);
        assert_eq!(info.height, height);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
```

If a `#[cfg(test)] mod tests` block already exists at the bottom of ffmpeg.rs, add the test inside it instead of creating a duplicate module.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video lossless_temp 2>&1 | tail -20
```

Expected: FAIL — `FrameEncoder::new_lossless_temp` does not exist.

- [ ] **Step 3: Add `new_lossless_temp()` constructor to `FrameEncoder`**

Add this method inside `impl FrameEncoder`, after the `new()` method:

```rust
    /// Create a lossless ffv1/mkv encoder for temporary intermediate storage.
    ///
    /// Output is a lossless Matroska video (no audio, no NVENC attempt).
    /// Intended for the Maxine-enhanced frame temp file between Pass 1 and Pass 2.
    pub fn new_lossless_temp(
        output: &Path,
        width: usize,
        height: usize,
        fps: f64,
    ) -> Result<Self, FfmpegError> {
        let w_s = width.to_string();
        let h_s = height.to_string();
        let fps_s = format!("{fps:.3}");
        let size_s = format!("{w_s}x{h_s}");
        let out_s = output.to_str().unwrap_or("temp.mkv");

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-s", &size_s,
            "-r", &fps_s,
            "-i", "pipe:0",
            "-map", "0:v",
            "-c:v", "ffv1",
            "-level", "3",
            out_s,
        ]);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(FfmpegError::NotFound)?;
        let stdin = child.stdin.take().ok_or_else(|| {
            FfmpegError::EncodeFailed("could not open lossless encoder stdin".to_string())
        })?;
        let stderr = child.stderr.take();

        if let Ok(Some(status)) = child.try_wait() {
            let msg = stderr.map(|mut s| {
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            }).unwrap_or_default();
            return Err(FfmpegError::EncodeFailed(format!(
                "ffv1 encoder exited immediately (code {:?}): {}",
                status.code(), msg.trim()
            )));
        }

        Ok(Self {
            child,
            stdin,
            stderr,
            frame_bytes: width * height * 3,
        })
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video lossless_temp 2>&1 | tail -20
```

Expected: `lossless_temp_encoder_creates_decodable_file` PASSES.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/ffmpeg.rs
git commit -m "feat(ffmpeg): add FrameEncoder::new_lossless_temp() for ffv1/mkv temp output"
```

---

### Task 7: grade.rs — Full Restructure for Maxine (issue #21)

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

This is the largest task. It restructures grade.rs in three areas:
1. New `--maxine` CLI args
2. Pass 1: full-res decode + enhance() per frame + temp encoder + proxy downscale
3. Pass 2: decode from temp file instead of raw input

- [ ] **Step 1: Add `--maxine` CLI args to `GradeArgs`**

In `GradeArgs` struct (after the `pub verbose: bool` field at line 84), add:

```rust
    /// Enable Maxine AI enhancement preprocessing (requires nvvfx SDK or DOREA_MAXINE_MOCK=1)
    #[arg(long)]
    pub maxine: bool,

    /// Disable Maxine artifact reduction before upscale [default: enabled when --maxine]
    #[arg(long)]
    pub no_maxine_artifact_reduction: bool,

    /// Maxine super-resolution upscale factor [default: 2]
    #[arg(long, default_value = "2")]
    pub maxine_upscale_factor: u32,
```

- [ ] **Step 2: Add a RAII temp file guard struct**

Before `pub fn run(args: GradeArgs)`, add:

```rust
/// RAII guard that deletes a temp file when dropped.
struct TempFileGuard(Option<std::path::PathBuf>);

impl TempFileGuard {
    fn new(path: std::path::PathBuf) -> Self {
        Self(Some(path))
    }
    /// Disarm the guard (don't delete on drop — used if deletion is done explicitly).
    #[allow(dead_code)]
    fn disarm(&mut self) { self.0 = None; }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(ref p) = self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}
```

- [ ] **Step 3: Add Maxine startup validation in `run()`**

At the top of `run()`, after the existing validation checks (after line 140), add:

```rust
    if args.maxine {
        let valid_factors = [2u32, 3, 4];
        if !valid_factors.contains(&args.maxine_upscale_factor) {
            anyhow::bail!(
                "--maxine-upscale-factor {} is not supported. Supported: {:?}",
                args.maxine_upscale_factor, valid_factors,
            );
        }
        log::info!(
            "Maxine enabled: upscale_factor={}, artifact_reduction={}",
            args.maxine_upscale_factor,
            !args.no_maxine_artifact_reduction,
        );
        log::info!(
            "Pass 1 will decode full-resolution (Maxine preprocessing). \
             Estimated temp file: ~{:.1} GB for this video.",
            // rough estimate: ffv1 at ~0.5 bits/px for natural video
            info.width as f64 * info.height as f64 * 3.0
                * info.frame_count as f64 * 0.5 / 8.0 / 1e9,
        );
    }
```

Note: place this block after `info` is available (after `let info = ffmpeg::probe(...)`).

- [ ] **Step 4: Spawn inference server before Pass 1 (when --maxine)**

Before the Pass 1 section (before line 172), add:

```rust
    // When Maxine is enabled: spawn ONE inference server (Maxine + RAUNE + Depth)
    // used for Pass 1 enhance calls and the calibration batch. Shut down after calibration.
    let mut maxine_server: Option<InferenceServer> = if args.maxine {
        let mut cfg = build_inference_config(&args);
        cfg.maxine = true;
        cfg.maxine_upscale_factor = args.maxine_upscale_factor;
        Some(
            InferenceServer::spawn(&cfg)
                .context("failed to spawn Maxine inference server")?,
        )
    } else {
        None
    };
```

Also add the temp file path variable:

```rust
    let maxine_temp_path: Option<std::path::PathBuf> = if args.maxine {
        Some(std::env::temp_dir().join(format!("dorea_maxine_{}.mkv", std::process::id())))
    } else {
        None
    };
    // Guard deletes temp file on drop (even on error / early return).
    let _maxine_temp_guard = maxine_temp_path.as_ref()
        .map(|p| TempFileGuard::new(p.clone()));
```

- [ ] **Step 5: Restructure Pass 1 to handle Maxine**

Replace the existing Pass 1 block (lines 172–213) with:

```rust
    // -----------------------------------------------------------------------
    // Pass 1: decode + optional Maxine enhance + proxy downscale + keyframe detect
    // -----------------------------------------------------------------------
    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(info.width, info.height, args.proxy_size);

    let mut keyframes: Vec<KeyframeEntry> = Vec::new();
    let mut detector: Box<dyn ChangeDetector> = Box::new(MseDetector::default());
    let mut frames_since_kf = 0usize;
    let scene_cut_threshold = args.depth_skip_threshold * 10.0;

    if let Some(ref mut inf_srv) = maxine_server {
        // Maxine path: full-res decode → enhance → write temp → proxy downscale → keyframe detect
        use dorea_video::resize::resize_rgb_bilinear;
        use dorea_video::ffmpeg::FrameEncoder;

        let temp_path = maxine_temp_path.as_ref().unwrap();
        let mut temp_enc = FrameEncoder::new_lossless_temp(
            temp_path, info.width, info.height, info.fps,
        ).context("failed to create Maxine temp encoder")?;

        let full_frames = ffmpeg::decode_frames(&args.input, &info)
            .context("failed to spawn full-res decoder for Maxine pass")?;

        for frame_result in full_frames {
            let frame = frame_result.context("Maxine pass frame decode error")?;

            let maxine_full = inf_srv.enhance(
                &frame.index.to_string(),
                &frame.pixels,
                frame.width,
                frame.height,
                !args.no_maxine_artifact_reduction,
                args.maxine_upscale_factor,
            ).unwrap_or_else(|e| {
                log::warn!("enhance() IPC failed for frame {} — using original: {e}", frame.index);
                frame.pixels.clone()
            });

            temp_enc.write_frame(&maxine_full)
                .context("failed to write Maxine-enhanced frame to temp file")?;

            let maxine_proxy = if proxy_w == frame.width && proxy_h == frame.height {
                maxine_full.clone()
            } else {
                resize_rgb_bilinear(&maxine_full, frame.width, frame.height, proxy_w, proxy_h)
            };

            let change = detector.score(&maxine_proxy);
            let scene_cut = change < f32::MAX && change > scene_cut_threshold;
            let is_keyframe = !interp_enabled
                || keyframes.is_empty()
                || scene_cut
                || frames_since_kf >= args.depth_max_interval
                || (change < f32::MAX && change > args.depth_skip_threshold);

            if is_keyframe {
                if scene_cut {
                    log::info!("Scene cut at frame {} (change={:.6})", frame.index, change);
                    detector.reset();
                }
                keyframes.push(KeyframeEntry {
                    frame_index: frame.index,
                    proxy_pixels: maxine_proxy.clone(),
                    scene_cut_before: scene_cut,
                });
                detector.set_reference(&maxine_proxy);
                frames_since_kf = 0;
            } else {
                frames_since_kf += 1;
            }
        }

        temp_enc.finish().context("failed to finalize Maxine temp file")?;
        log::info!(
            "Maxine Pass 1 complete: {} keyframes, temp file: {}",
            keyframes.len(),
            temp_path.display(),
        );
    } else {
        // No-Maxine path: proxy decode (existing behaviour)
        let proxy_frames = ffmpeg::decode_frames_scaled(&args.input, &info, proxy_w, proxy_h)
            .context("failed to spawn ffmpeg proxy decoder")?;

        for frame_result in proxy_frames {
            let frame = frame_result.context("proxy frame decode error")?;
            let change = detector.score(&frame.pixels);
            let scene_cut = change < f32::MAX && change > scene_cut_threshold;
            let is_keyframe = !interp_enabled
                || keyframes.is_empty()
                || scene_cut
                || frames_since_kf >= args.depth_max_interval
                || (change < f32::MAX && change > args.depth_skip_threshold);

            if is_keyframe {
                if scene_cut {
                    log::info!("Scene cut at frame {} (change={:.6})", frame.index, change);
                    detector.reset();
                }
                keyframes.push(KeyframeEntry {
                    frame_index: frame.index,
                    proxy_pixels: frame.pixels.clone(),
                    scene_cut_before: scene_cut,
                });
                detector.set_reference(&frame.pixels);
                frames_since_kf = 0;
            } else {
                frames_since_kf += 1;
            }
        }
        log::info!("Pass 1 complete: {} keyframes detected", keyframes.len());
    }
```

- [ ] **Step 6: Shut down Maxine server after calibration**

In the calibration section, after the `let _ = inf_server.shutdown();` call for the auto-calibrate path (around line 362), and after the pre-computed calibration path's `let _ = inf_server.shutdown();` (around line 279), add:

After each calibration server shutdown, also shut down the Maxine server:

```rust
    // Shut down Maxine server (VRAM freed for Pass 2 CUDA grading).
    if let Some(mut srv) = maxine_server.take() {
        let _ = srv.shutdown();
        log::info!("Maxine server shut down — VRAM freed for Pass 2");
    }
```

Place this after the calibration server is shut down in both branches (pre-computed and auto-calibrate).

Note: `maxine_server` must be `Option<InferenceServer>` declared with `let mut` so `.take()` works. Ensure it's declared with `let mut maxine_server` in Step 4.

- [ ] **Step 7: Restructure Pass 2 to decode from temp file when Maxine**

Replace the Pass 2 decode source (line 501: `let frames = ffmpeg::decode_frames(&args.input, &info)`) with:

```rust
    // -----------------------------------------------------------------------
    // Pass 2: grade + encode
    // Decode source: Maxine temp file (enhanced frames) or original input (raw).
    // -----------------------------------------------------------------------
    let decode_source = maxine_temp_path.as_deref().unwrap_or(args.input.as_path());
    let frames = ffmpeg::decode_frames(decode_source, &info)
        .context("failed to spawn ffmpeg full-res decoder")?;
```

No other changes to Pass 2 — `frame.pixels` already contains the correct data (Maxine-enhanced from temp file, or raw from original).

- [ ] **Step 8: Build**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo build 2>&1 | tail -30
```

Expected: clean build. Fix any compile errors (missing imports, type mismatches) before proceeding.

Add any missing imports at the top of grade.rs:
```rust
use dorea_video::inference::InferenceServer;
```
(if not already present — check existing imports at line 12).

- [ ] **Step 9: Run full test suite**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "feat(grade): Maxine preprocessing — full-res Pass 1, temp-file reuse in Pass 2"
```

---

### Task 8: Maxine Setup Guide (issue #22)

**Files:**
- Create: `docs/guides/maxine-setup.md`

- [ ] **Step 1: Create the guide**

Create `docs/guides/maxine-setup.md` (in the `repos/dorea/` repo):

```markdown
# NVIDIA Maxine VFX SDK Setup

Optional dependency for AI-enhanced pre-processing. When enabled, Maxine runs on
every input frame at full resolution before RAUNE and depth inference.

## Prerequisites

- NVIDIA GPU: Turing, Ampere, Ada, or Blackwell with Tensor Cores (RTX 3060 ✓)
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

3. Verify installation:

   ```bash
   python -c "import nvvfx; print('nvvfx OK')"
   ```

## Usage

Enable Maxine preprocessing when grading:

```bash
dorea grade --input footage.mp4 --maxine
```

CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--maxine` | off | Enable Maxine preprocessing |
| `--maxine-upscale-factor N` | 2 | Super-resolution factor (2, 3, 4) |
| `--no-maxine-artifact-reduction` | — | Skip artifact reduction step |

## What it does

Maxine runs **before** RAUNE and depth inference, on every frame at full resolution:

1. Artifact reduction at original resolution (if ≤1080p input)
2. 2× AI super-resolution (e.g. 1080p → 4K intermediate)
3. Area downsample back to original resolution

The Maxine-enhanced frames are stored in a **temporary lossless video file** (~2–4 GB for
5-min 1080p) and reused in the grading pass. Maxine runs once per frame, not twice.

All downstream algorithms (RAUNE, Depth Anything, LUT calibration, GPU grading) operate
on Maxine-enhanced frames for consistent quality.

## Disk space

The temp file is created in `$TMPDIR` and deleted automatically after grading.
Estimated size: `width × height × fps × duration_s × 0.5 bits/px`.

| Clip | Resolution | Duration | Approx temp size |
|------|------------|----------|-----------------|
| Short | 1080p | 1 min | ~0.5 GB |
| Medium | 1080p | 5 min | ~2–4 GB |
| Long | 4K | 10 min | ~15–20 GB |

Ensure `$TMPDIR` has sufficient free space before running with `--maxine`.

## VRAM budget

All models (Maxine + RAUNE + Depth) are loaded simultaneously in one Python process
during Pass 1 and calibration (~4.8–5.6 GB peak on RTX 3060 6 GB).

After calibration, the inference server shuts down — Pass 2 grading uses only
the CUDA grader (~300 MB).

If you run out of VRAM:
- Try `--no-maxine-artifact-reduction` to save ~500 MB
- Or disable Maxine entirely (`dorea grade --input footage.mp4` without `--maxine`)

## Known limitations

- **8-bit bottleneck**: Maxine accepts only uint8 input. Full pipeline operates in 8-bit
  color depth when Maxine is enabled (vs 10-bit without Maxine).
- **H.265 source**: Artifact reduction is trained for H.264. Effectiveness on H.265/HEVC
  footage is unvalidated, but expected to reduce common blocking/ringing artifacts.
- **Supported scale factors**: 2, 3, 4 only (`--maxine-upscale-factor`).
  Factors 3 and 4 require more VRAM — validate with `nvidia-smi` on first use.

## CI / Testing without the SDK

Set `DOREA_MAXINE_MOCK=1` to enable passthrough mode (identity transform — no nvvfx needed):

```bash
DOREA_MAXINE_MOCK=1 dorea grade --input test.mp4 --maxine
```

All IPC paths, the temp file creation/deletion lifecycle, and keyframe detection on
Maxine-proxied frames are exercised even in mock mode.
```

- [ ] **Step 2: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add docs/guides/maxine-setup.md
git commit -m "docs: add Maxine VFX SDK setup guide (pre-processing role, temp file)"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `cargo build` succeeds, no warnings
- [ ] `cargo test` all pass
- [ ] `dorea grade --help` shows `--maxine`, `--maxine-upscale-factor`, `--no-maxine-artifact-reduction`
- [ ] `dorea grade --input test.mp4 --maxine --maxine-upscale-factor 5` exits with "not supported" error
- [ ] `DOREA_MAXINE_MOCK=1 dorea grade --input test.mp4 --maxine` completes, temp file created and deleted
- [ ] `dorea grade --input test.mp4` (no `--maxine`) produces same output as before (byte-identical)
- [ ] Python tests pass: `DOREA_MAXINE_MOCK=1 python -m pytest python/tests/ -v`
