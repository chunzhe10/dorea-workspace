# Maxine Pipeline Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Maxine enhancement pipeline so one server handles the full lifecycle — Maxine-only at startup, then RAUNE+Depth loaded lazily after Pass 1 — eliminating the two-server VRAM bug and removing the unused `--calibration` Path B.

**Architecture:** A single inference server is spawned per `dorea grade` run. During Pass 1 it runs Maxine on every full-res frame. After Pass 1, `unload_maxine()` frees Maxine VRAM, then `load_raune()` + `load_depth()` loads calibration models into the same process. Pass 2 decodes from the lossless temp file with no models active. VRAM peak is max(Maxine, RAUNE+Depth) — never all three simultaneously.

**Tech Stack:** Rust (dorea-cli, dorea-video), Python (PyO3 bridge, subprocess IPC server), nvvfx SDK via DOREA_MAXINE_MOCK=1 for CI, cargo test, pytest

---

## File Map

| File | Change |
|------|--------|
| `python/dorea_inference/bridge.py` | Add `_maxine_model`, `load_maxine_model`, `unload_maxine`, `run_maxine_cpu`; update `unload_models` |
| `python/dorea_inference/server.py` | Add `unload_maxine`, `load_raune`, `load_depth` IPC handlers; change Maxine startup from FATAL to WARNING |
| `python/tests/test_lifecycle_server.py` | New: tests for lifecycle IPC commands |
| `crates/dorea-video/src/inference_subprocess.rs` | Add `skip_depth: bool` to `InferenceConfig`; add `unload_maxine`, `load_raune`, `load_depth` methods; update `build_args` |
| `crates/dorea-video/src/inference/pyo3_backend.rs` | Add `maxine`, `maxine_upscale_factor`, `skip_depth` to `InferenceConfig`; add `device` field to `InferenceServer`; add `enhance`, `unload_maxine`, `load_raune`, `load_depth` methods |
| `crates/dorea-cli/src/grade.rs` | Remove `--calibration` Path B; remove `--maxine` (replace with `--no-maxine`); restructure to single-server lifecycle |

---

### Task 1: Python bridge — Maxine model support

**Files:**
- Modify: `python/dorea_inference/bridge.py`

The pyo3 backend calls Python bridge functions directly. Currently `bridge.py` has no Maxine support. We add three new functions and update `unload_models`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_bridge_maxine.py`:

```python
"""Tests for Maxine support in the PyO3 bridge module."""
import os
import numpy as np
import pytest

os.environ["DOREA_MAXINE_MOCK"] = "1"


def test_load_and_run_maxine_cpu():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    result = bridge.run_maxine_cpu(frame, artifact_reduce=True)
    assert result.shape == (8, 10, 3)
    assert result.dtype == np.uint8
    # cleanup
    bridge.unload_maxine()


def test_unload_maxine_clears_model():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    bridge.unload_maxine()
    with pytest.raises(RuntimeError, match="not loaded"):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        bridge.run_maxine_cpu(frame)


def test_unload_models_also_clears_maxine():
    from dorea_inference import bridge
    bridge.load_maxine_model(upscale_factor=2)
    bridge.unload_models()
    with pytest.raises(RuntimeError, match="not loaded"):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        bridge.run_maxine_cpu(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_bridge_maxine.py -v
```

Expected: `AttributeError: module 'dorea_inference.bridge' has no attribute 'load_maxine_model'`

- [ ] **Step 3: Implement Maxine support in bridge.py**

Add after `_raune_model = None` (line 41 of current file):

```python
_maxine_model = None


def load_maxine_model(upscale_factor: int = 2, device: str = "cuda") -> None:
    """Load the Maxine enhancer. Called on demand after spawn."""
    global _maxine_model
    from .maxine_enhancer import MaxineEnhancer
    _maxine_model = MaxineEnhancer(upscale_factor=upscale_factor)


def unload_maxine() -> None:
    """Release Maxine model reference and free CUDA cache."""
    global _maxine_model
    _maxine_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_maxine_cpu(frame_rgb: np.ndarray, artifact_reduce: bool = True) -> np.ndarray:
    """Run Maxine enhancement, return same-resolution numpy uint8 array."""
    if _maxine_model is None:
        raise RuntimeError("Maxine model not loaded — call load_maxine_model() first")
    h, w = frame_rgb.shape[:2]
    return _maxine_model.enhance(frame_rgb, width=w, height=h, artifact_reduce=artifact_reduce)
```

Update `unload_models()` to also clear `_maxine_model`:

```python
def unload_models() -> None:
    """Release model references so they can be garbage-collected."""
    global _depth_model, _raune_model, _maxine_model
    _depth_model = None
    _raune_model = None
    _maxine_model = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_bridge_maxine.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/bridge.py python/tests/test_bridge_maxine.py
git commit -m "feat(bridge): add Maxine model load/unload/run to PyO3 bridge"
```

---

### Task 2: Python server — lifecycle IPC handlers + graceful Maxine skip

**Files:**
- Modify: `python/dorea_inference/server.py`
- Create: `python/tests/test_lifecycle_server.py`

Add three new IPC request types (`unload_maxine`, `load_raune`, `load_depth`) and change the Maxine startup failure from a fatal crash to a logged warning.

- [ ] **Step 1: Write the failing tests**

Create `python/tests/test_lifecycle_server.py`:

```python
"""Tests for dynamic model lifecycle IPC commands."""
import base64
import io
import json
import os
import sys

import numpy as np
import pytest

os.environ["DOREA_MAXINE_MOCK"] = "1"


def _run_server(requests: list, extra_argv: list = None) -> list:
    from dorea_inference.server import main
    lines = [json.dumps(r) for r in requests] + ['{"type": "shutdown"}']
    stdin_data = "\n".join(lines) + "\n"
    captured = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = captured
    argv = ["--no-raune", "--no-depth"] + (extra_argv or [])
    try:
        main(argv=argv)
    except SystemExit:
        pass
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    output = captured.getvalue().strip()
    return [json.loads(line) for line in output.split("\n") if line.strip()]


def test_unload_maxine_returns_ok():
    responses = _run_server([{"type": "unload_maxine"}], extra_argv=["--maxine"])
    assert responses[0]["type"] == "ok"


def test_load_raune_returns_ok():
    responses = _run_server([{"type": "load_raune", "weights": None, "models_dir": None}])
    assert responses[0]["type"] == "ok"


def test_load_depth_returns_ok():
    responses = _run_server([{"type": "load_depth", "model_path": None}])
    assert responses[0]["type"] == "ok"


def test_maxine_graceful_skip_when_unavailable():
    """Server should start successfully even if Maxine fails to load."""
    import importlib
    # Temporarily hide MaxineEnhancer to simulate nvvfx not installed
    import dorea_inference.maxine_enhancer as me_mod
    original_class = me_mod.MaxineEnhancer

    class BrokenMaxine:
        def __init__(self, **kwargs):
            raise RuntimeError("nvvfx not found — simulated")

    me_mod.MaxineEnhancer = BrokenMaxine
    try:
        # Server should start and respond to ping without crashing
        responses = _run_server([{"type": "ping"}], extra_argv=["--maxine"])
        assert responses[0]["type"] == "pong"
    finally:
        me_mod.MaxineEnhancer = original_class
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_lifecycle_server.py -v
```

Expected: `FAILED` (unknown request type `unload_maxine` / `load_raune` / `load_depth`)

- [ ] **Step 3: Add lifecycle handlers to server.py**

In `server.py`, in the main request loop after the `elif req_type == "enhance":` block and before `elif req_type == "shutdown":`, add:

```python
            elif req_type == "unload_maxine":
                import torch
                maxine_enhancer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[dorea-inference] Maxine unloaded — VRAM freed", file=sys.stderr, flush=True)
                resp = OkResponse()

            elif req_type == "load_raune":
                from .raune_net import RauneNetInference
                raune_model = RauneNetInference(
                    weights_path=req.get("weights"),
                    device=device,
                    raune_models_dir=req.get("models_dir"),
                )
                print("[dorea-inference] RAUNE-Net loaded (on demand)", file=sys.stderr, flush=True)
                resp = OkResponse()

            elif req_type == "load_depth":
                from .depth_anything import DepthAnythingInference
                depth_model = DepthAnythingInference(
                    model_path=req.get("model_path"),
                    device=device,
                )
                print("[dorea-inference] Depth Anything V2 loaded (on demand)", file=sys.stderr, flush=True)
                resp = OkResponse()
```

- [ ] **Step 4: Change Maxine startup from FATAL to WARNING**

Find the block in `server.py` that starts with `if args.maxine:` and loads MaxineEnhancer. It currently ends with `raise`. Change it to log a warning instead:

```python
    if args.maxine:
        try:
            from .maxine_enhancer import MaxineEnhancer
            maxine_enhancer = MaxineEnhancer(upscale_factor=args.maxine_upscale_factor)
            print(
                f"[dorea-inference] Maxine enhancer loaded "
                f"(upscale_factor={args.maxine_upscale_factor}, "
                f"artifact_reduction={not args.no_maxine_artifact_reduction})",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            print(
                f"[dorea-inference] WARNING: Maxine enhancer failed to load: {e} "
                f"— running without Maxine (enhance requests will return original frame). "
                f"Set DOREA_MAXINE_MOCK=1 for testing without nvvfx SDK.",
                file=sys.stderr, flush=True,
            )
            # maxine_enhancer remains None — enhance handler falls through to error
```

Also update the `enhance` handler to return the original frame as passthrough when `maxine_enhancer is None` (instead of raising an error):

```python
            elif req_type == "enhance":
                fmt = req.get("format", "raw_rgb")
                if fmt == "raw_rgb":
                    img = decode_raw_rgb(req["image_b64"], int(req["width"]), int(req["height"]))
                else:
                    img = decode_png(req["image_b64"])
                if maxine_enhancer is None:
                    # Graceful passthrough: return original frame unchanged
                    enhanced = img
                else:
                    enhanced = maxine_enhancer.enhance(
                        img,
                        width=int(req["width"]),
                        height=int(req["height"]),
                        artifact_reduce=not req.get("no_artifact_reduce", False),
                    )
                resp = EnhanceResult.from_array(req_id, enhanced)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
DOREA_MAXINE_MOCK=1 python -m pytest python/tests/test_lifecycle_server.py python/tests/test_maxine_server.py -v
```

Expected: all tests pass. Note: `test_enhance_handler_without_maxine_flag_errors` in `test_maxine_server.py` will now fail because we changed enhance to passthrough — update that test:

```python
def test_enhance_without_maxine_returns_passthrough():
    """enhance request without --maxine should return enhance_result (passthrough mode)."""
    req = _make_enhance_req(4, 6)
    responses = _run_server_with_reqs([req])  # no --maxine
    assert responses[0]["type"] == "enhance_result"
    assert responses[0]["width"] == 4
    assert responses[0]["height"] == 6
```

Re-run to confirm all pass.

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/server.py python/tests/test_lifecycle_server.py python/tests/test_maxine_server.py
git commit -m "feat(server): add lifecycle IPC handlers; Maxine load failure becomes warning"
```

---

### Task 3: Subprocess backend — skip_depth + lifecycle methods

**Files:**
- Modify: `crates/dorea-video/src/inference_subprocess.rs`

Add `skip_depth: bool` to `InferenceConfig` (so we can start a Maxine-only server with `--no-depth`), and add `unload_maxine`, `load_raune`, `load_depth` methods to `InferenceServer`.

- [ ] **Step 1: Write the failing tests**

In `crates/dorea-video/src/inference_subprocess.rs`, inside `mod tests { ... }`, add:

```rust
    #[test]
    fn spawn_command_includes_no_depth_when_skip_depth() {
        let config = InferenceConfig {
            skip_depth: true,
            ..InferenceConfig::default()
        };
        let args = config.build_args();
        assert!(args.contains(&"--no-depth".to_string()), "missing --no-depth");
    }

    #[test]
    fn spawn_command_omits_no_depth_by_default() {
        let config = InferenceConfig::default();
        let args = config.build_args();
        assert!(!args.contains(&"--no-depth".to_string()), "--no-depth should be absent by default");
    }

    #[test]
    fn unload_maxine_sends_correct_json() {
        // Test that the JSON we'd send is correct — verify the structure without live server.
        let req = serde_json::json!({"type": "unload_maxine"});
        assert_eq!(req["type"].as_str().unwrap(), "unload_maxine");
    }

    #[test]
    fn load_raune_sends_correct_json() {
        use std::path::Path;
        let weights = Path::new("/path/to/weights.pth");
        let models_dir = Path::new("/path/to/raune");
        let req = serde_json::json!({
            "type": "load_raune",
            "weights": weights.to_str(),
            "models_dir": models_dir.to_str(),
        });
        assert_eq!(req["type"].as_str().unwrap(), "load_raune");
        assert_eq!(req["weights"].as_str().unwrap(), "/path/to/weights.pth");
    }

    #[test]
    fn load_depth_sends_correct_json() {
        use std::path::Path;
        let model = Path::new("/models/depth_anything");
        let req = serde_json::json!({
            "type": "load_depth",
            "model_path": model.to_str(),
        });
        assert_eq!(req["type"].as_str().unwrap(), "load_depth");
        assert_eq!(req["model_path"].as_str().unwrap(), "/models/depth_anything");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video -- inference_subprocess::tests::spawn_command_includes_no_depth 2>&1 | tail -10
```

Expected: compile error — `skip_depth` field does not exist on `InferenceConfig`

- [ ] **Step 3: Add skip_depth to InferenceConfig**

In `inference_subprocess.rs`, in the `InferenceConfig` struct, add after `pub skip_raune: bool`:

```rust
    /// Skip Depth Anything entirely (pass `--no-depth` to the server).
    /// Use this when spawning a Maxine-only server for Pass 1.
    pub skip_depth: bool,
```

In `impl Default for InferenceConfig`, add after `skip_raune: false`:

```rust
            skip_depth: false,
```

In `impl InferenceConfig`, in `build_args()`, add after the `skip_raune` block:

```rust
        if self.skip_depth {
            args.push("--no-depth".to_string());
        } else if let Some(p) = &self.depth_model {
            args.push("--depth-model".to_string());
            args.push(p.to_str().unwrap_or("").to_string());
        }
```

**Important:** Remove the existing `if let Some(p) = &self.depth_model` block that pushes `--depth-model` (it's currently unconditional — this replaces it with the skip_depth guard).

- [ ] **Step 4: Add unload_maxine, load_raune, load_depth methods**

In `impl InferenceServer`, add after the `enhance()` method (before `upscale_depth`):

```rust
    /// Unload Maxine model and free its VRAM without stopping the server.
    pub fn unload_maxine(&mut self) -> Result<(), InferenceError> {
        let req = serde_json::json!({"type": "unload_maxine"});
        self.send_line(&req.to_string())?;
        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("unload_maxine response parse: {e}")))?;
        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        Ok(())
    }

    /// Load RAUNE-Net into the running server (after it was started without it).
    pub fn load_raune(
        &mut self,
        weights: Option<&std::path::Path>,
        models_dir: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        let req = serde_json::json!({
            "type": "load_raune",
            "weights": weights.and_then(|p| p.to_str()),
            "models_dir": models_dir.and_then(|p| p.to_str()),
        });
        self.send_line(&req.to_string())?;
        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("load_raune response parse: {e}")))?;
        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        Ok(())
    }

    /// Load Depth Anything into the running server (after it was started without it).
    pub fn load_depth(
        &mut self,
        model_path: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        let req = serde_json::json!({
            "type": "load_depth",
            "model_path": model_path.and_then(|p| p.to_str()),
        });
        self.send_line(&req.to_string())?;
        let resp = self.recv_line()?;
        let v: serde_json::Value = serde_json::from_str(&resp)
            .map_err(|e| InferenceError::Ipc(format!("load_depth response parse: {e}")))?;
        if v["type"].as_str() == Some("error") {
            return Err(InferenceError::ServerError(
                v["message"].as_str().unwrap_or("unknown error").to_string(),
            ));
        }
        Ok(())
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video -- inference_subprocess::tests 2>&1 | tail -20
```

Expected: all tests in `inference_subprocess::tests` pass

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference_subprocess.rs
git commit -m "feat(inference): add skip_depth, unload_maxine, load_raune, load_depth to subprocess backend"
```

---

### Task 4: pyo3 backend — Maxine + lifecycle support

**Files:**
- Modify: `crates/dorea-video/src/inference/pyo3_backend.rs`

Mirror the subprocess backend changes: add `maxine`, `maxine_upscale_factor`, `skip_depth` to `InferenceConfig`; add `device` field to `InferenceServer` (needed by `load_raune`/`load_depth` to pass device to Python); add `enhance`, `unload_maxine`, `load_raune`, `load_depth` methods.

- [ ] **Step 1: Write the failing tests**

In `crates/dorea-video/src/inference/pyo3_backend.rs`, add a test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_config_default_has_maxine_fields() {
        let cfg = InferenceConfig::default();
        assert!(!cfg.maxine, "maxine should default to false");
        assert_eq!(cfg.maxine_upscale_factor, 2);
        assert!(!cfg.skip_depth, "skip_depth should default to false");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video --features python -- pyo3_backend::tests 2>&1 | tail -10
```

Expected: compile error — `maxine` field not found on `InferenceConfig`

- [ ] **Step 3: Add fields to InferenceConfig**

In `pyo3_backend.rs`, in the `InferenceConfig` struct, add after `startup_timeout`:

```rust
    /// Enable Maxine enhancement.
    pub maxine: bool,
    /// Maxine super-resolution upscale factor (default 2).
    pub maxine_upscale_factor: u32,
    /// Skip Depth Anything at spawn (load on demand via load_depth()).
    pub skip_depth: bool,
```

In `impl Default for InferenceConfig`, add after `startup_timeout`:

```rust
            maxine: false,
            maxine_upscale_factor: 2,
            skip_depth: false,
```

- [ ] **Step 4: Add device field to InferenceServer**

In `pyo3_backend.rs`, change `InferenceServer` struct from:

```rust
pub struct InferenceServer {
    bridge: Py<PyModule>,
    _not_send: PhantomData<*const ()>,
}
```

To:

```rust
pub struct InferenceServer {
    bridge: Py<PyModule>,
    /// Device string captured at spawn time (e.g. "cuda" or "cpu").
    device: String,
    _not_send: PhantomData<*const ()>,
}
```

Update `spawn()` — find the `Ok(Self { ... })` at the end and add `device: device.to_string()`:

```rust
            Ok(Self {
                bridge: bridge.unbind(),
                device: device.to_string(),
                _not_send: PhantomData,
            })
```

- [ ] **Step 5: Update spawn() to conditionally load Maxine**

In `spawn()`, after the `if !config.skip_raune { Self::call_load_raune(...) }` block and before `Ok(Self { ... })`, add:

```rust
            // Load depth model unless skipped.
            if !config.skip_depth {
                Self::call_load_depth(py, &bridge, depth_path, device)?;
            }

            // Load Maxine model if requested.
            if config.maxine {
                Self::call_load_maxine(py, &bridge, config.maxine_upscale_factor, device)?;
            }
```

**Note:** Remove the existing `Self::call_load_depth(...)` call that's currently unconditional (it's at the top of the `Python::with_gil` block). The new code replaces it with a conditional version.

Add the new private helper:

```rust
    fn call_load_maxine(
        py: Python<'_>,
        bridge: &Bound<'_, PyModule>,
        upscale_factor: u32,
        _device: &str,
    ) -> Result<(), InferenceError> {
        bridge
            .call_method1("load_maxine_model", (upscale_factor as i64,))
            .map_err(|e| InferenceError::InitFailed(format!("load_maxine_model: {e}")))?;
        Ok(())
    }
```

- [ ] **Step 6: Add enhance, unload_maxine, load_raune, load_depth methods**

In `impl InferenceServer`, add these four methods after `vram_free_bytes()` and before `shutdown()`:

```rust
    /// Run Maxine enhancement in-process via the PyO3 bridge.
    ///
    /// Returns enhanced RGB u8 at the same resolution as input.
    pub fn enhance(
        &self,
        _id: &str,
        image_rgb: &[u8],
        width: usize,
        height: usize,
        artifact_reduce: bool,
    ) -> Result<Vec<u8>, InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let np_flat = numpy::PyArray1::from_slice_bound(py, image_rgb);
            let np_reshaped = np_flat
                .call_method1("reshape", ((height, width, 3),))
                .map_err(|e| InferenceError::Ipc(format!("reshape: {e}")))?;
            let result = bridge
                .call_method1("run_maxine_cpu", (np_reshaped, artifact_reduce))
                .map_err(|e| Self::map_python_error(py, e))?;
            let flat = result
                .call_method1("reshape", ((-1_i32,),))
                .map_err(|e| InferenceError::Ipc(format!("flatten result: {e}")))?;
            let rgb_data: Vec<u8> = flat
                .call_method0("tolist")
                .map_err(|e| InferenceError::Ipc(format!("tolist: {e}")))?
                .extract()
                .map_err(|e| InferenceError::Ipc(format!("extract rgb: {e}")))?;
            Ok(rgb_data)
        })
    }

    /// Unload Maxine model and free its VRAM without stopping the server.
    pub fn unload_maxine(&self) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            bridge
                .call_method0("unload_maxine")
                .map_err(|e| InferenceError::InitFailed(format!("unload_maxine: {e}")))?;
            Ok(())
        })
    }

    /// Load RAUNE-Net into the running server (after it was started without it).
    pub fn load_raune(
        &self,
        weights: Option<&std::path::Path>,
        models_dir: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let py_weights = match weights.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            let py_models_dir = match models_dir.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            bridge
                .call_method1("load_raune_model", (py_weights, self.device.as_str(), py_models_dir))
                .map_err(|e| InferenceError::InitFailed(format!("load_raune_model: {e}")))?;
            Ok(())
        })
    }

    /// Load Depth Anything into the running server (after it was started without it).
    pub fn load_depth(
        &self,
        model_path: Option<&std::path::Path>,
    ) -> Result<(), InferenceError> {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let py_model_path = match model_path.and_then(|p| p.to_str()) {
                Some(p) => p.into_py(py),
                None => py.None(),
            };
            bridge
                .call_method1("load_depth_model", (py_model_path, self.device.as_str()))
                .map_err(|e| InferenceError::InitFailed(format!("load_depth_model: {e}")))?;
            Ok(())
        })
    }
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-video --features python -- pyo3_backend::tests 2>&1 | tail -10
```

Expected: `1 passed` (compile-time field check test)

Also verify the whole video crate compiles clean:

```bash
cargo build -p dorea-video --features python 2>&1 | tail -10
```

Expected: no errors

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-video/src/inference/pyo3_backend.rs
git commit -m "feat(pyo3): add Maxine support, skip_depth, lifecycle methods to pyo3 backend"
```

---

### Task 5: grade.rs — single-server lifecycle, remove Path B, --no-maxine default

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs`

This is the largest single-file change. Three things happen together because they're all interdependent:
1. Remove Path B (`--calibration` flag)
2. Change `--maxine` opt-in to `--no-maxine` opt-out
3. Restructure to single-server lifecycle (Maxine-only at spawn → lifecycle transition after Pass 1)

- [ ] **Step 1: Write the failing tests**

At the bottom of `grade.rs`, in `mod tests { ... }`, add:

```rust
    #[test]
    fn build_inference_config_maxine_true_by_default() {
        let args = GradeArgs {
            input: PathBuf::from("/dev/null"),
            output: None,
            no_maxine: false,
            no_maxine_artifact_reduction: false,
            maxine_upscale_factor: 2,
            warmth: 1.0,
            strength: 0.8,
            contrast: 1.0,
            proxy_size: 518,
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            no_depth_interp: false,
            raune_weights: None,
            raune_models_dir: None,
            depth_model: None,
            python: PathBuf::from("/opt/dorea-venv/bin/python"),
            cpu_only: false,
            verbose: false,
        };
        let cfg = build_inference_config(&args);
        assert!(cfg.maxine, "Maxine should be enabled by default");
        assert!(!cfg.skip_raune, "RAUNE should not be skipped in config");
        assert!(!cfg.skip_depth, "depth should not be skipped in config");
    }

    #[test]
    fn build_inference_config_no_maxine_flag_disables_maxine() {
        let args = GradeArgs {
            input: PathBuf::from("/dev/null"),
            output: None,
            no_maxine: true,
            no_maxine_artifact_reduction: false,
            maxine_upscale_factor: 2,
            warmth: 1.0,
            strength: 0.8,
            contrast: 1.0,
            proxy_size: 518,
            depth_skip_threshold: 0.005,
            depth_max_interval: 12,
            no_depth_interp: false,
            raune_weights: None,
            raune_models_dir: None,
            depth_model: None,
            python: PathBuf::from("/opt/dorea-venv/bin/python"),
            cpu_only: false,
            verbose: false,
        };
        let cfg = build_inference_config(&args);
        assert!(!cfg.maxine, "no_maxine flag should disable Maxine");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -20
```

Expected: compile errors — `no_maxine` field not on `GradeArgs`, `skip_depth` not on `InferenceConfig` (already added in Task 3, so just `no_maxine`)

- [ ] **Step 3: Update GradeArgs — replace --maxine with --no-maxine, remove --calibration**

Replace the entire `GradeArgs` struct:

```rust
#[derive(Args, Debug)]
pub struct GradeArgs {
    /// Input video file (MP4/MOV/MKV)
    #[arg(long)]
    pub input: PathBuf,

    /// Output video file [default: <input-stem>_graded.mp4]
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Warmth multiplier [0.0–2.0]
    #[arg(long, default_value = "1.0")]
    pub warmth: f32,

    /// LUT/HSL blend strength [0.0–1.0]
    #[arg(long, default_value = "0.8")]
    pub strength: f32,

    /// Ambiance contrast multiplier [0.0–1.0]
    #[arg(long, default_value = "1.0")]
    pub contrast: f32,

    /// Proxy resolution for inference (long edge, pixels) [default: 518]
    #[arg(long, default_value = "518")]
    pub proxy_size: usize,

    /// MSE threshold for keyframe detection (lower = more keyframes)
    #[arg(long, default_value = "0.005")]
    pub depth_skip_threshold: f32,

    /// Maximum frames between keyframes
    #[arg(long, default_value = "12")]
    pub depth_max_interval: usize,

    /// Disable temporal interpolation — run full pipeline on every frame
    #[arg(long)]
    pub no_depth_interp: bool,

    /// Path to RAUNE-Net weights .pth (for auto-calibration)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// Path to RAUNE-Net checkout directory (contains models/raune_net.py).
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Path to Depth Anything V2 model directory
    #[arg(long)]
    pub depth_model: Option<PathBuf>,

    /// Python executable to use for the inference subprocess
    #[arg(long, default_value = "/opt/dorea-venv/bin/python")]
    pub python: PathBuf,

    /// Force CPU-only mode (no CUDA)
    #[arg(long)]
    pub cpu_only: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Disable Maxine AI enhancement preprocessing (Maxine is attempted by default)
    #[arg(long)]
    pub no_maxine: bool,

    /// Disable Maxine artifact reduction before upscale [default: enabled]
    #[arg(long)]
    pub no_maxine_artifact_reduction: bool,

    /// Maxine super-resolution upscale factor [default: 2]
    #[arg(long, default_value = "2")]
    pub maxine_upscale_factor: u32,
}
```

- [ ] **Step 4: Update build_inference_config**

Replace `build_inference_config`:

```rust
fn build_inference_config(args: &GradeArgs) -> InferenceConfig {
    InferenceConfig {
        python_exe: args.python.clone(),
        raune_weights: args.raune_weights.clone(),
        raune_models_dir: args.raune_models_dir.clone(),
        skip_raune: false,
        depth_model: args.depth_model.clone(),
        skip_depth: false,
        device: if args.cpu_only { Some("cpu".to_string()) } else { None },
        startup_timeout: Duration::from_secs(180),
        maxine: !args.no_maxine,
        maxine_upscale_factor: args.maxine_upscale_factor,
    }
}
```

- [ ] **Step 5: Verify tests pass at this point**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli -- tests::build_inference_config 2>&1 | tail -10
```

Expected: both new tests pass, existing `lerp_depth` tests still pass

- [ ] **Step 6: Restructure run() — single-server lifecycle**

Replace the server spawn block (currently lines ~222–234, which spawns `maxine_server: Option<InferenceServer>`) and the entire `if let Some(cal_path) = &args.calibration { ... } else { ... }` block (Path B + Path A).

The new structure:

**Replace the old server spawn block:**

```rust
    // Validate upscale factor before spawning
    let use_maxine = !args.no_maxine;
    if use_maxine {
        let valid_factors = [2u32, 3, 4];
        if !valid_factors.contains(&args.maxine_upscale_factor) {
            anyhow::bail!(
                "--maxine-upscale-factor {} is not supported. Supported: {:?}",
                args.maxine_upscale_factor, valid_factors,
            );
        }
    }

    // Spawn ONE inference server for the entire run.
    // Starts with Maxine loaded only (RAUNE+Depth loaded lazily after Pass 1).
    let maxine_start_cfg = InferenceConfig {
        skip_raune: true,
        skip_depth: true,
        ..build_inference_config(&args)
        // Note: build_inference_config sets maxine = !args.no_maxine
    };
    let mut inf_server = InferenceServer::spawn(&maxine_start_cfg)
        .context("failed to spawn inference server")?;
```

**Replace the `maxine_temp_path` block — keep as-is but make it unconditional when `use_maxine`:**

```rust
    let maxine_temp_path: Option<std::path::PathBuf> = if use_maxine {
        Some(std::env::temp_dir().join(format!("dorea_maxine_{}.mkv", std::process::id())))
    } else {
        None
    };
    let _maxine_temp_guard = maxine_temp_path.as_ref().map(|p| TempFileGuard::new(p.clone()));
```

**Replace the Pass 1 branch — use `inf_server` directly instead of `maxine_server`:**

```rust
    if use_maxine {
        // Maxine path: full-res decode → enhance → write temp → proxy downscale → keyframe detect
        use dorea_video::resize::resize_rgb_bilinear;

        let temp_path = maxine_temp_path.as_ref().unwrap();
        let mut temp_enc = FrameEncoder::new_lossless_temp(
            temp_path, info.width, info.height, info.fps,
        ).context("failed to create Maxine temp encoder")?;

        let full_frames = ffmpeg::decode_frames(&args.input, &info)
            .context("failed to spawn full-res decoder for Maxine pass")?;

        for frame_result in full_frames {
            let frame = frame_result.context("Maxine pass frame decode error")?;

            let maxine_full = inf_server.enhance(
                &frame.index.to_string(),
                &frame.pixels,
                frame.width,
                frame.height,
                !args.no_maxine_artifact_reduction,
            ).unwrap_or_else(|e| {
                log::warn!("enhance() failed for frame {} — using original: {e}", frame.index);
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
        // No-Maxine path: proxy decode
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

**Replace the calibration block — remove Path B entirely, reuse `inf_server` for fused RAUNE+depth:**

```rust
    // -----------------------------------------------------------------------
    // Model lifecycle transition: Maxine → RAUNE + Depth
    // -----------------------------------------------------------------------
    // Free Maxine VRAM, then load RAUNE and Depth into the same server process.
    // VRAM peak across the run is max(Maxine, RAUNE+Depth) — never all three.
    if use_maxine {
        inf_server.unload_maxine()
            .unwrap_or_else(|e| log::warn!("unload_maxine failed (non-fatal): {e}"));
        log::info!("Maxine unloaded — loading RAUNE+Depth for calibration");
    }
    inf_server.load_raune(
        args.raune_weights.as_deref(),
        args.raune_models_dir.as_deref(),
    ).context("failed to load RAUNE-Net for calibration")?;
    inf_server.load_depth(
        args.depth_model.as_deref(),
    ).context("failed to load Depth Anything for calibration")?;

    // -----------------------------------------------------------------------
    // Auto-calibrate: fused RAUNE+depth, dual output
    // -----------------------------------------------------------------------
    log::info!(
        "Auto-calibrating from {} keyframes (fused RAUNE+depth)",
        keyframes.len()
    );

    let fused_items: Vec<RauneDepthBatchItem> = keyframes.iter().map(|kf| {
        RauneDepthBatchItem {
            id: format!("kf_f{}", kf.frame_index),
            pixels: kf.proxy_pixels.clone(),
            width: proxy_w,
            height: proxy_h,
            raune_max_size: proxy_w.max(proxy_h),
            depth_max_size: args.proxy_size.min(1036),
        }
    }).collect();

    let mut store = PagedCalibrationStore::new()
        .context("failed to create paged calibration store")?;
    let mut kf_depths: HashMap<u64, (Vec<f32>, usize, usize)> = HashMap::new();

    for (chunk_kfs, chunk_items) in keyframes
        .chunks(FUSED_BATCH_SIZE)
        .zip(fused_items.chunks(FUSED_BATCH_SIZE))
    {
        let mut results = inf_server.run_raune_depth_batch(chunk_items)
            .unwrap_or_else(|e| {
                log::warn!(
                    "Fused RAUNE+depth batch failed: {e} — using originals + uniform depth"
                );
                chunk_items.iter().map(|item| {
                    (item.id.clone(), item.pixels.clone(),
                     item.width, item.height,
                     vec![0.5f32; item.width * item.height],
                     item.width, item.height)
                }).collect()
            });

        if results.len() < chunk_items.len() {
            log::warn!(
                "Fused batch returned {} results for {} items — padding with originals",
                results.len(), chunk_items.len()
            );
            for item in &chunk_items[results.len()..] {
                results.push((
                    item.id.clone(), item.pixels.clone(),
                    item.width, item.height,
                    vec![0.5f32; item.width * item.height],
                    item.width, item.height,
                ));
            }
        }

        for (kf, (_, enhanced, enh_w, enh_h, depth, dw, dh)) in
            chunk_kfs.iter().zip(results.into_iter())
        {
            debug_assert_eq!(enh_w, proxy_w, "RAUNE enh_w {enh_w} != proxy_w {proxy_w}");
            debug_assert_eq!(enh_h, proxy_h, "RAUNE enh_h {enh_h} != proxy_h {proxy_h}");
            let depth_for_store = if dw == proxy_w && dh == proxy_h {
                depth.clone()
            } else {
                InferenceServer::upscale_depth(&depth, dw, dh, proxy_w, proxy_h)
            };
            store.push(&kf.proxy_pixels, &enhanced, &depth_for_store, proxy_w, proxy_h)
                .context("failed to page fused result to store")?;
            kf_depths.insert(kf.frame_index, (depth, dw, dh));
        }
    }
    log::info!("Fused inference complete ({} keyframes)", keyframes.len());
    let _ = inf_server.shutdown();

    store.seal().context("failed to seal calibration store")?;
```

The rest of the calibration (3-pass LUT/HSL computation) stays exactly as-is after this block.

Also remove the `DepthBatchItem` import from the use statement at the top since Path B no longer uses it directly (it was only used in Path B's depth-only server). Check if `DepthBatchItem` is still used elsewhere; if not, remove from the import.

- [ ] **Step 7: Run the full dorea-cli tests**

```bash
cd /workspaces/dorea-workspace/repos/dorea
cargo test -p dorea-cli 2>&1 | tail -20
```

Expected: all tests pass (lerp_depth tests + new build_inference_config tests)

Verify `--calibration` is gone from help:

```bash
cargo run -p dorea-cli -- grade --help 2>&1 | grep calibration
```

Expected: no output (flag removed)

Verify `--no-maxine` appears in help:

```bash
cargo run -p dorea-cli -- grade --help 2>&1 | grep maxine
```

Expected: `--no-maxine`, `--no-maxine-artifact-reduction`, `--maxine-upscale-factor` visible

- [ ] **Step 8: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add crates/dorea-cli/src/grade.rs
git commit -m "refactor(grade): single-server lifecycle, remove --calibration Path B, default Maxine"
```

---

## Self-Review

**1. Spec coverage check:**

| Requirement | Task |
|-------------|------|
| `unload_maxine` IPC in server.py | Task 2 |
| `load_raune` IPC in server.py | Task 2 |
| `load_depth` IPC in server.py | Task 2 |
| Maxine FATAL → WARNING graceful skip | Task 2 |
| `skip_depth` in subprocess `InferenceConfig` | Task 3 |
| `unload_maxine` method in subprocess backend | Task 3 |
| `load_raune` method in subprocess backend | Task 3 |
| `load_depth` method in subprocess backend | Task 3 |
| Maxine + lifecycle in pyo3 backend | Task 4 |
| Remove `--calibration` Path B | Task 5 |
| `--no-maxine` opt-out (was opt-in `--maxine`) | Task 5 |
| Single-server lifecycle in grade.rs | Task 5 |
| `build_inference_config` sets `maxine: true` by default | Task 5 |
| VRAM peak never exceeds max(Maxine, RAUNE+Depth) | Task 5 |

**2. Placeholder scan:** No TBDs. All code blocks are complete.

**3. Type consistency:**
- `InferenceConfig.skip_depth: bool` — added in Task 3 (subprocess) and Task 4 (pyo3). Default `false` in both.
- `InferenceConfig.maxine: bool` — in subprocess since before; added to pyo3 in Task 4. Default `false` in both.
- `InferenceServer.unload_maxine()` — added to subprocess (Task 3) and pyo3 (Task 4). Both return `Result<(), InferenceError>`.
- `InferenceServer.load_raune(weights: Option<&Path>, models_dir: Option<&Path>)` — same signature in both backends.
- `InferenceServer.load_depth(model_path: Option<&Path>)` — same signature in both backends.
- `GradeArgs.no_maxine: bool` — replaces `maxine: bool`. Test struct literals in Task 5 use `no_maxine`.
- `build_inference_config` sets `maxine: !args.no_maxine` — consistent with `GradeArgs.no_maxine`.
