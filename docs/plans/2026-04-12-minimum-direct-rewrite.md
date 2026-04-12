# Minimum-Direct Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip `repos/dorea` down to a single-command direct-mode binary — delete 5 legacy Rust crates, 8 Python modules, 5 Python test files, and all LUT-pipeline code — while bumping direct-mode defaults to 1440p proxy / batch=8. As a side-effect, this unbreaks `cargo build` on main, which currently fails with 7 errors referencing a nonexistent `stages` module and `stage_mask` field.

**Architecture:** 3 commits on a single branch against `chunzhe10/dorea` main. Commit 1 does the whole dorea-cli refactor (slim Rust code, drop 5 crate deps, delete LUT subcommand files, delete `build.rs`, bump defaults) — this is the commit that unbreaks the build. Commit 2 is pure deletion (5 crate dirs, 8 Python modules, 5 test files, dead config sections). Commit 3 is a short README rewrite.

**Tech Stack:** Rust (cargo, clap 4, anyhow, serde, toml, log, env_logger), Python 3 (subprocess-only — `raune_filter.py` is untouched), `gh` CLI for PR and issue ops (always with `--repo chunzhe10/dorea` — defaults to the wrong repo in this workspace).

**Spec:** `docs/decisions/2026-04-11-minimum-direct-rewrite.md`

---

## Prerequisites

- Working directory: `/workspaces/dorea-workspace` (the workspace repo root)
- `repos/dorea` is on branch `feat/direct-default-1440p` (created earlier; no commits on top of main yet; will be renamed in Task 0)
- `origin/main` on dorea does NOT currently build — this is expected, commit 1 fixes it
- `/opt/dorea-venv` has PyAV, Triton, torch, torchvision, numpy installed (per workspace memory `finding_dorea_venv_rebuild_gaps`)
- `gh` CLI authenticated and able to write to `chunzhe10/dorea`
- RTX 3060 6 GB workstation accessible for smoke test (Task 4 — user-run)
- Test clip at `footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4`

**Safety note:** the plan NEVER runs `git reset --hard`, `git push --force`, `cargo clean`, or any destructive op without explicit user consent. If any step fails unexpectedly, stop and ask.

---

## File Structure

**Files created (0)** — this plan only rewrites and deletes.

**Files rewritten wholesale** (via Write tool):
- `crates/dorea-cli/src/main.rs`
- `crates/dorea-cli/src/lib.rs`
- `crates/dorea-cli/src/grade.rs`
- `crates/dorea-cli/src/config.rs`
- `crates/dorea-cli/src/pipeline/mod.rs`
- `crates/dorea-cli/src/pipeline/grading.rs`
- `README.md`

**Files edited surgically** (via Edit tool):
- `crates/dorea-cli/Cargo.toml`
- `Cargo.toml` (workspace root)
- `python/pyproject.toml`
- `dorea.toml`

**Files deleted** (8 Rust sources, 5 Rust crate dirs, 8 Python modules, 5 Python test files, 1 build script):
- `crates/dorea-cli/src/{calibrate,preview,probe,change_detect,optical_flow}.rs`
- `crates/dorea-cli/src/pipeline/{calibration,feature,keyframe}.rs`
- `crates/dorea-cli/build.rs`
- `crates/dorea-color/`, `crates/dorea-lut/`, `crates/dorea-hsl/`, `crates/dorea-cal/`, `crates/dorea-gpu/`
- `python/dorea_inference/{bridge,server,maxine_enhancer,__main__,depth_anything,yolo_seg,protocol,raune_net}.py`
- `python/tests/{test_bridge_maxine,test_lifecycle_server,test_maxine_server,test_maxine_protocol,test_infer_batch}.py`

---

## Task 0: Branch setup

**Files:** none — branch rename only.

- [ ] **Step 1: Confirm current state**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git status --short && git branch --show-current && git log --oneline -1
```

Expected:
- No output from `git status` (clean tree)
- Current branch: `feat/direct-default-1440p`
- HEAD at `3f3ca2b perf(direct): fp16 RAUNE inference + batch sizing CLI flag (#67)` (same as origin/main)

If the branch or HEAD differs, stop and ask the user.

- [ ] **Step 2: Rename the branch to `feat/minimum-direct`**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git branch -m feat/direct-default-1440p feat/minimum-direct && git branch --show-current
```

Expected: `feat/minimum-direct`. No upstream is set yet (we haven't pushed); no `git push` side-effects from the rename.

- [ ] **Step 3: Confirm no baseline build attempt — main is known-broken**

Do NOT run `cargo build` here. We already know it fails on main with 7 errors. Commit 1 is the fix. Running a broken build just burns time and produces noisy output.

---

## Task 1: Commit 1 — refactor dorea-cli to direct-only, bump defaults

**Files:**
- Rewrite: `crates/dorea-cli/src/pipeline/mod.rs`, `crates/dorea-cli/src/pipeline/grading.rs`, `crates/dorea-cli/src/config.rs`, `crates/dorea-cli/src/grade.rs`, `crates/dorea-cli/src/lib.rs`, `crates/dorea-cli/src/main.rs`
- Edit: `crates/dorea-cli/Cargo.toml`
- Delete: `crates/dorea-cli/src/{calibrate,preview,probe,change_detect,optical_flow}.rs`, `crates/dorea-cli/src/pipeline/{calibration,feature,keyframe}.rs`, `crates/dorea-cli/build.rs`

**Intent:** Single large commit that leaves the tree building cleanly. Ordering within the task keeps compilation broken throughout (intentional — we're going from broken main to building commit in one atomic change). Verify build at the end only.

- [ ] **Step 1: Rewrite `crates/dorea-cli/src/pipeline/mod.rs`**

Use the Write tool to replace the file with:

```rust
//! Direct-mode grading pipeline.

pub mod grading;

use std::path::PathBuf;
use dorea_video::ffmpeg::{InputEncoding, OutputCodec};

/// Resolved pipeline configuration passed to the grading stage.
pub struct PipelineConfig {
    pub input: PathBuf,
    pub input_encoding: InputEncoding,
    pub output_codec: OutputCodec,
}
```

- [ ] **Step 2: Rewrite `crates/dorea-cli/src/pipeline/grading.rs`**

Use the Write tool to replace the file with:

```rust
//! Direct mode: spawn Python single-process RAUNE filter.
//!
//! The Rust side is a thin wrapper that resolves video info, constructs
//! the subprocess command line, and waits for exit. All decode/encode/
//! inference runs inside `python -m dorea_inference.raune_filter`.

use anyhow::{Context, Result};
use dorea_video::ffmpeg::{OutputCodec, VideoInfo};

use crate::pipeline::PipelineConfig;

/// Direct-mode grading configuration.
pub struct DirectModeConfig {
    pub python: std::path::PathBuf,
    pub raune_weights: std::path::PathBuf,
    pub raune_models_dir: std::path::PathBuf,
    pub raune_proxy_size: usize,
    pub batch_size: usize,
    pub output: std::path::PathBuf,
}

/// Direct mode: single-process RAUNE with OKLab chroma transfer.
pub fn run_grading_stage_direct(
    cfg: &PipelineConfig,
    info: &VideoInfo,
    direct_cfg: &DirectModeConfig,
) -> Result<u64> {
    use std::process::{Command, Stdio};

    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(
        info.width, info.height, direct_cfg.raune_proxy_size,
    );

    // Map output codec to PyAV codec name
    let pyav_codec = match cfg.output_codec {
        OutputCodec::Hevc10 => "hevc",
        OutputCodec::H264 => "h264",
        _ => "prores_ks",
    };

    log::info!(
        "Direct mode: single-process OKLab transfer, RAUNE proxy {}x{} batch={}, full-res {}x{}, codec={}",
        proxy_w, proxy_h, direct_cfg.batch_size, info.width, info.height, pyav_codec,
    );

    let python_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(|p| p.parent())
        .map(|p| p.join("python"))
        .unwrap_or_default();

    let output_str = direct_cfg.output.to_string_lossy();

    let mut raune_proc = Command::new(&direct_cfg.python)
        .env("PYTHONPATH", &python_dir)
        .args([
            "-m", "dorea_inference.raune_filter",
            "--weights", direct_cfg.raune_weights.to_str().unwrap_or(""),
            "--models-dir", direct_cfg.raune_models_dir.to_str().unwrap_or(""),
            "--full-width", &info.width.to_string(),
            "--full-height", &info.height.to_string(),
            "--proxy-width", &proxy_w.to_string(),
            "--proxy-height", &proxy_h.to_string(),
            "--batch-size", &direct_cfg.batch_size.to_string(),
            "--input", cfg.input.to_str().unwrap_or(""),
            "--output", &output_str,
            "--output-codec", pyav_codec,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn raune_filter.py")?;

    let status = raune_proc.wait().context("raune_filter wait failed")?;
    if !status.success() {
        anyhow::bail!("raune_filter exited with {status}");
    }

    // The filter reports frame count via stderr; return total frames from info
    Ok(info.frame_count)
}
```

- [ ] **Step 3: Rewrite `crates/dorea-cli/src/config.rs`**

Use the Write tool to replace the file with:

```rust
//! `dorea.toml` configuration file loading.
//!
//! Resolution order (first match wins):
//!   1. Path in `$DOREA_CONFIG` env var
//!   2. `./dorea.toml` in the current working directory
//!   3. `~/.config/dorea/config.toml`
//!   4. Built-in defaults
//!
//! CLI flags always override config file values.

use std::path::PathBuf;
use serde::Deserialize;

/// Top-level `dorea.toml` config.
#[derive(Debug, Deserialize, Default)]
pub struct DoreaConfig {
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub grade: GradeDefaults,
}

/// Paths to AI model weights and the Python interpreter.
#[derive(Debug, Deserialize, Default)]
pub struct ModelsConfig {
    /// Python executable (default: `/opt/dorea-venv/bin/python`)
    pub python: Option<PathBuf>,
    /// RAUNE-Net weights `.pth`
    pub raune_weights: Option<PathBuf>,
    /// `sea_thru_poc` checkout directory (contains `models/raune_net.py`)
    pub raune_models_dir: Option<PathBuf>,
}

/// `dorea grade` defaults.
#[derive(Debug, Deserialize, Default)]
pub struct GradeDefaults {
    /// RAUNE proxy long-edge resolution for direct mode (default: 1440)
    pub raune_proxy_size: Option<usize>,
    /// Frames per RAUNE batch (default: 8)
    pub direct_batch_size: Option<usize>,
    /// Input encoding override (default: auto-detect from container/codec)
    pub input_encoding: Option<String>,
    /// Output codec override (default: "h264" for 8-bit, "prores" for 10-bit)
    pub output_codec: Option<String>,
}

impl DoreaConfig {
    /// Load config from the first location that exists.
    /// Returns `Default` if no config file is found (not an error).
    pub fn load() -> Self {
        if let Ok(path) = std::env::var("DOREA_CONFIG") {
            if let Some(cfg) = Self::try_load(&PathBuf::from(&path)) {
                log::debug!("Loaded config from $DOREA_CONFIG: {path}");
                return cfg;
            }
            log::warn!("$DOREA_CONFIG={path} does not exist or is invalid — using defaults");
        }

        if let Some(cfg) = Self::try_load(&PathBuf::from("dorea.toml")) {
            log::debug!("Loaded config from ./dorea.toml");
            return cfg;
        }

        let user_cfg = std::env::var("HOME").ok()
            .map(|h| PathBuf::from(h).join(".config").join("dorea").join("config.toml"));
        if let Some(ref path) = user_cfg {
            if let Some(cfg) = Self::try_load(path) {
                log::debug!("Loaded config from {}", path.display());
                return cfg;
            }
        }

        log::debug!("No dorea.toml found — using built-in defaults");
        Self::default()
    }

    fn try_load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        match toml::from_str(&text) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                log::warn!("Failed to parse {}: {e}", path.display());
                None
            }
        }
    }
}
```

- [ ] **Step 4: Rewrite `crates/dorea-cli/src/grade.rs`**

Use the Write tool to replace the file with:

```rust
// dorea grade — direct-mode RAUNE + OKLab delta grading.
//
// Resolves config, probes the input video, and spawns a Python subprocess
// that decodes, runs RAUNE at proxy resolution, computes the OKLab delta,
// upscales via bilinear interpolation, applies the delta in OKLab, and
// encodes the output. The heavy lifting lives in raune_filter.py; this
// binary is a thin orchestrator.

use std::path::PathBuf;
use clap::Parser;
use anyhow::{Context, Result};

use dorea_video::ffmpeg::{self, InputEncoding, OutputCodec};

use crate::pipeline::{self, PipelineConfig};
use crate::pipeline::grading::DirectModeConfig;

#[derive(Parser, Debug)]
#[command(name = "dorea", about = "Underwater video direct-mode color grading")]
pub struct GradeArgs {
    /// Input video file (MP4/MOV/MKV).
    pub input: PathBuf,

    /// Output video file. Default: <input-stem>_graded.<ext>
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// RAUNE weights path (config: [models].raune_weights)
    #[arg(long)]
    pub raune_weights: Option<PathBuf>,

    /// RAUNE models directory (config: [models].raune_models_dir)
    #[arg(long)]
    pub raune_models_dir: Option<PathBuf>,

    /// Python interpreter path (config: [models].python)
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// RAUNE proxy long-edge resolution (config: [grade].raune_proxy_size, default: 1440)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,

    /// Frames per RAUNE batch (config: [grade].direct_batch_size, default: 8).
    /// fp16 RAUNE halves activation memory vs fp32; batch=8 fp16 ≈ batch=4 fp32
    /// footprint — known-safe on RTX 3060 (6 GB). Values above 8 show diminishing
    /// returns; 16+ regresses throughput due to per-frame upload overhead.
    #[arg(long)]
    pub direct_batch_size: Option<usize>,

    /// Input encoding: dlog-m, ilog, srgb (config: [grade].input_encoding, default: auto)
    #[arg(long)]
    pub input_encoding: Option<String>,

    /// Output codec: prores, hevc10, h264 (config: [grade].output_codec, default: auto)
    #[arg(long)]
    pub output_codec: Option<String>,

    /// Enable verbose (debug) logging.
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: GradeArgs, cfg: &crate::config::DoreaConfig) -> Result<()> {
    // Resolve config → CLI → built-in defaults
    let python = args.python.clone()
        .or_else(|| cfg.models.python.clone())
        .unwrap_or_else(|| PathBuf::from("/opt/dorea-venv/bin/python"));
    let raune_weights = args.raune_weights.clone()
        .or_else(|| cfg.models.raune_weights.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "RAUNE weights required — set [models].raune_weights in dorea.toml or pass --raune-weights"
        ))?;
    let raune_models_dir = args.raune_models_dir.clone()
        .or_else(|| cfg.models.raune_models_dir.clone())
        .ok_or_else(|| anyhow::anyhow!(
            "RAUNE models dir required — set [models].raune_models_dir in dorea.toml or pass --raune-models-dir"
        ))?;

    let raune_proxy_size = args.raune_proxy_size
        .or(cfg.grade.raune_proxy_size)
        .unwrap_or(1440_usize);

    let direct_batch_size = args.direct_batch_size
        .or(cfg.grade.direct_batch_size)
        .unwrap_or(8);

    // Validate batch size
    if direct_batch_size == 0 {
        anyhow::bail!("--direct-batch-size must be >= 1");
    }
    if direct_batch_size > 32 {
        anyhow::bail!(
            "--direct-batch-size {direct_batch_size} exceeds safe limit of 32 \
             (would risk CUDA OOM on 6GB VRAM). Use a smaller value."
        );
    }
    if direct_batch_size > 8 {
        log::warn!(
            "--direct-batch-size={direct_batch_size}: values above 8 may regress \
             throughput on 6GB VRAM (RTX 3060) due to per-frame upload overhead \
             in _process_batch. Measured baseline: batch=8 ~4.36 fps, batch=16 ~3.47 fps."
        );
    }

    // Probe input
    let info = ffmpeg::probe(&args.input)
        .context("ffprobe failed — is ffmpeg installed?")?;
    log::info!(
        "Input: {}x{} @ {:.3}fps, {:.1}s ({} frames)",
        info.width, info.height, info.fps, info.duration_secs, info.frame_count
    );

    // Resolve input encoding: CLI flag → config → auto-detect
    let input_encoding = args.input_encoding.as_deref()
        .or(cfg.grade.input_encoding.as_deref())
        .map(|s| s.parse::<InputEncoding>().map_err(|e| anyhow::anyhow!("invalid --input-encoding: {e}")))
        .transpose()?
        .unwrap_or_else(|| InputEncoding::auto_detect(&info, &args.input));

    // Resolve output codec: CLI flag → config → auto based on input encoding
    let output_codec = args.output_codec.as_deref()
        .or(cfg.grade.output_codec.as_deref())
        .map(|s| s.parse::<OutputCodec>().map_err(|e| anyhow::anyhow!("invalid --output-codec: {e}")))
        .transpose()?
        .unwrap_or_else(|| {
            if input_encoding.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 }
        });

    log::info!("Encoding: input={input_encoding}, output={output_codec}, 10-bit={}", output_codec.is_10bit());

    let output = args.output.clone().unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = if output_codec == OutputCodec::ProRes { "mov" } else { "mp4" };
        args.input.with_file_name(format!("{stem}_graded.{ext}"))
    });

    log::info!("Grading: {} → {}", args.input.display(), output.display());

    let (proxy_w, proxy_h) = dorea_video::resize::proxy_dims(
        info.width, info.height, raune_proxy_size,
    );

    log::info!(
        "Direct mode: RAUNE proxy {}x{} (max {raune_proxy_size}), batch={direct_batch_size}, output {}x{}",
        proxy_w, proxy_h, info.width, info.height,
    );

    let pipeline_cfg = PipelineConfig {
        input: args.input.clone(),
        input_encoding,
        output_codec,
    };

    let direct_cfg = DirectModeConfig {
        python: python.clone(),
        raune_weights: raune_weights.clone(),
        raune_models_dir: raune_models_dir.clone(),
        raune_proxy_size,
        batch_size: direct_batch_size,
        output: output.clone(),
    };

    let frame_count = pipeline::grading::run_grading_stage_direct(
        &pipeline_cfg, &info, &direct_cfg,
    )?;

    if info.frame_count > 0 && frame_count < info.frame_count {
        log::warn!(
            "Incomplete grading: {frame_count} frames processed, {} expected",
            info.frame_count
        );
    }
    log::info!("Done. Graded {frame_count} frames → {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use dorea_video::ffmpeg::{InputEncoding, OutputCodec, VideoInfo};

    #[test]
    fn default_encoding_for_8bit_h264() {
        let info = VideoInfo {
            width: 1920, height: 1080, fps: 30.0, duration_secs: 10.0,
            frame_count: 300, has_audio: true,
            codec_name: "h264".to_string(),
            pix_fmt: "yuv420p".to_string(),
            bits_per_component: 8,
        };
        let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
        assert_eq!(enc, InputEncoding::Srgb);
        assert!(!enc.is_10bit());
        let codec = if enc.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 };
        assert_eq!(codec, OutputCodec::H264);
    }

    #[test]
    fn default_encoding_for_10bit_hevc() {
        let info = VideoInfo {
            width: 3840, height: 2160, fps: 29.97, duration_secs: 3.0,
            frame_count: 90, has_audio: true,
            codec_name: "hevc".to_string(),
            pix_fmt: "yuv420p10le".to_string(),
            bits_per_component: 10,
        };
        let enc = InputEncoding::auto_detect(&info, std::path::Path::new("clip.mp4"));
        assert_eq!(enc, InputEncoding::DLogM);
        assert!(enc.is_10bit());
        let codec = if enc.is_10bit() { OutputCodec::ProRes } else { OutputCodec::H264 };
        assert_eq!(codec, OutputCodec::ProRes);
    }
}
```

This file intentionally drops the 5 `lerp_depth`/`build_inference_config` tests that referenced helpers living in deleted code paths. The 2 encoding auto-detect tests only depend on `dorea_video::ffmpeg` (kept) and survive.

- [ ] **Step 5: Rewrite `crates/dorea-cli/src/lib.rs`**

Use the Write tool to replace the file with:

```rust
pub mod config;
pub mod grade;
pub mod pipeline;
```

- [ ] **Step 6: Rewrite `crates/dorea-cli/src/main.rs`**

Use the Write tool to replace the file with:

```rust
use clap::Parser;
use dorea_cli::{config::DoreaConfig, grade::{self, GradeArgs}};

fn main() -> anyhow::Result<()> {
    let args = GradeArgs::parse();

    env_logger::Builder::from_env(env_logger::Env::default())
        .filter_level(if args.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    let config = DoreaConfig::load();
    grade::run(args, &config)
}
```

- [ ] **Step 7: Delete LUT-pipeline source files in `dorea-cli`**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git rm \
  crates/dorea-cli/src/calibrate.rs \
  crates/dorea-cli/src/preview.rs \
  crates/dorea-cli/src/probe.rs \
  crates/dorea-cli/src/change_detect.rs \
  crates/dorea-cli/src/optical_flow.rs \
  crates/dorea-cli/src/pipeline/calibration.rs \
  crates/dorea-cli/src/pipeline/feature.rs \
  crates/dorea-cli/src/pipeline/keyframe.rs \
  crates/dorea-cli/build.rs
```

Expected: 9 files staged for removal.

- [ ] **Step 8: Edit `crates/dorea-cli/Cargo.toml` — drop dead deps and the cuda feature**

Use the Edit tool. Replace:

```toml
[dependencies]
dorea-color = { workspace = true }
dorea-lut = { workspace = true }
dorea-hsl = { workspace = true }
dorea-cal = { workspace = true }
dorea-gpu = { workspace = true }
dorea-video = { workspace = true }
clap = { version = "4", features = ["derive"] }
crossbeam-channel = "0.5"
image = "0.25"
anyhow = { workspace = true }
log = { workspace = true }
env_logger = "0.11"
memmap2 = "0.9"
serde = { workspace = true }
toml = { workspace = true }

[features]
# Mirrors dorea-gpu: activated by build.rs (nvcc detection), not manually.
# Declared here so Cargo recognises `#[cfg(feature = "cuda")]` as a valid condition.
cuda = []
```

With:

```toml
[dependencies]
dorea-video = { workspace = true }
clap = { version = "4", features = ["derive"] }
anyhow = { workspace = true }
log = { workspace = true }
env_logger = "0.11"
serde = { workspace = true }
toml = { workspace = true }
```

Removed: `dorea-color`, `dorea-lut`, `dorea-hsl`, `dorea-cal`, `dorea-gpu` (5 legacy crates), `crossbeam-channel` (was used by the 3-thread pipeline on the LUT path — the direct-mode 3-thread pipeline lives inside `raune_filter.py`, not in Rust), `image` (was used for keyframe image I/O), `memmap2` (was used for the paged calibration store), and the `[features] cuda = []` block (nothing left to gate).

- [ ] **Step 9: Build the crate in isolation**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -30
```

Expected: `Finished \`dev\` profile [unoptimized + debuginfo] target(s)`. Zero errors.

If the build fails:
- Errors about `dorea_gpu::`, `dorea_color::`, `dorea_lut::`, `dorea_hsl::`, `dorea_cal::`, `MaxineDefaults`, `InferDefaults`, `build_inference_config`, `lerp_depth`, or `PreviewDefaults` mean one of the rewrites was incomplete — compare the error line against the new file contents in this plan.
- Errors about `crossbeam-channel`, `image`, or `memmap2` unresolved mean one of those deps is still used somewhere — re-add to Cargo.toml temporarily, grep the crate for the usage, remove the usage, drop the dep. Most likely this doesn't happen since those deps were only used in deleted files.
- Errors about `dorea_video::resize::proxy_dims` NOT FOUND would be a real problem — that function exists in `dorea-video` per our inspection of `grading.rs` and must not be broken. If this happens, stop and ask the user.

- [ ] **Step 10: Build the full workspace**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release 2>&1 | tail -30
```

Expected: `Finished \`release\` profile [optimized] target(s)`. This will also attempt to build the 5 legacy crates (still present in the workspace) — they may emit warnings but should not fail. If any of them fail, that's fine for now — we're only gating on `dorea-cli` being green in this commit. Actually, wait: `cargo build` builds all workspace members by default, and if any fail, the whole command fails.

Verify first:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release -p dorea-cli 2>&1 | tail -5
```

If `dorea-cli` alone builds, that is sufficient for commit 1. The 5 legacy crates are deleted in commit 2 and we don't need them to build in the meantime. Skip the full workspace build until after commit 2.

- [ ] **Step 11: Run tests on `dorea-cli`**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli 2>&1 | tail -30
```

Expected: 2 tests pass (`default_encoding_for_8bit_h264`, `default_encoding_for_10bit_hevc`). No failures. No tests referencing `lerp_depth`, `build_inference_config`, or Maxine should exist in the output.

- [ ] **Step 12: Stage the changes and commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add \
  crates/dorea-cli/src/main.rs \
  crates/dorea-cli/src/lib.rs \
  crates/dorea-cli/src/grade.rs \
  crates/dorea-cli/src/config.rs \
  crates/dorea-cli/src/pipeline/mod.rs \
  crates/dorea-cli/src/pipeline/grading.rs \
  crates/dorea-cli/Cargo.toml \
  && git status --short
```

Expected `git status` output shows:
- `M  crates/dorea-cli/Cargo.toml`
- `M  crates/dorea-cli/src/config.rs`
- `M  crates/dorea-cli/src/grade.rs`
- `M  crates/dorea-cli/src/lib.rs`
- `M  crates/dorea-cli/src/main.rs`
- `M  crates/dorea-cli/src/pipeline/grading.rs`
- `M  crates/dorea-cli/src/pipeline/mod.rs`
- `D  crates/dorea-cli/build.rs`
- `D  crates/dorea-cli/src/calibrate.rs`
- `D  crates/dorea-cli/src/change_detect.rs`
- `D  crates/dorea-cli/src/optical_flow.rs`
- `D  crates/dorea-cli/src/pipeline/calibration.rs`
- `D  crates/dorea-cli/src/pipeline/feature.rs`
- `D  crates/dorea-cli/src/pipeline/keyframe.rs`
- `D  crates/dorea-cli/src/preview.rs`
- `D  crates/dorea-cli/src/probe.rs`

Then commit:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git commit -m "$(cat <<'EOF'
refactor(dorea-cli): collapse to direct-only, bump defaults

Strip the LUT/calibration/keyframe/depth-zones pipeline out of dorea-cli:
delete calibrate/preview/probe/change_detect/optical_flow subcommands,
delete pipeline::{calibration,feature,keyframe} stages, collapse main.rs
to a single-command binary (positional <input> replaces --input flag),
slim PipelineConfig to (input, input_encoding, output_codec), slim
GradeArgs to direct-mode fields, slim DoreaConfig to [models] + [grade],
drop build.rs + cuda feature (vestigial after dorea-gpu removal).

Bump direct-mode defaults: raune_proxy_size 1080 → 1440, direct_batch_size
4 → 8. Rationale: delta upscale bench (corvia 019d7a9d) showed bilinear
is best-ΔE at 1440p with all classical methods gaining ~8% ΔE vs 1080p;
fp16 RAUNE headroom from PR #67 makes batch=8 safe on 6GB VRAM.

Incidentally fixes main's build — the broken stages::format / stage_mask
/ AdaptiveGrader references lived in the deleted LUT code path (plus one
stray stage_mask: 0 in the direct-mode PipelineConfig literal, also
removed here).

Still in the tree: the 5 legacy crates (dorea-color, -lut, -hsl, -cal,
-gpu) and 8 Python modules that supported the LUT path. They are
orphans after this commit — nothing references them — and are deleted
wholesale in the follow-up commit.

BREAKING CHANGE: `dorea calibrate`, `dorea preview`, `dorea probe` are
gone. `dorea grade --input X` is gone. New surface is `dorea <input>`
with flags for output path, RAUNE paths, and proxy/batch tuning.

Refs docs/decisions/2026-04-11-minimum-direct-rewrite.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -1
```

Expected: one new commit on `feat/minimum-direct`, with the summary line `refactor(dorea-cli): collapse to direct-only, bump defaults`.

---

## Task 2: Commit 2 — delete legacy crates, orphan Python modules, stale config

**Files:**
- Delete: 5 crate directories under `crates/`
- Delete: 8 Python modules under `python/dorea_inference/`
- Delete: 5 Python test files under `python/tests/`
- Edit: `Cargo.toml` (workspace root), `python/pyproject.toml`, `dorea.toml`

**Intent:** Pure deletion. Zero behavior changes, no new code. Build must still pass after this commit (now the workspace has exactly 2 members: `dorea-cli` and `dorea-video`).

- [ ] **Step 1: Delete the 5 legacy Rust crate directories**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git rm -r \
  crates/dorea-color \
  crates/dorea-lut \
  crates/dorea-hsl \
  crates/dorea-cal \
  crates/dorea-gpu
```

Expected: a large number of files staged for removal (CUDA kernels, Rust sources, Cargo.toml files, etc.). Many hundreds of lines.

- [ ] **Step 2: Edit workspace `Cargo.toml` to shrink members list**

Use the Edit tool. Current content (from file inspection):

```toml
[workspace]
resolver = "2"
members = [
    "crates/dorea-cli",
    "crates/dorea-color",
    "crates/dorea-lut",
    "crates/dorea-hsl",
    "crates/dorea-cal",
    "crates/dorea-gpu",
    "crates/dorea-video",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"

[workspace.dependencies]
dorea-color = { path = "crates/dorea-color" }
dorea-lut = { path = "crates/dorea-lut" }
dorea-hsl = { path = "crates/dorea-hsl" }
dorea-cal = { path = "crates/dorea-cal" }
dorea-gpu = { path = "crates/dorea-gpu" }
dorea-video = { path = "crates/dorea-video" }
serde = { version = "1", features = ["derive"] }
bincode = "1"
thiserror = "1"
anyhow = "1"
log = "0.4"
rayon = "1"
```

Replace the `members = [ ... ]` block with:

```toml
members = [
    "crates/dorea-cli",
    "crates/dorea-video",
]
```

And replace the `[workspace.dependencies]` block with:

```toml
[workspace.dependencies]
dorea-video = { path = "crates/dorea-video" }
serde = { version = "1", features = ["derive"] }
bincode = "1"
thiserror = "1"
anyhow = "1"
log = "0.4"
rayon = "1"
```

(Removed: `dorea-color`, `dorea-lut`, `dorea-hsl`, `dorea-cal`, `dorea-gpu` from both members and workspace.dependencies.)

- [ ] **Step 3: Delete the 8 Python modules**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git rm \
  python/dorea_inference/bridge.py \
  python/dorea_inference/server.py \
  python/dorea_inference/maxine_enhancer.py \
  python/dorea_inference/__main__.py \
  python/dorea_inference/depth_anything.py \
  python/dorea_inference/yolo_seg.py \
  python/dorea_inference/protocol.py \
  python/dorea_inference/raune_net.py
```

Expected: 8 files staged for removal. `python/dorea_inference/` retains only `__init__.py` and `raune_filter.py` plus `__pycache__/`.

- [ ] **Step 4: Delete the 5 Python test files**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git rm \
  python/tests/test_bridge_maxine.py \
  python/tests/test_lifecycle_server.py \
  python/tests/test_maxine_server.py \
  python/tests/test_maxine_protocol.py \
  python/tests/test_infer_batch.py
```

Expected: 5 files staged for removal. `python/tests/` retains only `__init__.py` plus `__pycache__/`.

- [ ] **Step 5: Edit `python/pyproject.toml` — drop `transformers` and `Pillow`**

Use the Edit tool. Replace:

```toml
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.38",
    "Pillow>=10.0",
    "numpy>=1.24",
]
```

With:

```toml
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy>=1.24",
]
```

`transformers` was consumed only by `depth_anything.py` (deleted). `Pillow` was consumed only by `protocol.py` (deleted). `torch`, `torchvision`, `numpy` are all still required by `raune_filter.py`. `PyAV` and `triton` are not listed in this file (they're installed separately via `setup_bench.sh` per workspace memory) — do not add them.

- [ ] **Step 6: Edit `dorea.toml` — drop dead sections and fields**

Use the Write tool to replace the file entirely with:

```toml
# dorea.toml — local config for dorea pipeline
# CLI flags always override values here.
# Copy to ~/.config/dorea/config.toml for a user-level default.

[models]
python          = "/opt/dorea-venv/bin/python"
raune_weights   = "/workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
raune_models_dir = "/workspaces/dorea-workspace/working/sea_thru_poc"

[grade]
# raune_proxy_size = 1440    # RAUNE input long-edge (default: 1440)
# direct_batch_size = 8      # Frames per RAUNE batch (default: 8, max 32)
# input_encoding = "auto"    # auto-detect from container/codec; options: dlog-m, ilog, srgb
# output_codec = "auto"      # auto: prores for 10-bit, h264 for 8-bit; options: prores, hevc10, h264
```

Removed: `[models].depth_model`, the entire `[inference]` section, all LUT-pipeline `[grade]` fields (`warmth`, `strength`, `contrast`, `depth_skip_threshold`, etc.), the entire `[maxine]` section, the entire `[preview]` section.

- [ ] **Step 7: Full workspace build**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release 2>&1 | tail -20
```

Expected: `Finished \`release\` profile [optimized] target(s)`. The workspace now has exactly 2 crates, and both build cleanly.

If the build fails:
- `could not resolve dependency 'dorea-color'` or similar → the `Cargo.toml` edit in Step 2 missed an entry. Check `[workspace.dependencies]`.
- `could not find 'crates/dorea-gpu'` → one of the crate directories was not fully deleted. Run `ls crates/` to verify.

- [ ] **Step 8: Run all Rust tests**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test --release 2>&1 | tail -30
```

Expected: all tests pass across `dorea-cli` (2 encoding tests) and `dorea-video` (whatever tests it has — unchanged by this plan).

- [ ] **Step 9: Verify Python test collection doesn't import deleted modules**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && /opt/dorea-venv/bin/python -m pytest python/tests/ --collect-only 2>&1 | tail -20
```

Expected: zero test files collected (all 5 test files were deleted in Step 4). Output should show `collected 0 items`, NOT an import error. If pytest reports `ImportError` or `ModuleNotFoundError`, some test file was missed — grep `python/tests/` for imports of the deleted modules and delete any remaining ones.

- [ ] **Step 10: Verify `raune_filter.py` still runs standalone**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && PYTHONPATH=python /opt/dorea-venv/bin/python -m dorea_inference.raune_filter --help 2>&1 | tail -20
```

Expected: argparse help text listing `--weights`, `--models-dir`, `--full-width`, `--full-height`, `--proxy-width`, `--proxy-height`, `--batch-size`, `--input`, `--output`, `--output-codec`. No import errors. No reference to `bridge`, `server`, `maxine_enhancer`, or any deleted module.

If this fails with `ImportError: No module named 'dorea_inference.xxx'` → there's an unexpected transitive import inside `raune_filter.py` or its dependencies. Stop, investigate, ask the user.

- [ ] **Step 11: Stage and commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add Cargo.toml python/pyproject.toml dorea.toml && git status --short
```

Expected: a long list of `D` (deleted) entries plus the 3 `M` edits for `Cargo.toml`, `python/pyproject.toml`, `dorea.toml`.

Commit:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git commit -m "$(cat <<'EOF'
chore: delete legacy crates, orphan Python modules, stale config

Pure deletion commit. No behavior changes.

Rust:
- Delete 5 legacy crates: dorea-color, dorea-lut, dorea-hsl, dorea-cal,
  dorea-gpu. Workspace members list shrinks from 7 to 2 (dorea-cli +
  dorea-video). All 5 crates were orphaned after the previous commit.
- Update workspace Cargo.toml: remove 5 members, remove 5
  workspace.dependencies entries.

Python:
- Delete 8 dorea_inference modules: bridge.py, server.py,
  maxine_enhancer.py, __main__.py, depth_anything.py, yolo_seg.py,
  protocol.py, raune_net.py. None are imported by the surviving
  raune_filter.py (verified by grep during planning).
- Delete 5 test files that imported the removed modules:
  test_bridge_maxine, test_lifecycle_server, test_maxine_server,
  test_maxine_protocol, test_infer_batch.
- python/pyproject.toml: drop transformers (only used by
  depth_anything.py) and Pillow (only used by protocol.py).

Config:
- dorea.toml: drop [models].depth_model, [inference] section entirely,
  [maxine] section entirely, [preview] section entirely, and all LUT-
  pipeline [grade] fields. What remains: [models] python/raune_weights/
  raune_models_dir, and a [grade] section with the 4 direct-mode
  settings commented out (since the defaults are usable as-is).

Refs docs/decisions/2026-04-11-minimum-direct-rewrite.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -2
```

Expected: second commit on `feat/minimum-direct`.

---

## Task 3: Commit 3 — rewrite README

**Files:** Rewrite `README.md` in `repos/dorea`.

**Intent:** Replace the stale README (currently describes dorea grade as "Phase 3 — not yet implemented") with a short stub that reflects the new minimum-direct state.

- [ ] **Step 1: Rewrite `repos/dorea/README.md`**

Use the Write tool to replace the file entirely with:

```markdown
# Dorea

Automated underwater video color grading — single-pass direct-mode pipeline.
Named after the Dorado constellation.

## Architecture

A thin Rust CLI (`dorea`) that probes input video, resolves config, and
spawns a Python subprocess (`dorea_inference.raune_filter`) which does all
the heavy lifting: PyAV decode, batched RAUNE-Net inference at proxy
resolution on CUDA, OKLab delta computation, bilinear upscale to full
resolution, fused Triton OKLab transfer, PyAV encode — all in a single
producer-consumer 3-thread pipeline.

```
crates/
├── dorea-cli    — `dorea` binary (argument parsing, config, subprocess spawn)
└── dorea-video  — ffmpeg probe, InputEncoding auto-detect, OutputCodec enum

python/dorea_inference/
├── raune_filter.py  — all runtime logic: decode → RAUNE → OKLab delta → encode
└── __init__.py
```

## Hardware requirements

- NVIDIA GPU with ≥ 6 GB VRAM (RTX 3060 or better) — required for RAUNE-Net fp16 inference + Triton OKLab transfer.
- Linux workstation or devcontainer.
- FFmpeg with HEVC/H.264 support.

## Build

```bash
cargo build --release -p dorea-cli
```

The binary lands at `target/release/dorea`.

## Run

```bash
# Auto-detects encoding + output codec. Direct mode, 1440p proxy, batch=8.
dorea path/to/clip.mp4

# Explicit output path
dorea path/to/clip.mp4 --output path/to/graded.mov

# Verbose logging
dorea path/to/clip.mp4 --verbose
```

## Configuration

Create `dorea.toml` in the current working directory (or `~/.config/dorea/config.toml`):

```toml
[models]
python           = "/opt/dorea-venv/bin/python"
raune_weights    = "/path/to/RAUNENet/weights_95.pth"
raune_models_dir = "/path/to/sea_thru_poc"

[grade]
# raune_proxy_size = 1440    # long-edge pixels; values above 1440 may OOM on 6 GB VRAM
# direct_batch_size = 8      # frames per RAUNE forward pass; max 32
```

CLI flags always override config values. See `dorea --help` for the full list.

## Tunables

- `--raune-proxy-size N` — RAUNE input long-edge in pixels. Default **1440**. Lower = faster but noisier upscale; higher = better ΔE but more VRAM + compute. Delta upscale bench (1080p vs 1440p) showed ~8% ΔE improvement at 1440p with no upscale-stage cost.
- `--direct-batch-size N` — frames per RAUNE forward pass. Default **8**. fp16 activation memory is ~½ of fp32, so batch=8 fp16 fits in the same VRAM envelope as batch=4 fp32. Values above 8 show diminishing returns; `N > 16` regresses throughput.

## Breaking changes from previous `dorea`

This release removes the `calibrate`, `preview`, and `probe` subcommands and the entire 3D LUT / depth-zones / YOLO-seg / Maxine pipeline. `dorea grade` is gone; the new form is `dorea <input>` with `<input>` as a positional argument (no `--input` flag). Existing `dorea.toml` files will parse but fields for removed features (`[maxine]`, `[preview]`, `[inference]`, most of `[grade]`) are silently ignored.
```

- [ ] **Step 2: Commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add README.md && git commit -m "$(cat <<'EOF'
docs: rewrite README for minimum-direct

The previous README described dorea grade as "Phase 3 — not yet
implemented" and referenced the Sea-Thru POC as the algorithmic
source of truth. Both were badly stale. This replaces it with a
short accurate stub covering the new 2-crate architecture, build
instructions, the single-command `dorea <input>` invocation, the
dorea.toml config surface, the proxy-size / batch-size tunables,
and a breaking-changes section naming the removed subcommands.

Refs docs/decisions/2026-04-11-minimum-direct-rewrite.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -3
```

Expected: third and final commit on `feat/minimum-direct`. Branch now has 3 commits total on top of `origin/main`.

- [ ] **Step 3: Final build + test sanity check**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release 2>&1 | tail -5 && cargo test --release 2>&1 | tail -10
```

Expected: green build, green tests.

---

## Task 4: Smoke test on the RTX 3060 workstation (user-run)

**Files:** none. This task requires the user's GPU; a subagent cannot run it.

**Intent:** Confirm the binary works end-to-end on real hardware with real data. This is the single acceptance criterion that validates "direct mode at 1440p batch=8 does not OOM and produces a valid output file."

- [ ] **Step 1: Release build**

User runs in terminal A:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release -p dorea-cli 2>&1 | tail -5
```

Expected: `Finished \`release\` profile [optimized]`.

- [ ] **Step 2: `--help` output sanity**

User runs:
```bash
/workspaces/dorea-workspace/repos/dorea/target/release/dorea --help
```

Expected: positional `<INPUT>` in the usage line, no subcommand menu, flags include `--output`, `--raune-weights`, `--raune-models-dir`, `--python`, `--raune-proxy-size`, `--direct-batch-size`, `--input-encoding`, `--output-codec`, `--verbose`. No `--input` flag. No `grade`, `calibrate`, `preview`, or `probe` subcommands mentioned.

- [ ] **Step 3: Start nvidia-smi in terminal B**

User runs in a second terminal window:
```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv -l 1
```

Keep this running throughout Step 4.

- [ ] **Step 4: Smoke run — direct mode with defaults**

In terminal A:
```bash
cd /workspaces/dorea-workspace && ./repos/dorea/target/release/dorea \
  footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 \
  --output /tmp/smoke_minimum_direct.mp4 \
  2>&1 | tee /tmp/smoke_log.txt
```

Expected:
- Log line matching: `Direct mode: RAUNE proxy 2560x1440 (max 1440), batch=8, output 3840x2160`
- Log line: `Direct mode: single-process OKLab transfer, RAUNE proxy 2560x1440 batch=8, full-res 3840x2160, codec=...`
- Process runs for ~5–30 seconds on the 3-second clip, no OOM, no crash.
- Terminal B `nvidia-smi` peak `memory.used` stays below 6144 MiB (the RTX 3060 physical limit).
- Exit code 0.
- `/tmp/smoke_minimum_direct.mp4` exists and is non-zero size.

**If OOM occurs:** terminal A prints `CUDA out of memory`, return code non-zero. Record the peak memory seen in terminal B. Stop. The spec accepted this risk but said failure mode is loud; mitigation is `--direct-batch-size 4`. Re-run with that flag, confirm it fits, file a follow-up issue recommending we either leave the default at 8 (if the issue was transient) or revert to 4 as a patch commit before opening the PR.

**If the output file won't play** (e.g. ffplay shows a black frame, VLC errors): do not open the PR. The subprocess bridge is broken somewhere. Stop and investigate.

- [ ] **Step 5: Record smoke results in `/tmp/smoke_results.md`**

User creates the file with content like:

```markdown
# Smoke test — feat/minimum-direct

Host: RTX 3060 6 GB laptop, dorea devcontainer
Input: footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4
Output: /tmp/smoke_minimum_direct.mp4

- `cargo build --release`: OK
- `dorea --help`: single-command binary, positional <INPUT>, no subcommands — OK
- Direct run (no flags): completed in XXs, processed ~360 frames
- Peak VRAM (from terminal B nvidia-smi): XXXX MiB / 6144 MiB
- Output file: YYY bytes, plays cleanly in ffplay
- Log line verified: `Direct mode: RAUNE proxy 2560x1440 (max 1440), batch=8`
```

Fill in the `XX`, `XXXX`, `YYY` with actual numbers. This content gets pasted into the PR body in Task 5.

---

## Task 5: Pre-merge tag, push, open PR

**Files:** none. Git operations only.

**Intent:** Safe rollback via tag, push the branch, open the PR explicitly against `chunzhe10/dorea main`.

- [ ] **Step 1: Fetch latest main from origin**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git fetch origin main 2>&1 | tail -5
```

Expected: confirms `origin/main` is at the expected commit (`3f3ca2b` at time of writing).

- [ ] **Step 2: Tag `origin/main` as `pre-minimum-direct` before pushing anything**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git tag pre-minimum-direct origin/main && git push origin pre-minimum-direct
```

Expected: tag created locally AND pushed to origin. This is the rollback anchor — after the PR merges, `git reset --hard pre-minimum-direct` (with user consent, not automatic) brings main back to the pre-rewrite state without touching the deleted files' git history.

- [ ] **Step 3: Push the feature branch**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git push -u origin feat/minimum-direct 2>&1 | tail -10
```

Expected: branch created on origin.

- [ ] **Step 4: Open the PR with `gh`, explicitly targeting `chunzhe10/dorea main`**

User pastes the smoke-test results from `/tmp/smoke_results.md` into the `<paste-smoke-results-here>` placeholder below, then runs:

```bash
gh pr create --repo chunzhe10/dorea \
  --base main \
  --head feat/minimum-direct \
  --title "Minimum-direct rewrite: delete legacy, collapse to single-command binary" \
  --body "$(cat <<'EOF'
## Summary

- Delete 5 legacy Rust crates (`dorea-color`, `dorea-lut`, `dorea-hsl`, `dorea-cal`, `dorea-gpu`), 8 Python modules, and 5 test files. Workspace shrinks from 7 crates to 2.
- Collapse `dorea-cli` to a single-command binary. `dorea grade --input X` becomes `dorea X`. `calibrate`, `preview`, `probe` subcommands removed.
- Bump direct-mode defaults: `raune_proxy_size` 1080 → 1440, `direct_batch_size` 4 → 8.
- Delete vestigial `build.rs` + `[features] cuda = []` in `dorea-cli/Cargo.toml`.
- Incidentally unbreaks main's build (the 7-error `stages` / `stage_mask` / `AdaptiveGrader` state from PRs #63/#65/#67).

## Motivation

Three forces pushing toward the same outcome:

1. Delta upscale bench (corvia `019d7a9d`) showed bilinear is best-ΔE at 1440p proxy, with classical methods gaining ~8% ΔE vs 1080p. Direct mode uses bilinear.
2. Direct mode has received all the recent perf investment (PR #65 3-thread pipeline, PR #67 fp16 + batch sizing). The LUT pipeline is frozen and had dead-code Maxine plumbing hard-disabled across 4 source files.
3. `cargo build --release -p dorea-cli` on main currently fails with 7 errors: `stages` module not found (×4), `AdaptiveGrader` unresolved, `PipelineConfig.stage_mask` missing (×2). No branch in the repo has a fix. The broken references live in LUT-pipeline code that this PR deletes (plus one stray `stage_mask: 0` in the direct-mode path, also removed).

## Breaking change

This is a CLI-surface breaking change. Scripts that call `dorea grade`, `dorea calibrate`, `dorea preview`, `dorea probe`, or `dorea grade --input X` will break. The new shape is `dorea <input>` with positional input. Existing `dorea.toml` files parse without error — fields for removed features are silently ignored.

## Smoke test on RTX 3060 6 GB

<paste-smoke-results-here>

## Rollback

`origin/main` was tagged `pre-minimum-direct` before merging. Rollback is `git reset --hard pre-minimum-direct` (user-initiated, not automatic). Do not `git revert` — it would re-create 5 crate directories and conflict with any post-merge work in `dorea-cli`.

## Known follow-ups

- In-flight branches on origin that touch deleted paths will conflict-hell after merge: `feat/15-maxine-arch-improvements`, `feat/15-maxine-enhancement`, `feat/27-cuda-grader-buffer-prealloc`, `feat/28-combined-lut-cuda-texture`, `feat/oklab-grade-pipeline`, `feat/streaming-calibration`. User's call whether to close, rebase, or warn their owners.
- Post-land VRAM measurement on a long clip (not just the 3s smoke clip).

## Test plan

- [x] `cargo build --release` (2 crates: dorea-cli + dorea-video)
- [x] `cargo test --release` (2 tests: encoding auto-detect)
- [x] `dorea --help` — single-command binary, positional input, no subcommands
- [x] `dorea <clip>.mp4` — runs direct mode at 1440p batch=8 on RTX 3060, no OOM, produces valid output file
- [x] `python -m pytest python/tests/ --collect-only` — collects 0 items, no import errors
- [x] `python -m dorea_inference.raune_filter --help` — runs cleanly

## References

- Spec: `docs/decisions/2026-04-11-minimum-direct-rewrite.md` (workspace repo)
- Plan: `docs/plans/2026-04-12-minimum-direct-rewrite.md` (workspace repo)
- Superseded spec: `docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md`
- Bench findings: corvia `019d7a7b` (1080p first run), `019d7a9d` (1080 vs 1440 comparison)
- Predecessor PRs: #63 (OKLab grade + direct mode), #65 (3-thread pipeline), #67 (fp16 + batch sizing)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed. Record it.

- [ ] **Step 5: Verify the PR landed on the right repo**

Run:
```bash
gh pr list --repo chunzhe10/dorea --head feat/minimum-direct --json number,title,baseRefName
```

Expected: one entry with `"baseRefName": "main"` and the title from the create command. If the list is empty or shows up on `chunzhe10/dorea-workspace`, the `--repo` flag was dropped — stop and investigate.

---

## Task 6: Commit this plan to the workspace repo

**Files:** `docs/plans/2026-04-12-minimum-direct-rewrite.md` (this file).

**Intent:** Preserve the plan alongside the spec in the workspace repo's history.

- [ ] **Step 1: Confirm the plan file exists**

Run:
```bash
ls -la /workspaces/dorea-workspace/docs/plans/2026-04-12-minimum-direct-rewrite.md
```

Expected: file present.

- [ ] **Step 2: Commit**

Run (from the workspace root, NOT `repos/dorea`):

```bash
cd /workspaces/dorea-workspace && git add docs/plans/2026-04-12-minimum-direct-rewrite.md && git commit -m "$(cat <<'EOF'
docs: add minimum-direct rewrite implementation plan

3-commit plan (refactor, delete, README) for the minimum-direct
rewrite on chunzhe10/dorea. Backs the spec committed earlier
at docs/decisions/2026-04-11-minimum-direct-rewrite.md. Applied
all 5-persona review findings: merged commits 1+2 to keep the
tree building, fixed the grade.rs:258 stage_mask direct-branch
bug, added the missing Python module deletes (__main__.py,
test_lifecycle_server, test_maxine_server, depth_anything,
yolo_seg, protocol, raune_net, test_infer_batch, test_maxine_protocol),
dropped the vestigial build.rs and cuda feature.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" && git log --oneline -3
```

Expected: one new commit on the workspace `master`.

**Note:** this task can run before Task 0 if preferred (plan-first), or after the dorea PR is open. Either order works.

---

## Acceptance Criteria (final)

Before marking this plan done, confirm all of these:

- [ ] `cargo build --release` passes in `repos/dorea` on `feat/minimum-direct`
- [ ] `cargo test --release` passes (2 encoding tests plus whatever `dorea-video` contributes)
- [ ] Workspace `Cargo.toml` `members` = `["crates/dorea-cli", "crates/dorea-video"]`, exactly 2 entries
- [ ] `crates/dorea-color`, `crates/dorea-lut`, `crates/dorea-hsl`, `crates/dorea-cal`, `crates/dorea-gpu` directories do not exist
- [ ] `dorea --help` shows positional `<INPUT>` (no subcommand menu, no `--input` flag)
- [ ] `dorea <clip>.mp4` with no other flags runs direct mode at 1440p batch=8 on the RTX 3060 workstation without OOM and produces a valid output file (smoke test)
- [ ] `python -m pytest python/tests/ --collect-only` collects 0 items with no ImportError
- [ ] `python -m dorea_inference.raune_filter --help` runs cleanly
- [ ] `grep -rn "dorea_gpu\|dorea_color\|dorea_lut\|dorea_hsl\|dorea_cal\|maxine\|stages::format\|stage_mask\|AdaptiveGrader" crates/dorea-cli/src/` returns zero hits
- [ ] `grep -rn "bridge\|server\|maxine_enhancer\|depth_anything\|yolo_seg\|protocol" python/dorea_inference/` returns zero hits (aside from `__pycache__/`)
- [ ] `pre-minimum-direct` tag pushed to `origin/chunzhe10/dorea` before the PR merges
- [ ] PR opened against `chunzhe10/dorea main` (NOT `chunzhe10/dorea-workspace`)
- [ ] Plan file committed to `chunzhe10/dorea-workspace master`

**NOT acceptance criteria** (per spec's non-goals):
- No formal VRAM peak or throughput measurement beyond the smoke test
- No comparison against prior 1080p / batch=4 performance
- No preservation of any LUT-pipeline functionality
- No rewrite of `raune_filter.py`
