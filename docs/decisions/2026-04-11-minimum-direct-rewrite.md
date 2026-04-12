# Minimum-Direct Rewrite

**Date:** 2026-04-11 (drafted) / 2026-04-12 (revised after 5-persona review)
**Status:** Draft — awaiting user approval before implementation plan
**Scope:** `repos/dorea` — aggressive removal of legacy LUT pipeline. Workspace-side decision record only.
**Supersedes:** `2026-04-11-direct-pipeline-1440p-promotion.md`
**Related corvia entries:**
- `019d7a9d-d640-7470-849e-bfcbdc475522` — 1080p vs 1440p proxy bench (bilinear wins at 1440p, −8.3% ΔE)
- `2026-04-10-direct-mode-fp16-batch.md` — fp16 RAUNE + batch sizing (#67)
- `2026-04-10-direct-mode-3thread-pipeline.md` — 3-thread producer-consumer (#65)

## Summary

Strip `repos/dorea` down to just what the direct-mode RAUNE + OKLab-delta pipeline needs. Delete 5 Rust crates, 8 dorea-cli source files, 8 Python modules, 5 Python test files, and the LUT code path inside `grade.rs`. The binary becomes a single-purpose tool with no subcommand layer: `dorea <input>` runs the grade. Bump direct-mode defaults to 1440p proxy / batch=8 while we're already touching the code.

Incidentally fixes main's current broken-build state — the broken `stages` references live in LUT-pipeline code that's being deleted, and the broken `stage_mask` references (including one in the direct-mode path at `grade.rs:258`) are rewritten as part of `PipelineConfig` slimming.

Three commits (revised from 6 after review):
1. Rust: strip LUT code, delete dorea-cli LUT files, slim structs, bump defaults, drop vestigial `build.rs` + `cuda` feature.
2. Delete: 5 legacy crates, 8 Python modules, 5 test files, `transformers`/`Pillow` from `pyproject.toml`, `[maxine]`/`[inference]`/dead `[grade]` fields from `dorea.toml`, workspace `Cargo.toml` members.
3. README rewrite.

## Motivation

Three forces pushing toward the same outcome:

1. **Bench says direct+bilinear at 1440p is the win.** Corvia `019d7a9d`: at 1440p proxy, bilinear becomes best-ΔE among non-trivial upscales, classical methods collapse into a 1.8% band, learned SR (`sr_maxine`) is strictly worst.
2. **Direct mode has received all the recent perf investment.** `3b6626c` (3-thread pipeline, PR #65), `3f3ca2b` (fp16 + batch sizing, PR #67). The LUT pipeline is frozen and shipping with dead-code Maxine plumbing.
3. **Main is broken at HEAD.** `cargo build --release -p dorea-cli` on `main` fails with 7 errors — `stages` module not found (×4), `AdaptiveGrader` unresolved import, `PipelineConfig.stage_mask` missing (×2). The `stages` module does not exist in any branch. PR #63 merged a partial state that never compiled; PR #65 and #67 sit on top without touching it. The LUT pipeline is the concrete code that holds the broken references; deleting it (plus fixing the one direct-path `stage_mask: 0` reference) resolves the breakage as a side effect.

Promoting direct-mode-as-the-only-mode, at 1440p with batch=8, captures the bench finding, collapses the code surface to one path, and unbreaks main.

## Non-Goals

- **No algorithmic changes to direct mode.** Same RAUNE-Net weights, same bilinear upscale, same fp16 inference, same OKLab delta, same 3-thread pipeline, same Triton kernel. Only defaults change.
- **No new features.**
- **No migration path for existing `dorea.toml` files.** Fields for deleted sections are silently ignored (no `deny_unknown_fields` anywhere — verified in review).
- **No bench changes.** `benchmarks/upscale_bench/` stays on `feat/upscale-bench` unchanged.
- **No deferred `feat/upscale-bench` review items.** Out of scope.
- **No VRAM / throughput measurement.** Back-of-envelope only; smoke-run acceptance criterion only.
- **No rewrite of `raune_filter.py`.** The file stays as-is.

## Design

### What stays

**Crates (2):**
- `crates/dorea-cli` — slimmed (see below). **No `build.rs`**, **no `cuda` feature** — both vestigial after `dorea-gpu` is gone.
- `crates/dorea-video` — unchanged (`ffmpeg::probe`, `InputEncoding`, `OutputCodec`).

**`dorea-cli` files kept:**
- `src/main.rs` — single-command binary, no subcommand layer
- `src/lib.rs` — 3 modules: `config`, `grade`, `pipeline`
- `src/config.rs` — slimmed to direct-mode fields only
- `src/grade.rs` — LUT branch removed; also the `#[cfg(test)] mod tests` block (lines ~407–488) is dropped since its tests target deleted `build_inference_config` / Maxine helpers
- `src/pipeline/mod.rs` — `pub use grading::*;` and a slimmed `PipelineConfig` (no `stage_mask`, no LUT/depth/Maxine fields)
- `src/pipeline/grading.rs` — stripped to `DirectModeConfig` + `run_grading_stage_direct`
- `Cargo.toml` — deps on `dorea-color`, `dorea-lut`, `dorea-hsl`, `dorea-cal`, `dorea-gpu` removed; keep `dorea-video`; `[features] cuda = []` removed; no `[build-dependencies]` after `build.rs` is deleted

**Python kept:**
- `python/dorea_inference/__init__.py` — trivial `__version__`, unchanged
- `python/dorea_inference/raune_filter.py` — unchanged
- Whatever `raune_filter.py` imports transitively (numpy, torch, torchvision, PyAV, Triton — all external pip deps)

**Workspace root:**
- `Cargo.toml` — members list shrinks from 7 to 2 (`dorea-cli`, `dorea-video`); `[workspace.dependencies]` entries for the 5 deleted crates removed
- `dorea.toml` — `[maxine]` section removed; `[preview]` section removed; `[inference]` section removed (orphaned after rewrite — the direct path uses `[grade].raune_proxy_size`, not `[inference].proxy_size`); `[grade]` reduced to `raune_proxy_size` + `direct_batch_size`
- `python/pyproject.toml` — `transformers` and `Pillow` dependencies dropped (only consumed by the to-be-deleted `depth_anything.py` and `protocol.py`)
- `README.md` — minimal rewrite

### What gets deleted

**Crates (5 wholesale):**
- `crates/dorea-color/` — CIELab/OKLab/HSV/DLogM Rust math. Unused after LUT pipeline is gone; direct mode does OKLab in Python/Triton. Verified in review: also referenced by `dorea-cli/Cargo.toml:11` which is dropped as part of commit 1.
- `crates/dorea-lut/`
- `crates/dorea-hsl/`
- `crates/dorea-cal/`
- `crates/dorea-gpu/`

**`dorea-cli` files deleted:**
- `src/calibrate.rs`
- `src/preview.rs`
- `src/probe.rs` (per user decision: grade-only binary)
- `src/change_detect.rs`
- `src/optical_flow.rs`
- `src/pipeline/calibration.rs`
- `src/pipeline/feature.rs`
- `src/pipeline/keyframe.rs`
- `build.rs` (nvcc probe, vestigial after `dorea-gpu` delete)
- The `#[cfg(test)] mod tests` block inside `src/grade.rs` (LUT-pipeline tests that reference deleted helpers)

**Python modules deleted:**
- `python/dorea_inference/bridge.py` — server-mode glue
- `python/dorea_inference/server.py` — gRPC/pipe server for the LUT pipeline
- `python/dorea_inference/maxine_enhancer.py` — Maxine VideoSuperRes wrapper
- `python/dorea_inference/__main__.py` — imports `from .server import main`, dies when `server.py` is gone
- `python/dorea_inference/depth_anything.py` — depth inference for LUT depth zones
- `python/dorea_inference/yolo_seg.py` — YOLO-seg for class-mask grading
- `python/dorea_inference/protocol.py` — wire format, only consumed by bridge/server
- `python/dorea_inference/raune_net.py` — wrapper; `raune_filter.py` imports `from models.raune_net import RauneNet` via direct `sys.path` injection of the models dir, NOT through this wrapper

**Python tests deleted:**
- `python/tests/test_bridge_maxine.py`
- `python/tests/test_lifecycle_server.py` — imports `dorea_inference.server` + `.maxine_enhancer`
- `python/tests/test_maxine_server.py` — imports `dorea_inference.server`
- `python/tests/test_maxine_protocol.py` — imports `dorea_inference.protocol`
- `python/tests/test_infer_batch.py` — imports `dorea_inference.raune_net` + `.depth_anything` + `.protocol`

### Default changes

| Setting | Old | New |
|---|---:|---:|
| `raune_proxy_size` built-in default | `1080` | `1440` |
| `direct_batch_size` built-in default | `4` | `8` |

Rationale unchanged from superseded spec.

### Slimmed `GradeArgs` (canonical shape)

```rust
#[derive(Parser, Debug)]
#[command(name = "dorea", about = "Underwater video direct-mode color grading")]
pub struct GradeArgs {
    /// Input video file.
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

    /// Frames per RAUNE batch (config: [grade].direct_batch_size, default: 8)
    #[arg(long)]
    pub direct_batch_size: Option<usize>,

    /// Input encoding override (config: [grade].input_encoding)
    #[arg(long)]
    pub input_encoding: Option<String>,

    /// Output codec override (config: [grade].output_codec)
    #[arg(long)]
    pub output_codec: Option<String>,

    /// Verbose (debug) logging
    #[arg(long)]
    pub verbose: bool,
}
```

Gone: `warmth`, `strength`, `contrast`, `proxy_size`, `depth_skip_threshold`, `depth_max_interval`, `fused_batch_size`, `depth_zones`, `base_lut_zones`, `scene_threshold`, `maxine_upscale_factor`, `stages`, `flat`, `direct`, `mode`, `cpu_only`, `calibration`, and any other fields that existed only to support the LUT pipeline.

### Slimmed `PipelineConfig`

```rust
pub struct PipelineConfig {
    pub input: PathBuf,
    pub input_encoding: InputEncoding,
    pub output_codec: OutputCodec,
}
```

**No `stage_mask` field.** The current `grade.rs:258` literal `stage_mask: 0,` (in the direct-mode branch) is dropped. The current `grade.rs:337` literal `stage_mask,` (in the LUT branch) vanishes when the LUT branch is deleted. `PipelineConfig` has no such field anywhere after the rewrite.

### Slimmed `main.rs`

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

No `Command` enum. `GradeArgs` is the top-level CLI struct with `#[command(name = "dorea")]`. User runs `dorea input.mp4` with no subcommand.

## Commit structure (branch `feat/minimum-direct`, against `chunzhe10/dorea` main)

Three commits, each leaves the workspace building.

### Commit 1: `refactor(dorea-cli): collapse to direct-only, bump defaults`

Biggest commit — does all the in-place editing and file deletion inside `dorea-cli`. After this commit, `dorea-cli` no longer references any of the 5 legacy crates, so commit 2 can delete them cleanly.

Touches:
- `crates/dorea-cli/src/grade.rs` — delete everything after the `if direct_mode { ... return Ok(()); }` block (lines ~284 onwards), delete the LUT-specific arg fields, delete the `stage_mask: 0,` inside the direct block, delete the `#[cfg(test)] mod tests` at lines ~407–488. Slim `GradeArgs` to the canonical shape above. Bump `raune_proxy_size` fallback to 1440, bump `direct_batch_size` fallback to 8. Fix stale doc comments.
- `crates/dorea-cli/src/config.rs` — drop `MaxineDefaults`, drop all `GradeDefaults` fields except `raune_proxy_size`, `direct_batch_size`, `input_encoding`, `output_codec`; drop `InferDefaults` entirely (`[inference]` is being removed); drop `depth_model` from `ModelDefaults`.
- `crates/dorea-cli/src/pipeline/mod.rs` — slim `PipelineConfig` to the 3-field shape above; re-export only `grading::*`.
- `crates/dorea-cli/src/pipeline/grading.rs` — delete everything except `DirectModeConfig` and `run_grading_stage_direct`. Adjust `run_grading_stage_direct` to use the new slim `PipelineConfig`.
- `crates/dorea-cli/src/lib.rs` — reduce to `pub mod config; pub mod grade; pub mod pipeline;` (3 modules).
- `crates/dorea-cli/src/main.rs` — rewrite to the slim shape shown above (single-command binary).
- **Delete:** `crates/dorea-cli/src/{calibrate,preview,probe,change_detect,optical_flow}.rs`, `crates/dorea-cli/src/pipeline/{calibration,feature,keyframe}.rs`, `crates/dorea-cli/build.rs`.
- `crates/dorea-cli/Cargo.toml` — drop deps on `dorea-color`, `dorea-lut`, `dorea-hsl`, `dorea-cal`, `dorea-gpu`; drop `[features] cuda = []`; remove `build = "build.rs"` if present.

**Post-commit-1 state:** `cargo build --release -p dorea-cli` passes (this is the commit that unbreaks main). The 5 legacy crates still exist in the tree as orphans — nothing depends on them.

### Commit 2: `chore: delete legacy crates, orphan Python modules, stale config`

Pure deletion commit. No behavior changes, no in-place editing except to manifests and `dorea.toml`.

- Delete `crates/dorea-color/`, `crates/dorea-lut/`, `crates/dorea-hsl/`, `crates/dorea-cal/`, `crates/dorea-gpu/`.
- `Cargo.toml` (workspace root) — remove the 5 crate names from `members`, remove the 5 entries from `[workspace.dependencies]`.
- `Cargo.lock` — regenerates on next `cargo build`; included in commit as a generated artifact.
- Delete Python modules: `python/dorea_inference/{bridge,server,maxine_enhancer,__main__,depth_anything,yolo_seg,protocol,raune_net}.py`.
- Delete Python tests: `python/tests/{test_bridge_maxine,test_lifecycle_server,test_maxine_server,test_maxine_protocol,test_infer_batch}.py`.
- `python/pyproject.toml` — drop `transformers`, `Pillow` from the dependencies list.
- `dorea.toml` (repo root) — delete `[maxine]`, `[preview]`, `[inference]` sections; delete all `[grade]` fields except `raune_proxy_size` and `direct_batch_size` (if they're even present in the current file); `[models]` keeps `python`, `raune_weights`, `raune_models_dir`; drop `depth_model`.

**Post-commit-2 state:** `cargo build --release` passes, `cargo test` passes across the 2 remaining crates, `pytest python/tests/` passes (only tests against surviving modules run).

### Commit 3: `docs: rewrite README for minimum-direct`

Short stub replacing the currently-stale `repos/dorea/README.md` (which still calls `dorea grade` "Phase 3 — not yet implemented"). Must include:
- One-liner description: underwater video color grading, direct-mode only
- Build instructions: `cargo build --release`
- Usage: `dorea path/to/clip.mp4` (single positional, no subcommand)
- Hardware requirement: 6 GB VRAM (RTX 3060+)
- Config location: `dorea.toml` with `[models]` and `[grade]` sections
- Tunables: `--raune-proxy-size` (default 1440) and `--direct-batch-size` (default 8), with the "values above 8 may regress on 6 GB" caveat
- **Breaking changes section**: "This release removes the `calibrate`, `preview`, `probe` subcommands and the 3D LUT / depth-zones / YOLO-seg pipeline. `dorea grade` → `dorea`; positional `<input>` replaces the old `--input` flag. Existing `dorea.toml` files will parse but fields for removed features are silently ignored."

## Acceptance criteria

- `cd repos/dorea && cargo build --release` passes on the feature branch (currently fails on main).
- `cargo test` passes across `dorea-cli` and `dorea-video`.
- `dorea --help` shows positional `<INPUT>` usage, no subcommand menu.
- `./target/release/dorea footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4` runs on the RTX 3060 6 GB workstation without OOM and produces a valid output file that plays.
- `python -m pytest python/tests/ --collect-only` collects cleanly (no ImportError in deleted modules).
- `python -m dorea_inference.raune_filter --help` runs without error (direct invocation path the Rust side uses).
- `grep -r "dorea_gpu\|dorea_color\|dorea_lut\|dorea_hsl\|dorea_cal\|maxine\|stages::format\|stage_mask\|AdaptiveGrader" crates/dorea-cli/src/` returns no hits.
- `grep -r "bridge\|server\|maxine_enhancer" python/dorea_inference/` returns no hits.
- Workspace `Cargo.toml` `members` has exactly 2 entries: `crates/dorea-cli`, `crates/dorea-video`.

**Not acceptance criteria:** formal VRAM or throughput measurement, performance comparison to prior default, preservation of any LUT-pipeline functionality.

## Risks

- **Broken main base.** Commit 1 is the commit that unbreaks it. Revert during development: `git reset --hard origin/main` returns to the (broken) known state.
- **VRAM fit at 1440p batch=8** — back-of-envelope ~1.1–1.2× the known-safe 1080p batch=4 fp32 ceiling. Not re-measured. Failure mode is loud CUDA OOM.
- **Post-merge rollback** is `git revert -m 1 <merge-sha>` which re-creates 5 crate directories. Mitigation: **before merging, tag `origin/main` as `pre-minimum-direct`** so rollback becomes a reset-to-tag instead of a revert-with-conflicts.
- **In-flight origin branches will conflict-hell after merge.** Six branches touch deleted paths (identified in review):
  - `feat/15-maxine-arch-improvements` — `calibrate.rs`, `preview.rs`
  - `feat/15-maxine-enhancement` — `calibrate.rs`, `preview.rs`, `dorea-lut`
  - `feat/27-cuda-grader-buffer-prealloc` — entirely in `dorea-gpu`
  - `feat/28-combined-lut-cuda-texture` — entirely in `dorea-gpu` (~11 files of active work)
  - `feat/oklab-grade-pipeline` — `dorea-color`, `dorea-gpu`
  - `feat/streaming-calibration` — `dorea-lut`

  User's decision whether to close, rebase, or warn. Not a gate on this PR.
- **`python -m dorea_inference` entrypoint is being removed** via `__main__.py` deletion. The direct-mode Rust path invokes `python -m dorea_inference.raune_filter` (module path), which is unaffected. Any external script still using `python -m dorea_inference` will break.
- **Stale doc comment in `raune_filter.py:42`**: `# OKLab conversion (matches dorea-color/src/lab.rs)`. References a deleted crate but is only a comment, no runtime impact. Left alone per "no rewrite of `raune_filter.py`" non-goal.

## Out of scope (explicit)

- Any change to `benchmarks/upscale_bench/` on `feat/upscale-bench`.
- Any upscale-method change (bilinear wins at 1440p).
- Any change to `raune_filter.py` or the Triton kernel.
- Any change to the 3-thread producer-consumer pipeline.
- Migration helpers, deprecation shims, cross-version config translators.
- Post-merge measurement on a long clip (discovery work, separate issue if needed).
