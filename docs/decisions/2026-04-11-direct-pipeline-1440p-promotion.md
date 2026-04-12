# Direct Pipeline: 1440p Promotion + Default-Mode Flip

> **SUPERSEDED by `2026-04-11-minimum-direct-rewrite.md`.** Mid-brainstorm the user
> pivoted from "flip default + soft-deprecate LUT" to "delete legacy entirely,
> keep only minimum direct." This doc is kept for history — do not execute the
> plan below. The new scope also incidentally fixes main's broken `stages` /
> `stage_mask` references by deleting the code that held them.

# Direct Pipeline: 1440p Promotion + Default-Mode Flip

**Date:** 2026-04-11
**Status:** SUPERSEDED — see note above.
**Scope:** `repos/dorea` (Rust CLI + docs). Separate decision record in `dorea-workspace`.
**Related corvia entries:**
- `019d7a7b-458c-7033-ad34-56d84fdfb0a6` — Delta upscale bench, first real run (1080p, 9 methods)
- `019d7a9d-d640-7470-849e-bfcbdc475522` — Delta upscale bench, 1080p vs 1440p proxy comparison
- `2026-04-10-direct-mode-fp16-batch.md` — fp16 RAUNE inference + batch sizing (#67)
- `2026-04-10-direct-mode-3thread-pipeline.md` — 3-thread producer-consumer (#65)

## Summary

Three coupled changes to `dorea grade`, landing as one PR on `chunzhe10/dorea`:

1. **Flip default grading mode from `lut` to `direct`.** `dorea grade clip.mp4` with no flags runs direct mode. New `--mode direct|lut` flag is the canonical CLI surface. `--direct` boolean becomes a deprecated alias.
2. **Bump direct-mode RAUNE proxy default from 1080 to 1440, and `direct_batch_size` default from 4 to 8.** Also fix the stale `"default: 1920"` doc comment at `grade.rs:120`.
3. **Soft-deprecate the LUT pipeline.** `log::warn!` at startup when `mode = "lut"` is selected. README reorders direct-first; LUT moves under "Legacy / advanced." CI is unchanged — both paths still run on every push.

Maxine removal is tracked as a separate follow-up issue (see Follow-ups). No bench changes. No upscale-method changes. No new benchmarks. No cargo features. No file moves.

## Motivation

The delta upscale bench (`019d7a9d`) established two load-bearing facts:

- At **1440p proxy**, `bilinear` becomes the best-ΔE non-trivial upscale method (4.574 vs 4.988 at 1080p, **−8.3%**). All classical methods gain ~5–9% ΔE vs 1080p.
- Learned SR (`sr_maxine`) is **strictly worst** at both 1080p and 1440p and does not close the gap with more pixels.

The bench's upscale-only timing is insensitive to proxy size for bilinear (0.68 ms both), so the upscale stage itself pays nothing for 1440p. The quality improvement comes "for free" from the upscale's perspective.

Meanwhile, direct mode is now the high-performance path in dorea grade. It has received the recent perf investment: `3b6626c` (3-thread producer-consumer, #65), `3f3ca2b` (fp16 RAUNE + batch sizing, #67). It runs roughly 4+ fps on 4K 120fps HEVC after those two changes. The LUT pipeline is heavier, more complex, and depends on paths (Maxine enhancement, adaptive depth zones, motion-aware grading) that have known issues and are not currently exercised in production.

Promoting direct mode to default, at 1440p, captures the bench's quality finding and aligns the default CLI experience with the code path that actually receives ongoing perf work.

## Non-Goals

- **No upscale method change.** Bench says `bilinear` wins at 1440p. Not re-litigating.
- **No benchmark harness changes.** The bench stays on `feat/upscale-bench` unchanged. No new bench package in this work.
- **No deleted legacy code.** The LUT pipeline remains in-tree, builds on the default `cargo build` path, and still runs on every CI push.
- **No VRAM or throughput measurement** beyond a one-clip smoke test on the existing `DJI_20251101111428_0055_D_3s.MP4`. No SLO. No formal mini-bench. Back-of-envelope VRAM math only.
- **No deferred `feat/upscale-bench` review items.** `git_sha` header bug in `visualize.py`, SSIM silent torchmetrics fallback in `metrics.py`, `--gold-sanity-check` broad except in `run.py`, gold cache key missing `force_path`, dead `--verbose` flag, `frames_data` GPU retention — file as separate issues, do not touch in this branch.
- **No Maxine removal** in this PR. Follow-up issue tracks it separately.

## Design

### CLI surface

**New canonical flag:**
```
--mode <direct|lut>    Grading mode (default: direct)
```

Resolution order: `args.mode` → `cfg.grade.mode` → `"direct"` (new built-in default).

**Deprecated alias:**
```
--direct    [DEPRECATED] alias for --mode direct
```

When `--direct` is passed, the CLI logs once at startup:

> `--direct is deprecated; use --mode direct (now the default).`

and sets the resolved mode to `"direct"`.

**Soft deprecation of LUT mode:**

When the resolved mode is `"lut"` (whether from `--mode lut`, `cfg.grade.mode = "lut"`, or `--direct` not being passed but the user opting back into LUT explicitly), the CLI logs once at startup:

> `Legacy LUT pipeline selected. Direct mode is the default; see README for details.`

LUT mode continues to work exactly as today. No functional change — warning only.

### Default changes

| Setting | Location | Old | New |
|---|---|---:|---:|
| Grading mode (built-in default) | `grade.rs` mode-resolution path | `"lut"` (effective) | `"direct"` |
| `raune_proxy_size` (built-in default) | `grade.rs:210` `.unwrap_or(1080)` | `1080` | `1440` |
| `direct_batch_size` (built-in default) | `grade.rs:223` `.unwrap_or(4)` | `4` | `8` |
| Stale doc comment | `grade.rs:120` | `"default: 1920"` | `"default: 1440"` |
| Stale doc comment | `config.rs:56` | `"default: 1920"` | `"default: 1440"` |

The `direct_batch_size` bump from 4 to 8 is the deferred implementation of the decision in `2026-04-10-direct-mode-fp16-batch.md`. fp16 halves activation memory, so batch=8 fp16 ≈ batch=4 fp32 footprint. At 1440p, activation memory scales ~2.25× vs 1080p, giving batch=8 fp16 at 1440p ≈ 1.1–1.2× the known-safe ceiling of batch=4 fp32 at 1080p. Should fit; failure mode is loud (CUDA OOM), not silent.

### Files changed

**`crates/dorea-cli/src/grade.rs`:**
- Add `mode: Option<String>` field to `GradeArgs` with `#[arg(long, value_parser = ["direct", "lut"])]`.
- Mark existing `direct: bool` field as deprecated in its doc comment.
- Rewrite the mode-resolution gate so `mode` is canonical and `--direct` aliases to it with a deprecation warning.
- Change the final-fallback default from effectively `"lut"` to `"direct"`.
- Add the one-line `log::warn!` soft-deprecation when resolved mode is `"lut"`.
- Change `.unwrap_or(1080_usize)` → `.unwrap_or(1440_usize)` on the `raune_proxy_size` fallback.
- Change `.unwrap_or(4)` → `.unwrap_or(8)` on the `direct_batch_size` fallback.
- Fix line 120 stale doc comment (`"default: 1920"` → `"default: 1440"`).
- Update the line 124 batch-size doc comment to say `(default: 8)` and reference fp16 activation headroom.

**`crates/dorea-cli/src/config.rs`:**
- Line 54 doc comment: `Grading mode: "lut" (default) or "direct"` → `Grading mode: "direct" (default) or "lut" (legacy)`.
- Line 56 doc comment: `"default: 1920"` → `"default: 1440"`.
- No struct-shape changes.

**`README.md`** (in `repos/dorea`):
- Reorder grading section so direct mode is the first example. Lead with `dorea grade clip.mp4` as the default one-liner.
- Move the LUT pipeline description into a lower-ranked "Legacy: LUT pipeline" section.
- Add a paragraph under the legacy section: *"Direct mode is now the default. The LUT pipeline remains available for workflows that need its features (adaptive depth zones, motion-aware grading) but is no longer maintained for new features."*

**`CHANGELOG.md`** (if it exists):
- Breaking change entry: *"Default grading mode is now `direct`. Users relying on the previous LUT default must pass `--mode lut` or set `[grade].mode = "lut"` in `dorea.toml`."*
- Note direct-mode default bumps: `raune_proxy_size` 1080→1440, `direct_batch_size` 4→8.

**`dorea.toml.example`** (if it exists and mentions `[grade].mode`):
- Update any commented example to reflect the new default.

**CI workflows** (`.github/workflows/*.yml`):
- Audit for any job that invokes `dorea grade` and either assumes the old LUT default or passes `--direct`. Update or drop flags as appropriate. No structural matrix changes.

**Verification-only reads (may produce zero-line changes):**
- `crates/dorea-cli/src/pipeline/grading.rs::run_grading_stage_direct` — confirm nothing hardcodes 1080 in the subprocess invocation.
- `python/dorea_inference/raune_filter.py::_process_batch` (or equivalent) — confirm no hardcoded proxy buffer sizes. The bench verified 1440p works end-to-end at this layer, so no changes are expected.

### Files NOT changed

- `crates/dorea-cli/src/calibrate.rs`, `preview.rs`, `probe.rs`
- `crates/dorea-cli/src/pipeline/calibration.rs`, `feature.rs`, `keyframe.rs`, `mod.rs`
- `crates/dorea-gpu/**`
- Python modules (direct-mode Python subprocess already accepts `--raune-proxy-size`)
- `dorea.toml` default (separate from `dorea.toml.example`)
- `.mcp.json`, `AGENTS.md`, `CLAUDE.md`, corvia config

## Branch and commit structure

Single branch on `chunzhe10/dorea`: `feat/direct-default-1440p` (final name decided in plan phase).

Proposed commit layout, each individually revertible:

1. `refactor(grade): add --mode flag, deprecate --direct boolean` — additive. Establishes new CLI surface. `mode = "lut"` still resolves to LUT. No default change, no behavior change.
2. `feat(grade): flip default grading mode to direct` — the breaking-change commit. One-line default change + soft-deprecation `log::warn!` for LUT mode.
3. `perf(direct): bump raune proxy default to 1440p, batch size to 8` — the proxy + batch default bumps + stale comment fixes.
4. `docs: reorder README, direct-first` — README, CHANGELOG, any `dorea.toml.example` update.
5. `ci: audit grade invocations for new default mode` — conditional; only if the workflow audit finds something.

Commit 2 is the breaking-change boundary. Revert strategy: `git revert` the merge commit restores all defaults and CLI surface cleanly. No persistent state, no migration.

PR opens against `chunzhe10/dorea master` with `gh pr create --repo chunzhe10/dorea` (explicit repo flag — `gh` defaults to the wrong repo in this workspace).

Workspace-side: this decision record plus the implementation plan land as a separate commit on `chunzhe10/dorea-workspace master`. Not linked to the dorea PR; can land before or after.

## Acceptance criteria

- `cargo build --release` passes on the dorea branch.
- `cargo test` passes, including the existing Maxine-disabled assertion test (unchanged).
- `dorea grade --help` shows `--mode` as the canonical flag. `--direct` is present but documented as deprecated.
- `dorea grade clip.mp4` with no flags and no config override:
  - Runs direct mode at 1440p, batch=8.
  - Does not OOM on `DJI_20251101111428_0055_D_3s.MP4` on the RTX 3060 6GB workstation.
  - Completes without error.
- `dorea grade --mode lut clip.mp4` runs the LUT pipeline and emits the soft-deprecation warning once at startup.
- `dorea grade --direct clip.mp4` runs direct mode and emits the `--direct` deprecation alias warning.
- README's first grading example is direct mode with no flags.
- No new dependencies, no files moved under `legacy/`, no cargo features added.

**Not acceptance criteria:**
- No formal throughput measurement.
- No formal VRAM peak measurement beyond `nvidia-smi` eyeball during the smoke test.
- No comparison against prior 1080p performance numbers.

## Risks

**VRAM fit at 1440p batch=8.** Not measured. Back-of-envelope: ~1.1–1.2× the known-safe ceiling of batch=4 fp32 at 1080p. Probably fits. Failure mode is loud (CUDA OOM). Mitigation: existing batch-size validation in `grade.rs:230-242` still warns on batch > 8 and errors on batch > 32, so users hitting OOM can drop to `--direct-batch-size 4`. User accepted this risk (Q4 in brainstorm).

**Throughput regression.** RAUNE proxy inference at 1440p runs on ~2.25× more pixels. End-to-end throughput regression may be less than 2.25× thanks to the 3-thread pipeline overlapping GPU time with decode/encode, but it is non-zero. User accepted this risk (Q5 in brainstorm — no SLO, ship and see).

**Silent behavior change for scripts.** Any script that calls `dorea grade clip.mp4` bare and expects LUT-mode output (3D LUT export, adaptive-zone metadata) silently switches to direct mode on upgrade. These users do not see the soft-deprecation warning because they are getting direct mode. Mitigation: CHANGELOG breaking-change entry; no programmatic safeguard.

**Noisy `--direct` deprecation warning.** Scripts still passing `--direct` explicitly emit the alias warning on every run until updated. Acceptable.

## Follow-ups (separate issues, NOT in this PR)

1. **Maxine dead-code removal** (`chunzhe10/dorea`). Maxine is already hard-disabled at runtime (`grade.rs:334`, `grade.rs:402`, `calibrate.rs:204`, `preview.rs:111`, and the assertion at `grade.rs:450`). The Python `maxine_enhancer.py`, bridge.py Maxine functions, server.py handlers, `MaxineDefaults`, `maxine_upscale_factor`, `maxine_in_fused_batch`, hard-coded `maxine: false` fields, and `test_bridge_maxine.py` can all be deleted without functional change. Track as a separate cleanup issue; reference this spec as motivation.

2. **Post-land VRAM and throughput measurement** (`chunzhe10/dorea`). Run direct mode at 1440p batch=8 on a long real clip, capture `nvidia-smi` peak VRAM and wall-clock fps, record as a corvia finding. Not a gate on this PR.

3. **Deferred `feat/upscale-bench` review items** — already known, file as separate issues on that branch, do not touch in this PR.

## Out of scope (explicit)

- Upscale method changes (`bilinear` is the bench-validated winner at 1440p).
- New benchmark harness or bench package changes.
- Any changes to Maxine code in this PR.
- Moving LUT pipeline code under `legacy/`, adding cargo features, or gating compilation.
- Retiring or deleting the LUT pipeline.
- Any changes to `dorea.toml` `[inference].proxy_size = 518` (separate path, serves calibrate).
- Changes to the 3-thread queue capacities, decoder/encoder tuning, or VRAM scheduler.
