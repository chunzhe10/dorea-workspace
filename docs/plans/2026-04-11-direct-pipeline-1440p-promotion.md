# Direct Pipeline 1440p Promotion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip dorea grade's default mode from `lut` to `direct`, bump direct mode's RAUNE proxy default from 1080p to 1440p and batch size from 4 to 8, soft-deprecate the LUT pipeline.

**Architecture:** Single branch on `chunzhe10/dorea`, four small Rust commits + one docs commit, no Python changes, no new tests, no CI changes. Backed by delta upscale bench (bilinear wins at 1440p, −8.3% ΔE) and fp16 RAUNE activation headroom decision.

**Tech Stack:** Rust (`cargo build`, `cargo test`, `clap` args), Python (unchanged — already parameterized), `gh` CLI for PR and follow-up issues (always with `--repo chunzhe10/dorea`).

**Spec:** `docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md`

**Deviations from spec (discovered during plan phase):**
- `CHANGELOG.md` does not exist in `repos/dorea`. Spec's CHANGELOG entry task is dropped.
- `dorea.toml.example` does not exist. Spec's .example update task is dropped.
- `.github/workflows/` does not exist. Spec's CI-audit task (commit 5) is dropped.
- `README.md` is badly stale ("Phase 3 — not yet implemented"). Rather than reordering a non-existent grading section, Task 6 adds a new minimal grading section. A full README rewrite is a separate follow-up.
- Verification items 5 and 6 from the spec are confirmed **zero-change** during plan phase: `grading.rs` passes `--proxy-width`/`--proxy-height` parameterized on `raune_proxy_size`, and `raune_filter.py` reads them from argparse. No hardcodes. The plan still includes explicit re-verification as Task 5, but it's expected to produce no edits.

**Effective commit count:** 4 Rust/code commits + 1 docs commit = 5 commits on the branch.

---

## Prerequisites

- Working directory: `/workspaces/dorea-workspace`
- `repos/dorea` on `master`, clean tree
- `/opt/dorea-venv` has pytest, PyAV, and direct-mode Python deps installed (per workspace memory — may need `setup_bench.sh` if rebuilt)
- `gh` CLI authenticated
- RTX 3060 6GB workstation accessible (for smoke test in Task 7)
- Test clip available at `footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4`

---

## File Structure

**Files modified (all in `repos/dorea`):**
- `crates/dorea-cli/src/grade.rs` — CLI args, direct-mode gate, defaults, soft-deprecation warning, doc comments
- `crates/dorea-cli/src/config.rs` — two stale doc comments
- `README.md` — new minimal grading section

**Files verified but not edited (expected zero-change):**
- `crates/dorea-cli/src/pipeline/grading.rs` — `DirectModeConfig` and Python subprocess invocation
- `python/dorea_inference/raune_filter.py` — `run_single_process` proxy size handling

**Files NOT touched:**
- `crates/dorea-gpu/**`, `calibrate.rs`, `preview.rs`, `probe.rs`, other `pipeline/*.rs` modules, any Python module other than verification reads, `dorea.toml`

---

## Task 0: Create feature branch on repos/dorea

**Files:** none yet.

- [ ] **Step 1: Confirm clean working tree in repos/dorea**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git status --short && git branch --show-current
```

Expected: no output from `git status`, branch is `master`.

If there are uncommitted changes, stop and ask the user what to do — do NOT stash or discard.

- [ ] **Step 2: Pull latest master**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git fetch origin && git pull --ff-only origin master
```

Expected: fast-forward or already up to date.

- [ ] **Step 3: Create and check out the feature branch**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git checkout -b feat/direct-default-1440p
```

Expected: `Switched to a new branch 'feat/direct-default-1440p'`.

- [ ] **Step 4: Baseline build to confirm master compiles cleanly on this machine**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release -p dorea-cli 2>&1 | tail -20
```

Expected: `Finished \`release\` profile [optimized] target(s) in ...`. If this fails on master, stop — it's not a task regression, it's an environment issue.

---

## Task 1: Add `--mode` flag, deprecate `--direct` boolean

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (GradeArgs struct + mode resolution)
- Modify: `crates/dorea-cli/src/config.rs` (one doc comment)

**Intent:** Additive CLI change. No default behavior changes in this commit. `mode = "lut"` still resolves to LUT mode. `--direct` still works but emits a deprecation warning. `--mode direct|lut` becomes the canonical flag.

- [ ] **Step 1: Add `mode` field to `GradeArgs` in `grade.rs`**

In `crates/dorea-cli/src/grade.rs`, find the `GradeArgs` struct around line 116. The existing `direct` field looks like:

```rust
    /// Direct mode: per-frame RAUNE, no LUT pipeline (skips keyframe/calibration/CUDA stages)
    #[arg(long)]
    pub direct: bool,
```

Replace with:

```rust
    /// Grading mode: `direct` (per-frame RAUNE, default) or `lut` (legacy multi-stage pipeline)
    #[arg(long, value_parser = ["direct", "lut"])]
    pub mode: Option<String>,

    /// [DEPRECATED] alias for `--mode direct`. Emits a warning when used.
    #[arg(long)]
    pub direct: bool,
```

- [ ] **Step 2: Update the stale doc comment on `raune_proxy_size` and `direct_batch_size`**

Still in `grade.rs`, find lines 120–128:

```rust
    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1920)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,

    /// Frames per batch in direct mode (default: 4). On RTX 3060 (6GB), values
    /// above 8 show diminishing returns due to PCIe upload overhead in the
    /// per-frame loop in _process_batch. batch=8 ~4.36 fps, batch=16 ~3.47 fps.
    #[arg(long)]
    pub direct_batch_size: Option<usize>,
```

Replace with (new defaults will be applied in Task 3, but comment changes land here to keep doc-comment edits together):

```rust
    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1440)
    #[arg(long)]
    pub raune_proxy_size: Option<usize>,

    /// Frames per batch in direct mode (default: 8). fp16 RAUNE halves activation
    /// memory vs fp32, so batch=8 fp16 ≈ batch=4 fp32 footprint — known-safe on
    /// RTX 3060 (6GB). Values above 8 show diminishing returns due to per-frame
    /// upload overhead in `_process_batch`. Measured: batch=8 ~4.36 fps, batch=16 ~3.47 fps.
    #[arg(long)]
    pub direct_batch_size: Option<usize>,
```

- [ ] **Step 3: Update mode resolution logic in `grade.rs`**

Find line 206:

```rust
    let direct_mode = args.direct || cfg.grade.mode.as_deref() == Some("direct");
    if direct_mode {
```

Replace with:

```rust
    // Resolve grading mode: CLI --mode → CLI --direct alias → config → default "lut" (Task 2
    // flips the default to "direct").
    let resolved_mode: String = if let Some(m) = args.mode.as_deref() {
        m.to_string()
    } else if args.direct {
        log::warn!("--direct is deprecated; use --mode direct (now the default)");
        "direct".to_string()
    } else if let Some(m) = cfg.grade.mode.as_deref() {
        m.to_string()
    } else {
        "lut".to_string()
    };

    let direct_mode = resolved_mode == "direct";
    if direct_mode {
```

- [ ] **Step 4: Update the stale `[grade].mode` doc comment in `config.rs`**

In `crates/dorea-cli/src/config.rs`, find line 54:

```rust
    /// Grading mode: `"lut"` (default) or `"direct"` (per-frame RAUNE, no LUT pipeline)
    pub mode: Option<String>,
```

Replace with (Task 2 will make this accurate about the default):

```rust
    /// Grading mode: `"lut"` or `"direct"` (per-frame RAUNE, no LUT pipeline).
    /// Built-in default is set at CLI resolution time; see `grade.rs` for current value.
    pub mode: Option<String>,
```

Also find line 56:

```rust
    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1920)
    pub raune_proxy_size: Option<usize>,
```

Replace with:

```rust
    /// RAUNE proxy resolution for direct mode (long-edge pixels, default: 1440)
    pub raune_proxy_size: Option<usize>,
```

- [ ] **Step 5: Verify compilation**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -20
```

Expected: `Finished \`dev\` profile [unoptimized + debuginfo] target(s)`. No warnings about unused `args.direct`.

If `args.direct` triggers an "unused field" clippy warning, that's fine — it IS used in the deprecation alias branch. If compilation fails on the `value_parser = ["direct", "lut"]` attribute, check that `clap` version supports that syntax (it should; dorea-cli uses clap 4.x).

- [ ] **Step 6: Run existing tests**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli 2>&1 | tail -30
```

Expected: all tests pass. The `grade.rs:450` Maxine-disabled test is unchanged and should still pass.

- [ ] **Step 7: Commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add crates/dorea-cli/src/grade.rs crates/dorea-cli/src/config.rs && git commit -m "$(cat <<'EOF'
refactor(grade): add --mode flag, deprecate --direct boolean

Adds `--mode direct|lut` as the canonical grading mode flag. Keeps
`--direct` as a deprecated alias that emits a warning on use. Also
fixes stale doc comments on `raune_proxy_size` (was: default 1920)
and the direct_batch_size comment (documenting the fp16 headroom).

This commit is additive — no default behavior changes. `mode = "lut"`
still resolves to the LUT pipeline, same as before. Task 2 flips the
default; Task 3 bumps the proxy and batch defaults.

Refs docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit, two files changed.

---

## Task 2: Flip the default grading mode from `lut` to `direct`

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (default fallback + soft-deprecation warning)

**Intent:** The breaking-change commit. One-line default flip plus a single soft-deprecation warning emitted once at startup when resolved mode is `"lut"`. This is the commit that changes user-visible behavior.

- [ ] **Step 1: Flip the default in the resolved_mode fallback**

In `crates/dorea-cli/src/grade.rs`, find the block added in Task 1:

```rust
    } else if let Some(m) = cfg.grade.mode.as_deref() {
        m.to_string()
    } else {
        "lut".to_string()
    };
```

Replace the fallback literal:

```rust
    } else if let Some(m) = cfg.grade.mode.as_deref() {
        m.to_string()
    } else {
        "direct".to_string()
    };
```

- [ ] **Step 2: Add soft-deprecation warning for LUT mode**

Immediately after the `resolved_mode` block and before `let direct_mode = resolved_mode == "direct";`, add:

```rust
    if resolved_mode == "lut" {
        log::warn!(
            "Legacy LUT pipeline selected. Direct mode is the default; \
             see README for details."
        );
    }
```

The surrounding block should now look like:

```rust
    let resolved_mode: String = if let Some(m) = args.mode.as_deref() {
        m.to_string()
    } else if args.direct {
        log::warn!("--direct is deprecated; use --mode direct (now the default)");
        "direct".to_string()
    } else if let Some(m) = cfg.grade.mode.as_deref() {
        m.to_string()
    } else {
        "direct".to_string()
    };

    if resolved_mode == "lut" {
        log::warn!(
            "Legacy LUT pipeline selected. Direct mode is the default; \
             see README for details."
        );
    }

    let direct_mode = resolved_mode == "direct";
    if direct_mode {
```

- [ ] **Step 3: Verify compilation**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 4: Run tests**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo test -p dorea-cli 2>&1 | tail -30
```

Expected: all tests pass. None of the existing tests should assert on grade-mode default behavior (if one does, it was already tautological against the old default — see Step 5 for the bail-out procedure).

- [ ] **Step 5: If a test fails because it assumed LUT was the default**

If any test in `crates/dorea-cli/tests/` fails with an error along the lines of "expected LUT pipeline but got direct mode," that test was implicitly relying on the old default and needs to be updated to pass `--mode lut` explicitly. Update the test command invocation to add `--mode lut`, not the default. Do NOT change the code behavior back. If the user has explicit views on test philosophy here, ask before editing.

Expected: if Step 4 passed, skip Step 5.

- [ ] **Step 6: Commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add crates/dorea-cli/src/grade.rs && git commit -m "$(cat <<'EOF'
feat(grade): flip default grading mode to direct

Changes the built-in default mode from "lut" (effective) to "direct".
`dorea grade clip.mp4` with no flags now runs direct mode, skipping
the LUT/calibration/keyframe pipeline. Users can opt back into the
LUT pipeline via `--mode lut` or `[grade].mode = "lut"` in dorea.toml.

When LUT mode is selected, a soft-deprecation warning is emitted at
startup. The LUT pipeline remains functional and on the default
compile path — this only changes the default CLI behavior.

BREAKING CHANGE: scripts that run `dorea grade` bare and relied on
LUT-mode output (3D LUT export, adaptive-zone metadata, etc.) must
now pass `--mode lut` explicitly.

Refs docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit, one file changed.

---

## Task 3: Bump direct-mode RAUNE proxy default to 1440 and batch size to 8

**Files:**
- Modify: `crates/dorea-cli/src/grade.rs` (two `unwrap_or` calls in the direct-mode block)

**Intent:** The performance-tuning commit. Raises quality (−8.3% ΔE at 1440p per bench entry `019d7a9d`) and discharges the deferred fp16 batch bump from `2026-04-10-direct-mode-fp16-batch.md`.

- [ ] **Step 1: Bump `raune_proxy_size` default**

In `crates/dorea-cli/src/grade.rs`, find line 208:

```rust
        let raune_proxy_size = args.raune_proxy_size
            .or(cfg.grade.raune_proxy_size)
            .unwrap_or(1080_usize);
```

Replace with:

```rust
        let raune_proxy_size = args.raune_proxy_size
            .or(cfg.grade.raune_proxy_size)
            .unwrap_or(1440_usize);
```

- [ ] **Step 2: Bump `direct_batch_size` default**

Still in `grade.rs`, find line 221:

```rust
        let direct_batch_size = args.direct_batch_size
            .or(cfg.grade.direct_batch_size)
            .unwrap_or(4);
```

Replace with:

```rust
        let direct_batch_size = args.direct_batch_size
            .or(cfg.grade.direct_batch_size)
            .unwrap_or(8);
```

- [ ] **Step 3: Verify compilation and tests**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build -p dorea-cli 2>&1 | tail -10 && cargo test -p dorea-cli 2>&1 | tail -20
```

Expected: clean build, all tests pass. If any test asserts on old defaults (1080 or 4), it needs updating — same principle as Task 2 Step 5.

- [ ] **Step 4: Commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add crates/dorea-cli/src/grade.rs && git commit -m "$(cat <<'EOF'
perf(direct): bump raune proxy default to 1440p, batch size to 8

Raises the direct-mode RAUNE proxy default from 1080 to 1440. Delta
upscale bench (corvia 019d7a9d) showed bilinear becomes the best-ΔE
non-trivial upscale method at 1440p and all classical methods gain
~8% ΔE vs 1080p — a "free" quality win from the upscale side.

Also bumps direct_batch_size default from 4 to 8 to cash in the
deferred fp16 activation-memory headroom from PR #67. batch=8 fp16
≈ batch=4 fp32 footprint; 1440p activation is ~2.25× 1080p, so
1440p + batch=8 + fp16 ≈ 1.1-1.2× the known-safe 1080p + batch=4
+ fp32 ceiling. Back-of-envelope; measurement left as follow-up.

RAUNE inference at 1440p runs on ~2.25x more pixels, so per-frame
GPU time grows; the 3-thread producer-consumer pipeline partially
offsets this via decode/encode overlap. No SLO enforced.

Refs docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit, one file changed.

---

## Task 4: Verification reads (expected zero-change)

**Files:**
- Read: `crates/dorea-cli/src/pipeline/grading.rs` (`DirectModeConfig`, `run_grading_stage_direct`)
- Read: `python/dorea_inference/raune_filter.py` (`run_single_process`)

**Intent:** Confirm nothing hardcodes 1080 in the subprocess boundary. Confirmed during plan phase but re-verify on the committed branch to guard against any rebase drift. If anything IS hardcoded, fix it and commit.

- [ ] **Step 1: Re-verify `grading.rs` does not hardcode 1080**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && grep -n "1080\|raune_proxy" crates/dorea-cli/src/pipeline/grading.rs
```

Expected: only the `DirectModeConfig.raune_proxy_size` field at line ~319 and its pass-through into `proxy_dims` at line ~336 and the Python subprocess invocation at lines ~365-368 (`--proxy-width`, `--proxy-height`). No literal `1080`.

If there IS a hardcoded 1080 in this file, fix it by replacing with the parameterized value and commit with message `fix(direct): remove hardcoded 1080 proxy size in grading.rs`.

- [ ] **Step 2: Re-verify `raune_filter.py` does not hardcode 1080**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && grep -n "1080\|proxy_width\|proxy_height" python/dorea_inference/raune_filter.py | head -20
```

Expected: occurrences reference `args.proxy_width` and `args.proxy_height` — both parameterized via argparse at the script entry. No literal `1080`.

If there IS a hardcoded 1080 in this file, fix it and commit with message `fix(raune_filter): remove hardcoded 1080 proxy size`.

- [ ] **Step 3: Note verification result**

If both reads produced no edits, note in the PR description: *"Verification items 5/6 from the spec re-checked on branch; both confirmed parameterized, zero-change."* Move on to Task 5 without committing.

---

## Task 5: Add a minimal grading section to README

**Files:**
- Modify: `repos/dorea/README.md`

**Intent:** Current README is pre-pipeline (describes `dorea grade` as "Phase 3 — not yet implemented"). A full rewrite is out of scope. This task adds a small new section that (a) shows direct mode as the default one-liner and (b) mentions LUT mode as legacy. Rest of the README stays stale; a full rewrite is filed as a follow-up issue in Task 8.

- [ ] **Step 1: Add the new section**

In `repos/dorea/README.md`, find the `## Quick start` section around line 43. Immediately after that section's closing code fence and before `## Development`, insert the block below. The block uses 4-backtick outer fences here only so that the inner triple-backtick bash block renders correctly inside this plan — when you paste into README.md, **strip the outer 4-backtick fences and the language marker `markdown`**; keep only the content between them.

````markdown
## Grading modes

Dorea grade supports two modes. **Direct mode is the default.**

```bash
# Direct mode (default) — per-frame RAUNE at 1440p proxy, bilinear OKLab upscale,
# no calibration or keyframe stages. Fastest path, best ΔE vs RAUNE-at-4K.
dorea grade clip.mp4

# Legacy LUT pipeline — calibration → feature extraction → 3D LUT grading with
# adaptive depth zones and motion-aware content-adaptive grading. Retained for
# workflows that specifically need LUT export or adaptive-zone metadata.
dorea grade --mode lut clip.mp4
```

The LUT pipeline is still supported but emits a soft-deprecation warning at
startup. It is no longer the default; direct mode receives ongoing perf work
(fp16 RAUNE, 3-thread producer-consumer pipeline, 1440p proxy).

Direct-mode tunables:
- `--raune-proxy-size N` — RAUNE input long-edge in pixels (default: **1440**).
- `--direct-batch-size N` — frames per RAUNE forward (default: **8**). Values > 8
  are not recommended on 6 GB VRAM.
````

- [ ] **Step 2: Verify the file is well-formed**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && head -80 README.md | tail -40
```

Expected: the new section appears between `## Quick start` and `## Development` with no mangled markdown.

- [ ] **Step 3: Commit**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git add README.md && git commit -m "$(cat <<'EOF'
docs: add grading modes section, direct-first

Adds a minimal grading section to the README that teaches direct mode
as the default one-liner and labels the LUT pipeline as legacy. The
rest of the README remains stale (still describes v2 as Phase 3 work
in progress) — a full rewrite is tracked separately.

Refs docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit, one file changed.

---

## Task 6: Smoke test acceptance criteria on the workstation

**Files:** none.

**Intent:** Confirm the binary actually runs end-to-end with the new defaults on the RTX 3060 6GB workstation. This is not a gating benchmark — it is "did we ship a thing that runs cleanly" plus a `nvidia-smi` eyeball for the VRAM risk flagged in the spec.

- [ ] **Step 1: Release build**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && cargo build --release -p dorea-cli 2>&1 | tail -10
```

Expected: `Finished \`release\` profile [optimized]`.

- [ ] **Step 2: `--help` output sanity**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && ./target/release/dorea grade --help 2>&1 | grep -A 1 "mode\|direct"
```

Expected: `--mode <direct|lut>` appears in the help output; `--direct` appears with the deprecated marker.

- [ ] **Step 3: Smoke run — direct mode with no flags (new default path)**

In **terminal A**, start nvidia-smi polling:
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

In **terminal B**, run:
```bash
cd /workspaces/dorea-workspace && ./repos/dorea/target/release/dorea grade \
  footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 \
  --output /tmp/smoke_1440_b8.mp4 \
  2>&1 | tee /tmp/smoke_log.txt
```

Expected:
- Log line: `"Direct mode: single-process OKLab transfer, RAUNE proxy 2560x1440 batch=8, full-res ..."`
- No OOM, no crash, completes.
- `nvidia-smi` peak VRAM in terminal A stays under 6 GB total (the RTX 3060 has 6 GB physical).
- Output file `/tmp/smoke_1440_b8.mp4` exists and plays.

**If OOM occurs:** stop, record the peak VRAM seen in terminal A, and file this as a blocker on the PR. Revert Task 3 (or drop batch size back to 4) and re-run before merging. Do NOT merge with a known-OOM default.

- [ ] **Step 4: Smoke run — legacy LUT mode emits deprecation warning**

Start the LUT pipeline just long enough to see the first log line, then interrupt:

```bash
cd /workspaces/dorea-workspace && timeout 8s ./repos/dorea/target/release/dorea grade \
  --mode lut \
  footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 \
  --output /tmp/smoke_lut.mp4 \
  2>&1 | grep -i "legacy\|deprecat\|lut"
```

Expected: the soft-deprecation line `"Legacy LUT pipeline selected. Direct mode is the default; see README for details."` appears in the grep output. The 8-second timeout kills the process before calibration completes; that is fine — we only need the startup warning.

- [ ] **Step 5: Smoke run — `--direct` alias emits deprecation warning**

```bash
cd /workspaces/dorea-workspace && timeout 8s ./repos/dorea/target/release/dorea grade \
  --direct \
  footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 \
  --output /tmp/smoke_direct_alias.mp4 \
  2>&1 | grep -i "deprecat"
```

Expected: `"--direct is deprecated; use --mode direct (now the default)"` appears.

- [ ] **Step 6: Record smoke-test results in a scratch file**

Create `/tmp/smoke_results.md` with a one-paragraph summary of what each of steps 3, 4, 5 produced, including the peak VRAM observed in Step 3 from terminal A. This gets pasted into the PR description in Task 7.

Example content:
```markdown
- Step 3: direct-mode default run completed in XXs, processed all ~360 frames,
  peak VRAM from nvidia-smi: XXXX MiB / 6144 MiB. Output file exists, plays.
- Step 4: LUT mode soft-deprecation warning verified.
- Step 5: --direct alias deprecation warning verified.
```

---

## Task 7: Open PR on chunzhe10/dorea

**Files:** none.

- [ ] **Step 1: Push the branch**

Run:
```bash
cd /workspaces/dorea-workspace/repos/dorea && git push -u origin feat/direct-default-1440p
```

Expected: branch is created on origin.

- [ ] **Step 2: Open the PR with `gh`, explicitly targeting chunzhe10/dorea**

Run (copying the smoke-test results from `/tmp/smoke_results.md` into the body):

```bash
gh pr create --repo chunzhe10/dorea \
  --base master \
  --head feat/direct-default-1440p \
  --title "Direct mode is default; bump RAUNE proxy to 1440p, batch to 8" \
  --body "$(cat <<'EOF'
## Summary

- Flip default grading mode from `lut` to `direct` (`dorea grade clip.mp4` now runs direct mode).
- Bump direct-mode `raune_proxy_size` default from 1080 to 1440 (bench: bilinear best-ΔE at 1440p, −8.3% vs 1080p).
- Bump direct-mode `direct_batch_size` default from 4 to 8 (deferred fp16 batch bump from #67).
- Add `--mode direct|lut` as canonical CLI flag; `--direct` is deprecated alias.
- Soft-deprecate the LUT pipeline with a one-line startup warning. LUT pipeline is otherwise unchanged and still on the default compile path.

## Motivation

Delta upscale bench (corvia `019d7a9d`) showed that at 1440p proxy, `bilinear` is the new best-ΔE non-trivial upscale method and every classical method gains ~8% ΔE vs 1080p. Direct mode already uses bilinear. fp16 RAUNE (PR #67) created activation-memory headroom for `batch=8`. This PR ships both wins as the new defaults.

## Breaking change

Scripts that run `dorea grade clip.mp4` bare and expected LUT-mode output (3D LUT export, adaptive-zone metadata) must now pass `--mode lut` explicitly, or set `[grade].mode = "lut"` in `dorea.toml`.

## Risks

- **VRAM at 1440p batch=8:** back-of-envelope ~1.1–1.2× the known-safe 1080p batch=4 fp32 ceiling. Smoke test on the RTX 3060 6 GB workstation: <paste-smoke-results-here>.
- **Throughput regression:** RAUNE at 1440p processes ~2.25× more pixels per frame. Offset partially by 3-thread decode/encode overlap. No SLO enforced; ship and see.

## Test plan

- [x] `cargo build --release -p dorea-cli`
- [x] `cargo test -p dorea-cli`
- [x] `dorea grade clip.mp4` (no flags) → runs direct mode at 1440p batch=8, no OOM
- [x] `dorea grade --mode lut clip.mp4` → LUT pipeline + soft-deprecation warning
- [x] `dorea grade --direct clip.mp4` → direct mode + `--direct` deprecation warning
- [x] `dorea grade --help` → `--mode` is canonical

## Follow-ups (separate issues)

- Maxine dead-code removal
- Post-land VRAM/throughput measurement on a full-length clip
- README rewrite (currently describes grade as "Phase 3 — not yet implemented")

## References

- Decision record: `docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md` (workspace repo)
- Plan: `docs/plans/2026-04-11-direct-pipeline-1440p-promotion.md` (workspace repo)
- Bench findings: corvia `019d7a7b`, `019d7a9d`
- Predecessor PRs: #65 (3-thread pipeline), #67 (fp16 + batch sizing)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed. Paste the smoke-test results from `/tmp/smoke_results.md` into the placeholder `<paste-smoke-results-here>` section either via `gh pr edit` or by editing the body in the terminal before the command runs.

- [ ] **Step 3: Verify PR is on the right repo**

Run:
```bash
gh pr view --repo chunzhe10/dorea $(gh pr list --repo chunzhe10/dorea --head feat/direct-default-1440p --json number -q '.[0].number')
```

Expected: shows the PR we just created, targeting `chunzhe10/dorea master`. If it shows up on `chunzhe10/dorea-workspace`, the `--repo` flag was dropped — stop and check the `gh` invocation.

---

## Task 8: File follow-up issues on chunzhe10/dorea

**Files:** none.

**Intent:** Three separate GitHub issues, all with `--repo chunzhe10/dorea`, tracking deferred work explicitly marked out-of-scope in the spec.

- [ ] **Step 1: Maxine dead-code removal issue**

Run:
```bash
gh issue create --repo chunzhe10/dorea \
  --title "Remove Maxine dead code (already runtime-disabled)" \
  --body "$(cat <<'EOF'
Maxine is already hard-disabled at runtime in multiple code paths:

- \`crates/dorea-cli/src/grade.rs:334\` — \`maxine_in_fused_batch: false\`
- \`crates/dorea-cli/src/grade.rs:402\` — \`maxine: false\` (\"SDK not available in devcontainer\")
- \`crates/dorea-cli/src/grade.rs:450\` — \`assert!(!cfg.maxine, ...)\` test
- \`crates/dorea-cli/src/calibrate.rs:204\` — \`maxine: false\`
- \`crates/dorea-cli/src/preview.rs:111\` — \`maxine: false\`

Nothing passes \`maxine: true\` anywhere on the code path, so the Python Maxine modules (\`python/dorea_inference/maxine_enhancer.py\`, \`bridge.py\` Maxine functions, \`server.py\` Maxine handlers, \`python/tests/test_bridge_maxine.py\`) and the Rust plumbing (\`MaxineDefaults\`, \`maxine_upscale_factor\`, \`maxine_in_fused_batch\`, the hardcoded \`false\` assignments and the assertion) can all be deleted without any functional change.

Motivated by \`docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md\` in the dorea-workspace repo.

Scope:
- [ ] Delete \`python/dorea_inference/maxine_enhancer.py\`
- [ ] Delete \`python/tests/test_bridge_maxine.py\`
- [ ] Remove Maxine functions from \`python/dorea_inference/bridge.py\` (\`load_maxine_model\`, \`unload_maxine\`, \`run_maxine\`, the \`enable_maxine\` branch in \`run_fused_batch\`, the \`_maxine_model\` global)
- [ ] Remove Maxine handlers from \`python/dorea_inference/server.py\` (\`--maxine\`, \`--maxine-upscale-factor\`, \`--no-maxine-artifact-reduction\` args and the server handlers for them)
- [ ] Remove Rust \`MaxineDefaults\` struct from \`crates/dorea-cli/src/config.rs\`
- [ ] Remove \`maxine_upscale_factor\`, \`maxine_in_fused_batch\`, \`maxine\` fields from \`grade.rs\`, \`calibrate.rs\`, \`preview.rs\`, \`pipeline/mod.rs\`
- [ ] Remove the \`assert!(!cfg.maxine, ...)\` test at \`grade.rs:450\`
- [ ] Remove \`maxine\` entries from \`dorea.toml\` if present

Acceptance: \`cargo build --release\` passes, \`cargo test\` passes, \`python -m pytest python/tests/\` passes, no references to \`maxine\`/\`Maxine\`/\`nvvfx\` remain outside the bench package.
EOF
)"
```

- [ ] **Step 2: Post-land measurement issue**

Run:
```bash
gh issue create --repo chunzhe10/dorea \
  --title "Post-land: measure VRAM peak and throughput at direct 1440p batch=8" \
  --body "$(cat <<'EOF'
After \`feat/direct-default-1440p\` lands, measure real end-to-end behavior on a long clip (not the 3s bench clip):

- Run \`dorea grade\` on a full-length dive clip (5+ minutes) at the new default (direct mode, 1440p, batch=8)
- Capture \`nvidia-smi\` peak VRAM during the run
- Capture wall-clock fps
- Record as a corvia finding with \`source_origin = \"repo:dorea\"\`
- Compare against the previous 1080p/batch=4 baseline if available

Not a gate on the feature — this is post-land discovery. If peak VRAM is near the 6 GB ceiling or throughput has regressed significantly, file a follow-up to tune the defaults (e.g. drop batch to 6, drop proxy back to 1080p).

Motivated by \`docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md\` in the dorea-workspace repo (Risks section).
EOF
)"
```

- [ ] **Step 3: README rewrite issue**

Run:
```bash
gh issue create --repo chunzhe10/dorea \
  --title "Rewrite README — current one describes pre-pipeline v2 state" \
  --body "$(cat <<'EOF'
\`repos/dorea/README.md\` is badly stale. It describes dorea v2 as a Rust rewrite with \`dorea grade\` marked \"Phase 3 — not yet implemented\" and references the Sea-Thru POC as the algorithmic source of truth. The actual current code has:

- Full grade pipeline (LUT + direct modes)
- fp16 RAUNE inference
- 3-thread producer-consumer pipeline
- YOLO-seg class-mask grading
- 10-bit DJI D-Log M / Insta360 I-Log support
- Adaptive depth zones, motion-aware content-adaptive grading
- OKLab delta upscale (direct mode)

\`feat/direct-default-1440p\` added a minimal new \"Grading modes\" section to the README as a stopgap, but the rest of the file needs a real rewrite. Scope for that follow-up:

- [ ] Rewrite architecture diagram to match current crate layout
- [ ] Drop \"Phase 3 — not yet implemented\" language
- [ ] Document the 6 GB VRAM constraint and fp16 + batch sizing
- [ ] Document the \`[grade]\` and \`[inference]\` config sections
- [ ] Document YOLO-seg, 10-bit support, output codec auto-selection
- [ ] Add a \"Troubleshooting\" section for common VRAM / CUDA issues

Motivated by \`docs/decisions/2026-04-11-direct-pipeline-1440p-promotion.md\` in the dorea-workspace repo.
EOF
)"
```

Expected: three issue URLs printed.

- [ ] **Step 4: Confirm the issues landed on the right repo**

Run:
```bash
gh issue list --repo chunzhe10/dorea --limit 5
```

Expected: the three new issues appear at the top. If they landed on `chunzhe10/dorea-workspace`, the `--repo` flag was dropped somewhere — fix the issue URLs with `gh issue transfer` or close and re-file.

---

## Task 9: Commit the plan itself to dorea-workspace

**Files:**
- Create: `docs/plans/2026-04-11-direct-pipeline-1440p-promotion.md` (this file — it exists when you read this)

- [ ] **Step 1: Confirm the plan file exists at the expected path**

Run:
```bash
ls -la /workspaces/dorea-workspace/docs/plans/2026-04-11-direct-pipeline-1440p-promotion.md
```

Expected: file exists, owned by the current user.

- [ ] **Step 2: Commit the plan to the workspace repo**

This step runs from the workspace root (NOT `repos/dorea`):

```bash
cd /workspaces/dorea-workspace && git add docs/plans/2026-04-11-direct-pipeline-1440p-promotion.md && git commit -m "$(cat <<'EOF'
docs: add direct-pipeline 1440p promotion implementation plan

Task-by-task plan for the direct-default + 1440p proxy + batch=8
work on chunzhe10/dorea. Backs the decision record committed
earlier in docs/decisions/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit, one file added.

**Note:** the workspace commit is intentionally not linked to the dorea PR. They can land in either order. This step can also run before Task 0 if the executor prefers to have the plan on master before working in the dorea repo.

---

## Acceptance Criteria (final — quotes the spec)

Before closing the PR and marking this plan complete, confirm:

- [ ] `cargo build --release -p dorea-cli` passes on the branch
- [ ] `cargo test -p dorea-cli` passes (Maxine-disabled assertion at `grade.rs:450` unchanged)
- [ ] `dorea grade --help` shows `--mode` as canonical, `--direct` as deprecated
- [ ] `dorea grade clip.mp4` (no flags, no config override) runs direct mode at 1440p batch=8 without OOM on the 6 GB workstation
- [ ] `dorea grade --mode lut clip.mp4` runs LUT pipeline and emits the soft-deprecation warning once at startup
- [ ] `dorea grade --direct clip.mp4` runs direct mode and emits the alias deprecation warning
- [ ] README has the new grading section with direct mode as the first example
- [ ] PR opened against `chunzhe10/dorea` (NOT `chunzhe10/dorea-workspace`)
- [ ] Three follow-up issues filed on `chunzhe10/dorea`
- [ ] Plan file committed to `chunzhe10/dorea-workspace`

**NOT acceptance criteria** (explicitly per spec):
- No formal throughput measurement
- No formal VRAM peak measurement beyond `nvidia-smi` eyeball during smoke test
- No comparison against prior 1080p performance
- No Maxine code changes
