# Delta Upscale Benchmark — Design Spec

**Date:** 2026-04-10
**Status:** Draft — awaiting user approval before implementation plan
**Scope:** `repos/dorea` (Python, benchmark only; no changes to `raune_filter.py` in this spec)
**Related corvia entries:**
- `019d7501` — fp16 RAUNE final decision (2026-04-10)
- `019d74bb` — Direct Mode Performance Deep Dive (2026-04-10)
- `019d74c9` — Direct Mode Bottleneck CORRECTION (2026-04-10)

## Summary

Build a re-runnable benchmark that measures the quality and performance of **9 candidate methods** for upscaling the OKLab delta produced by RAUNE in `raune_filter.py::_process_batch`, using **RAUNE-at-4K** as the ground-truth reference. Output is a contact sheet (summary grid, per-frame detail sheets, error heatmaps) and a CSV of metrics. The tool lives at `repos/dorea/benchmarks/upscale_bench/` and is structured so adding a new upscale method requires writing exactly one function.

## Motivation

`raune_filter.py::_process_batch()` currently computes the RAUNE OKLab delta at proxy resolution (1080p) and upscales it to full resolution (2160p) with plain bilinear interpolation:

```python
# line 514 in raune_filter.py
delta_full = F.interpolate(delta_lab, size=(fh, fw), mode="bilinear", align_corners=False)
```

Visual inspection of the output suggests the bilinear upscale introduces edge halos and chroma bleed where RAUNE's correction changes locally (e.g. across object boundaries). The current fp16 pipeline delivers 4.25 fps at the GPU-stage ceiling, so further speedups require a different axis. Quality-preserving improvements to the delta upscale are on the table and need empirical comparison before we commit to a specific replacement.

This spec does not implement a replacement. It implements the benchmark needed to decide which replacement to implement.

## Goals

1. Produce an objective comparison of 9 delta-upscale candidates on 3 hand-selected frames from `DJI_20251101111428_0055_D_3s.MP4`.
2. Measure both **quality** (delta-space error vs. gold, final-image ΔE2000, SSIM) and **performance** (wall time for upscale-only and end-to-end).
3. Emit a visual contact sheet that makes the trade-offs legible at a glance (grid summary, per-frame zoomed crops, error heatmaps).
4. Be re-runnable as the pipeline evolves — adding a new method is a single-function addition; re-running is `python -m benchmarks.upscale_bench.run`.
5. Cache the expensive gold-standard computation so iterating on method implementations is fast.

## Non-goals

- **No production integration** in this spec. Wiring the winning method into `raune_filter.py` is a separate PR, after results are reviewed.
- **No regression / CI gating.** This is a characterization bench, not a test suite. Numbers are expected to change as RAUNE or methods evolve.
- **No automatic winner selection.** Humans pick the winner by looking at the contact sheet and the metrics CSV.
- **No speed optimization of the bench itself.** Correctness and clarity over throughput. First-run cost is acceptable because gold is cached.

## Architecture

### Package layout

```
repos/dorea/benchmarks/upscale_bench/
├── __init__.py
├── methods.py           # @register("name") registry + 9 method functions
├── gold.py              # compute_gold(frame, model) → (raune_4k, delta_4k)
├── metrics.py           # delta_l1/l2/max/p95, ΔE2000, SSIM, timing helpers
├── visualize.py         # summary grid, per-frame sheets, error heatmaps
├── frame_select.py      # heuristic auto-selection + decode
├── run.py               # CLI entry point, orchestration
├── README.md            # usage + "how to add a method" cookbook
└── tests/
    ├── __init__.py
    ├── test_methods.py  # per-method smoke tests on synthetic input
    └── test_e2e.py      # end-to-end pytest subprocess runs
```

### Method contract

Every upscale method is a function registered via a decorator. The contract is fixed so the driver can treat all methods uniformly:

```python
@register("bilinear")
def bilinear(
    delta_proxy: torch.Tensor,       # (1, 3, ph, pw) fp32 OKLab delta on CUDA
    orig_full:   torch.Tensor,       # (1, 3, fh, fw) fp32 RGB [0,1] on CUDA (for edge-aware methods)
    full_size:   tuple[int, int],    # (fh, fw)
) -> torch.Tensor:                   # (1, 3, fh, fw) fp32 OKLab delta on CUDA
    return F.interpolate(delta_proxy, size=full_size, mode="bilinear", align_corners=False)
```

**Contract rules:**
- Input: `delta_proxy` (at proxy resolution in OKLab) + `orig_full` (full-res RGB guide, always passed; methods that don't need it ignore it).
- Output: upscaled delta at `full_size`, in OKLab, `(1, 3, H, W)` fp32, on CUDA. Same shape and dtype contract every time.
- The driver applies the returned delta to `orig_full` via the existing `triton_oklab_transfer` from `raune_filter.py`, so the downstream RGB-conversion path is bit-identical across methods. The only thing that varies is the upscale function itself. This eliminates measurement noise from reimplementing the transfer.
- Method functions must not mutate their inputs.
- Method functions must be pure — no module-level state, no cached tensors between frames. The driver handles warm-up and gold caching; methods do not cache anything between calls.
- Any preprocessing a method needs (e.g. `joint_bilateral`'s RGB→OKLab luma guide conversion) happens **inside** the method function and is **included** in its timing. This is the fair comparison: a production implementation would pay the same cost per frame.
- Adding a new method: one function, one decorator, zero other files touched.

### Driver flow (`run.py`)

```
1. parse CLI args
2. log environment (python/torch/cuda/driver versions, GPU, weights SHA, clip SHA, git SHA)
3. load RAUNE model once (fp16, InstanceNorm → fp32 per #67)
4. select frames:
     if --frames: decode explicit indices
     else:        run frame_select.auto_select(clip_path) → [idx_edge, idx_smooth, idx_chroma]
5. for each frame_idx:
     a. decode frame_4k (fp32 RGB [0,1] on CUDA)
     b. compute_gold(frame_4k, raune_model) → (raune_4k, delta_4k)
        - cache-check: working/upscale_bench/gold/{weights_sha[:8]}_{frame_idx}.pt
        - on miss: try native 4K RAUNE; on OOM fall back to 2×2 tiled (1920×1080, 128px overlap, feathered)
        - write cache
     c. resize frame_4k → frame_proxy (1080p)
        run RAUNE on frame_proxy → raune_proxy
        compute delta_proxy = lab(raune_proxy) - lab(orig_proxy)
     d. for each registered method:
          - warmup 3 runs
          - 10 timed runs, cuda.synchronize() bracketed
          - run once more to capture delta_method (warmup output discarded)
          - compute final_method via triton_oklab_transfer(frame_4k, delta_method)
          - compute metrics against gold
          - append row to results[]
6. generate visualizations:
     - summary_grid.png      (visualize.generate_grid)
     - per_frame/*.png       (visualize.generate_per_frame_sheet × 3)
     - heatmaps/*.png        (visualize.generate_heatmap × 3 × 9)
7. write metrics.csv
8. write run_report.md
```

Total expected runtime on first run with all 9 methods + 3 frames: ~1–3 minutes (dominated by gold computation and Real-ESRGAN model download). Re-runs with cached gold: ~30 seconds.

## The 9 methods

**Baseline (always shown, not counted as one of the 7 core methods):**

### `bilinear`
Current production behavior. `F.interpolate(mode="bilinear", align_corners=False)`. Reference row — every other method is compared against this and against the gold.

### Core methods (7)

#### 1. `bicubic`
One-line swap from baseline: `F.interpolate(mode="bicubic", align_corners=False)`. Better frequency response than bilinear, still not edge-aware. No new dependencies. If it wins, ship a 1-character config change and move on.

#### 2. `lanczos3`
3-lobe Lanczos filter. Implementation: precompute a separable 1-D Lanczos kernel, apply via `F.conv2d` row-then-column. PyTorch does not provide Lanczos in `F.interpolate`, so this requires a small helper. The classical "best non-edge-aware upscale". Included alongside bicubic because the delta to bicubic is small but non-zero, and the user wants both.

#### 3. `joint_bilateral`
Luma-guided joint bilateral upsample. Triton kernel (the pipeline already ships Triton; the loop structure mirrors the existing OKLab transfer kernel in `raune_filter.py`). For each output pixel:

```
sum_w = 0; sum_d = 0
for each of the 4 neighboring proxy-grid samples n:
    w_spatial = exp(-spatial_distance(out, n)² / σ_spatial²)
    w_guide   = exp(-|L_full(out) - L_guide(n)|² / σ_luma²)
    w = w_spatial * w_guide
    sum_w += w
    sum_d += w * delta_proxy(n)
delta_full(out) = sum_d / sum_w
```

Hyperparameters:
- `σ_spatial = 1.5` proxy pixels
- `σ_luma = 0.1` on OKLab-L in [0, 1]

`L_guide(n)` is the luma at the proxy-grid point, obtained by downsampling `orig_full`'s OKLab-L to proxy dimensions once up front and reusing. This is the method I expect to win on edges specifically.

#### 4. `guided_filter`
He/Sun/Tang 2010 guided filter with `orig_full`'s OKLab-L as the guide. Implemented in PyTorch using separable box filters (cheap on GPU, closed-form, no tuning loop).

Hyperparameters:
- `r = 8` (box filter radius, in full-resolution pixels)
- `ε = 1e-4` (regularization)

Conceptually similar to joint bilateral but with different trade-offs: smoother in flat regions, less aggressive at hard edges. Worth testing alongside joint bilateral because the winner between them is content-dependent.

#### 5. `asymmetric_bilateral`
Same Triton kernel as `joint_bilateral` but with per-channel spatial sigma:
- `σ_spatial_L = 1.5` (same as joint_bilateral)
- `σ_spatial_ab = 3.0` (chroma blurred more than luma)

Theory: OKLab-L corrections must track luminance edges precisely, but a/b corrections can be smoother without being visible — human color acuity is lower than luma acuity. If this beats plain `joint_bilateral`, we learn that chroma-bleed tolerance is the lever that matters most.

#### 6. `higher_proxy` (control, not an upscale)
Not strictly an upscale method — a **control condition**. Runs RAUNE at `proxy_size = 2160` (native 4K) instead of 1080p. Uses the same tiled fallback path as `gold.py` if native-4K OOMs.

The strategic value of this row:

- If `higher_proxy` wins by a lot: the bottleneck is **information loss in RAUNE** (1080p never saw the full-res frequencies), and no amount of fancy upscale can recover what RAUNE never saw.
- If `higher_proxy` ≈ `joint_bilateral`: classical upscale has hit its ceiling and only learned SR can do more.
- If `higher_proxy` ≈ `bilinear`: RAUNE at 4K is no better than RAUNE at 1080p on this content — out-of-distribution effects dominate.

This is the most informative row for the strategic question: "what's the actual ceiling?"

#### 7. `sr_maxine`
Learned super-resolution on the *graded proxy*, not on the delta.

Flow:
```
1. apply delta_proxy to orig_proxy (1080p) → graded_proxy (1080p)
2. NVIDIA Maxine VideoSuperRes(quality=HIGH) upscales graded_proxy 1080 → 2160
3. compute delta_method = lab(graded_4k) - lab(orig_full)
4. driver applies delta_method via triton_oklab_transfer as usual
```

Rationale: a learned SR model trained on natural images is in-distribution when upscaling a natural image (`graded_proxy`), but out-of-distribution when upscaling a signed delta field. Testing "SR the image, not the delta" is the honest question.

Maxine verified working in this session:
- Package: `nvidia-vfx` on PyPI (imports as `nvvfx`)
- SDK version: 1.2.0
- Wheel ABI: `cp312-abi3-manylinux_2_27_x86_64` (compatible with venv's Python 3.13)
- Smoke test passed end-to-end on RTX 3060: 256×256 → 512×512 upscale ran clean

### Optional (8th method)

#### `sr_realesrgan`
Same flow as `sr_maxine` but with Real-ESRGAN `RealESRGAN_x2plus` in place of Maxine. Tests aggressive learned SR as a counterpoint to Maxine's conservative behavior.

New dependencies: `basicsr`, `realesrgan`, one ~150 MB model download. If either import fails, this method is **soft-skipped** (warning logged, method dropped from the run, other methods continue). This is the only soft-skip in the bench.

## Gold standard

### Test source
`footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4` — 3840×2160, HEVC yuv420p10le, 120 fps, 360 frames. Same clip referenced in every prior direct-mode finding.

### Frame selection

**Auto-select (`frame_select.auto_select`):**

1. Decode every 30th frame as a 1080p thumbnail.
2. Score each thumbnail on three heuristics:
   - **Edge density**: `mean(|sobel_x| + |sobel_y|)` on OKLab-L
   - **Smoothness**: `-mean(var(32×32 blocks))` on OKLab-L (negative of variance → high score = flat)
   - **Chroma magnitude**: `mean(sqrt(a² + b²))` in OKLab
3. Pick the top-1 frame for each criterion. Return 3 absolute frame indices.

The result is deterministic (same clip → same 3 frames every run), so re-runs produce comparable numbers. Different clips will produce different picks.

**CLI override:** `--frames 30,180,300` bypasses auto-select and uses the explicit indices.

### Gold computation (`gold.py`)

`compute_gold(frame_4k: torch.Tensor, raune_model) → dict`:

```
1. Try native 4K RAUNE:
     with torch.no_grad():
       try:
         raune_4k = raune_model((frame_4k * 2 - 1).half()).float()
         raune_4k = ((raune_4k + 1) / 2).clamp(0, 1)
         path_used = "native_4k"
       except torch.cuda.OutOfMemoryError:
         torch.cuda.empty_cache()
         goto tiled

2. Tiled fallback:
     - split frame_4k into 2×2 tiles, each 1984×1144 ( = (3840+128)/2 × (2160+128)/2 ),
       positioned so the two inner seams overlap by exactly 128 pixels and the
       outer edges have no overlap
     - run RAUNE on each tile independently
     - feather the overlaps with linear alpha (0 → 1 across the 128-pixel overlap band)
     - stitch back into a 3840×2160 raune_4k
     - path_used = "tiled_2x2_1984x1144_o128"

3. Compute delta_4k = lab(raune_4k) - lab(frame_4k) in OKLab

4. Return {
     "raune_4k":  raune_4k,
     "delta_4k":  delta_4k,
     "path_used": path_used,
     "weights_sha1_prefix": WEIGHTS_SHA[:8],
   }
```

### Gold caching

File: `working/upscale_bench/gold/{weights_sha1[:8]}_{frame_idx}.pt`

Contents: `torch.save` of a dict with `raune_4k`, `delta_4k`, `path_used`, and a version field (bumped if the cache format changes).

- Cache key includes the weights SHA prefix — swap weights and caches auto-invalidate.
- Cache key includes frame index — different frames don't collide.
- `--regen-gold` forces recomputation (ignore cache, overwrite).
- Corrupt cache: if `torch.load` throws, log a warning, recompute, and overwrite.

### Gold sanity check

`--gold-sanity-check` runs **both** native and tiled paths (native-forced by temporarily disabling the tiled fallback, then tiled-forced), and reports per-pixel disagreement in OKLab between them. Expected on identical hardware: disagreement is tiny but nonzero due to numerical ordering. Large disagreement (e.g. `delta_max > 0.02` on OKLab-L) is a red flag — the gold is not reliable and the bench results that depend on it should be discounted.

Native and tiled output both get cached under distinct filenames when this flag is set.

### Honest caveat about the gold

RAUNE-at-4K — whether native or tiled — is still an out-of-distribution input for a model likely trained on small patches. The gold is "what RAUNE-at-4K produces", not "what a perfect upscale method would produce". The bench does not claim otherwise. The run report explicitly calls this out and displays which gold path was used per frame.

## Metrics

### Per-(frame × method) measurements

**Delta-space error** (how well does each upscale reconstruct the ideal OKLab delta?):
- `delta_l1_L` — mean absolute error on L channel
- `delta_l1_a` — mean absolute error on a channel
- `delta_l1_b` — mean absolute error on b channel
- `delta_l1` — mean absolute error overall (average of the three)
- `delta_l2` — RMSE overall
- `delta_max` — max absolute error anywhere in the frame (worst-case halo)
- `delta_p95` — 95th percentile of `|delta_method - delta_gold|` (halo behavior without outlier dominance)

**Final-image error** (what the viewer sees after the delta is applied):
- `final_delta_e` — mean ΔE2000 between `final_method` and `raune_4k`, computed in **CIELab** (D65 white point), not OKLab. CIELab is the industry-standard color space for ΔE2000; both images are converted from sRGB → CIELab for this metric only.
- `final_delta_e_p95` — 95th percentile of per-pixel ΔE2000
- `ssim` — single-number SSIM computed on **BT.709 luma** (Y from Y'CbCr, not OKLab-L). Uses `torchmetrics.image.StructuralSimilarityIndexMeasure` if available, else a small inlined implementation.

**Performance**:
- `wall_time_ms_upscale` — median of 10 runs after 3 warm-ups, upscale function only, `torch.cuda.synchronize()` bracketed
- `wall_time_ms_end_to_end` — same timing methodology, full (upscale + `triton_oklab_transfer` + GPU→CPU→uint8 pipeline). Two timings because `sr_maxine` does its main work in the transfer path rather than the upscale itself; upscale-only would mislead.

### Aggregation

Per-method summary across frames, written as the last block of rows in `metrics.csv` and in the `run_report.md`:
- Mean of each metric across the 3 frames
- Stddev of each metric across the 3 frames
- Rank position on each metric (1 = best)

## Outputs

### Directory layout

```
working/upscale_bench/
├── gold/
│   └── {weights_sha[:8]}_{frame_idx}.pt           # cached gold
├── heatmaps/
│   └── frame{idx}_{method}.png                    # 3 × 9 = 27 files
├── per_frame/
│   └── frame{idx}_sheet.png                       # 3 files
├── summary_grid.png                               # contact sheet A
├── metrics.csv                                    # machine-readable
└── run_report.md                                  # human-readable summary
```

Everything under `working/` is gitignored (existing entry in `.gitignore`). The script itself is committed.

### Summary grid (`summary_grid.png`) — contact sheet A

~3000 × 5500 px. Layout:

- **Header band** (top): run date, weights SHA prefix, clip name, gold path used per frame (e.g. "frame 90: native_4k, frame 180: tiled_2x2, frame 270: tiled_2x2"), git SHA, any OOM events.
- **Main grid**: rows = 3 frames, columns = 11 (orig_proxy, gold, bilinear, bicubic, lanczos3, joint_bilateral, guided_filter, asymmetric_bilateral, higher_proxy, sr_maxine, sr_realesrgan). Each cell is the full final image downsampled to ~450 px wide.
- **Cell annotations**: thin colored border. Green = row winner on `final_delta_e`. Light gray = bilinear baseline. No border = other.
- **Column headers**: method name + median `wall_time_ms_end_to_end`, so the speed cost is visible without flipping to the CSV.
- **Footer band** (bottom): compact per-method aggregate table — mean ± stddev of the key metrics across frames.

### Per-frame sheets (`per_frame/frame{idx}_sheet.png`) — 3 files

~2400 × 2400 px per frame. Layout:

- **Top band**: full gold image at ~1920 px wide, with a red rectangle indicating the auto-selected crop region.
- **Middle band**: row of 11 small thumbnails, same order as the summary grid.
- **Crop band**: row of 11 × 400×400 crops, taken from the same spatial location in every thumbnail.
- **Crop location selection**: automatic per frame. Compute `variance_across_methods(delta_method at each pixel)`, find the 400×400 window with the highest total variance, crop there. This zooms into the region where methods disagree most — the "argument area" — without manual intervention.
- **Metrics strip** (bottom): compact table of all metrics for this frame alone, all methods.

### Error heatmaps (`heatmaps/frame{idx}_{method}.png`) — 27 files

~1920 × 1080 px per heatmap. Each pixel shows `|delta_method(x,y) - delta_gold(x,y)|` — averaged across L/a/b channels — as a colormap.

- Colormap: `turbo`
- Normalization: `[0, frame_p95]` **per frame**, shared across all methods on that frame. Method-by-method comparison on a single frame is directly visual (10× worse = 10× hotter, not clipped).
- Small legend bar overlaid.
- Filename encodes both frame index and method, for straightforward grepping.

### `metrics.csv`

One row per (frame, method) pair. Columns:

```
frame_idx,method,delta_l1_L,delta_l1_a,delta_l1_b,delta_l1,delta_l2,delta_max,delta_p95,final_delta_e,final_delta_e_p95,ssim,wall_time_ms_upscale,wall_time_ms_end_to_end,status
```

`status` is `ok`, `oom`, or `skipped`. Aggregate summary rows at the bottom with `frame_idx = "ALL_MEAN"` and `"ALL_STDDEV"`.

### `run_report.md`

Auto-generated at end of run. Sections:

1. **Environment**: Python/torch/cuda versions, GPU, VRAM, weights SHA, clip SHA, git SHA, dirty flag, nvvfx version, Real-ESRGAN version if present.
2. **Run metadata**: date, frames used, gold paths per frame, OOM events.
3. **Winners** (highlighted):
   - Best `final_delta_e`: `<method>` (<value>)
   - Best `delta_l1`: `<method>` (<value>)
   - Best `wall_time_ms_upscale`: `<method>` (<value>)
   - Best `wall_time_ms_end_to_end`: `<method>` (<value>)
4. **Per-method table**: mean ± stddev across frames for all metrics, sorted by `final_delta_e`.
5. **Per-frame thumbnails**: inline embedded per-frame sheet PNGs (via relative markdown image refs).
6. **Notes / caveats**: out-of-distribution warning for RAUNE-at-4K, gold-sanity-check results if flag was set.
7. **Footer**: "Open `summary_grid.png` for the full visual comparison."

This is the file to skim first after a run.

## Setup (one-time)

Three independent steps, executed once per fresh clone or after a devcontainer rebuild. Each is idempotent.

### 1. Git-LFS for model weights

```bash
# one-time per clone
cd /workspaces/dorea-workspace
git lfs install

cd repos/dorea
# stores the pattern in .gitattributes; idempotent
git lfs track "models/raune_net/*.pth"
git lfs track "models/realesrgan/*.pth"
git add .gitattributes
```

### 2. Download RAUNE weights (`scripts/download_raune_weights.sh`)

New script at `repos/dorea/scripts/download_raune_weights.sh`:

```bash
#!/bin/bash
set -euo pipefail
DEST="/workspaces/dorea-workspace/repos/dorea/models/raune_net/weights_95.pth"
EXPECTED_SHA256="<pinned after first download>"
GDRIVE_FOLDER_ID="1pjEh6s6-a3p7qBtkONSlYLmKrfgD6rBk"

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ]; then
    actual=$(sha256sum "$DEST" | awk '{print $1}')
    if [ "$actual" = "$EXPECTED_SHA256" ]; then
        echo "weights already present and verified: $DEST"
        exit 0
    else
        echo "weights present but checksum mismatch — re-downloading"
    fi
fi

# Ensure gdown is installed in the dorea venv
/opt/dorea-venv/bin/pip show gdown >/dev/null 2>&1 || \
    /opt/dorea-venv/bin/pip install gdown

# Pull the whole pretrained folder; pick out just RAUNENet/test/weights_95.pth
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
/opt/dorea-venv/bin/gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID" -O "$TMP"
cp "$TMP"/*/RAUNENet/test/weights_95.pth "$DEST"

actual=$(sha256sum "$DEST" | awk '{print $1}')
echo "downloaded $DEST"
echo "sha256: $actual"
if [ -z "$EXPECTED_SHA256" ] || [ "$EXPECTED_SHA256" = "<pinned after first download>" ]; then
    echo "NOTE: no pinned SHA256 yet. Record this value and update this script."
elif [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: sha256 mismatch. expected $EXPECTED_SHA256" >&2
    exit 1
fi
```

First run: downloads, prints the SHA256, user updates the `EXPECTED_SHA256` constant and commits. Subsequent runs: verify and skip.

After first download:

```bash
cd repos/dorea
git add models/raune_net/weights_95.pth  # stored as LFS pointer
git commit -m "chore: add RAUNE pretrained weights via git-lfs"
```

Update `dorea.toml`:
```diff
- raune_weights   = "/workspaces/dorea-workspace/working/sea_thru_poc/models/RAUNE-Net/pretrained/RAUNENet/test/weights_95.pth"
+ raune_weights   = "/workspaces/dorea-workspace/repos/dorea/models/raune_net/weights_95.pth"
- raune_models_dir = "/workspaces/dorea-workspace/working/sea_thru_poc"
+ raune_models_dir = "/workspaces/dorea-workspace/repos/dorea/models/raune_net"
```

`raune_models_dir` must contain a `models/raune_net.py` file (the `RAUNENet` class). Currently that class lives at `python/dorea_inference/raune_net.py`. Either copy or symlink it into `models/raune_net/models/raune_net.py` as part of the setup script, or have `scripts/download_raune_weights.sh` do the layout fix.

Decision: do the layout fix in the setup script. Simpler than adding import-path logic.

### 3. Install bench dependencies (`scripts/setup_bench.sh`)

New script at `repos/dorea/scripts/setup_bench.sh`:

```bash
#!/bin/bash
set -euo pipefail
VENV_PIP="/opt/dorea-venv/bin/pip"

# Core: already installed in Dockerfile, but ensure
$VENV_PIP install --quiet gdown

# Maxine: nvidia-vfx wheel-stub pulls real binary from pypi.nvidia.com
if ! /opt/dorea-venv/bin/python -c "import nvvfx" 2>/dev/null; then
    $VENV_PIP install nvidia-vfx
fi

# Real-ESRGAN: optional, soft-failing dep
$VENV_PIP install --quiet basicsr realesrgan || \
    echo "WARNING: Real-ESRGAN install failed; sr_realesrgan will be skipped at bench time"

# torchmetrics for SSIM (optional; inlined fallback if missing)
$VENV_PIP install --quiet torchmetrics || true

# git-lfs for weights
command -v git-lfs >/dev/null 2>&1 || \
    (apt-get update && apt-get install -y git-lfs)
git lfs install

echo "setup_bench.sh: complete"
```

Idempotent; safe to re-run after any devcontainer rebuild.

## CLI

```
python -m benchmarks.upscale_bench.run [OPTIONS]

Options:
  --clip PATH               Default: footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4
  --frames auto|N,N,N       Default: auto (heuristic-picked); override: "30,180,300"
  --methods all|A,B,C       Default: all; e.g. "bilinear,bicubic,joint_bilateral"
  --out-dir DIR             Default: working/upscale_bench/
  --proxy-size N            Default: 1080 (matches production)
  --regen-gold              Force gold recomputation, ignore cache
  --gold-sanity-check       Run both native and tiled gold, report disagreement
  --timing-runs N           Default: 10
  --timing-warmup N         Default: 3
  --verbose                 Log per-frame per-method progress
```

Defaults are chosen so `python -m benchmarks.upscale_bench.run` with zero args does the full benchmark on the canonical clip with all methods.

## Error handling

Follows the project's `feedback_fail_fast` memory — no silent fallbacks except where explicitly sensible.

| Condition | Behavior |
|---|---|
| RAUNE weights missing | Hard error with "run `scripts/download_raune_weights.sh`" hint |
| CUDA unavailable | Hard error — this is a GPU benchmark, CPU has no meaning |
| `nvvfx` import fails | **Hard error** with "run `scripts/setup_bench.sh` or `pip install nvidia-vfx`" hint |
| `realesrgan` import fails | **Soft skip** — warning logged, method dropped, other methods continue (only soft-skip in the bench) |
| OOM during native-4K gold pass | Auto-fallback to tiled path, log which was used |
| OOM during tiled gold pass | Hard error — VRAM is fundamentally insufficient |
| OOM during `higher_proxy` | Same fallback as gold (shared code path), log, continue |
| OOM during any other method | Mark row `failed` in `metrics.csv`, render an "OOM" placeholder cell in contact sheets, continue with other methods |
| Corrupt gold cache | Warn, regenerate, continue |
| Unknown method in `--methods` list | Hard error with list of registered methods |
| Unknown frame index in `--frames` | Hard error — no guessing |

The `realesrgan` soft-skip is the sole exception to fail-fast. Maxine is hard-required per explicit user instruction.

## Reproducibility

At startup, `run.py` logs and writes into `run_report.md`:

- Python version, `torch.__version__`, `torch.version.cuda`, driver version from `nvidia-smi`
- GPU name, VRAM total/free
- `nvvfx.get_sdk_version()`
- `realesrgan.__version__` if present
- Workspace git SHA + dirty-tree flag
- `repos/dorea` git SHA + dirty-tree flag
- RAUNE weights SHA1 (first 16 chars)
- Clip path + SHA1 (first 16 chars)
- Random seed (fixed at 0)

Determinism settings:
```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

Timing methodology:
- N warm-up runs (default 3) — discarded. Absorbs first-call JIT cost (Triton kernels, torch.compile, cuDNN algorithm autotuning).
- N timed runs (default 10), each bracketed by `torch.cuda.synchronize()` before `time.perf_counter()` and after.
- Report **median**, not mean. Median is more robust against GC pauses and algorithm autotuning noise.

## Testing

### Per-method smoke tests (`tests/test_methods.py`)

```python
import pytest
import torch
from benchmarks.upscale_bench.methods import REGISTRY

@pytest.mark.parametrize("method_name", list(REGISTRY.keys()))
def test_method_smoke(method_name):
    method = REGISTRY[method_name]
    delta_proxy = torch.randn(1, 3, 36, 64, device="cuda", dtype=torch.float32) * 0.1
    orig_full = torch.rand(1, 3, 72, 128, device="cuda", dtype=torch.float32)
    out = method(delta_proxy, orig_full, full_size=(72, 128))
    assert out.shape == (1, 3, 72, 128)
    assert out.dtype == torch.float32
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1.0  # OKLab deltas are small
```

Tests that rely on heavy deps (`nvvfx`, `realesrgan`) use `pytest.importorskip`.

### End-to-end smoke test (`tests/test_e2e.py`)

One pytest that invokes `python -m benchmarks.upscale_bench.run --methods bilinear,bicubic --frames <single frame> --out-dir <tmp>` via `subprocess` on a tiny synthetic 10-frame 4K clip (generated once via `ffmpeg` at test collection time, cached in `tests/fixtures/`). Asserts:

- `metrics.csv` exists and has exactly 2 method rows
- `summary_grid.png` exists and is a valid PNG
- `run_report.md` exists
- Return code is 0

This catches integration breakage (bad CLI wiring, wrong output paths) without being slow.

### What's deliberately not tested

- Numerical regression of bench output — numbers change as methods and RAUNE evolve
- Maxine and Real-ESRGAN smoke tests — heavy deps, skipped in CI by default
- Visualization correctness — humans look at the contact sheet

## Follow-ups (out of scope for this spec)

Filed as separate issues once the first bench run completes:

1. **Integrate the winning method into `raune_filter.py`** — replace the `F.interpolate(mode="bilinear")` call with the winner. Separate PR so the integration can be reviewed independently of the benchmark.
2. **Wire Maxine into the devcontainer Dockerfile** — add `pip install nvidia-vfx` to the Dockerfile so `/opt/dorea-venv` survives rebuilds without manual reinstall.
3. **Fix the outdated `docs/guides/maxine-setup.md`** — the guide references an NGC download step that's no longer needed with the `nvidia-vfx` wheel-stub package.
4. **Pin `EXPECTED_SHA256` in `download_raune_weights.sh`** after the first download records it.

These are noted in the run_report.md footer the first time the bench runs end-to-end.
