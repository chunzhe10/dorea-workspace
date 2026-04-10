# Underwater Image Enhancement Model Comparison POC — Design Spec

**Date:** 2026-04-10
**Status:** Draft
**Scope:** repos/dorea (research script, not production pipeline)
**Related:** Direct mode fp16 RAUNE-Net path (`repos/dorea/python/dorea_inference/raune_filter.py`)

---

## Summary

Build a standalone benchmark script that compares RAUNE-Net against three
lightweight alternatives (FA+Net, Shallow-UWnet, Color-Accurate UIE 2025) at
fp16 on the RTX 3060. The script produces a full-resolution contact sheet PNG
for visual inspection and a markdown table of speed/quality metrics. The goal
is to decide whether any alternative beats current fp16 RAUNE-Net on
throughput without unacceptable quality loss, to enable multi-instance
parallelism via a smaller model.

This is **throwaway research code**. It lives under `scripts/poc/` and does
not touch the production grading pipeline or the inference subprocess.

---

## Motivation

Direct mode currently runs fp16 RAUNE-Net (`.half()` + InstanceNorm kept in
fp32 + `torch.compile(mode="reduce-overhead", dynamic=False)`) at batch 4 on
4K proxies. RAUNE-Net's ~30 residual blocks are the dominant per-frame cost
now that depth estimation has been removed from the per-frame path. A
smaller model would enable multi-instance / CUDA-stream parallelism on the
3060's 6GB VRAM, potentially yielding higher end-to-end throughput than
single-instance RAUNE regardless of per-frame speedup.

We need quantitative numbers and visual quality evidence before committing
to any replacement.

---

## Scope

**In scope:**
- Benchmark RAUNE-Net (baseline) + FA+Net + Shallow-UWnet + Color-Accurate UIE
- fp16 quantization for all models (matching direct mode's recipe)
- Two variants per model: strict parity with RAUNE's full recipe, and
  best-effort per-model fp16 deployment
- Full-resolution contact sheet (8 frames × 5 columns) from one canonical
  dive clip
- Speed metrics: time-to-first-frame, mean/p50/p95 per-batch latency,
  throughput, peak VRAM
- Quality metrics: UIQM (reference-free), SSIM vs RAUNE baseline

**Out of scope:**
- Multi-instance parallelism itself — this POC *informs* that decision but
  does not implement it
- Integration with `dorea-cli` or the inference subprocess
- Video output / temporal stability (deferred to future pass if a winner
  emerges)
- int8 or TensorRT export (fp16 only for this round)
- Transformer-based UIE models (ViT-ClarityNet, HMENet) — they don't compose
  well with `torch.compile(reduce-overhead)` and defeat the throughput goal
- Automated model weight downloads (security/determinism)

---

## Architecture

### File layout

All new files under `repos/dorea/scripts/poc/` (new directory):

```
repos/dorea/scripts/poc/
├── uie_bench.py              # Main entrypoint — orchestrates everything
├── models/
│   ├── __init__.py
│   ├── base.py               # UIEModel ABC + shared fp16 helpers
│   ├── raune.py              # RAUNE-Net wrapper (reuses repo's RauneNet)
│   ├── fa_net.py             # FA+Net wrapper
│   ├── shallow_uwnet.py      # Shallow-UWnet wrapper
│   └── color_accurate.py     # Color-Accurate UIE 2025 wrapper
├── metrics.py                # UIQM, SSIM, timing, VRAM peak
├── render.py                 # Contact sheet composition
├── tests/
│   ├── test_shape_contract.py
│   ├── test_metrics.py
│   └── test_render.py
└── README.md
```

### Weights directory

Hand-placed under `working/poc_weights/` (gitignored):

```
working/poc_weights/
├── raune/              # symlink to existing working/sea_thru_poc weights
├── fa_net/
├── shallow_uwnet/
└── color_accurate/
```

The script **does not download**. On startup it checks each requested
model's weight path and fails that cell (not the whole run) with a clear
message citing the upstream URL from `scripts/poc/README.md`.

### Output directory

Timestamped subdirectory under `working/poc_out/`:

```
working/poc_out/2026-04-10T14-30-00/
├── contact_sheet.png
├── bench.json
├── bench.md
└── logs/
    └── <model>_<variant>.log
```

### Isolation

Nothing in `scripts/poc/` is imported by production code. The POC's RAUNE
wrapper imports `models.raune_net` from the same upstream RAUNE-Net checkout
that direct mode uses (via the `--raune-models-dir` / `[models].raune_models_dir`
path) — so the baseline is literally identical — but does not import from
`python/dorea_inference/`. Pytest under `scripts/poc/tests/` never touches
the production codebase.

---

## Component 1: Model interface

### `UIEModel` ABC (`models/base.py`)

```python
class Variant(Enum):
    STRICT_PARITY = "strict_parity"
    BEST_EFFORT = "best_effort"

class UIEModel(ABC):
    name: str

    @abstractmethod
    def load(self, variant: Variant, device: torch.device) -> None:
        """Load weights and apply the fp16 recipe for this variant."""

    @abstractmethod
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Input:  (N,3,H,W) fp16 [0,1] on CUDA.
           Output: (N,3,H,W) fp16 [0,1] on CUDA."""

    @abstractmethod
    def preferred_proxy_size(self) -> int:
        """Long-edge pixel count the model wants. Used by the bench loop
        to size the CUDA input tensor."""
```

The interface takes an **already-shaped, normalized, fp16 CUDA tensor** and
returns the same. No per-model preprocessing dance inside the timing loop.
The bench loop owns decode, resize, normalization. The model wrapper owns
weight loading and the forward pass only. This is the only way latency
measurements are apples-to-apples: if one wrapper internally does extra
`ToTensor`/`normalize` and another doesn't, the measurement is of plumbing,
not networks.

### Shared helper

`models/base.py::force_norm_fp32(model)` walks a module and forces
`InstanceNorm{1,2,3}d`, `BatchNorm{1,2,3}d`, and `GroupNorm` layers back to
fp32 after a global `.half()`. This generalizes the RAUNE-Net rule at
`repos/dorea/python/dorea_inference/raune_filter.py:680–685`.

### Variants

| Model | STRICT_PARITY | BEST_EFFORT |
|---|---|---|
| RAUNE | `.half()` + norm→fp32 + `torch.compile(reduce-overhead, dynamic=False)` | *same as strict* (baseline) |
| FA+Net | `.half()` + norm→fp32 + `torch.compile(reduce-overhead)` | `.half()` eager |
| Shallow-UWnet | `.half()` + norm→fp32 + `torch.compile(reduce-overhead)` | `.half()` eager |
| Color-Accurate UIE | `.half()` + norm→fp32 + `torch.compile(reduce-overhead)` | `.half()` eager |

RAUNE's BEST_EFFORT row is explicitly skipped in the matrix (it would
re-time the exact same config) and reported as "SKIP (same as strict)".

If a candidate model has no norm layers (plausible for the 3.9K-param
Color-Accurate UIE), the two variants differ only in `torch.compile` mode.
That's expected and surfaces in the output as near-identical rows.

---

## Component 2: Benchmark loop (`uie_bench.py`)

### Flow

1. Parse args, resolve input clip path and output directory
2. Decode all 360 frames via PyAV once at full 3840×2160, keep as uint8
   numpy arrays in RAM (~8.9 GB — fits on this workstation)
3. Sample 8 contact-sheet frame indices at `[0, 45, 90, 135, 180, 225, 270, 315]`
4. Precompute per-model contact-sheet inputs (resize the 8 frames to each
   model's `preferred_proxy_size`, normalize, transfer to CUDA, keep as fp16)
5. For each `(model, variant)` pair in the matrix:
   a. `torch.cuda.empty_cache()` + `reset_peak_memory_stats()`
   b. Start `t0 = perf_counter()`
   c. `model.load(variant, device=cuda)`
   d. First forward on 1 frame → `time_to_first_frame_s = perf_counter() - t0`
   e. Warmup: 3 batches at batch_size 4 (not timed) → 12 frames consumed
   f. Measured run: 87 batches at batch_size 4 (the remaining 348 frames),
      time each batch individually with `torch.cuda.synchronize()` before
      start and before stop timer
   g. Compute mean / p50 / p95 latency from the 87-sample vector
   h. Throughput = `(87 * batch_size) / total_warm_time_s`
   i. `vram_peak_mib = torch.cuda.max_memory_allocated() / 2**20`
   j. Run contact-sheet inference on the precomputed 8-frame input
   k. Move contact-sheet outputs to CPU uint8, keep in memory for render
   l. Compute UIQM per frame (CPU, numpy) and SSIM vs RAUNE's BEST_EFFORT
      output on the same frame indices
   m. Record BenchResult
   n. `del model; torch.cuda.empty_cache()`; assert
      `max_memory_allocated() < 5.5 GiB` after cleanup
6. Render `contact_sheet.png` from BEST_EFFORT outputs only
7. Write `bench.json` and `bench.md`
8. Print the markdown table and contact sheet path to stdout

### Key choices

**Whole clip in RAM.** 3840 × 2160 × 3 × 360 ≈ 8.9 GB. Eliminates HEVC 10-bit
PyAV decode from the timing loop — decode is CPU-bound and varies run to
run, it would be a confound.

**Batch size 4.** Direct mode's `dorea.toml` default `fused_batch_size = 32`
is for the fused RAUNE+depth path at a smaller proxy. Honest batch size for
4K proxies with fp16 RAUNE-Net on the 3060 is 4. Exposed as
`--batch-size` for tuning, defaults to 4.

**360 frames / 87 measured batches.** p95 of 87 samples is the ~83rd value,
stable enough for decision-making. Shorter bursts give noisy p95.

**Sequential loading is mandatory.** Never two models loaded at once. The
`del model; torch.cuda.empty_cache()` between cells plus the post-cleanup
VRAM assertion catches leaks that would cascade into OOM for later cells.

**Timing discipline.** `torch.cuda.synchronize()` before each start and stop
(CUDA kernels are async, `perf_counter` alone measures kernel launch, not
execution). First batch excluded from latency stats (CUDA-graph capture
lives there for compiled variants).

### BenchResult dataclass (`metrics.py`)

```python
@dataclass
class BenchResult:
    model: str                     # "raune", "fa_net", "shallow_uwnet", "color_accurate"
    variant: str                   # "strict_parity" | "best_effort"
    proxy_size: tuple[int, int]    # (H, W) actually used
    batch_size: int
    time_to_first_frame_s: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_fps: float
    vram_peak_mib: float
    uiqm_mean: float | None
    ssim_vs_raune_mean: float | None
    error: str | None              # set iff this cell failed
```

---

## Component 3: Metrics (`metrics.py`)

**UIQM** — vendored ~80-line numpy implementation (chrominance measure +
sharpness + contrast) rather than adding `piq` or `pytorch-iqa` as new
dependencies. UIQM is well-defined and self-contained. The source file will
cite Panetta et al. 2016 in a docstring.

**SSIM** — `skimage.metrics.structural_similarity`. skimage is already in
the venv. Compute per-channel mean against RAUNE's BEST_EFFORT output tile
for the same frame index. RAUNE's own SSIM row is 1.0 by definition.

**Timing helpers** — a small `cuda_timer` context manager that handles
synchronization correctly, and a `vram_snapshot` helper that wraps
`reset_peak_memory_stats` / `max_memory_allocated`.

---

## Component 4: Contact sheet render (`render.py`)

### Layout

- **8 rows × 5 columns.** Rows = sampled frames (indices 0, 45, 90, 135, 180,
  225, 270, 315). Columns = `original | raune | fa_net | shallow_uwnet |
  color_accurate` in fixed order. Original on the left for anchoring.
- **Tile resolution.** Gated by `--tile-size`:
  - `native` (default): each tile at its model's native inference resolution,
    padded with dark background to the max tile size for column alignment.
    Original column downsampled to match. Shows exactly what each model
    produced, no upscaling artifacts. Honest comparison of networks.
  - `4k`: each enhanced output bicubic-upscaled (PIL Lanczos) to 3840×2160 to
    match the original. Shows the "shipped" look (direct mode upscales the
    proxy back to full res for grading). Less honest about the network but
    more honest about the product. Will warn before writing (~200 MB PNG).
- **Borders.** 4 px dark gray (rgb 32, 32, 32) between tiles.
- **Column headers.** 80 px strip at top, model name + variant, white on dark,
  PIL default font (no font file dependency).
- **Row labels.** 120 px left margin, frame index (`f=000`, `f=045`, …).
- **Per-tile overlay.** Bottom-right, small black box with
  `p50=62ms UIQM=3.24` in white. Skipped for the original column.
- **Format.** PNG, default PIL compression.

### Which variant is shown

BEST_EFFORT only. The two-tier `bench.md` table already answers the
"did strict parity hurt speed?" question. Rendering both variants would
double columns and clutter the image. A safety check compares strict vs
best-effort outputs per model (max L1 difference); if any exceeds 2/255,
log a warning — this catches numeric drift from `torch.compile` changes.

### Error cell rendering

If a cell failed, its entire column is rendered as dark red tiles with the
error message wrapped and centered in white. The contact sheet is always
complete — no cell ever makes the sheet un-render. Metric overlays show
`—` for failed cells.

### Function signature

```python
def build_contact_sheet(
    original_frames: list[np.ndarray],             # 8 × (2160, 3840, 3) uint8
    model_outputs: dict[str, list[np.ndarray]],    # {"raune": 8 × ..., ...}
    bench_results: dict[str, BenchResult],         # for overlay text + error state
    tile_size: Literal["native", "4k"],
    out_path: Path,
) -> None:
```

Pure function — numpy in, PNG out. No torch, no CUDA.

---

## Error handling

**Per-cell isolation.** Every `(model, variant)` pair runs inside a
try/except boundary. Failures are captured into `BenchResult.error` and the
rest of the matrix continues. Rationale: if FA+Net's weights aren't
downloaded yet, RAUNE and Shallow-UWnet numbers from the same run are still
valuable.

**No silent fallbacks within a cell.** Matching `feedback_fail_fast.md`:
if `torch.compile` fails inside STRICT_PARITY, the cell fails — it does not
silently swap to eager. The BEST_EFFORT row for the same model still runs
independently.

| Situation | Behavior |
|---|---|
| Weights file missing | Cell fails; other cells continue |
| `torch.compile` failure in STRICT_PARITY | Cell fails; BEST_EFFORT still runs |
| CUDA OOM during load | Cell fails; cleanup; next cell starts clean |
| CUDA OOM during forward | Same |
| Input video missing / decode failure | **Hard abort** |
| CUDA unavailable | **Hard abort** |
| Shape contract / VRAM budget assertion | **Hard abort** (indicates POC bug) |

Hard aborts exit non-zero with one clear error line. Cell failures print a
warning line and continue.

---

## Testing

Three small pytest files under `scripts/poc/tests/`. No production imports.

1. **`test_shape_contract.py`** — each model wrapper runs on a synthetic
   `(4, 3, 256, 256)` fp16 tensor, both variants. Asserts output shape
   matches input, dtype is fp16, values are finite. Uses `pytest.importorskip`
   for weight files so it runs even before weights are downloaded. Catches
   "wired the model wrong" in seconds.

2. **`test_metrics.py`** — UIQM and SSIM on known inputs. Identity image
   pair → SSIM = 1.0. UIQM value on a fixed synthetic image matches a
   pre-computed reference value (tolerance 1e-3). Catches vendored UIQM
   regressions.

3. **`test_render.py`** — renders a contact sheet from synthetic numpy
   inputs with one cell marked as error. Asserts PNG exists, is non-trivial
   size (>10 KB), and a pixel sampled from the expected error-cell region
   is red. Catches layout drift.

No integration test for the bench loop itself — that's what we run to *get*
answers.

Run with:

```bash
/opt/dorea-venv/bin/python -m pytest repos/dorea/scripts/poc/tests/ -v
```

---

## Run UX

### Invocation

```bash
cd /workspaces/dorea-workspace
/opt/dorea-venv/bin/python repos/dorea/scripts/poc/uie_bench.py \
    --input footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4 \
    --out working/poc_out \
    [--models raune,fa_net,shallow_uwnet,color_accurate] \
    [--variants strict_parity,best_effort] \
    [--batch-size 4] \
    [--tile-size native]
```

Defaults are tuned so the minimal command is just `--input <clip>`.

### Progress output (one line per cell)

```
[1/8] raune          / strict_parity ... loading ... compiling ... warm ... 62ms p50, 64 FPS, 1842 MiB, UIQM 3.24, SSIM 1.00
[2/8] raune          / best_effort   ... SKIP (same as strict for raune)
[3/8] fa_net         / strict_parity ... FAILED: weights not found at working/poc_weights/fa_net/fa_plus_net.pth
[4/8] fa_net         / best_effort   ... FAILED: (same)
[5/8] shallow_uwnet  / strict_parity ... loading ... compiling ... warm ... 18ms p50, 220 FPS, 580 MiB, UIQM 3.01, SSIM 0.82
...
```

### Final summary

Prints the `bench.md` table to stdout and the contact sheet path. Nothing
else.

---

## bench.md format

Sorted by `throughput_fps` descending. Example:

```
| Model          | Variant       | TTFF (s) | p50 (ms) | p95 (ms) |  FPS | VRAM (MiB) | UIQM | SSIM→RAUNE |
|----------------|---------------|---------:|---------:|---------:|-----:|-----------:|-----:|-----------:|
| fa_net         | best_effort   |      1.2 |        8 |       11 |  485 |        312 | 3.12 |       0.87 |
| fa_net         | strict_parity |      4.8 |        9 |       12 |  445 |        318 | 3.12 |       0.87 |
| color_accurate | best_effort   |      0.9 |        6 |        9 |  600 |        180 | 3.05 |       0.79 |
| shallow_uwnet  | best_effort   |      1.5 |       18 |       24 |  220 |        580 | 3.01 |       0.82 |
| raune          | strict_parity |      9.4 |       62 |       78 |   64 |       1842 | 3.24 |       1.00 |
```

---

## Open questions

None. All decisions locked during brainstorming. Weights URLs and exact file
names will be filled into `scripts/poc/README.md` during implementation
(first task in the implementation plan).

---

## Success criteria

The POC succeeds if, after one run against
`footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4`, you can answer
both of:

1. **Is there a model that is ≥2× faster than RAUNE-Net at fp16 while
   producing a contact sheet you'd ship?** (The ≥2× threshold is the break-
   even where multi-instance overhead becomes worth chasing.)
2. **How much VRAM headroom does that model leave for a second instance?**
   (Need ≤2.5 GB peak to leave room for 2 instances on the 6 GB card.)

If the answers are "no" and "doesn't matter", the POC has still succeeded —
it told you to stop looking and invest elsewhere.
