# Bench Harness Noise Floor — 2026-04-13

Two back-to-back runs of `scripts/bench/run.py` on the same git commit
(`b885609`), same hardware, same clip, N=5 samples each, 4 warmup frames,
no code changes between runs.

**Hardware:** RTX 3060 Laptop GPU, driver 570.211.01
**PCIe:** reported gen1 x8 (idle power state), measured 9.6-10.4 GB/s pinned
**Clip:** `/tmp/test_clip_30f.mp4` (30 frames, 4K H.264)
**Config:** 960x540 proxy, batch=4, `--tensorrt`

## Results

| Metric | cal-a (mean ± CI) | cal-b (mean ± CI) | Δ | p | flag |
|---|---:|---:|---:|---:|:---:|
| steady_state_fps | 4.452 ± 0.047 | 4.399 ± 0.037 | **-1.2%** | 0.041 | ** |
| gpu_kernel_ms_per_frame | 193.96 ± 2.77 | 195.72 ± 1.66 | +0.9% | 0.177 | — |
| gpu_thread_wall_ms_per_frame | 193.98 ± 2.79 | 195.76 ± 1.68 | +0.9% | 0.175 | — |
| wall_ms_per_frame | 224.76 ± 2.12 | 227.30 ± 1.64 | **+1.1%** | 0.032 | ** |
| decode_thread_wall_ms_per_frame | 25.62 ± 1.74 | 26.08 ± 2.96 | +1.8% | 0.722 | — |
| encode_thread_wall_ms_per_frame | 45.88 ± 1.61 | 46.20 ± 1.51 | +0.7% | 0.698 | — |
| time_to_first_frame_ms | 2754 ± 98 | 2785 ± 69 | +1.1% | 0.498 | — |

## Measured Noise Floor (per-run 95% CI half-width, relative)

| Metric | CI |
|---|---:|
| steady_state_fps | ±1.0% |
| gpu_kernel_ms_per_frame | ±1.0% |
| wall_ms_per_frame | ±0.9% |
| decode_thread_wall_ms_per_frame | ±9% (high variance) |
| encode_thread_wall_ms_per_frame | ±3.4% |

## Interpretation

The calibration flagged `steady_state_fps` (-1.2%, p=0.041) and
`wall_ms_per_frame` (+1.1%, p=0.032) as statistically significant with `**`
despite comparing **identical code**. Meanwhile `gpu_kernel_ms_per_frame`
(+0.9%, p=0.18) correctly showed as within noise.

**This reveals a systematic drift of ~1% between back-to-back runs**, most
likely driven by thermal state or GPU DVFS. The GPU kernel work itself is
stable; the variance comes from something else in the pipeline — likely
CPU-side scheduling or encode thread fluctuation.

**Practical implication:** the default significance markers are too
sensitive for this workstation. A `**` marker on a < 1.5% delta should NOT
be trusted as a real improvement without additional runs.

## Revised Decision Thresholds for This Workstation

| Delta (absolute) | p-value | Trust level |
|---|---|---|
| < 1.5% | any | **Within noise** — do not trust |
| 1.5%–3% | < 0.01 | Ambiguous — rerun with N=10 to confirm |
| 1.5%–3% | ≥ 0.01 | Within noise |
| > 3% | < 0.01 | **Likely real** |
| > 3% | < 0.05 | Probably real — verify with another run |
| > 5% | < 0.01 | **High confidence real** |

**Rule of thumb:** multiply the measured delta by the p-value to get a
crude "confidence score". Below 0.03 is worth investigating; above 0.1 is
almost certainly real.

## Validation of Prior Work

This noise floor vindicates the PR #77 (pinned memory) claim of +5.8% FPS:
the measured delta was > 4× the noise floor and was repeatable. The 1.5×
TRT speedup (PR #73) was ~50× the noise floor — unambiguously real.

## Recommendations

1. **Increase N_samples to 10** for headline benchmarks. N=10 reduces the
   CI by √2, tightening the noise floor to ~0.7% and making 1-2% deltas
   distinguishable.
2. **Eventually implement `--pin`** (GPU clock lock, CPU governor,
   drop_caches) as a Phase 2 task. Pinning should cut thermal drift
   contribution and tighten the noise floor to the ~0.3-0.5% range.
3. **Longer benchmark clip** (300+ frames at 4K) would amortize warmup
   effects more and reduce per-sample variance. The current 30-frame clip
   runs for only ~7 seconds — short enough that sample-to-sample drift
   dominates.
4. **Never trust a `**` flag below 1.5% delta** until one of the above
   mitigations is implemented and re-calibrated.

## Source Files

- `cal-a` result: `scripts/bench/results/2026-04-13T02-24-18Z_b885609_cal-a.json`
- `cal-b` result: `scripts/bench/results/2026-04-13T02-25-12Z_b885609_cal-b.json`
- Compare output: above table, generated via `scripts/bench/compare.py`

## Notes

- `decode_thread_wall_ms_per_frame` has very high variance (±9% CI) which
  is fine — the decode thread is 93% idle so small absolute timing noise
  becomes large relative noise. Don't use this metric for regression gates.
- `time_to_first_frame_ms` includes TRT engine deserialization (~1.5s) and
  Triton JIT compile. For short 30-frame clips this dominates the first
  sample's per-frame average. Use `steady_state_fps` for optimization
  decisions.
- PCIe is reported as gen1 x8 by NVML when the GPU is idle (power saving).
  The measured bandwidth (~10 GB/s pinned) is healthier than gen1 x8
  theoretical (~4 GB/s), confirming the link upshifts under load.
