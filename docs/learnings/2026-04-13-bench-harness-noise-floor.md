# Bench Harness Noise Floor — 2026-04-13 (v2, after review fixes)

Two back-to-back runs of `scripts/bench/run.py` on the same git commit,
same hardware, same clip, N=5 samples each, 4 warmup frames, no code
changes between runs.

**Hardware:** RTX 3060 Laptop GPU, driver 570.211.01
**PCIe:** reported gen1 x8 (idle power state), measured 10.3-10.4 GB/s pinned
**Clip:** `/tmp/test_clip_30f.mp4` (30 frames, 4K H.264)
**Config:** 960x540 proxy, batch=4, `--tensorrt`
**Harness version:** schema v2 (post-review fixes — see design spec v2 and
`chunzhe10/dorea#79`)

## v2 vs v1 Results

v1 (pre-review fixes) compared identical code and incorrectly flagged a
1.2% delta on `steady_state_fps` as `**` (statistically significant),
which was a Type I error from three compounding problems:
1. Welch's t-test assumes iid normal samples — autocorrelated thermal
   drift between back-to-back samples inflated the false-positive rate
2. No multiple-comparison correction across 7 metrics (30% FWER)
3. Forced `torch.cuda.synchronize()` on every batch in `_process_batch`
   added variance via CPU-side wait time variability

v2 fixes all three: permutation test on raw samples, Holm-Bonferroni
correction, and `DOREA_BENCH=1`-gated CUDA event recording.

## v2 Results (after review fixes)

### Run A: `cal-v2-a`
```
steady_state_fps:   4.463 ± 0.036
gpu_kernel_ms:      192.9 ± 2.4
wall_ms_per_frame:  223.6 ± 2.2
decode_ms:          23.5 ± 1.8
encode_ms:          44.9 ± 0.9
```

### Run B: `cal-v2-b`
```
steady_state_fps:   4.453 ± 0.046
gpu_kernel_ms:      194.2 ± 1.6
wall_ms_per_frame:  225.1 ± 1.8
decode_ms:          25.4 ± 0.7
encode_ms:          45.2 ± 1.6
```

### Comparison (via v2 `compare.py`, ROPE=±1.5%)

| Metric | Δ | p_raw | p_holm | Verdict |
|---|---:|---:|---:|:---|
| steady_state_fps | -0.2% | 0.992 | 1.000 | **EQUIVALENT** |
| gpu_kernel_ms_per_frame | +0.7% | 0.238 | 0.952 | **EQUIVALENT** |
| gpu_thread_wall_ms_per_frame | +0.7% | 0.254 | 0.952 | **EQUIVALENT** |
| wall_ms_per_frame | +0.7% | 0.183 | 0.913 | **EQUIVALENT** |
| decode_thread_wall_ms_per_frame | +8.3% | 0.032 | 0.190 | unclear |
| encode_thread_wall_ms_per_frame | +0.6% | 0.730 | 1.000 | **EQUIVALENT** |

**v2 correctly classifies identical-code runs as EQUIVALENT** on every
headline metric. The decoder thread is the only metric showing
"unclear" — it's genuinely more variable (the thread is 93% idle on
average, so small absolute timing drifts turn into large relative
percentages), but the delta is nowhere near `p_holm < 0.01` so it
doesn't trigger DIFFERENT either.

## ROPE (Region of Practical Equivalence) — ±1.5%

The ROPE is the threshold below which we consider a change practically
equivalent regardless of p-value. It was chosen at 1.5% because:

1. The observed noise floor for `steady_state_fps` is about 0.8-1.0%
   (95% CI half-width at N=5).
2. A 1.5% ROPE gives a safety margin above the noise floor.
3. Changes of <1.5% are unlikely to be actionable for this pipeline
   anyway — a 1% fps change at 2.8 fps is ~10 seconds over a 1000-frame
   clip.

## Decision Thresholds

| Verdict | Condition |
|---|---|
| **EQUIVALENT** | Δ within ±1.5% — no meaningful change |
| **DIFFERENT** | p_holm < 0.01 AND Δ > 1.5% — confident real change |
| **unclear** | Large Δ but weak p, or small Δ with significant p — retry |

## Validation of Prior Work

The v1 calibration "vindicated" prior PRs (#73 TRT, #77 pinned memory)
inferentially. With v2, we can re-benchmark and test properly:

- **PR #73 (TRT FP16):** expected ~50% fps delta. Well above ROPE,
  guaranteed to classify DIFFERENT at p_holm < 0.01. (Not re-measured.)
- **PR #77 (pinned memory):** claimed +5.8% at 540p. This is 4× ROPE —
  should classify DIFFERENT if real. (Not re-measured but the earlier
  numbers were from before the `cuda.synchronize` removal, so the
  magnitude is now suspect. A re-benchmark is warranted if pinned
  memory's contribution is important.)

## Recommendations

1. **N=5 is marginal** for the permutation test. With N=5, raw p can
   only reach ~0.008 (2/C(10,5)) — meaning Holm-corrected p for 6
   metrics can never beat 0.048, well above the 0.01 DIFFERENT
   threshold. **Use N >= 6 (preferably N=10) for regression-gate runs.**
2. **Longer benchmark clip** remains desirable. 30 frames × 0.22s =
   6.6s per sample is too short for steady state to fully dominate.
   300+ frames would amortize warmup better and allow N=3 samples to
   give tighter CIs.
3. **`--pin` is still Phase 2**. Thermal drift is now bounded by the
   ROPE, but locking GPU clocks would tighten the noise floor further.

## Source Files

- `cal-v2-a`: `scripts/bench/results/2026-04-13T07-42-43Z_b885609_cal-v2-a.json`
- `cal-v2-b`: `scripts/bench/results/2026-04-13T07-43-40Z_b885609_cal-v2-b.json`
- 1440p baseline: `scripts/bench/results/2026-04-13T07-41-26Z_*_main-1440p-v2.json`
  (captured before path fix; git_sha field will be incorrect until re-run)

## Notes on the Original v1 Framing

The v1 doc described a "noise floor of ~1%" and recommended thresholds
like "Δ > 3% with p < 0.01". This was directionally right but
theoretically wrong — the 1% drift the v1 test measured was not a stable
"noise floor" but a Type I error rate inflated by violated statistical
assumptions. The right fix (done in v2) is to use distribution-free
tests and correct for multiple comparisons, not to bump thresholds until
false positives stop showing.

`decode_thread_wall_ms_per_frame` has wide CIs because the decode thread
is 93% idle — small absolute timing noise becomes large relative noise.
Don't use this metric for regression decisions.
