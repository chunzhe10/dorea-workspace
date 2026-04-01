# LUT Generation Learnings (2026-04-01)

Captured during dev-loop for issue #3. Corvia MCP was unavailable, so findings
are recorded here for manual ingestion.

## Algorithm Bugs Found During First Real Run

### 1. Gain normalization suppresses red channel
**What:** `compute_correction_params()` normalized gain to max=1.0, which set the
strongest channel (blue, underwater) to 1.0 and pushed red to ~0.5. With subsequent
red boost capped at 30%, effective red gain was only 0.57.
**Fix:** Removed normalization. Clip to [0.5, 1.5] range instead.
**Why it matters:** Any normalization-to-max approach will suppress the weakest channel
in color-imbalanced footage. For underwater, always expect red to be weakest.

### 2. Per-channel gamma penalizes weak channels
**What:** Gamma was derived from midtone color balance: weak channels got higher gamma
(less expansion), strong channels got lower gamma (more expansion). This created a
negative feedback loop for red.
**Fix:** Tightened the balance clamp from [0.5, 2.0] to [0.7, 1.3] and gamma range
from [0.3, 0.8] to [0.35, 0.65].
**Why it matters:** Per-channel gamma couples color balance with contrast. For a base
LUT, prefer uniform gamma with separate color balance (via gain).

### 3. Highlight rolloff is essential for D-Log M
**What:** D-Log M stores 14+ stops of dynamic range in a flat curve. Expanding with
gamma < 1 pushes highlights past 1.0. Without a soft shoulder, highlights clip hard.
**Fix:** Added exponential rolloff at shoulder=0.85:
`corrected = shoulder + headroom * (1 - exp(-excess / headroom))`
**Why it matters:** Any D-Log-to-Rec.709 conversion needs highlight protection.
The shoulder=0.85 starting point preserves detail without visible compression artifacts.

## Color Science Insights

### 4. Beer-Lambert underwater red absorption
Red gain needed to restore R/G parity (pure water, per Beer-Lambert):
- 3m depth: 2.18x
- 5m depth: 3.67x
- 8m depth: 8.0x
- 10m depth: 14.5x

Real tropical water absorbs faster (dissolved organics). Our 2.0x is conservative
for 3m and insufficient for 5m. Final R/G of 0.67 reads as "uncorrected" to most
viewers. Target 0.80-0.85 for "underwater ambiance" look.

### 5. Circular reasoning in reference-based gain derivation
The LUT script analyzes corrected reference images to derive gains. Since references
are already blue-biased (even after correction), the derived gains reinforce the blue
bias. Better approach: derive gains from the *correction needed* (raw vs target),
not from the target appearance alone.

### 6. Saturation ordering
Saturation after highlight rolloff can push compressed highlights above 1.0, causing
hue shifts at the final clip. Correct order: gamma -> saturation -> rolloff -> clip.

## Pipeline Insights

### 7. D-Log M JPEG encoding assumption
D-Log M keyframes extracted by ffmpeg contain raw D-Log M values in sRGB byte encoding
(no EOTF applied). `generate_references.py` correctly treats pixel values as D-Log M,
while `00_generate_lut.py` correctly treats reference images as sRGB. This asymmetry
is correct but undocumented and fragile.

### 8. Two-repo workflow
Pipeline code lives in `repos/dorea/` (chunzhe10/dorea.git), workspace files in the
workspace repo (chunzhe10/dorea-workspace.git). Changes require commits and PRs in
both repos. The `repos/dorea/` directory is gitignored in the workspace.
