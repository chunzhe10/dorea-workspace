# First LUT Generation — Design Decision

**Date:** 2026-04-01
**Issue:** #3 — Generate first real LUT from reference images
**Status:** Approved

## Problem

`luts/` is empty. `00_generate_lut.py` is fully implemented (632 lines) but has
never been run because there are no reference images in `references/`.

## Decision: Bootstrap references from existing dive footage

### Approach

Create a helper script (`scripts/generate_references.py`) that applies principled
underwater color correction to raw D-Log M keyframes, producing the "target look"
reference images that `00_generate_lut.py` needs.

### Why this approach

1. **Real footage**: Uses actual DJI Action 4 D-Log M keyframes from 2025-11-01 dive
   (28 frames across 2 clips), preserving realistic underwater color distribution
2. **Reproducible**: No external image dependencies. Parameters are documented and
   tunable. Re-run with different parameters = different look.
3. **Color science grounded**: Correction parameters based on known underwater light
   absorption characteristics (red attenuates ~50% at 3m depth)
4. **Not circular**: Reference images define the TARGET aesthetic; `00_generate_lut.py`
   then derives an optimized lift/gamma/gain LUT to reproduce that target from raw
   D-Log M input. Different correction models, same goal.

### Alternatives rejected

- **Download external images**: Copyright risk, not reproducible, may not match
  DJI Action 4 color profile
- **Synthetic gradients**: Unrealistic color distribution, poor zone statistics
- **Use raw D-Log M as references**: Would produce identity-like LUT (flat → flat)

### Reference generation parameters

Based on underwater photography color science:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| D-Log M decode | Standard DJI curve | Convert flat encoding to linear light |
| Red gain | 1.35 | Compensate for water absorption at 3-5m depth |
| Green gain | 1.05 | Slight boost for natural balance |
| Blue gain | 0.95 | Reduce cyan/blue cast from scattering |
| Gamma | 0.50 | Expand D-Log M flat contrast to natural look |
| Saturation | 1.4x | Restore vibrancy lost in log encoding |
| Shadow lift | +0.01 | Prevent shadow crushing |

### Pipeline flow

```
working/keyframes/2025-11-01/*.jpg  (D-Log M raw)
        │
        ▼
scripts/generate_references.py  (color correction)
        │
        ▼
references/look_v1/*.jpg  (sRGB target look)
        │
        ▼
scripts/00_generate_lut.py  (LUT derivation)
        │
        ▼
luts/underwater_base.cube  (33×33×33 3D LUT)
```

### Validation

1. Format: LUT is valid 33×33×33 .cube file
2. Visual: Apply LUT to raw keyframes via ffmpeg → visible red channel recovery
3. Quality: Before/after comparison shows natural underwater colors
