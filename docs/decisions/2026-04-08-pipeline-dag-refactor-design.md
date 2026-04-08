# Pipeline DAG Refactor Design

**Date:** 2026-04-08
**Issue:** chunzhe10/dorea#56

## Decision

Refactor `grade.rs::run` (~527 lines of sequential logic) into four named
stages under `crates/dorea-cli/src/pipeline/`, each with typed inputs/outputs.
`grade.rs::run` becomes ~80 lines of orchestration.

## Stages

1. **KeyframeStage** — proxy decode → change detect → keyframe select
2. **FeatureStage** — fused RAUNE+Maxine+depth inference → paged store
3. **CalibrationStage** — zone compute → segment detect → LUT build → HSL → grader init
4. **GradingStage** — full-res decode → depth interpolate → blend_t → grade → encode

## File Structure

| File | Content |
|------|---------|
| `pipeline/mod.rs` | Boundary I/O structs (no traits — stages are concrete, not generic) |
| `pipeline/keyframe.rs` | `KeyframeEntry`, `run_keyframe_stage()` |
| `pipeline/feature.rs` | `PagedCalibrationStore`, `run_feature_stage()` |
| `pipeline/calibration.rs` | `SegmentCalibration`, `run_calibration_stage()` |
| `pipeline/grading.rs` | `lerp_depth`, `run_grading_stage()` |
| `grade.rs` | Config resolution, probe, encoder setup, stage orchestration |

## Design Decisions

1. **Free functions, not trait objects.** Each stage is a `pub fn run_*_stage(input) -> Result<Output>`.
   The Node/Stage traits from the original issue add abstraction without benefit — the pipeline
   is strictly sequential, never recomposed, and the stages have different signatures.

2. **Boundary structs carry data between stages.** `KeyframeStageOutput`, `FeatureStageOutput`,
   `CalibrationStageOutput` — each consumed by the next stage.

3. **PagedCalibrationStore moves to feature.rs.** It's an implementation detail of the feature
   stage, not shared.

4. **Config struct passed to each stage.** A `PipelineConfig` struct in mod.rs holds all resolved
   config values. Stages receive `&PipelineConfig` instead of reconstructing from GradeArgs.

5. **GradingStage takes ownership of encoder.** It writes frames directly. Returns final stats.
