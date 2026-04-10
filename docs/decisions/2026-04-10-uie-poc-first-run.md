# UIE POC first run

**Date:** 2026-04-10

Initial end-to-end smoke run of `repos/dorea/scripts/poc/uie_bench.py` against
`footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4`.

Per-model results are in `working/poc_out/<timestamp>/bench.md` and
`contact_sheet.png`. See design spec
`docs/decisions/2026-04-10-uie-comparison-poc-design.md` and impl plan
`docs/plans/2026-04-10-uie-comparison-poc-impl.md`.

The Color-Accurate UIE cell failed gracefully (weights not yet vendored); the
other three models populated successfully. Next step: visual review and
decision on whether any candidate is worth pursuing for multi-instance
parallelism.
