"""Cross-case evaluation utilities.

Pure-computation helpers shared across pipeline/{rulebase,imitation}/case*/.
Case-specific glue (model loading, dataset construction) stays in each case
under its `evaluation/` directory and only calls into helpers exported here.
"""

from __future__ import annotations

from .metrics import (
    FromArrays,
    ShipsArrays,
    TargetArrays,
    best_f1_threshold,
    expected_calibration_error,
    expected_calibration_error_multiclass,
    from_metrics,
    ships_metrics,
    target_metrics,
)
from .snapshot_update import capture_observation, write_snapshot
from .vs_baseline import summarize_records, wilson_ci

__all__ = [
    "FromArrays",
    "ShipsArrays",
    "TargetArrays",
    "best_f1_threshold",
    "capture_observation",
    "expected_calibration_error",
    "expected_calibration_error_multiclass",
    "from_metrics",
    "ships_metrics",
    "summarize_records",
    "target_metrics",
    "wilson_ci",
    "write_snapshot",
]
