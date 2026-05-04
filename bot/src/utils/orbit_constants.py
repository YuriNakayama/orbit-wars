"""Orbit Wars physics constants shared across pipeline cases.

These mirror the per-case copies in `pipeline/{rulebase,imitation}/case*/baseline/core/`
and `pipeline/imitation/case*/policy/geometry.py`. The shared definition is the
single source of truth for trajectory_safety; per-case copies remain in place
to keep each case directory self-contained for Kaggle submission packaging.
"""

from __future__ import annotations

CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
SUN_SAFETY: float = 1.5
MAX_SPEED: float = 6.0
ROTATION_LIMIT: float = 50.0
HORIZON: int = 110
INTERCEPT_TOLERANCE: int = 1
LAUNCH_CLEARANCE: float = 0.1
COMET_APPEARANCE_TURNS: tuple[int, ...] = (50, 150, 250, 350, 450)
