"""Fleet movement kinematics shared across pipeline cases.

Copied verbatim from pipeline/rulebase/case1/baseline/core/physics.py::fleet_speed
to break the circular dependency between trajectory_safety and any specific case.
"""

from __future__ import annotations

import math

from .orbit_constants import MAX_SPEED


def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return float(1.0 + (MAX_SPEED - 1.0) * (ratio**1.5))
