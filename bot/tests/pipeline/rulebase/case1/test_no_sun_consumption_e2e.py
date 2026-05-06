"""E2E smoke test: rulebase/case1 must not send any fleet into the sun.

Plays a 10-turn match between baseline_v1 and a noop opponent under a fixed
seed, then walks every step's `fleets` list and asserts no fleet's position
is within `SUN_R + safety` of the centre. This catches the post-aim Bug 1
(sun consumption) at the simulator level, beyond the synthetic decoder tests.

Marked @pytest.mark.integration — depends on kaggle_environments runtime.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from pipeline.rulebase.case1.baseline import agent  # noqa: E402
from pipeline.rulebase.case1.baseline.core.config import (  # noqa: E402
    CENTER_X,
    CENTER_Y,
    SUN_R,
)

_SUN_PROBE_SAFETY = 0.5


def _noop(_obs: object) -> list[list[int | float]]:
    return []


def _walk_fleets(env: Any) -> list[tuple[int, int, float, float]]:
    """Return [(step_idx, fleet_id, x, y), ...] across all steps."""
    out: list[tuple[int, int, float, float]] = []
    for step_idx, step in enumerate(env.steps):
        for entry in step:
            obs = entry.get("observation") or {}
            for fleet in obs.get("fleets") or []:
                if len(fleet) < 4:
                    continue
                fid = int(fleet[0])
                fx = float(fleet[2])
                fy = float(fleet[3])
                out.append((step_idx, fid, fx, fy))
    return out


@pytest.mark.integration
def test_no_fleet_enters_sun_zone_under_baseline_v1() -> None:
    env = kaggle_environments.make(
        "orbit_wars",
        configuration={"agents": 2, "seed": 0, "episodeSteps": 12},
    )
    env.run([agent, _noop])
    assert len(env.steps) > 0

    threshold = SUN_R + _SUN_PROBE_SAFETY
    violations: list[tuple[int, int, float, float, float]] = []
    for step_idx, fid, fx, fy in _walk_fleets(env):
        d = math.hypot(fx - CENTER_X, fy - CENTER_Y)
        if d < threshold:
            violations.append((step_idx, fid, fx, fy, d))

    assert not violations, (
        f"Fleet inside sun zone (threshold={threshold}): "
        f"{violations[:5]}{'...' if len(violations) > 5 else ''}"
    )
