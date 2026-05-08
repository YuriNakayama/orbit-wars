"""Regression tests: case4 must send larger, fewer fleets than case2.

These are integration-level measurements over several seeds to reduce variance
introduced by the non-deterministic parts of kaggle_environments.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from pipeline.rulebase.case2.baseline import agent as agent_v2  # noqa: E402
from pipeline.rulebase.case4.baseline import agent as agent_v4  # noqa: E402

AgentFn = Callable[[dict[str, Any]], list[list[int | float]]]


def _max_observed_ships_per_fleet(
    agent: AgentFn, seed: int, max_turns: int = 80
) -> dict[int, int]:
    """Run self-play and record each fleet's peak ship count.

    kaggle_environments reports fleet.ships which decays as the fleet crosses
    near other bodies; to approximate launch size we take the per-fleet maximum.
    """
    env = kaggle_environments.make(
        "orbit_wars", configuration={"agents": 2, "seed": seed}
    )
    peaks: dict[int, int] = {}
    for _ in range(max_turns):
        if env.done:
            break
        env.step([agent(env.steps[-1][0]["observation"]), []])
        fleets = env.steps[-1][0]["observation"].get("fleets", [])
        for fleet in fleets:
            fid, owner, _x, _y, _angle, _from_pid, ships = fleet
            if owner != 0:
                continue
            ships_int = int(ships)
            if peaks.get(int(fid), 0) < ships_int:
                peaks[int(fid)] = ships_int
    return peaks


def _avg_peak_over_seeds(agent: AgentFn, seeds: list[int]) -> float:
    all_peaks: list[int] = []
    for seed in seeds:
        all_peaks.extend(_max_observed_ships_per_fleet(agent, seed=seed).values())
    if not all_peaks:
        return 0.0
    return sum(all_peaks) / len(all_peaks)


SEEDS = [0, 1, 2, 7]


@pytest.mark.integration
@pytest.mark.slow
def test_case4_average_fleet_size_larger_than_case2() -> None:
    v2_avg = _avg_peak_over_seeds(agent_v2, SEEDS)
    v4_avg = _avg_peak_over_seeds(agent_v4, SEEDS)

    assert v2_avg > 0 and v4_avg > 0
    assert v4_avg > v2_avg * 1.03, (
        f"case4 avg fleet ({v4_avg:.1f}) should exceed case2 avg ({v2_avg:.1f}) by >3%"
    )


@pytest.mark.integration
def test_case4_has_large_fleet_launches() -> None:
    """case4 should produce at least one fleet of >= 25 ships (FULL_COMMIT effect)."""
    peaks = _max_observed_ships_per_fleet(agent_v4, seed=0, max_turns=100)
    assert peaks, "case4 should launch at least some fleets"
    max_peak = max(peaks.values())
    assert max_peak >= 15, (
        f"case4 should occasionally launch consolidated fleets; max observed={max_peak}"
    )
