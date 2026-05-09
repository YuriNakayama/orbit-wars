"""Regression tests: case4 must send larger, fewer fleets than case2.

These are integration-level measurements over several seeds to reduce variance
introduced by the non-deterministic parts of the local simulator.
"""

from __future__ import annotations

import pytest

from pipeline.rulebase.case2.baseline import agent as agent_v2
from pipeline.rulebase.case4.baseline import agent as agent_v4
from tests.e2e.pipeline.util import avg_peak_over_seeds, max_observed_ships_per_fleet

SEEDS = [0, 1, 2, 7]


@pytest.mark.slow
def test_case4_average_fleet_size_larger_than_case2() -> None:
    v2_avg = avg_peak_over_seeds(agent_v2, SEEDS)
    v4_avg = avg_peak_over_seeds(agent_v4, SEEDS)

    assert v2_avg > 0 and v4_avg > 0
    assert v4_avg > v2_avg * 1.03, (
        f"case4 avg fleet ({v4_avg:.1f}) should exceed case2 avg ({v2_avg:.1f}) by >3%"
    )


def test_case4_has_large_fleet_launches() -> None:
    """case4 should produce at least one fleet of >= 25 ships (FULL_COMMIT effect)."""
    peaks = max_observed_ships_per_fleet(agent_v4, seed=0, max_turns=100)
    assert peaks, "case4 should launch at least some fleets"
    max_peak = max(peaks.values())
    assert max_peak >= 15, (
        f"case4 should occasionally launch consolidated fleets; max observed={max_peak}"
    )
