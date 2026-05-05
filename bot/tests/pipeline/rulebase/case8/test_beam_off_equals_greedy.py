"""When BEAM_ENABLED=False AND THRASH_FILTER_ENABLED=False, case8 must produce
the same moves as case7 (greedy + no score modifications)."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case7.baseline import agent as agent_v7
from pipeline.rulebase.case8.baseline import agent as agent_v8
from pipeline.rulebase.case8.baseline.core import config as case8_config


@pytest.fixture(autouse=True)
def _disable_beam_and_thrash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force beam path off AND thrash filter off so case8 walks the
    pure greedy code path that matches case7."""
    monkeypatch.setattr(case8_config, "BEAM_ENABLED", False)
    monkeypatch.setattr(case8_config, "THRASH_FILTER_ENABLED", False)
    # Both flags are imported at module-load by their consumers; patch there too.
    from pipeline.rulebase.case8.baseline import strategy, strategy_helpers

    monkeypatch.setattr(strategy, "BEAM_ENABLED", False)
    monkeypatch.setattr(strategy_helpers, "THRASH_FILTER_ENABLED", False)


def _make_obs() -> dict[str, Any]:
    return {
        "player": 0,
        "step": 30,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 40, 2],
            [1, 1, 80.0, 50.0, 3.0, 40, 2],
            [2, -1, 50.0, 30.0, 3.0, 8, 1],
            [3, -1, 50.0, 70.0, 3.0, 8, 1],
            [4, 0, 30.0, 60.0, 3.0, 12, 1],
            [5, 1, 70.0, 40.0, 3.0, 12, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [
            [0, 0, 20.0, 50.0, 3.0, 10, 2],
            [1, 1, 80.0, 50.0, 3.0, 10, 2],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
            [3, -1, 50.0, 70.0, 3.0, 5, 1],
            [4, -1, 30.0, 60.0, 3.0, 5, 1],
            [5, -1, 70.0, 40.0, 3.0, 5, 1],
        ],
        "comets": [],
        "comet_planet_ids": [],
    }


def test_beam_off_matches_case7_greedy() -> None:
    obs = _make_obs()
    moves_v7 = agent_v7(obs)
    moves_v8 = agent_v8(obs)
    assert moves_v8 == moves_v7
