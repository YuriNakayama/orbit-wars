"""When NAIVE_MCTS_ENABLED=False, case12 must produce the same moves as case4."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case4.baseline import agent as agent_v4
from pipeline.rulebase.case12.baseline import agent as agent_v12
from pipeline.rulebase.case12.baseline.core import config as case12_config


@pytest.fixture(autouse=True)
def _disable_mcts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(case12_config, "NAIVE_MCTS_ENABLED", False)
    from pipeline.rulebase.case12.baseline import strategy

    monkeypatch.setattr(strategy, "NAIVE_MCTS_ENABLED", False)


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


def test_mcts_off_matches_case4() -> None:
    obs = _make_obs()
    moves_v4 = agent_v4(obs)
    moves_v12 = agent_v12(obs)
    assert moves_v12 == moves_v4
