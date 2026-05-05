"""When PORTFOLIO_ENABLED=False, case11 must produce the same moves as case4."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case4.baseline import agent as agent_v4
from pipeline.rulebase.case11.baseline import agent as agent_v11
from pipeline.rulebase.case11.baseline.core import config as case11_config


@pytest.fixture(autouse=True)
def _disable_portfolio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the PGS path off so case11 walks the case4 mission/movement
    pipeline. Patch both config and the strategy module's binding."""
    monkeypatch.setattr(case11_config, "PORTFOLIO_ENABLED", False)
    from pipeline.rulebase.case11.baseline import strategy

    monkeypatch.setattr(strategy, "PORTFOLIO_ENABLED", False)


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


def test_portfolio_off_matches_case4() -> None:
    obs = _make_obs()
    moves_v4 = agent_v4(obs)
    moves_v11 = agent_v11(obs)
    assert moves_v11 == moves_v4
