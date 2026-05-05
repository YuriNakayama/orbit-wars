"""Even with an extremely tight time budget, the beam returns valid moves
(early-stop using best-so-far seeded from the greedy ordering)."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case8.baseline import agent as agent_v8
from pipeline.rulebase.case8.baseline.core import config as case8_config


@pytest.fixture(autouse=True)
def _tight_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget so small that beam expansion will exit immediately after seeding."""
    from pipeline.rulebase.case8.baseline import strategy

    monkeypatch.setattr(case8_config, "BEAM_ENABLED", True)
    monkeypatch.setattr(case8_config, "BEAM_TIME_BUDGET_S", 0.001)
    monkeypatch.setattr(strategy, "BEAM_ENABLED", True)
    monkeypatch.setattr(strategy, "BEAM_TIME_BUDGET_S", 0.001)


def _make_obs() -> dict[str, Any]:
    return {
        "player": 0,
        "step": 30,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 40, 2],
            [1, 1, 80.0, 50.0, 3.0, 40, 2],
            [2, -1, 50.0, 30.0, 3.0, 8, 1],
            [3, -1, 50.0, 70.0, 3.0, 8, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [
            [0, 0, 20.0, 50.0, 3.0, 10, 2],
            [1, 1, 80.0, 50.0, 3.0, 10, 2],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
            [3, -1, 50.0, 70.0, 3.0, 5, 1],
        ],
        "comets": [],
        "comet_planet_ids": [],
    }


def test_tight_budget_returns_valid_moves() -> None:
    moves = agent_v8(_make_obs())
    assert isinstance(moves, list)
    for move in moves:
        assert isinstance(move, list) and len(move) == 3
        src_id, angle, ships = move
        assert isinstance(src_id, int)
        assert isinstance(angle, (int, float))
        assert isinstance(ships, int) and ships >= 1
