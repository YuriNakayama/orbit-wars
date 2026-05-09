from __future__ import annotations

from pipeline.rulebase.case6.baseline import agent


def test_agent_returns_valid_action_shape() -> None:
    """Unit-level sanity: feed a synthetic obs and check return shape."""
    obs = {
        "player": 0,
        "step": 20,
        "planets": [
            [0, 0, 20.0, 50.0, 3.0, 30, 2],
            [1, 1, 80.0, 50.0, 3.0, 30, 2],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
            [3, -1, 50.0, 70.0, 3.0, 5, 1],
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
    actions = agent(obs)
    assert isinstance(actions, list)
    for move in actions:
        assert isinstance(move, list) and len(move) == 3
        src_id, angle, ships = move
        assert isinstance(src_id, int)
        assert isinstance(angle, (int, float))
        assert isinstance(ships, int)
        assert ships >= 1
