"""Smoke tests for the case8 (baseline_v8) agent."""

from __future__ import annotations

import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from pipeline.rulebase.case8.baseline import agent  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
def test_agent_runs_1v1_to_done() -> None:
    """Full self-play episode — beam + true2p_light makes per-turn cost
    sensitive to pytest-cov tracing overhead, so this is `slow` and excluded
    from `dev/test-bot` (which runs `-m "not slow"`). Real rust-backed runs
    via `python -m dataset run` measure turn_p95 ≤ 0.7s as required."""
    env = kaggle_environments.make("orbit_wars", configuration={"agents": 2, "seed": 0})
    env.run([agent, agent])
    assert len(env.steps) > 0
    final = env.steps[-1]
    statuses = [s.get("status") for s in final]
    assert all(status in ("DONE", "INACTIVE") for status in statuses)


def test_agent_returns_valid_action_shape() -> None:
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
        assert isinstance(ships, int) and ships >= 1
