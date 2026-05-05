"""Integration + snapshot tests for the case10 agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

kaggle_environments = pytest.importorskip("kaggle_environments")

from pipeline.rulebase.case10.baseline import agent  # noqa: E402

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_OBS = SNAPSHOT_DIR / "obs_seed0_turn10.json"
SNAPSHOT_ACTION = SNAPSHOT_DIR / "action_seed0_turn10.json"


@pytest.mark.integration
def test_agent_runs_1v1_to_done() -> None:
    env = kaggle_environments.make("orbit_wars", configuration={"agents": 2, "seed": 0})
    env.run([agent, agent])
    assert len(env.steps) > 0
    final = env.steps[-1]
    statuses = [s.get("status") for s in final]
    assert all(status in ("DONE", "INACTIVE") for status in statuses)


@pytest.mark.integration
def test_agent_snapshot_first_turn_action() -> None:
    """Agent is deterministic given a fixed obs — verify the captured action."""
    if not SNAPSHOT_OBS.exists() or not SNAPSHOT_ACTION.exists():
        pytest.skip(
            "snapshot files not found — run "
            "`uv run python -m pipeline.rulebase.case10.evaluation"
            ".snapshot_update` first"
        )
    obs = json.loads(SNAPSHOT_OBS.read_text(encoding="utf-8"))
    expected = json.loads(SNAPSHOT_ACTION.read_text(encoding="utf-8"))
    actual = agent(obs)
    assert actual == expected


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
