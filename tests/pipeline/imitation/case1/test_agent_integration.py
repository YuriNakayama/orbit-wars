"""Integration test: imitation/case1 agent runs in kaggle_environments."""

from __future__ import annotations

import pytest
from kaggle_environments import make

from pipeline.imitation.case1.policy.agent import agent


@pytest.mark.slow
def test_agent_runs_full_episode_vs_random() -> None:
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    statuses = [s["status"] for s in env.state]
    for status in statuses:
        assert status in {"DONE", "INACTIVE"}, f"unexpected status: {status}"


@pytest.mark.slow
def test_agent_runs_full_episode_self_play() -> None:
    env = make("orbit_wars", debug=True)
    env.run([agent, agent])
    statuses = [s["status"] for s in env.state]
    for status in statuses:
        assert status in {"DONE", "INACTIVE"}, f"unexpected status: {status}"
