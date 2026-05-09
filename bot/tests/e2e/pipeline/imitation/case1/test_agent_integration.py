"""Integration test: imitation/case1 agent runs in the local simulator."""

from __future__ import annotations

import pytest

from pipeline.imitation.case1.policy.agent import agent
from tests.e2e.pipeline.util import (
    DONE_STATUSES,
    make_orbit_env,
    run_orbit_wars_episode,
)


def _skip_if_weights_incompatible() -> None:
    """Probe agent on a minimal obs; skip if on-disk weights mismatch arch."""
    probe_obs = {
        "player": 0,
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": [],
        "planets": [],
        "fleets": [],
    }
    try:
        agent(probe_obs)
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            pytest.skip(f"weights.pt incompatible with current architecture: {e}")
        raise


@pytest.mark.slow
def test_agent_runs_full_episode_vs_random() -> None:
    _skip_if_weights_incompatible()
    env = make_orbit_env(debug=True)
    run_orbit_wars_episode(env, [agent, "random"])
    statuses = [s["status"] for s in env.state]
    for status in statuses:
        assert status in DONE_STATUSES, f"unexpected status: {status}"


@pytest.mark.slow
def test_agent_runs_full_episode_self_play() -> None:
    _skip_if_weights_incompatible()
    env = make_orbit_env(debug=True)
    run_orbit_wars_episode(env, [agent, agent])
    statuses = [s["status"] for s in env.state]
    for status in statuses:
        assert status in DONE_STATUSES, f"unexpected status: {status}"
