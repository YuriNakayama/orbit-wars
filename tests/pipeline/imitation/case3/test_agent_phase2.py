"""Episode-boundary tests for pipeline.imitation.case3.policy.agent_phase2."""

from __future__ import annotations

from pipeline.imitation.case3.policy import agent_phase2
from pipeline.imitation.case3.policy.featurizer_phase2 import LaunchEvent


def test_history_resets_on_step_zero() -> None:
    agent_phase2._HISTORY.recent_launches.append(
        LaunchEvent(step=42, owner=1, ships=12)
    )
    agent_phase2._HISTORY.last_step = 42
    obs = {"step": 0}
    agent_phase2._maybe_reset_history(obs)
    assert len(agent_phase2._HISTORY.recent_launches) == 0
    assert agent_phase2._HISTORY.last_step is None


def test_history_resets_on_step_regression() -> None:
    agent_phase2._HISTORY.last_step = 50
    agent_phase2._HISTORY.recent_launches.append(LaunchEvent(step=50, owner=1, ships=5))
    obs = {"step": 10}
    agent_phase2._maybe_reset_history(obs)
    assert len(agent_phase2._HISTORY.recent_launches) == 0


def test_history_does_not_reset_on_normal_progression() -> None:
    agent_phase2._HISTORY.clear()
    agent_phase2._HISTORY.last_step = 10
    agent_phase2._HISTORY.recent_launches.append(LaunchEvent(step=10, owner=1, ships=8))
    obs = {"step": 11}
    agent_phase2._maybe_reset_history(obs)
    # No reset; events preserved.
    assert len(agent_phase2._HISTORY.recent_launches) == 1
