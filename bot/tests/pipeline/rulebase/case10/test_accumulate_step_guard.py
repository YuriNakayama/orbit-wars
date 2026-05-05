"""Unit tests for the case10 ACCUMULATE_MIN_LAUNCH_STEP guard.

Verifies that `_build_accumulate` returns empty when world.step < threshold,
which kills the t14 trap from project_case7_t14_trap.
"""

from __future__ import annotations

import pytest

from pipeline.rulebase.case10.baseline.core import config as cfg
from pipeline.rulebase.case10.baseline.missions.stay import build_accumulate

# Reuse the fixture helpers from the regression test.
from tests.pipeline.rulebase.case10.test_accumulate import _make_obs, _world


@pytest.fixture(autouse=True)
def _enable_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "ACCUMULATE_ENABLED", True)
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_LAUNCH_STEP", 30)


def test_step_below_threshold_returns_empty() -> None:
    """At step=14 (the canonical t14 trap fingerprint), accumulate must NOT fire."""
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 60, 2)],  # 60 ships = at KNEE threshold
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
        step=14,
    )
    world = _world(obs)
    holds, fire_missions, _decisions, held_ids = build_accumulate(world, {})
    assert holds == {}
    assert fire_missions == []
    assert held_ids == set()


def test_step_at_threshold_runs_normally() -> None:
    """At step=30 (the threshold), accumulate is allowed to evaluate sources.

    This does not assert it fires (depends on target geometry), only that the
    guard does not short-circuit. Compare against step=14 above which always
    returns empty regardless of board state.
    """
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 60, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
        step=30,
    )
    world = _world(obs)
    holds, fire_missions, decisions, _ = build_accumulate(world, {})
    # Either fired or held something OR a decision was logged.
    # The exact branch depends on geometry; we only assert the guard path
    # didn't return immediately by checking that >=1 decision was recorded
    # OR holds/fires non-empty.
    assert len(decisions) >= 1 or holds != {} or fire_missions != [], (
        "step=30 should reach the accumulate evaluation loop"
    )


def test_guard_disabled_with_zero_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACCUMULATE_MIN_LAUNCH_STEP=0 makes the guard a noop (case7 equivalent)."""
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_LAUNCH_STEP", 0)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 60, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
        step=14,
    )
    world = _world(obs)
    _holds, _fire, decisions, _held = build_accumulate(world, {})
    # With guard disabled the evaluation loop runs; at minimum a decision
    # is logged for the source.
    assert len(decisions) >= 1
