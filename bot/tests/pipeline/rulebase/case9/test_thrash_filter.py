"""Unit tests for the case9 planet thrash filter (case8 iter3 v1 ported)."""

from __future__ import annotations

import pytest

from pipeline.rulebase.case9.baseline.core.types import Fleet, Planet
from pipeline.rulebase.case9.baseline.core.world_model import WorldModel
from pipeline.rulebase.case9.baseline.strategy_helpers import apply_score_modifiers


def _make_world(
    *,
    recently_lost: dict[int, int] | None = None,
    mission_commits: dict[int, list[int]] | None = None,
    step: int = 50,
) -> WorldModel:
    planets = [
        Planet(0, 0, 20.0, 50.0, 3.0, 40, 2),
        Planet(1, 1, 80.0, 50.0, 3.0, 40, 2),
        Planet(2, -1, 50.0, 30.0, 3.0, 8, 1),  # capture target
    ]
    fleets: list[Fleet] = []
    return WorldModel(
        player=0,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id={p.id: p for p in planets},
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
        predicted_arrivals={},
        opponent_threat_score={},
        recently_lost=recently_lost or {},
        mission_commits=mission_commits or {},
    )


@pytest.fixture(autouse=True)
def _enable_thrash(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.rulebase.case9.baseline import strategy_helpers

    monkeypatch.setattr(strategy_helpers, "THRASH_FILTER_ENABLED", True)
    monkeypatch.setattr(strategy_helpers, "THRASH_WINDOW", 10)
    monkeypatch.setattr(strategy_helpers, "THRASH_SCORE_MULT", 0.3)
    monkeypatch.setattr(strategy_helpers, "THRASH_REPEAT_COMMIT_LIMIT", 999)


def test_recently_lost_within_window_decays_capture_score() -> None:
    world_clean = _make_world()
    world_thrash = _make_world(recently_lost={2: 47})  # step=50, lost at 47

    target = world_clean.planet_by_id[2]
    base = 100.0
    score_clean = apply_score_modifiers(base, target, "capture", world_clean)
    score_thrash = apply_score_modifiers(base, target, "capture", world_thrash)

    assert score_thrash == pytest.approx(score_clean * 0.3)


def test_recently_lost_outside_window_does_not_decay() -> None:
    world = _make_world(recently_lost={2: 30}, step=50)  # 20 turns old

    target = world.planet_by_id[2]
    base = 100.0
    score_clean = apply_score_modifiers(base, target, "capture", _make_world(step=50))
    score = apply_score_modifiers(base, target, "capture", world)

    assert score == pytest.approx(score_clean)


def test_mission_commits_path_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With THRASH_REPEAT_COMMIT_LIMIT=999 (case9 default), commits never trigger
    the filter on their own — only recently_lost does."""
    from pipeline.rulebase.case9.baseline import strategy_helpers

    monkeypatch.setattr(strategy_helpers, "THRASH_REPEAT_COMMIT_LIMIT", 999)
    world = _make_world(mission_commits={2: [44, 47, 48, 49]}, step=50)

    target = world.planet_by_id[2]
    base = 100.0
    score_clean = apply_score_modifiers(base, target, "capture", _make_world(step=50))
    score = apply_score_modifiers(base, target, "capture", world)

    assert score == pytest.approx(score_clean)


def test_filter_does_not_apply_to_reinforce_or_harass() -> None:
    world = _make_world(recently_lost={2: 47})

    target = world.planet_by_id[2]
    base = 100.0
    for non_attack in ("reinforce", "harass"):
        score_clean = apply_score_modifiers(base, target, non_attack, _make_world())
        score_thrash = apply_score_modifiers(base, target, non_attack, world)
        assert score_thrash == pytest.approx(score_clean), (
            f"{non_attack} should not be affected by thrash filter"
        )


def test_filter_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.rulebase.case9.baseline import strategy_helpers

    monkeypatch.setattr(strategy_helpers, "THRASH_FILTER_ENABLED", False)
    world = _make_world(recently_lost={2: 47}, step=50)

    target = world.planet_by_id[2]
    base = 100.0
    score_clean = apply_score_modifiers(base, target, "capture", _make_world())
    score = apply_score_modifiers(base, target, "capture", world)

    assert score == pytest.approx(score_clean)
