"""Unit tests for the iter2 planner evaluator (mission_score mode)."""

from __future__ import annotations

from pipeline.rulebase.case8.baseline.core.types import Fleet, Mission, Planet
from pipeline.rulebase.case8.baseline.core.world_model import WorldModel
from pipeline.rulebase.case8.baseline.planner.evaluator import (
    score_commitments,
    score_commitments_legacy,
)


def _make_world() -> WorldModel:
    planets = [
        Planet(0, 0, 20.0, 50.0, 3.0, 40, 2),
        Planet(1, 1, 80.0, 50.0, 3.0, 40, 2),
        Planet(2, -1, 50.0, 30.0, 3.0, 8, 1),
    ]
    fleets: list[Fleet] = []
    return WorldModel(
        player=0,
        step=10,
        planets=planets,
        fleets=fleets,
        initial_by_id={p.id: p for p in planets},
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
        predicted_arrivals={},
        opponent_threat_score={},
    )


def test_mission_score_sums_accepted_only() -> None:
    """`score_commitments` returns sum of accepted_missions' scores; rejected
    missions don't contribute."""
    world = _make_world()
    weights = (1.0, 0.5, 0.3)
    horizon = 3

    accepted = [
        Mission(kind="capture", score=12.0, target_id=2, turns=1, options=[]),
        Mission(kind="snipe", score=4.5, target_id=1, turns=2, options=[]),
    ]
    score = score_commitments(world, {}, accepted, horizon, weights)
    assert score == 16.5

    score_empty = score_commitments(world, {}, [], horizon, weights)
    assert score_empty == 0.0


def test_legacy_evaluator_capture_no_worse_than_empty() -> None:
    """Legacy net-ships evaluator should not regress on capturing a neutral."""
    world = _make_world()
    weights = (1.0, 0.5, 0.3)
    horizon = 3

    empty_score = score_commitments_legacy(world, {}, horizon, weights)
    commitments = {2: [(1, 0, 9)]}
    capture_score = score_commitments_legacy(world, commitments, horizon, weights)

    assert capture_score >= empty_score
