"""Unit test: PGS prefers a productive script over idle when capture is feasible."""

from __future__ import annotations

from pipeline.rulebase.case11.baseline.core.types import Fleet, Planet
from pipeline.rulebase.case11.baseline.core.world_model import WorldModel
from pipeline.rulebase.case11.baseline.planner.evaluator import evaluate_assignment
from pipeline.rulebase.case11.baseline.planner.portfolio_greedy import run_pgs


def _make_world() -> WorldModel:
    """Tiny scenario: self has 1 home (40 ships) near a neutral with 5 ships."""
    planets = [
        Planet(0, 0, 20.0, 50.0, 3.0, 40, 2),  # self home
        Planet(1, 1, 80.0, 50.0, 3.0, 40, 2),  # enemy home
        Planet(2, -1, 35.0, 50.0, 3.0, 5, 1),  # near neutral
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


def test_pgs_prefers_capture_over_idle() -> None:
    world = _make_world()
    moves = run_pgs(world, horizon=8, iters=3, time_budget_s=1.0)
    # PGS should pick capture_safe or capture_aggressive (returning at least
    # one move) rather than leave everyone idle.
    assert len(moves) >= 1
    src_id, _angle, ships = moves[0]
    assert int(src_id) == 0  # only self planet
    assert int(ships) >= 1


def test_evaluate_assignment_rewards_capture() -> None:
    world = _make_world()
    score_idle = evaluate_assignment(world, {0: "idle"}, horizon=8)
    score_capture = evaluate_assignment(world, {0: "capture_safe"}, horizon=8)
    # Capturing the neutral should yield a higher horizon-end score.
    assert score_capture >= score_idle
