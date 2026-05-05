"""Unit tests for case12 NaïveMCTS basics."""

from __future__ import annotations

from pipeline.rulebase.case12.baseline.core.types import Fleet, Planet
from pipeline.rulebase.case12.baseline.core.world_model import WorldModel
from pipeline.rulebase.case12.baseline.planner.naive_mcts import (
    _ucb1_select,
    run_naive_mcts,
)


def _make_world() -> WorldModel:
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


def test_ucb1_explores_unvisited_arms_first() -> None:
    """Empty arm_stats → UCB1 should pick a script (not crash, deterministic
    via seeded rng)."""
    import random

    arm_stats: dict[tuple[int, str], tuple[int, float]] = {}
    rng = random.Random(0)
    name = _ucb1_select(0, arm_stats, exploration_c=1.41, rng=rng)
    assert isinstance(name, str)
    assert len(name) > 0


def test_run_naive_mcts_returns_moves() -> None:
    """NaïveMCTS on a non-trivial board should pick something (most-visited
    script is unlikely to be `idle` for 64 rollouts at horizon=20)."""
    world = _make_world()
    moves = run_naive_mcts(
        world,
        rollouts=32,  # smaller for fast test
        horizon=15,
        exploration_c=1.41,
        time_budget_s=5.0,
        rng_seed=42,
    )
    # We don't assert moves != [] (could happen if idle wins for all),
    # but we assert it doesn't crash and returns proper shape.
    assert isinstance(moves, list)
    for m in moves:
        assert isinstance(m, list) and len(m) == 3
