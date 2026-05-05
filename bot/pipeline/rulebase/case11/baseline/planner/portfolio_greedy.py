"""Portfolio Greedy Search (Churchill & Buro 2013) for case11.

Algorithm:
1. Initial assignment: every source gets `script_idle`.
2. Score the initial assignment with `evaluate_assignment`.
3. Hill climb: for each source, try each script; pick the one that
   maximizes the score; commit the swap and continue. Repeat for at most
   `iters` rounds or until no swap improves the score.
4. Return the moves materialized from the final assignment.

Time budget: caller supplies `time_budget_s`; we abort the climb early if
exceeded and return the best-so-far.
"""

from __future__ import annotations

import time

from ..core.world_model import WorldModel
from .evaluator import _moves_from_assignment, evaluate_assignment
from .scripts import SCRIPTS


def run_pgs(
    world: WorldModel,
    *,
    horizon: int,
    iters: int,
    time_budget_s: float,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.3),
) -> list[list[int | float]]:
    """Run PGS and return the moves for the best assignment found."""
    sources = list(world.my_planets)
    if not sources:
        return []

    deadline = time.perf_counter() + max(0.0, time_budget_s)
    script_names = [s.name for s in SCRIPTS]

    # Initial assignment: all idle.
    current: dict[int, str] = {src.id: "idle" for src in sources}
    current_score = evaluate_assignment(world, current, horizon, weights)
    best_assignment = dict(current)
    best_score = current_score

    for _ in range(iters):
        if time.perf_counter() >= deadline:
            break

        improved = False
        for src in sources:
            if time.perf_counter() >= deadline:
                break
            best_local_script = current[src.id]
            best_local_score = current_score
            for script_name in script_names:
                if script_name == current[src.id]:
                    continue
                trial = dict(current)
                trial[src.id] = script_name
                score = evaluate_assignment(world, trial, horizon, weights)
                if score > best_local_score:
                    best_local_script = script_name
                    best_local_score = score
            if best_local_script != current[src.id]:
                current[src.id] = best_local_script
                current_score = best_local_score
                improved = True
                if best_local_score > best_score:
                    best_score = best_local_score
                    best_assignment = dict(current)
        if not improved:
            break

    # Materialize moves from best_assignment.
    raw_moves = _moves_from_assignment(world, best_assignment)
    return [[int(m[0]), float(m[1]), int(m[2])] for m in raw_moves]
