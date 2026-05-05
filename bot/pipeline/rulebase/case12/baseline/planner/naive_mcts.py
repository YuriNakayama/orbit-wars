"""Naïve MCTS / CMAB-based search for case12.

Algorithm (Ontañón 2017):
1. Treat each source planet's script choice as an independent arm
   (Combinatorial Multi-armed Bandit).
2. For `rollouts` iterations:
   - For each source, pick a script via UCB1 over per-source arm stats
   - Combined assignment is evaluated by `evaluate_assignment` (playout)
   - Backprop reward to each (src, script) arm
3. Final assignment: per source, pick the most-visited script.
4. Materialize moves.

Time budget: caller supplies `time_budget_s`; rollouts terminate early
when exceeded and best-so-far assignment is returned.
"""

from __future__ import annotations

import math
import random
import time

from ..core.world_model import WorldModel
from .evaluator import _moves_from_assignment, evaluate_assignment
from .scripts import SCRIPTS


def _ucb1_select(
    src_id: int,
    arm_stats: dict[tuple[int, str], tuple[int, float]],
    exploration_c: float,
    rng: random.Random,
) -> str:
    """Select a script for `src_id` via UCB1 over per-source arm stats.

    Unvisited arms are tried first (UCB1 standard tie-breaker). When all
    arms have been visited at least once, picks the one with highest
    `mean + c * sqrt(ln(N) / n)`.
    """
    script_names = [s.name for s in SCRIPTS]
    total_visits = sum(v for (s, _), (v, _) in arm_stats.items() if s == src_id)
    if total_visits == 0:
        return rng.choice(script_names)

    # If any arm is unvisited for this source, explore it.
    for name in script_names:
        v, _ = arm_stats.get((src_id, name), (0, 0.0))
        if v == 0:
            return name

    best, best_score = script_names[0], -math.inf
    log_total = math.log(total_visits)
    for name in script_names:
        v, r = arm_stats[(src_id, name)]
        avg = r / v
        ucb = avg + exploration_c * math.sqrt(log_total / v)
        if ucb > best_score:
            best, best_score = name, ucb
    return best


def run_naive_mcts(
    world: WorldModel,
    *,
    rollouts: int,
    horizon: int,
    exploration_c: float,
    time_budget_s: float,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.3),
    rng_seed: int | None = None,
) -> list[list[int | float]]:
    """Run NaïveMCTS and return moves materialized from the most-visited
    assignment per source. Returns empty list when no my_planets."""
    sources = list(world.my_planets)
    if not sources:
        return []

    rng = random.Random(rng_seed if rng_seed is not None else world.step)
    arm_stats: dict[tuple[int, str], tuple[int, float]] = {}
    deadline = time.perf_counter() + max(0.0, time_budget_s)

    best_assignment: dict[int, str] = {src.id: "idle" for src in sources}
    best_reward = evaluate_assignment(world, best_assignment, horizon, weights)

    completed = 0
    for _ in range(rollouts):
        if time.perf_counter() >= deadline:
            break

        # Sample assignment via per-source UCB1
        assignment: dict[int, str] = {}
        for src in sources:
            assignment[src.id] = _ucb1_select(src.id, arm_stats, exploration_c, rng)

        # Evaluate
        reward = evaluate_assignment(world, assignment, horizon, weights)

        # Backprop to all arms used in this rollout
        for src_id, script_name in assignment.items():
            v, r = arm_stats.get((src_id, script_name), (0, 0.0))
            arm_stats[(src_id, script_name)] = (v + 1, r + reward)

        if reward > best_reward:
            best_reward = reward
            best_assignment = dict(assignment)
        completed += 1

    # Final: per source, pick the most-visited script (Ontañón 2017's
    # recommendation; falls back to best_assignment if no arm visited).
    final: dict[int, str] = {}
    for src in sources:
        best_name, best_visits = "idle", 0
        for s in SCRIPTS:
            v, _ = arm_stats.get((src.id, s.name), (0, 0.0))
            if v > best_visits:
                best_name, best_visits = s.name, v
        # If this source had no rollout visits at all, fall back to the
        # best-reward assignment we found during sampling.
        if best_visits == 0:
            best_name = best_assignment.get(src.id, "idle")
        final[src.id] = best_name

    raw_moves = _moves_from_assignment(world, final)
    return [[int(m[0]), float(m[1]), int(m[2])] for m in raw_moves]
