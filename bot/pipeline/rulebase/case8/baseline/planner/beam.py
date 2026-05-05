"""Beam search over mission orderings.

Search formulation:
- Start: empty ordering, with the top-K (K=BEAM_BRANCH_LIMIT) score-sorted
  missions as the candidate pool.
- Expand: at each depth, each beam member tries each remaining candidate
  mission as the next commit (1 branch per remaining candidate). The resulting
  partial ordering is materialized via `commit_missions_in_order` and scored
  by `score_commitments` over `BEAM_HORIZON` turns.
- Prune: keep top BEAM_WIDTH partial orderings.
- Stop: when no member can be extended (depth == K) or when wall-clock exceeds
  `BEAM_TIME_BUDGET_S`. Return the best ordering's MovesPlan.

Greedy fallback (caller side): when `BEAM_ENABLED=False` the caller skips this
entirely and uses the score-desc ordering directly. We also use the score-desc
ordering as the seed beam member so the beam is guaranteed to do no worse than
greedy in expectation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..core.types import Mission
from ..core.world_model import WorldModel
from .candidate import MovesPlan, commit_missions_in_order
from .evaluator import score_commitments, score_commitments_legacy


def _select_evaluator(
    mode: str,
) -> Callable[
    [
        WorldModel,
        dict[int, list[tuple[int, int, int]]],
        list[Mission],
        int,
        tuple[float, float, float],
    ],
    float,
]:
    if mode == "mission_score":
        return score_commitments
    if mode == "legacy_net_ships":
        # Wrap legacy signature so the caller path is uniform.
        def _legacy(
            world: WorldModel,
            commits: dict[int, list[tuple[int, int, int]]],
            _missions: list[Mission],
            horizon: int,
            weights: tuple[float, float, float],
        ) -> float:
            return score_commitments_legacy(world, commits, horizon, weights)

        return _legacy
    raise ValueError(f"unknown BEAM_EVALUATOR_MODE: {mode!r}")


def run_beam(
    world: WorldModel,
    missions: list[Mission],
    modes: dict[str, Any],
    merged_holds: dict[int, int],
    process_single: Callable[..., None],
    process_multi: Callable[..., None],
    single_source_kinds: frozenset[str],
    *,
    horizon: int,
    width: int,
    branch_limit: int,
    time_budget_s: float,
    weights: tuple[float, float, float],
    evaluator_mode: str = "mission_score",
) -> MovesPlan:
    """Run beam search and return the best MovesPlan.

    `missions` should already be score-desc sorted (caller responsibility);
    we slice the top `branch_limit` as the candidate pool.
    """
    if not missions:
        return commit_missions_in_order(
            world,
            [],
            modes,
            merged_holds,
            process_single,
            process_multi,
            single_source_kinds,
        )

    evaluate = _select_evaluator(evaluator_mode)

    pool = missions[: max(1, branch_limit)]
    pool_size = len(pool)
    deadline = time.perf_counter() + max(0.0, time_budget_s)

    # Beam member: (ordering as tuple of pool indices, score, MovesPlan).
    # Seed with the score-desc full ordering (greedy) so beam can't underperform.
    greedy_plan = commit_missions_in_order(
        world,
        list(pool),
        modes,
        merged_holds,
        process_single,
        process_multi,
        single_source_kinds,
    )
    greedy_score = evaluate(
        world,
        greedy_plan.planned_commitments,
        greedy_plan.accepted_missions,
        horizon,
        weights,
    )
    best_score = greedy_score
    best_plan = greedy_plan

    # Beam starts empty; at depth 0 we expand into single-mission orderings.
    beam: list[tuple[tuple[int, ...], float, MovesPlan]] = [
        ((), greedy_score, greedy_plan)
    ]

    for _depth in range(pool_size):
        if time.perf_counter() >= deadline:
            break

        next_beam: list[tuple[tuple[int, ...], float, MovesPlan]] = []
        for ordering, _score, _plan in beam:
            used = set(ordering)
            for idx in range(pool_size):
                if idx in used:
                    continue
                if time.perf_counter() >= deadline:
                    break
                new_order = ordering + (idx,)
                ordered = [pool[i] for i in new_order]
                plan = commit_missions_in_order(
                    world,
                    ordered,
                    modes,
                    merged_holds,
                    process_single,
                    process_multi,
                    single_source_kinds,
                )
                cand_score = evaluate(
                    world,
                    plan.planned_commitments,
                    plan.accepted_missions,
                    horizon,
                    weights,
                )
                next_beam.append((new_order, cand_score, plan))
                if cand_score > best_score:
                    best_score = cand_score
                    best_plan = plan

        if not next_beam:
            break
        next_beam.sort(key=lambda item: -item[1])
        beam = next_beam[: max(1, width)]

    return best_plan
