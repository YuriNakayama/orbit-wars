"""Candidate plan execution: run the existing per-mission commit logic with
an externally supplied mission ordering, returning the resulting moves and
planned_commitments WITHOUT applying followup/evacuation/rear_guard movements.

This lets the beam search compare different orderings on equal footing —
the movement emit is applied once at the end on the winning ordering.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Mission
from ..core.world_model import WorldModel


@dataclass
class MovesPlan:
    moves: list[list[int | float]] = field(default_factory=list)
    planned_commitments: dict[int, list[tuple[int, int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    spent_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    accepted_missions: list[Mission] = field(default_factory=list)


def commit_missions_in_order(
    world: WorldModel,
    ordered_missions: list[Mission],
    modes: dict[str, Any],
    merged_holds: dict[int, int],
    process_single: Callable[..., None],
    process_multi: Callable[..., None],
    single_source_kinds: frozenset[str],
) -> MovesPlan:
    """Walk `ordered_missions` through the per-mission commit primitives.

    `process_single` / `process_multi` are the existing
    `_process_single_source_mission` / `_process_multi_source_mission`
    closures from `strategy.py`, but bound to fresh per-plan state via the
    closures created below. This isolates each plan's bookkeeping so the beam
    search can evaluate orderings independently.
    """
    plan = MovesPlan()

    def source_inventory_left(source_id: int) -> int:
        return world.source_inventory_left(source_id, plan.spent_total)

    def source_attack_left(source_id: int) -> int:
        raw = world.source_attack_left(source_id, plan.spent_total)
        held = merged_holds.get(source_id, 0)
        return max(0, raw - held)

    def append_move(src_id: int, angle: float, ships: int) -> int:
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        plan.moves.append([src_id, float(angle), int(send)])
        plan.spent_total[src_id] += send
        return send

    for mission in ordered_missions:
        target = world.planet_by_id[mission.target_id]
        moves_before = len(plan.moves)
        if mission.kind in single_source_kinds:
            process_single(
                mission,
                target,
                world,
                modes,
                plan.planned_commitments,
                source_inventory_left,
                source_attack_left,
                append_move,
            )
        else:
            process_multi(
                mission,
                target,
                world,
                plan.planned_commitments,
                source_attack_left,
                append_move,
            )
        if len(plan.moves) > moves_before:
            plan.accepted_missions.append(mission)

    return plan
