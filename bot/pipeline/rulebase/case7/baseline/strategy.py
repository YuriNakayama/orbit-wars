# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Top-level planning: plan_moves orchestrates missions and emits kaggle actions.

The planner is split into a thin orchestrator (`plan_moves`) that drives an
inner per-mission processor. `_process_single_source_mission` handles
"single", "snipe", "reinforce", "crash_exploit", "harass", "accumulate_fire"
missions. `_process_multi_source_mission` handles the multi-source attack
family.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .core.config import REINFORCE_SAFETY_MARGIN
from .core.types import Mission, Planet
from .core.world_model import WorldModel
from .missions import collect_missions
from .missions.stay import build_accumulate, build_stay_holds
from .movements.evacuation import emit_evacuation_moves
from .movements.followup import emit_followup_moves
from .movements.rear_guard import emit_rear_guard_moves
from .strategy_helpers import build_modes, preferred_send

SINGLE_SOURCE_MISSION_KINDS: frozenset[str] = frozenset(
    {"single", "snipe", "reinforce", "crash_exploit", "harass", "accumulate_fire"}
)


def _process_single_source_mission(
    mission: Mission,
    target: Planet,
    world: WorldModel,
    modes: dict[str, Any],
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    source_inventory_left: Callable[[int], int],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    """Resolve a single-source mission and append the resulting move (if any)."""
    option = mission.options[0]

    left = (
        source_inventory_left(option.src_id)
        if mission.kind == "reinforce"
        else source_attack_left(option.src_id)
    )
    if left <= 0:
        return

    arrival_turn = option.turns

    if mission.kind == "reinforce":
        missing = world.reinforcement_needed_for(
            option.target_id, arrival_turn, planned_commitments
        )
    else:
        missing = world.ships_needed_to_capture(
            target.id, arrival_turn, planned_commitments
        )
    if missing <= 0:
        return

    send_limit = min(left, option.send_cap)
    if send_limit < missing:
        return

    if mission.kind in ("snipe", "crash_exploit", "harass"):
        send = missing
    elif mission.kind == "accumulate_fire":
        send = min(send_limit, max(missing, option.send_cap))
    elif mission.kind == "reinforce":
        send = min(send_limit, missing + REINFORCE_SAFETY_MARGIN)
    else:
        send = min(
            send_limit,
            max(
                missing,
                preferred_send(
                    target,
                    missing,
                    arrival_turn,
                    send_limit,
                    world,
                    modes,
                ),
            ),
        )
    if send < missing:
        return

    sent = append_move(option.src_id, option.angle, send)
    if sent < missing:
        return
    planned_commitments[target.id].append((arrival_turn, world.player, int(sent)))


def _process_multi_source_mission(
    mission: Mission,
    target: Planet,
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    """Resolve a multi-source mission and append per-option moves (if any)."""
    limits: list[int] = []
    for option in mission.options:
        left = source_attack_left(option.src_id)
        limits.append(min(left, option.send_cap))
    if min(limits) <= 0:
        return

    missing = world.ships_needed_to_capture(
        target.id, mission.turns, planned_commitments
    )
    if missing <= 0:
        return
    if sum(limits) < missing:
        return

    ordered = sorted(
        zip(mission.options, limits, strict=True),
        key=lambda item: (-item[1], item[0].turns, item[0].src_id),
    )
    remaining = missing
    sends: dict[int, int] = {}
    for option, limit in ordered:
        send = min(limit, remaining)
        sends[option.src_id] = send
        remaining -= send
    if remaining > 0:
        return

    committed: list[tuple[int, int, int]] = []
    for option, _ in ordered:
        send = sends.get(option.src_id, 0)
        if send <= 0:
            continue
        actual = append_move(option.src_id, option.angle, send)
        if actual <= 0:
            continue
        committed.append((option.turns, world.player, int(actual)))
    if sum(item[2] for item in committed) < missing:
        return
    planned_commitments[target.id].extend(committed)


def _enforce_inventory_cap(
    moves: list[list[int | float]], world: WorldModel
) -> list[list[int | float]]:
    """Final pass: clamp per-source spend to actual ships available."""
    final_moves: list[list[int | float]] = []
    used_final: dict[int, int] = defaultdict(int)
    for src_id, angle, ships in moves:
        source = world.planet_by_id[int(src_id)]
        max_allowed = int(source.ships) - used_final[int(src_id)]
        send = min(int(ships), max_allowed)
        if send >= 1:
            final_moves.append([int(src_id), float(angle), int(send)])
            used_final[int(src_id)] += send
    return final_moves


def plan_moves(
    world: WorldModel,
    consecutive_holds: dict[int, int] | None = None,
    accumulate_holds_consec: dict[int, int] | None = None,
) -> tuple[list[list[int | float]], set[int], set[int]]:
    """Plan one turn of moves.

    Returns (moves, burst_held_src_ids, accumulate_held_src_ids). The second
    element contains source planet ids that were burst-held (case6 STAY_BURST,
    1-turn arbitrage) this turn; the third contains source ids that were
    accumulate-held (case7 multi-turn accumulate). Callers track them in
    independent counters for STAY_BURST_MAX_HOLD_TURNS / ACCUMULATE_MAX_HOLD_TURNS.
    """
    modes = build_modes(world)
    planned_commitments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    moves: list[list[int | float]] = []
    spent_total: dict[int, int] = defaultdict(int)

    stay_holds, stay_decisions = build_stay_holds(world, consecutive_holds)
    burst_held_src_ids: set[int] = {
        d.src_id for d in stay_decisions if d.kind == "burst"
    }

    accumulate_holds, accumulate_fire_missions, _, accumulate_held_src_ids = (
        build_accumulate(world, planned_commitments, accumulate_holds_consec)
    )

    merged_holds: dict[int, int] = dict(stay_holds)
    for src_id, ships in accumulate_holds.items():
        merged_holds[src_id] = max(merged_holds.get(src_id, 0), ships)

    def source_inventory_left(source_id: int) -> int:
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id: int) -> int:
        raw = world.source_attack_left(source_id, spent_total)
        held = merged_holds.get(source_id, 0)
        return max(0, raw - held)

    def append_move(src_id: int, angle: float, ships: int) -> int:
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    missions = collect_missions(
        world,
        planned_commitments,
        modes,
        source_inventory_left,
        source_attack_left,
    )
    missions.extend(accumulate_fire_missions)
    missions.sort(key=lambda item: -item.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in SINGLE_SOURCE_MISSION_KINDS:
            _process_single_source_mission(
                mission,
                target,
                world,
                modes,
                planned_commitments,
                source_inventory_left,
                source_attack_left,
                append_move,
            )
        else:
            _process_multi_source_mission(
                mission,
                target,
                world,
                planned_commitments,
                source_attack_left,
                append_move,
            )

    emit_followup_moves(
        world, planned_commitments, modes, source_attack_left, append_move
    )
    emit_evacuation_moves(
        world, planned_commitments, modes, source_inventory_left, append_move
    )
    emit_rear_guard_moves(world, modes, source_attack_left, append_move)

    return (
        _enforce_inventory_cap(moves, world),
        burst_held_src_ids,
        accumulate_held_src_ids,
    )
