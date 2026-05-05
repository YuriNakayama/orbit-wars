"""Followup attack: use leftover ships for opportunistic captures post-dispatch."""

from __future__ import annotations

from typing import Any, Callable

from ..core.config import (
    ATTACK_COST_TURN_WEIGHT,
    FOLLOWUP_MIN_SHIPS,
    LATE_CAPTURE_BUFFER,
    LOW_VALUE_COMET_PRODUCTION,
    PARTIAL_SOURCE_MIN_SHIPS,
)
from ..core.types import Planet
from ..core.world_model import WorldModel
from ..strategy_helpers import (
    opening_filter,
    preferred_send,
    score_attack,
)


def emit_followup_moves(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    if world.is_very_late:
        return

    for src in world.my_planets:
        src_left = source_attack_left(src.id)
        if src_left < FOLLOWUP_MIN_SHIPS:
            continue

        best: tuple[float, Planet, int] | None = None
        for target in world.planets:
            if target.id == src.id or target.owner == world.player:
                continue
            if (
                target.id in world.comet_ids
                and target.production <= LOW_VALUE_COMET_PRODUCTION
            ):
                continue

            rough_ships = max(
                1,
                min(
                    src_left,
                    max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1),
                ),
            )
            rough_aim = world.plan_shot(src.id, target.id, rough_ships)
            if rough_aim is None:
                continue

            est_turns = rough_aim[1]
            if (
                world.is_late
                and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER
            ):
                continue

            rough_needed = world.ships_needed_to_capture(
                target.id, est_turns, planned_commitments
            )
            if rough_needed <= 0:
                continue
            if opening_filter(target, est_turns, rough_needed, src_left, world):
                continue

            send = preferred_send(
                target, rough_needed, est_turns, src_left, world, modes
            )
            if send < rough_needed:
                continue

            score = score_attack(
                target,
                send,
                est_turns,
                "capture",
                ATTACK_COST_TURN_WEIGHT,
                world,
                modes,
            )
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, target, send)

        if best is None:
            continue

        _, target, send = best
        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            continue

        angle, turns, _, _ = aim
        missing = world.ships_needed_to_capture(target.id, turns, planned_commitments)
        if missing <= 0:
            continue

        src_left = source_attack_left(src.id)
        send = min(
            src_left,
            max(
                missing,
                preferred_send(target, missing, turns, src_left, world, modes),
            ),
        )
        if send < missing:
            continue

        actual = append_move(src.id, angle, send)
        if actual < missing:
            continue
        planned_commitments[target.id].append((turns, world.player, int(actual)))
