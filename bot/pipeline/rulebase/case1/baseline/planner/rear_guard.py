"""Rear-guard advance: ferry rear-line ships toward the front."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.config import (
    REAR_DISTANCE_RATIO,
    REAR_MAX_TRAVEL_TURNS,
    REAR_SEND_MIN_SHIPS,
    REAR_SEND_RATIO_FOUR_PLAYER,
    REAR_SEND_RATIO_TWO_PLAYER,
    REAR_SOURCE_MIN_SHIPS,
    REAR_STAGE_PROGRESS,
)
from ..core.types import Planet
from ..core.world_model import WorldModel, nearest_distance_to_set
from ..strategy_helpers import planet_distance


def _frontier_targets(world: WorldModel) -> list[Planet]:
    if world.enemy_planets:
        return world.enemy_planets
    return world.static_neutral_planets or world.neutral_planets


def emit_rear_guard_moves(
    world: WorldModel,
    modes: dict[str, Any],
    source_attack_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    """Ferry rear-line ships toward the closest front planet (skipped late game)."""
    if not (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
    ):
        return

    frontier = _frontier_targets(world)
    if not frontier:
        return

    frontier_distance = {
        planet.id: nearest_distance_to_set(planet.x, planet.y, frontier)
        for planet in world.my_planets
    }
    safe_fronts = [
        planet
        for planet in world.my_planets
        if planet.id not in world.doomed_candidates
    ]
    if not safe_fronts:
        return

    front_anchor = min(safe_fronts, key=lambda planet: frontier_distance[planet.id])
    send_ratio = (
        REAR_SEND_RATIO_FOUR_PLAYER
        if world.is_four_player
        else REAR_SEND_RATIO_TWO_PLAYER
    )
    if modes["is_finishing"]:
        send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)

    for rear in sorted(
        world.my_planets,
        key=lambda planet: -frontier_distance[planet.id],
    ):
        if rear.id == front_anchor.id or rear.id in world.doomed_candidates:
            continue
        if source_attack_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
            continue
        if (
            frontier_distance[rear.id]
            < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO
        ):
            continue

        stage_candidates = [
            planet
            for planet in safe_fronts
            if planet.id != rear.id
            and frontier_distance[planet.id]
            < frontier_distance[rear.id] * REAR_STAGE_PROGRESS
        ]
        if stage_candidates:
            front = min(
                stage_candidates,
                key=lambda planet: planet_distance(rear, planet),
            )
        else:
            objective = min(
                frontier,
                key=lambda target: planet_distance(rear, target),
            )
            remaining_fronts = [
                planet for planet in safe_fronts if planet.id != rear.id
            ]
            if not remaining_fronts:
                continue
            front = min(
                remaining_fronts,
                key=lambda planet: planet_distance(planet, objective),
            )

        if front.id == rear.id:
            continue

        send = int(source_attack_left(rear.id) * send_ratio)
        if send < REAR_SEND_MIN_SHIPS:
            continue

        aim = world.plan_shot(rear.id, front.id, send)
        if aim is None:
            continue

        angle, turns, _, _ = aim
        if turns > REAR_MAX_TRAVEL_TURNS:
            continue
        append_move(rear.id, angle, send)
