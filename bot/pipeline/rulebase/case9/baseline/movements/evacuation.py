"""Evacuation: route doomed planets' ships into a capture or retreat to a safe ally."""

from __future__ import annotations

from typing import Any, Callable

from ..core.types import Planet
from ..core.world_model import WorldModel, nearest_distance_to_set
from ..strategy_helpers import planet_distance, target_value


def emit_evacuation_moves(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
    source_inventory_left: Callable[[int], int],
    append_move: Callable[[int, float, int], int],
) -> None:
    if not world.doomed_candidates:
        return

    frontier_targets: list[Planet]
    if world.enemy_planets:
        frontier_targets = world.enemy_planets
    else:
        frontier_targets = world.static_neutral_planets or world.neutral_planets

    if frontier_targets:
        frontier_distance = {
            planet.id: nearest_distance_to_set(planet.x, planet.y, frontier_targets)
            for planet in world.my_planets
        }
    else:
        frontier_distance = {planet.id: 10**9 for planet in world.my_planets}

    for planet in world.my_planets:
        if planet.id not in world.doomed_candidates:
            continue

        if planned_commitments.get(planet.id):
            incoming = sum(
                ships
                for _, owner, ships in planned_commitments[planet.id]
                if owner == world.player
            )
            if incoming > 0:
                continue

        available_now = source_inventory_left(planet.id)
        if available_now < world.reserve.get(planet.id, 0):
            continue

        best_capture: tuple[float, int, float, int, int] | None = None
        for target in world.planets:
            if target.id == planet.id or target.owner == world.player:
                continue
            probe_aim = world.plan_shot(planet.id, target.id, available_now)
            if probe_aim is None:
                continue
            probe_turns = probe_aim[1]
            if probe_turns > world.remaining_steps - 2:
                continue
            need = world.ships_needed_to_capture(
                target.id, probe_turns, planned_commitments
            )
            if need <= 0 or need > available_now:
                continue
            final_aim = world.plan_shot(planet.id, target.id, need)
            if final_aim is None:
                continue
            angle, turns, _, _ = final_aim
            value = target_value(target, turns, "capture", world, modes)
            if value <= 0:
                continue
            score = value / (need + turns + 1.0)
            if target.owner not in (-1, world.player):
                score *= 1.05
            if best_capture is None or score > best_capture[0]:
                best_capture = (score, target.id, angle, turns, need)

        if best_capture is not None:
            _, target_id, angle, turns, need = best_capture
            actual = append_move(planet.id, angle, need)
            if actual >= 1:
                planned_commitments[target_id].append(
                    (turns, world.player, int(actual))
                )
            continue

        safe_allies = [
            ally
            for ally in world.my_planets
            if ally.id != planet.id and ally.id not in world.doomed_candidates
        ]
        if not safe_allies:
            continue

        retreat_target = min(
            safe_allies,
            key=lambda ally: (
                frontier_distance.get(ally.id, 10**9),
                planet_distance(planet, ally),
            ),
        )
        aim = world.plan_shot(planet.id, retreat_target.id, available_now)
        if aim is None:
            continue
        angle, _, _, _ = aim
        append_move(planet.id, angle, available_now)
