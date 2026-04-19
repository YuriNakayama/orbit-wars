"""Harass mission: briefly seize high-production enemy planets to drain production."""

from __future__ import annotations

from typing import Any

from ..core.config import (
    HARASS_COST_TURN_WEIGHT,
    HARASS_ENABLED,
    HARASS_MAX_TRAVEL_TURNS,
    HARASS_MIN_SRC_RESERVE,
    HARASS_MIN_TARGET_PRODUCTION,
    HARASS_MIN_TARGET_SHIPS,
    HARASS_PRODUCTION_STEAL_TURNS,
    HARASS_VALUE_MULT,
    VERY_LATE_CAPTURE_BUFFER,
)
from ..core.types import Mission, ShotOption
from ..core.world_model import WorldModel


def build_harass_missions(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
) -> list[Mission]:
    """Generate one-shot captures on high-prod enemies to steal production briefly."""
    if not HARASS_ENABLED or world.is_very_late:
        return []
    if not world.enemy_planets:
        return []

    missions: list[Mission] = []

    for target in world.enemy_planets:
        if target.id in world.comet_ids:
            continue
        if int(target.production) < HARASS_MIN_TARGET_PRODUCTION:
            continue
        if int(target.ships) < HARASS_MIN_TARGET_SHIPS:
            continue

        best: Mission | None = None
        for src in world.my_planets:
            if src.id == target.id:
                continue
            if int(src.ships) < HARASS_MIN_SRC_RESERVE:
                continue

            probe = max(int(target.ships) + 2, HARASS_MIN_TARGET_SHIPS + 2)
            if probe >= int(src.ships):
                continue
            aim = world.plan_shot(src.id, target.id, probe)
            if aim is None:
                continue
            _, turns, _, _ = aim
            if turns > HARASS_MAX_TRAVEL_TURNS:
                continue
            if turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER:
                continue

            need = world.ships_needed_to_capture(target.id, turns, planned_commitments)
            if need <= 0 or need > int(src.ships) - HARASS_MIN_SRC_RESERVE + probe:
                continue
            if need >= int(src.ships):
                continue

            final = world.plan_shot(src.id, target.id, need)
            if final is None:
                continue
            angle, turns, _, _ = final
            if turns > HARASS_MAX_TRAVEL_TURNS:
                continue

            need = world.ships_needed_to_capture(target.id, turns, planned_commitments)
            if need <= 0 or need >= int(src.ships):
                continue

            stolen_production = (
                int(target.production) * HARASS_PRODUCTION_STEAL_TURNS * HARASS_VALUE_MULT
            )
            score = stolen_production / (need + turns * HARASS_COST_TURN_WEIGHT + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=need,
                mission="harass",
            )
            mission = Mission(
                kind="harass",
                score=score,
                target_id=target.id,
                turns=turns,
                options=[option],
            )
            if best is None or mission.score > best.score:
                best = mission

        if best is not None:
            missions.append(best)

    return missions
