# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Build reinforcement missions for threatened friendly planets."""

from __future__ import annotations

from typing import Any, Callable

from pipeline.case1.baseline.core.config import (
    PARTIAL_SOURCE_MIN_SHIPS,
    REINFORCE_ENABLED,
    REINFORCE_MAX_SOURCE_FRACTION,
    REINFORCE_MAX_TRAVEL_TURNS,
    REINFORCE_SAFETY_MARGIN,
)
from pipeline.case1.baseline.core.types import Mission, ShotOption
from pipeline.case1.baseline.core.world_model import WorldModel
from pipeline.case1.baseline.strategy_helpers import target_value


def build_reinforcement_missions(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
    source_budget_fn: Callable[[int], int],
) -> list[Mission]:
    if not REINFORCE_ENABLED or not world.threatened_candidates:
        return []

    missions: list[Mission] = []
    for target_id, info in world.threatened_candidates.items():
        target = world.planet_by_id[target_id]
        fall_turn = info["fall_turn"]
        if fall_turn is None or fall_turn > REINFORCE_MAX_TRAVEL_TURNS + 5:
            continue

        best_mission: Mission | None = None
        for src in world.my_planets:
            if src.id == target_id:
                continue
            budget = source_budget_fn(src.id)
            if budget <= 0:
                continue
            source_cap = min(
                budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION)
            )
            if source_cap <= 0:
                continue

            probe_ships = max(
                PARTIAL_SOURCE_MIN_SHIPS,
                int(info["deficit_hint"]) + REINFORCE_SAFETY_MARGIN,
            )
            probe_ships = min(probe_ships, source_cap)
            if probe_ships <= 0:
                continue

            aim = world.plan_shot(src.id, target.id, probe_ships)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > REINFORCE_MAX_TRAVEL_TURNS:
                continue
            if turns > fall_turn:
                continue

            need = world.reinforcement_needed_for(
                target_id, turns, planned_commitments
            )
            if need <= 0:
                continue
            send = min(source_cap, need + REINFORCE_SAFETY_MARGIN)
            if send < need:
                continue

            final = world.plan_shot(src.id, target.id, send)
            if final is None:
                continue
            angle, turns, _, _ = final
            if turns > fall_turn:
                continue

            value = target_value(target, turns, "reinforce", world, modes)
            if value <= 0:
                continue
            score = value / (send + turns * 0.35 + 1.0)

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target_id,
                angle=angle,
                turns=turns,
                needed=need,
                send_cap=send,
                mission="reinforce",
            )
            mission = Mission(
                kind="reinforce",
                score=score,
                target_id=target_id,
                turns=turns,
                options=[option],
            )
            if best_mission is None or mission.score > best_mission.score:
                best_mission = mission

        if best_mission is not None:
            missions.append(best_mission)

    return missions
