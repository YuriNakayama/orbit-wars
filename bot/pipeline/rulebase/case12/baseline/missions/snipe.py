# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Build a snipe mission: steal a neutral whose enemy wave is inbound."""

from __future__ import annotations

import math
from typing import Any

from ..core.config import (
    COMET_MAX_CHASE_TURNS,
    PARTIAL_SOURCE_MIN_SHIPS,
    SNIPE_COST_TURN_WEIGHT,
)
from ..core.types import Mission, Planet, ShotOption
from ..core.world_model import WorldModel
from ..strategy_helpers import score_attack


def build_snipe_mission(
    src: Planet,
    target: Planet,
    src_available: int,
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
) -> Mission | None:
    if target.owner != -1:
        return None

    enemy_etas = sorted(
        {
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        }
    )
    if not enemy_etas:
        return None

    probe = min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 8))
    rough = world.plan_shot(src.id, target.id, probe)
    if rough is None:
        return None

    for enemy_eta in enemy_etas[:3]:
        if abs(rough[1] - enemy_eta) > 1:
            continue

        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue

        need = world.ships_needed_to_capture(target.id, sync_turn, planned_commitments)
        if need <= 0 or need > src_available:
            continue

        final = world.plan_shot(src.id, target.id, need)
        if final is None:
            continue

        angle, turns, _, _ = final
        if abs(turns - enemy_eta) > 1:
            continue

        sync_turn = max(turns, enemy_eta)
        need = world.ships_needed_to_capture(target.id, sync_turn, planned_commitments)
        if need <= 0 or need > src_available:
            continue

        score = score_attack(
            target, need, sync_turn, "snipe", SNIPE_COST_TURN_WEIGHT, world, modes
        )
        if score <= 0:
            continue
        option = ShotOption(
            score=score,
            src_id=src.id,
            target_id=target.id,
            angle=angle,
            turns=turns,
            needed=need,
            send_cap=need,
            mission="snipe",
        )
        return Mission(
            kind="snipe",
            score=score,
            target_id=target.id,
            turns=sync_turn,
            options=[option],
        )

    return None
