"""Build capture missions and record per-target shot options for swarm pairing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from ..core.config import (
    ATTACK_COST_TURN_WEIGHT,
    COMET_MAX_CHASE_TURNS,
    PARTIAL_SOURCE_MIN_SHIPS,
    VERY_LATE_CAPTURE_BUFFER,
)
from ..core.types import Mission, ShotOption
from ..core.world_model import WorldModel
from ..strategy_helpers import (
    opening_filter,
    preferred_send,
    score_attack,
)
from .snipe import build_snipe_mission


def build_capture_and_snipe_missions(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
    source_attack_left: Callable[[int], int],
) -> tuple[list[Mission], dict[int, list[ShotOption]]]:
    """Generate single-source capture missions and snipe candidates.

    Returns the mission list plus `source_options_by_target` used by the swarm
    builder for multi-source pairing.
    """
    missions: list[Mission] = []
    source_options_by_target: dict[int, list[ShotOption]] = defaultdict(list)

    for src in world.my_planets:
        src_available = source_attack_left(src.id)
        if src_available <= 0:
            continue

        for target in world.planets:
            if target.id == src.id or target.owner == world.player:
                continue

            rough_ships = max(
                1,
                min(
                    src_available,
                    max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1),
                ),
            )
            rough_aim = world.plan_shot(src.id, target.id, rough_ships)
            if rough_aim is None:
                continue

            rough_turns = rough_aim[1]
            if (
                world.is_very_late
                and rough_turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER
            ):
                continue
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if rough_turns >= life or rough_turns > COMET_MAX_CHASE_TURNS:
                    continue

            rough_needed = world.ships_needed_to_capture(
                target.id, rough_turns, planned_commitments
            )
            if rough_needed <= 0:
                continue
            if opening_filter(target, rough_turns, rough_needed, src_available, world):
                continue

            send_guess = preferred_send(
                target, rough_needed, rough_turns, src_available, world, modes
            )
            aim = world.plan_shot(src.id, target.id, max(1, send_guess))
            if aim is None:
                continue

            angle, turns, _, _ = aim
            if (
                world.is_very_late
                and turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER
            ):
                continue
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if turns >= life or turns > COMET_MAX_CHASE_TURNS:
                    continue

            needed = world.ships_needed_to_capture(
                target.id, turns, planned_commitments
            )
            if needed <= 0:
                continue
            if opening_filter(target, turns, needed, src_available, world):
                continue

            send_cap = min(
                src_available,
                preferred_send(target, needed, turns, src_available, world, modes),
            )
            if send_cap < 1:
                continue
            if send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            expected_send = max(
                needed,
                min(
                    send_cap,
                    preferred_send(target, needed, turns, send_cap, world, modes),
                ),
            )
            score = score_attack(
                target,
                expected_send,
                turns,
                "capture",
                ATTACK_COST_TURN_WEIGHT,
                world,
                modes,
            )
            if score <= 0:
                continue

            option = ShotOption(
                score=score,
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                turns=turns,
                needed=needed,
                send_cap=send_cap,
                mission="capture",
            )
            source_options_by_target[target.id].append(option)

            if send_cap >= needed:
                missions.append(
                    Mission(
                        kind="single",
                        score=score,
                        target_id=target.id,
                        turns=turns,
                        options=[option],
                    )
                )

            snipe = build_snipe_mission(
                src, target, src_available, world, planned_commitments, modes
            )
            if snipe is not None:
                missions.append(snipe)

    return missions, source_options_by_target
