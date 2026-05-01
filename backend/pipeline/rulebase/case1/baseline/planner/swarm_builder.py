"""2-source / 3-source swarm mission discovery."""

from __future__ import annotations

from typing import Any

from ..core.config import (
    ATTACK_COST_TURN_WEIGHT,
    HOSTILE_SWARM_ETA_TOLERANCE,
    MULTI_SOURCE_ETA_TOLERANCE,
    MULTI_SOURCE_PLAN_PENALTY,
    MULTI_SOURCE_TOP_K,
    THREE_SOURCE_ETA_TOLERANCE,
    THREE_SOURCE_MIN_TARGET_SHIPS,
    THREE_SOURCE_PLAN_PENALTY,
    THREE_SOURCE_SWARM_ENABLED,
)
from ..core.types import Mission, ShotOption
from ..core.world_model import WorldModel
from ..strategy_helpers import apply_score_modifiers, target_value


def build_swarm_missions(
    world: WorldModel,
    source_options_by_target: dict[int, list[ShotOption]],
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
) -> list[Mission]:
    """Iterate per-target options and emit pair- + trio-source swarm missions."""
    missions: list[Mission] = []

    for target_id, options in source_options_by_target.items():
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[:MULTI_SOURCE_TOP_K]

        hostile_target = target.owner not in (-1, world.player)
        eta_tolerance_2 = (
            HOSTILE_SWARM_ETA_TOLERANCE
            if hostile_target
            else MULTI_SOURCE_ETA_TOLERANCE
        )

        for i in range(len(top_options)):
            for j in range(i + 1, len(top_options)):
                first = top_options[i]
                second = top_options[j]
                if first.src_id == second.src_id:
                    continue
                if abs(first.turns - second.turns) > eta_tolerance_2:
                    continue

                joint_turn = max(first.turns, second.turns)
                need = world.ships_needed_to_capture(
                    target_id, joint_turn, planned_commitments
                )
                if need <= 0:
                    continue
                if first.send_cap >= need or second.send_cap >= need:
                    continue

                total_cap = first.send_cap + second.send_cap
                if total_cap < need:
                    continue

                value = target_value(target, joint_turn, "swarm", world, modes)
                if value <= 0:
                    continue

                pair_score = apply_score_modifiers(
                    value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "swarm",
                    world,
                )
                pair_score *= MULTI_SOURCE_PLAN_PENALTY
                missions.append(
                    Mission(
                        kind="swarm",
                        score=pair_score,
                        target_id=target_id,
                        turns=joint_turn,
                        options=[first, second],
                    )
                )

        if (
            THREE_SOURCE_SWARM_ENABLED
            and hostile_target
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top_options) >= 3
        ):
            for i in range(len(top_options)):
                for j in range(i + 1, len(top_options)):
                    for k in range(j + 1, len(top_options)):
                        trio = [top_options[i], top_options[j], top_options[k]]
                        src_ids = {opt.src_id for opt in trio}
                        if len(src_ids) < 3:
                            continue
                        eta_spread = max(opt.turns for opt in trio) - min(
                            opt.turns for opt in trio
                        )
                        if eta_spread > THREE_SOURCE_ETA_TOLERANCE:
                            continue

                        joint_turn = max(opt.turns for opt in trio)
                        need = world.ships_needed_to_capture(
                            target_id, joint_turn, planned_commitments
                        )
                        if need <= 0:
                            continue
                        total_cap = sum(opt.send_cap for opt in trio)
                        if total_cap < need:
                            continue
                        if any(
                            trio[a].send_cap + trio[b].send_cap >= need
                            for a in range(3)
                            for b in range(a + 1, 3)
                        ):
                            continue

                        value = target_value(target, joint_turn, "swarm", world, modes)
                        if value <= 0:
                            continue
                        trio_score = apply_score_modifiers(
                            value / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
                            target,
                            "swarm",
                            world,
                        )
                        trio_score *= THREE_SOURCE_PLAN_PENALTY
                        missions.append(
                            Mission(
                                kind="swarm",
                                score=trio_score,
                                target_id=target_id,
                                turns=joint_turn,
                                options=trio,
                            )
                        )

    return missions
