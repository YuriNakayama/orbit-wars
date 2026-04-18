# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Top-level planning: plan_moves orchestrates missions and emits kaggle actions."""

from __future__ import annotations

from collections import defaultdict

from pipeline.case1.baseline.core.config import (
    ATTACK_COST_TURN_WEIGHT,
    COMET_MAX_CHASE_TURNS,
    FOLLOWUP_MIN_SHIPS,
    HOSTILE_SWARM_ETA_TOLERANCE,
    LATE_CAPTURE_BUFFER,
    LOW_VALUE_COMET_PRODUCTION,
    MULTI_SOURCE_ETA_TOLERANCE,
    MULTI_SOURCE_PLAN_PENALTY,
    MULTI_SOURCE_TOP_K,
    PARTIAL_SOURCE_MIN_SHIPS,
    REAR_DISTANCE_RATIO,
    REAR_MAX_TRAVEL_TURNS,
    REAR_SEND_MIN_SHIPS,
    REAR_SEND_RATIO_FOUR_PLAYER,
    REAR_SEND_RATIO_TWO_PLAYER,
    REAR_SOURCE_MIN_SHIPS,
    REAR_STAGE_PROGRESS,
    REINFORCE_SAFETY_MARGIN,
    THREE_SOURCE_ETA_TOLERANCE,
    THREE_SOURCE_MIN_TARGET_SHIPS,
    THREE_SOURCE_PLAN_PENALTY,
    THREE_SOURCE_SWARM_ENABLED,
    VERY_LATE_CAPTURE_BUFFER,
)
from pipeline.case1.baseline.core.types import Mission, Planet, ShotOption
from pipeline.case1.baseline.core.world_model import (
    WorldModel,
    nearest_distance_to_set,
)
from pipeline.case1.baseline.missions.crash_exploit import (
    build_crash_exploit_missions,
)
from pipeline.case1.baseline.missions.reinforcement import (
    build_reinforcement_missions,
)
from pipeline.case1.baseline.missions.snipe import build_snipe_mission
from pipeline.case1.baseline.strategy_helpers import (
    apply_score_modifiers,
    build_modes,
    opening_filter,
    planet_distance,
    preferred_send,
    target_value,
)


def plan_moves(world: WorldModel) -> list[list[int | float]]:
    modes = build_modes(world)
    planned_commitments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    source_options_by_target: dict[int, list[ShotOption]] = defaultdict(list)
    missions: list[Mission] = []
    moves: list[list[int | float]] = []
    spent_total: dict[int, int] = defaultdict(int)

    def source_inventory_left(source_id: int) -> int:
        return world.source_inventory_left(source_id, spent_total)

    def source_attack_left(source_id: int) -> int:
        return world.source_attack_left(source_id, spent_total)

    def append_move(src_id: int, angle: float, ships: int) -> int:
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    reinforce_missions = build_reinforcement_missions(
        world, planned_commitments, modes, source_inventory_left
    )
    missions.extend(reinforce_missions)

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
            if opening_filter(
                target, rough_turns, rough_needed, src_available, world
            ):
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
                preferred_send(
                    target, needed, turns, src_available, world, modes
                ),
            )
            if send_cap < 1:
                continue
            if send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue

            value = target_value(target, turns, "capture", world, modes)
            if value <= 0:
                continue

            expected_send = max(
                needed,
                min(
                    send_cap,
                    preferred_send(
                        target, needed, turns, send_cap, world, modes
                    ),
                ),
            )
            score = apply_score_modifiers(
                value / (expected_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                target,
                "capture",
                world,
            )

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

    for target_id, options in source_options_by_target.items():
        if len(options) < 2:
            continue

        target = world.planet_by_id[target_id]
        top_options = sorted(options, key=lambda item: -item.score)[
            :MULTI_SOURCE_TOP_K
        ]

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

                        value = target_value(
                            target, joint_turn, "swarm", world, modes
                        )
                        if value <= 0:
                            continue
                        trio_score = apply_score_modifiers(
                            value
                            / (need + joint_turn * ATTACK_COST_TURN_WEIGHT + 1.0),
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

    crash_missions = build_crash_exploit_missions(
        world, planned_commitments, modes
    )
    missions.extend(crash_missions)

    missions.sort(key=lambda item: -item.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "reinforce", "crash_exploit"):
            option = mission.options[0]

            if mission.kind == "reinforce":
                left = source_inventory_left(option.src_id)
            else:
                left = source_attack_left(option.src_id)
            if left <= 0:
                continue

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
                continue

            send_limit = min(left, option.send_cap)
            if send_limit < missing:
                continue

            if mission.kind in ("snipe", "crash_exploit"):
                send = missing
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
                continue

            sent = append_move(option.src_id, option.angle, send)
            if sent < missing:
                continue
            planned_commitments[target.id].append(
                (arrival_turn, world.player, int(sent))
            )
            continue

        limits: list[int] = []
        for option in mission.options:
            left = source_attack_left(option.src_id)
            limits.append(min(left, option.send_cap))
        if min(limits) <= 0:
            continue

        missing = world.ships_needed_to_capture(
            target.id, mission.turns, planned_commitments
        )
        if missing <= 0:
            continue
        if sum(limits) < missing:
            continue

        ordered = sorted(
            zip(mission.options, limits, strict=True),
            key=lambda item: (item[0].turns, -item[1], item[0].src_id),
        )
        remaining = missing
        sends: dict[int, int] = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(
                other_limit for _, other_limit in ordered[idx + 1 :]
            )
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue

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
            continue
        planned_commitments[target.id].extend(committed)

    if not world.is_very_late:
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
                if opening_filter(
                    target, est_turns, rough_needed, src_left, world
                ):
                    continue

                send = preferred_send(
                    target, rough_needed, est_turns, src_left, world, modes
                )
                if send < rough_needed:
                    continue

                value = target_value(target, est_turns, "capture", world, modes)
                if value <= 0:
                    continue

                score = apply_score_modifiers(
                    value / (send + est_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, send)

            if best is None:
                continue

            _, target, send = best
            aim = world.plan_shot(src.id, target.id, send)
            if aim is None:
                continue

            angle, turns, _, _ = aim
            missing = world.ships_needed_to_capture(
                target.id, turns, planned_commitments
            )
            if missing <= 0:
                continue

            src_left = source_attack_left(src.id)
            send = min(
                src_left,
                max(
                    missing,
                    preferred_send(
                        target, missing, turns, src_left, world, modes
                    ),
                ),
            )
            if send < missing:
                continue

            actual = append_move(src.id, angle, send)
            if actual < missing:
                continue
            planned_commitments[target.id].append(
                (turns, world.player, int(actual))
            )

    if world.doomed_candidates:
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        if frontier_targets:
            frontier_distance = {
                planet.id: nearest_distance_to_set(
                    planet.x, planet.y, frontier_targets
                )
                for planet in world.my_planets
            }
        else:
            frontier_distance = {
                planet.id: 10**9 for planet in world.my_planets
            }

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
                score = target_value(
                    target, turns, "capture", world, modes
                ) / (need + turns + 1.0)
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
            aim = world.plan_shot(
                planet.id, retreat_target.id, available_now
            )
            if aim is None:
                continue
            angle, _, _, _ = aim
            append_move(planet.id, angle, available_now)

    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
    ):
        frontier_targets = (
            world.enemy_planets
            if world.enemy_planets
            else (world.static_neutral_planets or world.neutral_planets)
        )
        frontier_distance = {
            planet.id: nearest_distance_to_set(
                planet.x, planet.y, frontier_targets
            )
            for planet in world.my_planets
        }
        safe_fronts = [
            planet
            for planet in world.my_planets
            if planet.id not in world.doomed_candidates
        ]
        if safe_fronts:
            front_anchor = min(
                safe_fronts, key=lambda planet: frontier_distance[planet.id]
            )
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
                if (
                    rear.id == front_anchor.id
                    or rear.id in world.doomed_candidates
                ):
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
                        frontier_targets,
                        key=lambda target: planet_distance(rear, target),
                    )
                    remaining_fronts = [
                        planet
                        for planet in safe_fronts
                        if planet.id != rear.id
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
