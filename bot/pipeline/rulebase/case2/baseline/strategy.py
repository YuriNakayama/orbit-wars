# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Top-level planning: plan_moves orchestrates missions and emits kaggle actions."""

from __future__ import annotations

from collections import defaultdict

from .core.config import REINFORCE_SAFETY_MARGIN
from .core.world_model import WorldModel
from .missions import collect_missions
from .movements.evacuation import emit_evacuation_moves
from .movements.followup import emit_followup_moves
from .movements.rear_guard import emit_rear_guard_moves
from .strategy_helpers import build_modes, preferred_send


def plan_moves(world: WorldModel) -> list[list[int | float]]:
    modes = build_modes(world)
    # Seed the aim cache for the capture-probe grid in one batched vmap (JAX
    # backend only; no-op under Python). This turns the O(P^2) capture probe
    # sweep from P^2 individual jit dispatches into a single kernel launch.
    world.warm_capture_probes()
    planned_commitments: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
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

    missions = collect_missions(
        world,
        planned_commitments,
        modes,
        source_inventory_left,
        source_attack_left,
    )
    missions.sort(key=lambda item: -item.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]

        if mission.kind in ("single", "snipe", "reinforce", "crash_exploit", "harass"):
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

            if mission.kind in ("snipe", "crash_exploit", "harass"):
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
            remaining_other = sum(other_limit for _, other_limit in ordered[idx + 1 :])
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

    emit_followup_moves(
        world, planned_commitments, modes, source_attack_left, append_move
    )
    emit_evacuation_moves(
        world, planned_commitments, modes, source_inventory_left, append_move
    )
    emit_rear_guard_moves(world, modes, source_attack_left, append_move)

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
