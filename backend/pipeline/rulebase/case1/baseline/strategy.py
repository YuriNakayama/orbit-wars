# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Top-level planning: plan_moves orchestrates planner units and emits actions.

The legacy 702-line `plan_moves` was decomposed into the modules under
`baseline/planner/`:

- `option_collector` — enumerates per (src, target) shot options + single + snipe.
- `swarm_builder` — pair- + trio-source swarm missions on top of those options.
- `mission_resolver` — single- vs multi-source mission allocation.
- `followup` — per-source best secondary capture after main missions.
- `evacuation` — doomed planets (capture or retreat).
- `rear_guard` — ferry rear ships toward the front.
- `finalizer` — clamp per-source spend to actual ships available.

This file glues the units together and remains the single public entrypoint.
"""

from __future__ import annotations

from collections import defaultdict

from .core.world_model import WorldModel
from .missions.crash_exploit import build_crash_exploit_missions
from .missions.reinforcement import build_reinforcement_missions
from .planner import (
    build_swarm_missions,
    collect_capture_options_and_missions,
    emit_evacuation_moves,
    emit_followup_moves,
    emit_rear_guard_moves,
    enforce_inventory_cap,
    process_multi_source_mission,
    process_single_source_mission,
)
from .planner.mission_resolver import SINGLE_SOURCE_MISSION_KINDS
from .strategy_helpers import build_modes


def plan_moves(world: WorldModel) -> list[list[int | float]]:
    modes = build_modes(world)
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

    missions = list(
        build_reinforcement_missions(
            world, planned_commitments, modes, source_inventory_left
        )
    )
    capture_missions, source_options_by_target = collect_capture_options_and_missions(
        world, planned_commitments, modes, source_attack_left
    )
    missions.extend(capture_missions)
    missions.extend(
        build_swarm_missions(
            world, source_options_by_target, planned_commitments, modes
        )
    )
    missions.extend(build_crash_exploit_missions(world, planned_commitments, modes))

    missions.sort(key=lambda item: -item.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in SINGLE_SOURCE_MISSION_KINDS:
            process_single_source_mission(
                mission,
                target,
                world,
                modes,
                planned_commitments,
                source_inventory_left,
                source_attack_left,
                append_move,
            )
        else:
            process_multi_source_mission(
                mission,
                target,
                world,
                planned_commitments,
                source_attack_left,
                append_move,
            )

    emit_followup_moves(
        world, planned_commitments, modes, source_attack_left, append_move
    )
    emit_evacuation_moves(
        world, planned_commitments, modes, source_inventory_left, append_move
    )
    emit_rear_guard_moves(world, modes, source_attack_left, append_move)

    return enforce_inventory_cap(moves, world)
