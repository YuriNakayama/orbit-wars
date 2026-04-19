"""Mission builder registry.

`collect_missions` は全 mission builder を既定順で呼び出し、
合算した Mission リストを返す。新 mission を追加する際は本モジュールに
builder 呼び出しを 1 行追加するだけでよい。
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.types import Mission
from ..core.world_model import WorldModel
from .capture import build_capture_and_snipe_missions
from .crash_exploit import build_crash_exploit_missions
from .reinforcement import build_reinforcement_missions
from .swarm import build_swarm_missions


def collect_missions(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    modes: dict[str, Any],
    source_inventory_left: Callable[[int], int],
    source_attack_left: Callable[[int], int],
) -> list[Mission]:
    """Run every mission builder and return the combined mission list.

    capture は swarm 生成に必要な ``source_options_by_target`` を返すので、
    他より先に呼び出し、結果を swarm builder に渡す。
    """
    missions: list[Mission] = []

    missions.extend(
        build_reinforcement_missions(
            world, planned_commitments, modes, source_inventory_left
        )
    )

    capture_missions, source_options_by_target = build_capture_and_snipe_missions(
        world, planned_commitments, modes, source_attack_left
    )
    missions.extend(capture_missions)

    missions.extend(
        build_swarm_missions(
            world, planned_commitments, modes, source_options_by_target
        )
    )

    missions.extend(build_crash_exploit_missions(world, planned_commitments, modes))

    return missions


__all__ = ["collect_missions"]
