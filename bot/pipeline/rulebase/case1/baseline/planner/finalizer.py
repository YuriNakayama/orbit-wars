"""Final pass: clamp per-source spend to actual ships available."""

from __future__ import annotations

from collections import defaultdict

from ..core.world_model import WorldModel


def enforce_inventory_cap(
    moves: list[list[int | float]], world: WorldModel
) -> list[list[int | float]]:
    """Drop / shrink moves whose cumulative ships exceed the source's inventory."""
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
