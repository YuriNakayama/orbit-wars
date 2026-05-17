"""SampledAction → Kaggle action list.

Resolves each (src, target, ships_bucket) triple to a concrete launch through
`aim_with_prediction`. Mirrors the lightweight portion of
`imitation/case1.decoder` but without the template / safety filter layer; the
PPO policy is expected to learn target value directly.
"""

from __future__ import annotations

import math
from typing import Any

from .geometry import Planet, aim_with_prediction
from .sampling import SampledAction
from .types import WorldSnapshot

SHIPS_BUCKET_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def _build_planet(row: list[Any]) -> Planet:
    return Planet(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def decode(
    action: SampledAction,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
) -> list[list[int | float]]:
    raw_planets = list(obs.get("planets", []) or [])
    pid_to_planet = {int(row[0]): _build_planet(row) for row in raw_planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    actions: list[list[int | float]] = []
    committed: dict[int, int] = {}

    from_choice = action.from_choice.tolist()
    target_choice = action.target_choice.tolist()
    ships_choice = action.ships_choice.tolist()

    for slot, pid in enumerate(snapshot.planet_ids):
        if slot >= len(from_choice) or not from_choice[slot]:
            continue
        src = pid_to_planet.get(int(pid))
        if src is None or src.owner != snapshot.player:
            continue
        tgt_slot = int(target_choice[slot])
        if tgt_slot < 0 or tgt_slot >= len(snapshot.planet_ids) or tgt_slot == slot:
            continue
        target_pid = int(snapshot.planet_ids[tgt_slot])
        target = pid_to_planet.get(target_pid)
        if target is None:
            continue
        bucket = int(ships_choice[slot])
        bucket = max(0, min(bucket, len(SHIPS_BUCKET_FRACTIONS) - 1))
        ships = int(math.floor(SHIPS_BUCKET_FRACTIONS[bucket] * src.ships))
        if ships <= 0 and src.ships > 0:
            ships = 1
        if ships <= 0:
            continue
        if ships > src.ships:
            ships = src.ships
        # Avoid piling multiple fires onto the same target this turn.
        if committed.get(target_pid, 0) > target.ships * 3:
            continue
        aim = aim_with_prediction(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
        if aim is None:
            continue
        angle, _turns, _ix, _iy = aim
        committed[target_pid] = committed.get(target_pid, 0) + ships
        actions.append([int(src.id), float(angle), int(ships)])
    return actions
