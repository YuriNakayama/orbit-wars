"""Policy output → Kaggle action list."""

from __future__ import annotations

import math
from typing import Any

import torch

from .featurizer import MAX_PLANETS
from .geometry import Planet, aim_with_prediction
from .types import PolicyOutput, WorldSnapshot

NO_OP_LABEL = MAX_PLANETS

# bucket index → fraction of source ships to send (5 buckets, midpoints)
SHIPS_BUCKET_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)


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
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    from_threshold: float = 0.5,
) -> list[list[int | float]]:
    """Greedy decode the model output into a Kaggle action list."""
    from_prob = torch.sigmoid(output.from_logits[0])  # (P,)
    target_argmax = output.target_logits[0].argmax(dim=-1)  # (P,)
    ships_argmax = output.ships_logits[0].argmax(dim=-1)  # (P,)

    raw_planets = list(obs.get("planets", []) or [])
    planets_by_idx = [_build_planet(row) for row in raw_planets]
    pid_to_planet = {p.id: p for p in planets_by_idx}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])

    actions: list[list[int | float]] = []
    for src_pid in snapshot.my_planet_ids:
        global_slot = snapshot.planet_ids.index(src_pid)
        if float(from_prob[global_slot].item()) < from_threshold:
            continue
        tgt_slot = int(target_argmax[global_slot].item())
        if tgt_slot >= MAX_PLANETS:
            continue  # no-op slot picked
        if tgt_slot >= len(snapshot.planet_ids):
            continue  # padding slot — defensive
        tgt_pid = snapshot.planet_ids[tgt_slot]
        if tgt_pid == src_pid:
            continue
        bucket = int(ships_argmax[global_slot].item())
        bucket = max(0, min(bucket, len(SHIPS_BUCKET_FRACTIONS) - 1))
        src = pid_to_planet.get(src_pid)
        target = pid_to_planet.get(tgt_pid)
        if src is None or target is None:
            continue
        ships = int(math.floor(SHIPS_BUCKET_FRACTIONS[bucket] * src.ships))
        if ships <= 0:
            continue
        if ships > src.ships:
            ships = src.ships
        aim = aim_with_prediction(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if aim is None:
            continue
        angle = aim[0]
        actions.append([src_pid, float(angle), int(ships)])
    return actions
