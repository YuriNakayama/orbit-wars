"""obs (dict) → BatchFeatures (torch.Tensor) for imitation/case1 IL baseline.

Pure function — no torch.nn, no autograd, no random state.
Used by both training preprocess and runtime agent inference.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .templates import TEMPLATE_CTX_DIM, template_context_features
from .types import BatchFeatures, WorldSnapshot

PLANET_FEAT_DIM = 11
GLOBAL_FEAT_DIM = 6
MAX_PLANETS = 36
BOARD_SIZE = 100.0
HORIZON_TURNS = 30  # for incoming-fleet eta normalization


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
    # Mirrors pipeline/rulebase/case1/baseline/core/physics.py::fleet_speed.
    # Fewer ships fly faster; we only need a monotone proxy here.
    return max(0.5, 2.0 - 0.05 * math.sqrt(max(1, ships)))


def _fleet_target_eta(
    fleet_x: float,
    fleet_y: float,
    angle: float,
    ships: int,
    planet_x: float,
    planet_y: float,
    planet_radius: float,
) -> float | None:
    dx = planet_x - fleet_x
    dy = planet_y - fleet_y
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    radius_sq = planet_radius * planet_radius
    if perp_sq >= radius_sq:
        return None
    hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
    speed = _fleet_speed(ships)
    if speed <= 0:
        return None
    return hit_d / speed


def featurize(obs: Any) -> tuple[BatchFeatures, WorldSnapshot]:
    """Convert a single observation dict to a BatchFeatures of batch_size=1."""
    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = list(_read(obs, "planets", []) or [])
    raw_fleets = list(_read(obs, "fleets", []) or [])
    raw_comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    ang_vel = float(_read(obs, "angular_velocity", 0.0) or 0.0)

    n = min(len(raw_planets), MAX_PLANETS)
    planet_feats = torch.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=torch.float32)
    planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    my_planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    target_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    template_ctx = torch.zeros((MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=torch.float32)

    planet_ids: list[int] = []
    my_planet_ids: list[int] = []

    incoming = [[0.0, 0.0, 0.0] for _ in range(MAX_PLANETS)]
    nearest_eta = [HORIZON_TURNS + 1.0] * MAX_PLANETS

    planet_index_by_id: dict[int, int] = {}
    for slot in range(n):
        pid = int(raw_planets[slot][0])
        planet_index_by_id[pid] = slot

    for fleet_row in raw_fleets:
        fid, fowner, fx, fy, fangle, _from_pid, fships = fleet_row
        f_owner = int(fowner)
        f_ships = int(fships)
        f_x = float(fx)
        f_y = float(fy)
        f_angle = float(fangle)
        for slot in range(n):
            pid_, _, px, py, pradius, _, _ = raw_planets[slot]
            eta = _fleet_target_eta(
                f_x,
                f_y,
                f_angle,
                f_ships,
                float(px),
                float(py),
                float(pradius),
            )
            if eta is None or eta > HORIZON_TURNS:
                continue
            if f_owner == player:
                incoming[slot][0] += f_ships
            elif f_owner == -1:
                incoming[slot][2] += f_ships
            else:
                incoming[slot][1] += f_ships
            if eta < nearest_eta[slot]:
                nearest_eta[slot] = eta

    for slot in range(n):
        pid, owner, px, py, radius, ships, production = raw_planets[slot]
        owner_i = int(owner)
        ships_i = int(ships)
        production_i = int(production)
        is_mine = owner_i == player
        is_neutral = owner_i == -1
        is_enemy = (not is_mine) and (not is_neutral)
        is_comet = int(pid) in raw_comet_ids

        eta_norm = min(nearest_eta[slot], HORIZON_TURNS + 1.0) / (HORIZON_TURNS + 1.0)
        feats = [
            float(px) / BOARD_SIZE,
            float(py) / BOARD_SIZE,
            float(radius) / 5.0,
            math.log1p(max(0, ships_i)),
            math.log1p(max(0, production_i)),
            1.0 if is_mine else 0.0,
            1.0 if is_enemy else 0.0,
            1.0 if is_neutral else 0.0,
            1.0 if is_comet else 0.0,
            math.log1p(incoming[slot][1]) - math.log1p(incoming[slot][0]),
            eta_norm,
        ]
        for j in range(PLANET_FEAT_DIM):
            planet_feats[slot, j] = feats[j]
        planet_mask[slot] = True
        if is_mine:
            my_planet_mask[slot] = True
            my_planet_ids.append(int(pid))
            ctx = template_context_features(
                list(raw_planets[slot]), raw_planets, player, BOARD_SIZE
            )
            for j in range(TEMPLATE_CTX_DIM):
                template_ctx[slot, j] = ctx[j]
        if not is_mine:
            target_mask[slot] = True
        planet_ids.append(int(pid))

    # global features
    my_total_ships = 0.0
    enemy_total_ships = 0.0
    neutral_total_ships = 0.0
    my_total_prod = 0.0
    enemy_total_prod = 0.0
    for slot in range(n):
        _, owner, _, _, _, ships, production = raw_planets[slot]
        owner_i = int(owner)
        if owner_i == player:
            my_total_ships += float(ships)
            my_total_prod += float(production)
        elif owner_i == -1:
            neutral_total_ships += float(ships)
        else:
            enemy_total_ships += float(ships)
            enemy_total_prod += float(production)

    global_feats = torch.tensor(
        [
            float(step) / 500.0,
            ang_vel * 10.0,
            math.log1p(my_total_ships),
            math.log1p(enemy_total_ships),
            math.log1p(neutral_total_ships),
            math.log1p(my_total_prod) - math.log1p(enemy_total_prod),
        ],
        dtype=torch.float32,
    )

    batch = BatchFeatures(
        planet_feats=planet_feats.unsqueeze(0),
        planet_mask=planet_mask.unsqueeze(0),
        my_planet_mask=my_planet_mask.unsqueeze(0),
        target_mask=target_mask.unsqueeze(0),
        global_feats=global_feats.unsqueeze(0),
        template_ctx=template_ctx.unsqueeze(0),
    )
    snapshot = WorldSnapshot(
        planet_ids=tuple(planet_ids),
        my_planet_ids=tuple(my_planet_ids),
        player=player,
        step=step,
    )
    return batch, snapshot
