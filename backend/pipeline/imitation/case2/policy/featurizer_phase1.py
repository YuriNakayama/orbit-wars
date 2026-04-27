"""Phase 1 featurizer for imitation/case2.

Extends the baseline featurizer (PLANET_FEAT_DIM=18, GLOBAL_FEAT_DIM=11) with
features translated from CNN-solution literature into per-node / global form:

- Mapping #3 (distance maps): nearest_ally_dist, nearest_neutral_dist
- Mapping #4 (orbital prediction channels): (dx, dy) at t+{1, 2, 4, 8}
- Mapping #5 (incoming fleet ETA & ship sums): split ally/enemy
- Mapping #6 (threat pressure short/long): re-define col 16 as short-range
  (<= 8u) and add long-range (> 8u, <= 25u) as new col 32
- Mapping #9 (next comet wave eta): global col 11
- Mapping #10 (totals fractions / score diff): global cols 12..15

Resulting dims:
  PLANET_FEAT_DIM = 33  (baseline 18 + 15)
  GLOBAL_FEAT_DIM = 16  (baseline 11 + 5)
  TEMPLATE_CTX_DIM unchanged

Pure function — no torch.nn / autograd. The baseline featurizer
(`featurizer.py`) is left untouched so baseline and phase1 can be trained
and evaluated side-by-side on independent parquets.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .geometry import (
    CENTER_X,
    CENTER_Y,
    Planet,
    predict_comet_position,
    predict_planet_position,
)
from .templates import TEMPLATE_CTX_DIM, template_context_features
from .types import BatchFeatures, WorldSnapshot

PLANET_FEAT_DIM = 33
GLOBAL_FEAT_DIM = 16
MAX_PLANETS = 36
BOARD_SIZE = 100.0
HORIZON_TURNS = 30
SUN_X = CENTER_X
SUN_Y = CENTER_Y
ROTATION_LIMIT = 50.0
DIAG = math.sqrt(2.0) * BOARD_SIZE
NEIGHBOR_RADIUS_SHORT = 8.0
NEIGHBOR_RADIUS_LONG = 25.0
COMET_WAVES = (50, 150, 250, 350, 450)
COMET_WINDOW = 30
ORBIT_HORIZONS = (1, 2, 4, 8)
LOG_NORM_DENOM = 6.0  # log1p(403) ~= 6, used to bring ship-sum features to ~[0, 1]


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
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


def _comet_active(step: int) -> bool:
    return any(0 <= step - w < COMET_WINDOW for w in COMET_WAVES)


def _next_comet_eta(step: int) -> int:
    """Turns until the next comet wave (>= 0). Returns 0 if currently in window."""
    if _comet_active(step):
        return 0
    upcoming = [w - step for w in COMET_WAVES if w >= step]
    if not upcoming:
        return 100  # cap: no more comets in episode
    return min(upcoming)


def _build_planet_obj(row: list[Any]) -> Planet:
    return Planet(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def _orbit_predictions(
    planet: Planet,
    initial_by_id: dict[int, Planet],
    angular_velocity: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> list[tuple[float, float]]:
    """Predicted (dx, dy) at t+1/2/4/8 relative to current position, normalized."""
    out: list[tuple[float, float]] = []
    for turns in ORBIT_HORIZONS:
        if planet.id in comet_ids:
            pos = predict_comet_position(planet.id, comets, turns)
            if pos is None:
                out.append((0.0, 0.0))
                continue
            nx, ny = pos
        else:
            nx, ny = predict_planet_position(
                planet, initial_by_id, angular_velocity, turns
            )
        out.append(
            (
                (nx - planet.x) / BOARD_SIZE,
                (ny - planet.y) / BOARD_SIZE,
            )
        )
    return out


def featurize(obs: Any) -> tuple[BatchFeatures, WorldSnapshot]:
    """Convert a single obs dict to a BatchFeatures of batch_size=1."""
    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = list(_read(obs, "planets", []) or [])
    raw_fleets = list(_read(obs, "fleets", []) or [])
    raw_comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    raw_comets = list(_read(obs, "comets", []) or [])
    raw_initial_planets = list(_read(obs, "initial_planets", []) or [])
    ang_vel = float(_read(obs, "angular_velocity", 0.0) or 0.0)

    n = min(len(raw_planets), MAX_PLANETS)
    planet_feats = torch.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=torch.float32)
    planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    my_planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    target_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    template_ctx = torch.zeros((MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=torch.float32)

    planet_ids: list[int] = []
    my_planet_ids: list[int] = []

    initial_by_id: dict[int, Planet] = {
        int(row[0]): _build_planet_obj(list(row)) for row in raw_initial_planets
    }

    # Pass 0: per-fleet ETA contributions to each planet (separated by owner).
    incoming_ally_ships = [0.0] * MAX_PLANETS
    incoming_enemy_ships = [0.0] * MAX_PLANETS
    incoming_neutral_ships = [0.0] * MAX_PLANETS
    nearest_eta = [HORIZON_TURNS + 1.0] * MAX_PLANETS
    incoming_ally_eta_min = [HORIZON_TURNS + 1.0] * MAX_PLANETS
    incoming_enemy_eta_min = [HORIZON_TURNS + 1.0] * MAX_PLANETS

    for fleet_row in raw_fleets:
        _, fowner, fx, fy, fangle, _from_pid, fships = fleet_row
        f_owner = int(fowner)
        f_ships = int(fships)
        f_x = float(fx)
        f_y = float(fy)
        f_angle = float(fangle)
        for slot in range(n):
            _, _, px, py, pradius, _, _ = raw_planets[slot]
            eta = _fleet_target_eta(
                f_x, f_y, f_angle, f_ships, float(px), float(py), float(pradius)
            )
            if eta is None or eta > HORIZON_TURNS:
                continue
            if f_owner == player:
                incoming_ally_ships[slot] += f_ships
                if eta < incoming_ally_eta_min[slot]:
                    incoming_ally_eta_min[slot] = eta
            elif f_owner == -1:
                incoming_neutral_ships[slot] += f_ships
            else:
                incoming_enemy_ships[slot] += f_ships
                if eta < incoming_enemy_eta_min[slot]:
                    incoming_enemy_eta_min[slot] = eta
            if eta < nearest_eta[slot]:
                nearest_eta[slot] = eta

    # Pass 1: pairwise distances (ally/enemy/neutral nearest, threat density).
    nearest_ally_dist = [DIAG] * MAX_PLANETS
    nearest_enemy_dist = [DIAG] * MAX_PLANETS
    nearest_neutral_dist = [DIAG] * MAX_PLANETS
    support_density = [0.0] * MAX_PLANETS  # ally ships within long radius
    threat_pressure_short = [0.0] * MAX_PLANETS  # enemy ships+fleets within short
    threat_pressure_long = [0.0] * MAX_PLANETS  # enemy ships+fleets within long
    for i in range(n):
        _, _, px_i, py_i, _, _, _ = raw_planets[i]
        for j in range(n):
            if i == j:
                continue
            _, owner_j, px_j, py_j, _, ships_j, _ = raw_planets[j]
            owner_jj = int(owner_j)
            d = math.hypot(float(px_i) - float(px_j), float(py_i) - float(py_j))
            if owner_jj == player:
                if d < nearest_ally_dist[i]:
                    nearest_ally_dist[i] = d
                if d <= NEIGHBOR_RADIUS_LONG:
                    support_density[i] += float(ships_j)
            elif owner_jj == -1:
                if d < nearest_neutral_dist[i]:
                    nearest_neutral_dist[i] = d
            else:
                if d < nearest_enemy_dist[i]:
                    nearest_enemy_dist[i] = d
                if d <= NEIGHBOR_RADIUS_SHORT:
                    threat_pressure_short[i] += float(ships_j)
                elif d <= NEIGHBOR_RADIUS_LONG:
                    threat_pressure_long[i] += float(ships_j)
        for fleet_row in raw_fleets:
            _, fowner, fx, fy, _, _, fships = fleet_row
            if int(fowner) == player or int(fowner) == -1:
                continue
            d = math.hypot(float(px_i) - float(fx), float(py_i) - float(fy))
            if d <= NEIGHBOR_RADIUS_SHORT:
                threat_pressure_short[i] += float(fships)
            elif d <= NEIGHBOR_RADIUS_LONG:
                threat_pressure_long[i] += float(fships)

    my_count = 0
    enemy_count = 0
    neutral_count = 0

    for slot in range(n):
        pid, owner, px, py, radius, ships, production = raw_planets[slot]
        owner_i = int(owner)
        ships_i = int(ships)
        production_i = int(production)
        is_mine = owner_i == player
        is_neutral = owner_i == -1
        is_enemy = (not is_mine) and (not is_neutral)
        is_comet = int(pid) in raw_comet_ids
        if is_mine:
            my_count += 1
        elif is_neutral:
            neutral_count += 1
        else:
            enemy_count += 1

        eta_norm = min(nearest_eta[slot], HORIZON_TURNS + 1.0) / (HORIZON_TURNS + 1.0)
        sun_dist = math.hypot(float(px) - SUN_X, float(py) - SUN_Y)
        is_static = 1.0 if (sun_dist + float(radius) >= ROTATION_LIMIT) else 0.0
        prod_per_ship = float(production_i) / float(max(1, ships_i))
        net_signed = (incoming_enemy_ships[slot] - incoming_ally_ships[slot]) / float(
            max(1, ships_i)
        )
        # Orbit prediction: 4 horizons -> 8 dims
        planet_obj = _build_planet_obj(list(raw_planets[slot]))
        orbit = _orbit_predictions(
            planet_obj, initial_by_id, ang_vel, raw_comets, raw_comet_ids
        )
        ally_eta_norm = min(incoming_ally_eta_min[slot], HORIZON_TURNS + 1.0) / (
            HORIZON_TURNS + 1.0
        )
        enemy_eta_norm = min(incoming_enemy_eta_min[slot], HORIZON_TURNS + 1.0) / (
            HORIZON_TURNS + 1.0
        )

        feats: list[float] = [
            float(px) / BOARD_SIZE,
            float(py) / BOARD_SIZE,
            float(radius) / 5.0,
            math.log1p(max(0, ships_i)),
            math.log1p(max(0, production_i)),
            1.0 if is_mine else 0.0,
            1.0 if is_enemy else 0.0,
            1.0 if is_neutral else 0.0,
            1.0 if is_comet else 0.0,
            math.log1p(incoming_enemy_ships[slot])
            - math.log1p(incoming_ally_ships[slot]),
            eta_norm,
            sun_dist / DIAG,
            is_static,
            min(prod_per_ship, 5.0) / 5.0,
            nearest_enemy_dist[slot] / DIAG,
            math.log1p(support_density[slot]) / LOG_NORM_DENOM,
            math.log1p(threat_pressure_short[slot]) / LOG_NORM_DENOM,
            max(-3.0, min(3.0, net_signed)) / 3.0,
            # ----- Phase 1 additions -----
            nearest_ally_dist[slot] / DIAG,
            nearest_neutral_dist[slot] / DIAG,
            orbit[0][0],
            orbit[0][1],
            orbit[1][0],
            orbit[1][1],
            orbit[2][0],
            orbit[2][1],
            orbit[3][0],
            orbit[3][1],
            ally_eta_norm,
            enemy_eta_norm,
            math.log1p(incoming_ally_ships[slot]) / LOG_NORM_DENOM,
            math.log1p(incoming_enemy_ships[slot]) / LOG_NORM_DENOM,
            math.log1p(threat_pressure_long[slot]) / LOG_NORM_DENOM,
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

    total_planets = max(1, n)
    total_ships = my_total_ships + enemy_total_ships + neutral_total_ships
    total_prod = my_total_prod + enemy_total_prod
    phase_mid = 1.0 if 100 <= step < 300 else 0.0
    phase_late = 1.0 if step >= 300 else 0.0
    score_diff = math.log1p(my_total_ships) - math.log1p(enemy_total_ships)
    next_comet_eta_norm = float(_next_comet_eta(step)) / 100.0
    global_feats = torch.tensor(
        [
            float(step) / 500.0,
            ang_vel * 10.0,
            math.log1p(my_total_ships),
            math.log1p(enemy_total_ships),
            math.log1p(neutral_total_ships),
            math.log1p(my_total_prod) - math.log1p(enemy_total_prod),
            float(my_count) / float(total_planets),
            float(enemy_count) / float(total_planets),
            1.0 if _comet_active(step) else 0.0,
            phase_mid,
            phase_late,
            # ----- Phase 1 additions -----
            min(1.0, next_comet_eta_norm),
            (my_total_ships / total_ships) if total_ships > 0 else 0.0,
            (enemy_total_ships / total_ships) if total_ships > 0 else 0.0,
            (my_total_prod / total_prod) if total_prod > 0 else 0.0,
            max(-3.0, min(3.0, score_diff)) / 3.0,
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
