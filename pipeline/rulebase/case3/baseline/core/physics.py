# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Fleet speed, orbital prediction, comet tracking, and safe-intercept search."""

from __future__ import annotations

import math
from typing import Any

from .config import (
    CENTER_X,
    CENTER_Y,
    HORIZON,
    INTERCEPT_TOLERANCE,
    MAX_SPEED,
    ROTATION_LIMIT,
    SAFE_INTERCEPT_HALF_STEP,
)
from .geometry import dist, safe_angle_and_distance
from .types import Planet


def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return float(1.0 + (MAX_SPEED - 1.0) * (ratio**1.5))


def is_static_planet(planet: Planet) -> bool:
    r = dist(planet.x, planet.y, CENTER_X, CENTER_Y)
    return r + planet.radius >= ROTATION_LIMIT


def predict_planet_position(
    planet: Planet,
    initial_by_id: dict[int, Planet],
    angular_velocity: float,
    turns: int,
) -> tuple[float, float]:
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_comet_position(
    planet_id: int, comets: list[dict[str, Any]], turns: int
) -> tuple[float, float] | None:
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None


def comet_remaining_life(planet_id: int, comets: list[dict[str, Any]]) -> int:
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return int(max(0, len(paths[idx]) - path_index))
    return 0


def estimate_arrival(
    sx: float,
    sy: float,
    sr: float,
    tx: float,
    ty: float,
    tr: float,
    ships: int,
) -> tuple[float, int] | None:
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(
    sx: float,
    sy: float,
    sr: float,
    tx: float,
    ty: float,
    tr: float,
    ships: int,
) -> int:
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10**9
    return est[1]


def predict_target_position(
    target: Planet,
    turns: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, float] | None:
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def predict_target_position_fractional(
    target: Planet,
    turns: float,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, float] | None:
    lo = int(math.floor(turns))
    hi = lo + 1
    frac = turns - lo
    pos_lo = predict_target_position(
        target, max(0, lo), initial_by_id, ang_vel, comets, comet_ids
    )
    if pos_lo is None:
        return None
    if frac <= 1e-9:
        return pos_lo
    pos_hi = predict_target_position(
        target, hi, initial_by_id, ang_vel, comets, comet_ids
    )
    if pos_hi is None:
        return pos_lo
    return (
        pos_lo[0] + (pos_hi[0] - pos_lo[0]) * frac,
        pos_lo[1] + (pos_hi[1] - pos_lo[1]) * frac,
    )


def _iter_candidate_turns(max_turns: int) -> list[float]:
    if SAFE_INTERCEPT_HALF_STEP:
        half_steps = int(max_turns * 2)
        return [i / 2.0 for i in range(2, half_steps + 1)]
    return [float(t) for t in range(1, max_turns + 1)]


def search_safe_intercept(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, int, float, float] | None:
    best: tuple[float, int, float, float] | None = None
    best_score: tuple[float, int, float] | None = None
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in _iter_candidate_turns(max_turns):
        pos = predict_target_position_fractional(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            continue
        est = estimate_arrival(
            src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships
        )
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, int(math.ceil(candidate_turns)))
        actual_pos = predict_target_position(
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x,
            src.y,
            src.radius,
            actual_pos[0],
            actual_pos[1],
            target.radius,
            ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (float(delta), confirm[1], candidate_turns)
        if best is None or best_score is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_with_prediction(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, int, float, float] | None:
    est = estimate_arrival(
        src.x, src.y, src.radius, target.x, target.y, target.radius, ships
    )
    if est is None:
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        pos = predict_target_position(
            target, turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(
            src.x, src.y, src.radius, ntx, nty, target.radius, ships
        )
        if next_est is None:
            return search_safe_intercept(
                src, target, ships, initial_by_id, ang_vel, comets, comet_ids
            )
        if (
            abs(ntx - tx) < 0.3
            and abs(nty - ty) < 0.3
            and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE
        ):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
    return final_est[0], final_est[1], tx, ty
