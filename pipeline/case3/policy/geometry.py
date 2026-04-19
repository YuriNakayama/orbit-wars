# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Aim/intercept geometry for case3.

Independent copy of the subset of pipeline/case1/baseline/core/{geometry,
physics,config}.py needed to compute angles for fired fleets. Keeping this
self-contained guarantees case3 has no runtime dependency on case1.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
SUN_SAFETY: float = 1.5
MAX_SPEED: float = 6.0
ROTATION_LIMIT: float = 50.0
HORIZON: int = 110
INTERCEPT_TOLERANCE: int = 1
LAUNCH_CLEARANCE: float = 0.1


class Planet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def _segment_hits_sun(
    x1: float, y1: float, x2: float, y2: float, safety: float = SUN_SAFETY
) -> bool:
    return (
        _point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety
    )


def _launch_point(sx: float, sy: float, sr: float, angle: float) -> tuple[float, float]:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def safe_angle_and_distance(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float
) -> tuple[float, float] | None:
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = _launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    if _segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


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
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


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


def _predict_target(
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


def _search_safe_intercept(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, int, float, float] | None:
    best: tuple[float, int, float, float] | None = None
    best_score: tuple[int, int, int] | None = None
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = _predict_target(
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
        actual_turns = max(turns, candidate_turns)
        actual_pos = _predict_target(
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
        score = (delta, confirm[1], candidate_turns)
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
        return _search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        pos = _predict_target(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(
            src.x, src.y, src.radius, ntx, nty, target.radius, ships
        )
        if next_est is None:
            return _search_safe_intercept(
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
        return _search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
    return final_est[0], final_est[1], tx, ty
