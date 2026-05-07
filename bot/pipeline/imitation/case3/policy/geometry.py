# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Aim/intercept geometry for imitation/case3.

Independent copy of the subset of pipeline/rulebase/case1/baseline/core/{geometry,
physics,config}.py needed to compute angles for fired fleets. Keeping this
self-contained guarantees imitation/case3 has no runtime dependency on rulebase.
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
SAFE_INTERCEPT_HALF_STEP: bool = True


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


def _predict_target_fractional(
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
    pos_lo = _predict_target(
        target, max(0, lo), initial_by_id, ang_vel, comets, comet_ids
    )
    if pos_lo is None:
        return None
    if frac <= 1e-9:
        return pos_lo
    pos_hi = _predict_target(target, hi, initial_by_id, ang_vel, comets, comet_ids)
    if pos_hi is None:
        return pos_lo
    return (
        pos_lo[0] + (pos_hi[0] - pos_lo[0]) * frac,
        pos_lo[1] + (pos_hi[1] - pos_lo[1]) * frac,
    )


def _iter_candidate_turns(max_turns: int) -> list[float]:
    if SAFE_INTERCEPT_HALF_STEP:
        return [i / 2.0 for i in range(2, int(max_turns * 2) + 1)]
    return [float(t) for t in range(1, max_turns + 1)]


def _first_engine_hit_turn(
    src: Planet,
    target: Planet,
    angle: float,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
    turn_lo: int,
    turn_hi: int,
) -> int | None:
    """Replay Orbit Wars' two sweep checks and return the first hit turn."""
    if turn_hi < turn_lo:
        return None
    speed = fleet_speed(max(1, ships))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    start_x, start_y = _launch_point(src.x, src.y, src.radius, angle)
    start = max(1, turn_lo)

    def fleet_at(t: int) -> tuple[float, float]:
        return start_x + cos_a * speed * t, start_y + sin_a * speed * t

    fx_prev, fy_prev = fleet_at(start - 1)
    for t in range(start, turn_hi + 1):
        fx, fy = fleet_at(t)
        planet_prev = _predict_target(
            target, t - 1, initial_by_id, ang_vel, comets, comet_ids
        )
        if planet_prev is not None:
            d = _point_to_segment_distance(
                planet_prev[0], planet_prev[1], fx_prev, fy_prev, fx, fy
            )
            if d < target.radius:
                return t
        planet_now = _predict_target(
            target, t, initial_by_id, ang_vel, comets, comet_ids
        )
        if planet_prev is not None and planet_now is not None:
            d = _point_to_segment_distance(
                fx, fy, planet_prev[0], planet_prev[1], planet_now[0], planet_now[1]
            )
            if d < target.radius:
                return t
        fx_prev, fy_prev = fx, fy
    return None


_HIT_SEARCH_WINDOW: int = 4


def _hit_turn_for_target_position(
    src: Planet,
    target: Planet,
    aim_pos: tuple[float, float],
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
    max_turns: int,
) -> tuple[float, int] | None:
    safe = safe_angle_and_distance(
        src.x, src.y, src.radius, aim_pos[0], aim_pos[1], target.radius
    )
    if safe is None:
        return None
    angle, _ = safe
    est = estimate_arrival(
        src.x, src.y, src.radius, aim_pos[0], aim_pos[1], target.radius, ships
    )
    if est is None:
        return None
    est_turn = est[1]
    hit = _first_engine_hit_turn(
        src,
        target,
        angle,
        ships,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        max(1, est_turn - _HIT_SEARCH_WINDOW),
        min(max_turns, est_turn + _HIT_SEARCH_WINDOW),
    )
    if hit is None:
        return None
    return angle, hit


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
    best_score: tuple[int, float] | None = None
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns <= 0:
        return None

    for candidate_turns in _iter_candidate_turns(max_turns):
        pos = _predict_target_fractional(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            continue
        result = _hit_turn_for_target_position(
            src,
            target,
            pos,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
            max_turns,
        )
        if result is None:
            continue
        angle, hit_turn = result
        actual_pos = _predict_target(
            target, hit_turn, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is None:
            continue
        score = (hit_turn, abs(hit_turn - candidate_turns))
        if best is None or best_score is None or score < best_score:
            best_score = score
            best = (angle, hit_turn, actual_pos[0], actual_pos[1])
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
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns <= 0:
        return None

    direct = _hit_turn_for_target_position(
        src,
        target,
        (target.x, target.y),
        ships,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        max_turns,
    )
    if direct is not None:
        angle, hit_turn = direct
        actual_pos = _predict_target(
            target, hit_turn, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is not None:
            return angle, hit_turn, actual_pos[0], actual_pos[1]

    est = estimate_arrival(
        src.x, src.y, src.radius, target.x, target.y, target.radius, ships
    )
    if est is None:
        return _search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
    _, turns_guess = est
    for _ in range(5):
        pos = _predict_target(
            target, turns_guess, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            break
        attempt = _hit_turn_for_target_position(
            src,
            target,
            pos,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
            max_turns,
        )
        if attempt is not None:
            angle, hit_turn = attempt
            actual_pos = _predict_target(
                target, hit_turn, initial_by_id, ang_vel, comets, comet_ids
            )
            if actual_pos is not None:
                return angle, hit_turn, actual_pos[0], actual_pos[1]
        next_est = estimate_arrival(
            src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships
        )
        if next_est is None:
            break
        new_turns = next_est[1]
        if new_turns == turns_guess:
            break
        turns_guess = new_turns

    return _search_safe_intercept(
        src, target, ships, initial_by_id, ang_vel, comets, comet_ids
    )
