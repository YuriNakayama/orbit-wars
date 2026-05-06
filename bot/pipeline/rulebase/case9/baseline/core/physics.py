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


def _point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Mirror of kaggle_environments.envs.orbit_wars.point_to_segment_distance."""
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return math.hypot(px - projx, py - projy)


_LAUNCH_OFFSET: float = 0.1  # engine starts fleets at planet surface + 0.1


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
    """Replay fleet over [turn_lo, turn_hi]; return first hit turn or None.

    Mirrors kaggle_environments.envs.orbit_wars per-turn order:
      1. fleet moves from pos[t-1] to pos[t]
         collision: seg_dist(planet_at_(t-1), fleet_seg) < planet.radius
      2. planet moves from pos[t-1] to pos[t]
         sweep: seg_dist(fleet_at_t, planet_seg) < planet.radius

    Fleet start point is `src.center + (src.radius + 0.1) * direction`,
    not `src.center` — short-range shots otherwise predict the wrong hit turn.
    """
    if turn_hi < turn_lo:
        return None
    speed = fleet_speed(max(1, ships))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    offset = src.radius + _LAUNCH_OFFSET
    start_x = src.x + cos_a * offset
    start_y = src.y + sin_a * offset

    start = max(1, turn_lo)

    def fleet_at(t: int) -> tuple[float, float]:
        return (start_x + cos_a * speed * t, start_y + sin_a * speed * t)

    def target_at(t: int) -> tuple[float, float] | None:
        return predict_target_position(
            target, t, initial_by_id, ang_vel, comets, comet_ids
        )

    fx_prev, fy_prev = fleet_at(start - 1)
    for t in range(start, turn_hi + 1):
        fx, fy = fleet_at(t)

        # Step 1: fleet moves; engine compares against planet at (t-1).
        planet_prev = target_at(t - 1)
        if planet_prev is not None:
            d = _point_to_segment_distance(
                planet_prev[0], planet_prev[1], fx_prev, fy_prev, fx, fy
            )
            if d < target.radius:
                return t

        # Step 2: planet moves; engine sweep checks fleet (already at t)
        # against planet's (t-1)→t segment.
        planet_now = target_at(t)
        if planet_prev is not None and planet_now is not None:
            d = _point_to_segment_distance(
                fx, fy, planet_prev[0], planet_prev[1], planet_now[0], planet_now[1]
            )
            if d < target.radius:
                return t

        fx_prev, fy_prev = fx, fy
    return None


# Search half-window around the geometric estimate. Lead-aim error rarely
# exceeds a few turns for well-conditioned shots, so a tight window keeps
# the per-aim cost O(window) instead of O(HORIZON).
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
    """Try aiming directly at aim_pos; return (angle, hit_turn) if engine hits.

    Uses the geometric arrival estimate to bracket the segment replay rather
    than walking the full HORIZON.
    """
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
    lo = max(1, est_turn - _HIT_SEARCH_WINDOW)
    hi = min(max_turns, est_turn + _HIT_SEARCH_WINDOW)
    hit = _first_engine_hit_turn(
        src,
        target,
        angle,
        ships,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        lo,
        hi,
    )
    if hit is None:
        return None
    return angle, hit


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
    best_score: tuple[int, float] | None = None
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns <= 0:
        return None

    for candidate_turns in _iter_candidate_turns(max_turns):
        pos = predict_target_position_fractional(
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
        actual_pos = predict_target_position(
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

    # First try: aim directly at target's current position. Fast path that
    # works for static / slowly-moving targets and seeds the lead-aim loop.
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
        actual_pos = predict_target_position(
            target, hit_turn, initial_by_id, ang_vel, comets, comet_ids
        )
        if actual_pos is not None:
            return angle, hit_turn, actual_pos[0], actual_pos[1]

    # Lead-aim refinement: estimate arrival turn, predict target position then,
    # aim there; iterate until the engine actually intercepts.
    est = estimate_arrival(
        src.x, src.y, src.radius, target.x, target.y, target.radius, ships
    )
    if est is None:
        return search_safe_intercept(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )

    _, turns_guess = est
    for _ in range(5):
        pos = predict_target_position(
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
            actual_pos = predict_target_position(
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

    return search_safe_intercept(
        src, target, ships, initial_by_id, ang_vel, comets, comet_ids
    )
