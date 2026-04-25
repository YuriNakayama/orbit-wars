"""Pure geometry / physics helpers (sun-safe routing, arrival prediction).

Verbatim port from the LB1224 notebook by Roman Tamrazov (Apache License 2.0).
"""

from __future__ import annotations

import math

from .config import (
    CENTER_X,
    CENTER_Y,
    HORIZON,
    INTERCEPT_TOLERANCE,
    LAUNCH_CLEARANCE,
    MAX_SPEED,
    ROTATION_LIMIT,
    ROUTE_SEARCH_HORIZON,
    SUN_R,
    SUN_SAFETY,
)


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (min(ratio, 1.0) ** 1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return (
        point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety
    )


def launch_point(sx, sy, sr, angle):
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_d = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    return (
        angle,
        start_x,
        start_y,
        start_x + math.cos(angle) * hit_d,
        start_y + math.sin(angle) * hit_d,
        hit_d,
    )


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, sx2, sy2, ex, ey, hit_d = actual_path_geometry(sx, sy, sr, tx, ty, tr)
    if segment_hits_sun(sx2, sy2, ex, ey):
        return None
    return angle, hit_d


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    ang = (
        math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X) + angular_velocity * turns
    )
    return CENTER_X + r * math.cos(ang), CENTER_Y + r * math.sin(ang)


def predict_comet_position(planet_id, comets, turns):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
        if idx >= len(paths):
            return None
        fi = pi + int(turns)
        path = paths[idx]
        return (path[fi][0], path[fi][1]) if 0 <= fi < len(path) else None
    return None


def comet_remaining_life(planet_id, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - pi)
    return 0


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    return dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius < ROTATION_LIMIT


def search_safe_intercept(
    src, target, ships, initial_by_id, ang_vel, comets, comet_ids
):
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    for ct in range(1, max_turns + 1):
        pos = predict_target_position(
            target, ct, initial_by_id, ang_vel, comets, comet_ids
        )
        if pos is None:
            continue
        est = estimate_arrival(
            src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships
        )
        if est is None:
            continue
        _, turns = est
        if abs(turns - ct) > INTERCEPT_TOLERANCE:
            continue
        at = max(turns, ct)
        ap = predict_target_position(
            target, at, initial_by_id, ang_vel, comets, comet_ids
        )
        if ap is None:
            continue
        confirm = estimate_arrival(
            src.x, src.y, src.radius, ap[0], ap[1], target.radius, ships
        )
        if confirm is None:
            continue
        delta = abs(confirm[1] - at)
        if delta > INTERCEPT_TOLERANCE:
            continue
        sc = (delta, confirm[1], ct)
        if best is None or sc < best_score:
            best_score = sc
            best = (confirm[0], confirm[1], ap[0], ap[1])
    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    est = estimate_arrival(
        src.x, src.y, src.radius, target.x, target.y, target.radius, ships
    )
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
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
            if not target_can_move(target, initial_by_id, comet_ids):
                return None
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
