"""Post-aim trajectory safety filters (rulebase/case1 copy).

Mirrors `bot/src/utils/trajectory_safety.py` and the imitation case5 copy
(`pipeline/imitation/case5/policy/safety.py`). Duplicated here so the case
remains self-contained for Kaggle submission packaging.

Hooked into WorldModel.plan_shot — failing any filter causes plan_shot to
return None instead of an unsafe (angle, turns, ix, iy) tuple.

Bugs addressed:
  Bug 1 sun consumption — is_trajectory_sun_safe samples integer turns.
  Bug 2 planet miss — intercept_holds_within_tolerance verifies the target
        stays within target.radius * miss_radius_factor of the predicted
        intercept across ±INTERCEPT_TOLERANCE turns.
  Bug 3 comet handling — fleet_crosses_other_comet detects collision with
        non-target comets; target_reachable_before_comet_expiry rejects
        comet targets that despawn before arrival.
"""

from __future__ import annotations

import math
from typing import Any

from .config import (
    CENTER_X,
    CENTER_Y,
    INTERCEPT_TOLERANCE,
    ROTATION_LIMIT,
    SUN_R,
    SUN_SAFETY,
)
from .physics import comet_remaining_life, fleet_speed
from .types import Planet

COMET_APPEARANCE_TURNS: tuple[int, ...] = (50, 150, 250, 350, 450)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def is_trajectory_sun_safe(
    launch_x: float,
    launch_y: float,
    angle: float,
    turns: int,
    ships: int,
    safety: float = SUN_SAFETY,
) -> bool:
    if turns < 0:
        return False
    speed = fleet_speed(max(1, ships))
    dx = math.cos(angle)
    dy = math.sin(angle)
    threshold = SUN_R + safety
    for t in range(turns + 1):
        px = launch_x + dx * speed * t
        py = launch_y + dy * speed * t
        if _dist(px, py, CENTER_X, CENTER_Y) < threshold:
            return False
    return True


def _predict_planet_position(
    target: Planet,
    initial: Planet | None,
    angular_velocity: float,
    turns: int,
) -> tuple[float, float]:
    if initial is None:
        return target.x, target.y
    r = _dist(initial.x, initial.y, CENTER_X, CENTER_Y)
    if r + initial.radius >= ROTATION_LIMIT:
        return target.x, target.y
    cur_ang = math.atan2(target.y - CENTER_Y, target.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


def _predict_comet_position(
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


def _comet_radius(comets: list[dict[str, Any]], default: float = 2.0) -> float:
    for group in comets:
        r = group.get("radius")
        if r is not None:
            return float(r)
    return default


def intercept_holds_within_tolerance(
    target: Planet,
    predicted_turns: int,
    predicted_pos: tuple[float, float],
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
    miss_radius_factor: float = 0.7,
    tolerance: int = INTERCEPT_TOLERANCE,
) -> bool:
    px, py = predicted_pos
    keep_within = max(1e-3, float(target.radius) * float(miss_radius_factor))
    initial = initial_by_id.get(target.id)
    is_comet = target.id in comet_ids
    for delta in range(-tolerance, tolerance + 1):
        t = predicted_turns + delta
        if t < 0:
            continue
        if is_comet:
            pos = _predict_comet_position(target.id, comets, t)
            if pos is None:
                return False
        else:
            pos = _predict_planet_position(target, initial, ang_vel, t)
        if _dist(pos[0], pos[1], px, py) > keep_within:
            return False
    return True


def fleet_crosses_other_comet(
    launch_x: float,
    launch_y: float,
    angle: float,
    turns: int,
    ships: int,
    current_step: int,
    comets: list[dict[str, Any]],
    exclude_planet_id: int,
    safety: float = 1.0,
) -> bool:
    _ = current_step
    if turns < 0:
        return False
    speed = fleet_speed(max(1, ships))
    dx = math.cos(angle)
    dy = math.sin(angle)
    comet_r = _comet_radius(comets)
    threshold = comet_r + safety
    for group in comets:
        pids = list(group.get("planet_ids") or [])
        for cid in pids:
            if cid == exclude_planet_id:
                continue
            for t in range(turns + 1):
                cpos = _predict_comet_position(cid, comets, t)
                if cpos is None:
                    continue
                fx = launch_x + dx * speed * t
                fy = launch_y + dy * speed * t
                if _dist(fx, fy, cpos[0], cpos[1]) < threshold:
                    return True
    return False


def target_reachable_before_comet_expiry(
    target_id: int,
    predicted_turns: int,
    comets: list[dict[str, Any]],
    safety_margin: int = 1,
) -> bool:
    is_comet = False
    for group in comets:
        if target_id in (group.get("planet_ids") or []):
            is_comet = True
            break
    if not is_comet:
        return True
    return comet_remaining_life(target_id, comets) > predicted_turns + safety_margin


def comet_appearance_imminent(
    current_step: int,
    predicted_turns: int,
    appearance_turns: tuple[int, ...] = COMET_APPEARANCE_TURNS,
    look_ahead_buffer: int = 5,
) -> int | None:
    end = current_step + predicted_turns + look_ahead_buffer
    for t in appearance_turns:
        if current_step <= t <= end:
            return t
    return None
