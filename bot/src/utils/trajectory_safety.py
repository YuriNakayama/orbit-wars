"""Post-model safety filters for fleet trajectories.

Each helper is a pure predicate (or descriptor) that callers — decoder.py for
imitation cases, world_model.plan_shot for rulebase cases — apply after
aim_with_prediction returns a candidate (angle, turns) pair. Failing any one
filter means the candidate action should be dropped.

Bugs addressed:
  Bug 1 (sun consumption): is_trajectory_sun_safe samples the fleet position
        at every integer turn and rejects if any is within SUN_R + safety.
        Stronger than the existing segment-only check.
  Bug 2 (planet miss): intercept_holds_within_tolerance re-predicts the target
        position at predicted_turns ± INTERCEPT_TOLERANCE and confirms the
        target's centre stays within target.radius * miss_radius_factor of
        the predicted intercept point.
  Bug 3 (comet handling):
    - target_reachable_before_comet_expiry: comet target lifetime > arrival.
    - fleet_crosses_other_comet: trajectory does not collide with any
      non-target comet during flight.
    - comet_appearance_imminent: lookahead for upcoming spawn turns
      (50/150/250/350/450) so the decoder can defer fragile actions.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from .fleet_kinematics import fleet_speed
from .orbit_constants import (
    CENTER_X,
    CENTER_Y,
    COMET_APPEARANCE_TURNS,
    INTERCEPT_TOLERANCE,
    ROTATION_LIMIT,
    SUN_R,
    SUN_SAFETY,
)


class Planet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


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
    """Reject if the fleet's position at any integer turn is within sun safety.

    Stronger than a segment-only check: catches grazing trajectories whose
    minimum distance to the sun centre dips inside `SUN_R + safety` at an
    intermediate sampled turn.
    """
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


def comet_remaining_life(
    planet_id: int, comets: list[dict[str, Any]]
) -> int:
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
    """Confirm target stays within `target.radius * miss_radius_factor`
    of `predicted_pos` for every turn in [predicted - tol, predicted + tol].
    """
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
                # comet expired within window - intercept does not hold.
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
    """Sample fleet position at each integer turn 0..turns, and for every
    comet planet (excluding `exclude_planet_id`) check whether the fleet is
    within `comet.radius + safety` of the comet's predicted position at
    that turn. Return True on first collision detected.

    `current_step` is unused at the moment (`comets[].path_index` already
    encodes the current pointer) but kept for forward compatibility with
    appearance-aware logic.
    """
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
    """If `target_id` is a comet, require remaining life > predicted_turns + margin.
    Non-comet targets are always reachable from this filter's perspective.
    """
    # Determine if target is among any comet group.
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
    """Return the appearance turn that falls in
    [current_step, current_step + predicted_turns + look_ahead_buffer], else None.
    """
    end = current_step + predicted_turns + look_ahead_buffer
    for t in appearance_turns:
        if current_step <= t <= end:
            return t
    return None
