"""JAX port of plan_shot's post-aim guards (baseline/core/safety.py, non-comet).

WorldModel.plan_shot wraps aim_with_prediction with 4 guards; the JAX agent
must apply the same to avoid firing un-shootable trajectories (over-fire root
cause). This module ports the two that matter for non-comet targets:
  - is_trajectory_sun_safe: the fleet path stays clear of the sun for all turns.
  - intercept_holds_within_tolerance: the moving target stays within
    0.7*radius of the predicted impact across +/-INTERCEPT_TOLERANCE turns.
The two comet guards are vacuously True for non-comet targets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics_jax import fleet_speed, predict_planet_position

Arr = jax.Array

CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
SUN_SAFETY: float = 1.5
LAUNCH_CLEARANCE: float = 0.1
INTERCEPT_TOLERANCE: int = 1
HORIZON: int = 110
MISS_RADIUS_FACTOR: float = 0.7

_SUN_STEPS = jnp.arange(0, HORIZON + 1, dtype=jnp.int32)


def is_trajectory_sun_safe(
    launch_x: Arr,
    launch_y: Arr,
    angle: Arr,
    turns: Arr,
    ships: Arr,
    safety: float = SUN_SAFETY,
) -> Arr:
    """Fleet path clear of sun for t in 0..turns. turns<0 → unsafe."""
    speed = fleet_speed(jnp.maximum(ships, 1))
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)
    threshold = SUN_R + safety

    def at_step(t: Arr) -> Arr:
        px = launch_x + dx * speed * t
        py = launch_y + dy * speed * t
        d = jnp.hypot(px - CENTER_X, py - CENTER_Y)
        # only steps t<=turns matter; others are masked safe
        return (t <= turns) & (d < threshold)

    hits = jax.vmap(at_step)(_SUN_STEPS)
    return (turns >= 0) & ~jnp.any(hits)


def intercept_holds_within_tolerance(
    cur_x: Arr,
    cur_y: Arr,
    init_x: Arr,
    init_y: Arr,
    init_r: Arr,
    target_r: Arr,
    predicted_turns: Arr,
    pred_x: Arr,
    pred_y: Arr,
    ang_vel: Arr,
    tolerance: int = INTERCEPT_TOLERANCE,
) -> Arr:
    """Target stays within 0.7*radius of (pred_x,pred_y) over +/-tolerance turns.

    Non-comet: uses predict_planet_position. Steps with t<0 are skipped (Python
    `continue`).
    """
    keep_within = jnp.maximum(1e-3, target_r * MISS_RADIUS_FACTOR)
    deltas = jnp.arange(-tolerance, tolerance + 1, dtype=jnp.int32)

    def at_delta(delta: Arr) -> Arr:
        t = predicted_turns + delta
        ox, oy = predict_planet_position(
            cur_x, cur_y, init_x, init_y, init_r, ang_vel, t
        )
        too_far = jnp.hypot(ox - pred_x, oy - pred_y) > keep_within
        # t<0 is skipped (never a failure); t>=0 with too_far fails
        return (t >= 0) & too_far

    fails = jax.vmap(at_delta)(deltas)
    return ~jnp.any(fails)


def plan_shot_ok(
    sx: Arr,
    sy: Arr,
    sr: Arr,
    cur_x: Arr,
    cur_y: Arr,
    init_x: Arr,
    init_y: Arr,
    init_r: Arr,
    target_r: Arr,
    angle: Arr,
    turns: Arr,
    ix: Arr,
    iy: Arr,
    ships: Arr,
    ang_vel: Arr,
    aim_ok: Arr,
) -> Arr:
    """plan_shot validity for a NON-COMET target: aim_ok AND both guards pass."""
    clearance = sr + LAUNCH_CLEARANCE
    launch_x = sx + jnp.cos(angle) * clearance
    launch_y = sy + jnp.sin(angle) * clearance
    sun_ok = is_trajectory_sun_safe(launch_x, launch_y, angle, turns, ships)
    tol_ok = intercept_holds_within_tolerance(
        cur_x, cur_y, init_x, init_y, init_r, target_r, turns, ix, iy, ang_vel
    )
    return aim_ok & sun_ok & tol_ok
