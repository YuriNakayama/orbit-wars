"""JAX-native port of baseline/core/physics.py (pure-prediction layer).

Faithful to case1's formulas (NOTE: case1 fleet_speed is log-based with
MAX_SPEED=6.0 — distinct from the lite port's sqrt formula; using the wrong
speed is a known source of action divergence). Comet prediction is host-resolved
(ragged paths) and handled separately; this module covers the static/rotating
planet path + estimate_arrival, all jit/vmap-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry_jax import dist, safe_angle_and_distance

# Mirror baseline/core/config.py
CENTER_X: float = 50.0
CENTER_Y: float = 50.0
MAX_SPEED: float = 6.0
ROTATION_LIMIT: float = 50.0
INTERCEPT_TOLERANCE: int = 1
HORIZON_AIM: int = 110


def fleet_speed(ships: jax.Array) -> jax.Array:
    """case1 log speed: 1 if ships<=1 else 1+(MAX_SPEED-1)*clip(log_ratio,0,1)**1.5."""
    ships_f = jnp.asarray(ships).astype(jnp.float_)
    ratio = jnp.log(jnp.maximum(ships_f, 1.0)) / jnp.log(1000.0)
    ratio = jnp.clip(ratio, 0.0, 1.0)
    speed = 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)
    return jnp.where(ships_f <= 1.0, 1.0, speed)


def is_static_planet(px: jax.Array, py: jax.Array, radius: jax.Array) -> jax.Array:
    r = dist(px, py, jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y))
    return r + radius >= ROTATION_LIMIT


def predict_planet_position(
    cur_x: jax.Array,
    cur_y: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_radius: jax.Array,
    angular_velocity: jax.Array,
    turns: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Rotating planets orbit; static (r+radius>=ROTATION_LIMIT) stay put.

    Mirrors Python predict_planet_position. Caller supplies the initial-planet
    position/radius (the dict lookup is resolved upstream). If no initial entry
    existed, Python returns (cur_x, cur_y); caller passes init==cur in that case.
    """
    r = dist(init_x, init_y, jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y))
    is_static = (r + init_radius) >= ROTATION_LIMIT
    cur_ang = jnp.arctan2(cur_y - CENTER_Y, cur_x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    rot_x = CENTER_X + r * jnp.cos(new_ang)
    rot_y = CENTER_Y + r * jnp.sin(new_ang)
    return jnp.where(is_static, cur_x, rot_x), jnp.where(is_static, cur_y, rot_y)


def estimate_arrival(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (angle, turns, valid). valid=False mirrors Python `None`.

    turns = max(1, ceil(total_d / fleet_speed(max(1, ships)))).
    """
    angle, total_d, safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    speed = fleet_speed(jnp.maximum(jnp.asarray(ships), 1))
    turns = jnp.maximum(1, jnp.ceil(total_d / speed)).astype(jnp.int32)
    return angle, turns, safe
