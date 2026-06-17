# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the vectorizable scalar helpers in `baseline/core/physics.py`.

Ported (pure, jit/vmap-friendly):

* `fleet_speed`              -> `fleet_speed_jax`
* `is_static_planet`         -> `is_static_planet_jax`
* `predict_planet_position`  -> `predict_planet_position_jax`
* `estimate_arrival`         -> `estimate_arrival_jax`
* `travel_time`              -> `travel_time_jax`

Deferred (data-dependent Python control flow that does not vectorize cleanly):

* `predict_comet_position`, `comet_remaining_life` — iterate over a variable
  list of comet-group dicts and do `list.index` / ragged path lookups. These
  are host-side bookkeeping, not numeric hot paths; left in Python.
* `predict_target_position(_fractional)` — dispatches on a `comet_ids` set.
* `search_safe_intercept`, `aim_with_prediction` — variable-length candidate
  search with early-exit refinement loops and `None` short-circuits. Vectorizing
  these is a follow-up (scan over a fixed `MAX_CANDIDATES` grid with masks); the
  numeric primitives they call (`estimate_arrival_jax`, `dist_jax`) are ported
  so that work is unblocked.

Deviation from source: functions that return `None` in Python (`estimate_arrival`,
`travel_time` via sentinel `10**9`) instead return a `valid` boolean alongside
the value. `predict_planet_position` takes the resolved initial planet
x/y/radius as arrays rather than a `dict[int, Planet]` lookup (the lookup stays
on the host).

`turns` is computed with `ceil`; the Python source uses `int(math.ceil(...))`,
which we mirror with `jnp.ceil(...).astype(int32)`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    MAX_SPEED,
    ROTATION_LIMIT,
)
from .geometry_jax import dist_jax, safe_angle_and_distance_jax

BIG_TURNS: int = 10**9


def fleet_speed_jax(ships: jax.Array) -> jax.Array:
    """Mirror of fleet_speed. ships<=1 -> 1.0, else log-ratio curve."""
    ships_f = ships.astype(jnp.float32)
    ratio = jnp.log(jnp.maximum(ships_f, 1.0)) / jnp.log(1000.0)
    ratio = jnp.clip(ratio, 0.0, 1.0)
    # Tuning: ratio**1.5 == ratio * sqrt(ratio). The explicit mul+sqrt fuses to
    # cheaper XLA ops than a generic `pow`, and stays bit-equivalent within the
    # float32 1e-4 parity band (verified by test_fleet_speed_parity). ratio is
    # clipped to [0, 1] so sqrt is always defined.
    curved = 1.0 + (MAX_SPEED - 1.0) * (ratio * jnp.sqrt(ratio))
    return jnp.where(ships <= 1, 1.0, curved)


def is_static_planet_jax(px: jax.Array, py: jax.Array, radius: jax.Array) -> jax.Array:
    r = dist_jax(px, py, jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y))
    return (r + radius) >= ROTATION_LIMIT


def predict_planet_position_jax(
    cur_x: jax.Array,
    cur_y: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_radius: jax.Array,
    angular_velocity: jax.Array,
    turns: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Predict an orbiting planet's future position.

    Static planets (r + radius >= ROTATION_LIMIT) keep their current position,
    matching the Python early return. The `init is None` Python branch (unknown
    planet) is a host-side concern: the caller passes the current position as
    `init_*` to reproduce the `return planet.x, planet.y` fallback.
    """
    r = dist_jax(init_x, init_y, jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y))
    static = (r + init_radius) >= ROTATION_LIMIT
    cur_ang = jnp.arctan2(cur_y - CENTER_Y, cur_x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    rot_x = CENTER_X + r * jnp.cos(new_ang)
    rot_y = CENTER_Y + r * jnp.sin(new_ang)
    return jnp.where(static, cur_x, rot_x), jnp.where(static, cur_y, rot_y)


def estimate_arrival_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (angle, turns, valid). valid==False mirrors Python None."""
    angle, total_d, valid = safe_angle_and_distance_jax(sx, sy, sr, tx, ty, tr)
    speed = fleet_speed_jax(jnp.maximum(ships, 1))
    raw_turns = jnp.ceil(total_d / speed).astype(jnp.int32)
    turns = jnp.maximum(1, raw_turns)
    return angle, turns, valid


def travel_time_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
) -> jax.Array:
    """Mirror of travel_time: 10**9 when the path is sun-blocked, else turns."""
    _, turns, valid = estimate_arrival_jax(sx, sy, sr, tx, ty, tr, ships)
    return jnp.where(valid, turns, BIG_TURNS)
