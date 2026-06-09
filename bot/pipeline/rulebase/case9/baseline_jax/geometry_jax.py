# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of `baseline/core/geometry.py` (2D geometry helpers).

Every function here is a pure, elementwise-vectorizable translation of its
Python counterpart and is `jax.jit` / `jax.vmap` friendly: no Python control
flow on traced values (the `seg_len_sq <= 1e-9` branch becomes `jnp.where`).

Intentional deviation from the Python source:

* `safe_angle_and_distance` returns `None` when the launch path crosses the
  sun. Under JAX's fixed-shape contract we cannot return `None`, so
  `safe_angle_and_distance_jax` returns an `(angle, hit_distance, valid)` tuple
  where `valid` is a boolean array; `valid == False` is equivalent to the
  Python `None`. The parity test maps the two representations.

All inputs are treated as float32 scalars/arrays. Constants are imported from
the Python `core.config` so the two stay in lockstep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    SUN_R,
    SUN_SAFETY,
)


def dist_jax(ax: jax.Array, ay: jax.Array, bx: jax.Array, by: jax.Array) -> jax.Array:
    """math.hypot(ax - bx, ay - by)."""
    return jnp.hypot(ax - bx, ay - by)


def point_to_segment_distance_jax(
    px: jax.Array,
    py: jax.Array,
    x1: jax.Array,
    y1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
) -> jax.Array:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    # Degenerate segment -> distance to the endpoint (matches the Python guard).
    degenerate = seg_len_sq <= 1e-9
    safe_len = jnp.where(degenerate, 1.0, seg_len_sq)
    t = ((px - x1) * dx + (py - y1) * dy) / safe_len
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    seg_dist = dist_jax(px, py, proj_x, proj_y)
    endpoint_dist = dist_jax(px, py, x1, y1)
    return jnp.where(degenerate, endpoint_dist, seg_dist)


def segment_hits_sun_jax(
    x1: jax.Array,
    y1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    safety: float = SUN_SAFETY,
) -> jax.Array:
    # The Python source ignores its `safety` parameter and always uses the
    # module-level SUN_SAFETY constant. We replicate that exactly.
    d = point_to_segment_distance_jax(
        jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y), x1, y1, x2, y2
    )
    return d < (SUN_R + SUN_SAFETY)


def launch_point_jax(
    sx: jax.Array, sy: jax.Array, sr: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + jnp.cos(angle) * clearance, sy + jnp.sin(angle) * clearance


def actual_path_geometry_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    angle = jnp.arctan2(ty - sy, tx - sx)
    start_x, start_y = launch_point_jax(sx, sy, sr, angle)
    hit_distance = jnp.maximum(
        0.0, dist_jax(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr
    )
    end_x = start_x + jnp.cos(angle) * hit_distance
    end_y = start_y + jnp.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (angle, hit_distance, valid). valid==False mirrors Python None."""
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry_jax(
        sx, sy, sr, tx, ty, tr
    )
    hits = segment_hits_sun_jax(start_x, start_y, end_x, end_y)
    valid = jnp.logical_not(hits)
    return angle, hit_distance, valid
