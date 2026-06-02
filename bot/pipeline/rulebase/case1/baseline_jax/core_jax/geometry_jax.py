"""JAX-native port of baseline/core/geometry.py — vectorized, jit/vmap-friendly.

Each function matches the Python original within float32 tol (x64 exact). All
inputs are JAX arrays (scalars or broadcastable); branches are expressed as
`jnp.where`, and `safe_angle_and_distance` returns an explicit `valid` flag
instead of `None`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Mirror baseline/core/config.py (replicated to avoid importing the Python sim).
CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
SUN_SAFETY: float = 1.5
LAUNCH_CLEARANCE: float = 0.1


def dist(ax: jax.Array, ay: jax.Array, bx: jax.Array, by: jax.Array) -> jax.Array:
    return jnp.hypot(ax - bx, ay - by)


def point_to_segment_distance(
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
    safe_len = jnp.where(seg_len_sq <= 1e-9, 1.0, seg_len_sq)
    t = ((px - x1) * dx + (py - y1) * dy) / safe_len
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    seg_dist = dist(px, py, proj_x, proj_y)
    return jnp.where(seg_len_sq <= 1e-9, dist(px, py, x1, y1), seg_dist)


def segment_hits_sun(
    x1: jax.Array,
    y1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    safety: float = SUN_SAFETY,
) -> jax.Array:
    d = point_to_segment_distance(
        jnp.asarray(CENTER_X), jnp.asarray(CENTER_Y), x1, y1, x2, y2
    )
    return d < SUN_R + safety


def launch_point(
    sx: jax.Array, sy: jax.Array, sr: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + jnp.cos(angle) * clearance, sy + jnp.sin(angle) * clearance


def actual_path_geometry(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    angle = jnp.arctan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = jnp.maximum(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + jnp.cos(angle) * hit_distance
    end_y = start_y + jnp.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    tr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (angle, hit_distance, valid). valid=False mirrors Python `None`."""
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr
    )
    valid = ~segment_hits_sun(start_x, start_y, end_x, end_y)
    return angle, hit_distance, valid
