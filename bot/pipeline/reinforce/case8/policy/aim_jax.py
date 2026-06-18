# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Pure-JAX orbital-intercept aim for the case8 training rollout.

Ports the ROTATING-PLANET fixed-point intercept loop of
`policy/geometry.py::aim_with_prediction` (with its helpers `estimate_arrival`,
`safe_angle_and_distance`, `fleet_speed`, `predict_planet_position`) into a
jit/vmap-friendly function so the training rollout aims at where an orbiting
planet WILL BE at fleet arrival — matching the eval/submission decoder, which
fixes the confirmed train/eval mismatch (the rollout used a straight bearing to
the target's *current* position, systematically missing orbiting planets).

INTENTIONAL DEVIATIONS FROM THE PYTHON REFERENCE (v1 scope):
  - Comet-path interception (`predict_comet_position` + ragged path lookup) is
    NOT ported. Comets fall back to the naive `atan2(target_now - source_now)`
    bearing. The caller decides comet-vs-planet; this function only handles the
    rotating/static planet case.
  - The brute-force `_search_safe_intercept` fallback is NOT ported. When the
    straight shot would hit the sun (`safe_angle_and_distance` -> None in the
    Python ref), this function falls back to the naive bearing instead of
    searching. A mask flag (`sun_safe`) is returned so the caller can choose.
  - The fixed-point loop runs a FIXED 5 iterations (matching the Python
    `for _ in range(5)`) with no early-exit convergence break, so it stays a
    single static trace. The Python loop early-exits on convergence; for a
    slowly-converging case the two can differ by a fraction of a degree (hence
    the 1e-2 rad parity tolerance, vs the usual 1e-4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    MAX_SPEED,
    SUN_R,
    SUN_SAFETY,
)

_INTERCEPT_ITERS = 5
_CENTER = jnp.asarray([CENTER_X, CENTER_Y], dtype=jnp.float32)
_DEFAULT_RADIUS = jnp.float32(2.0)  # typical planet radius (parity default)


def _fleet_speed(ships: jax.Array) -> jax.Array:
    """JAX port of geometry.fleet_speed (ships>=1 assumed by the caller clamp)."""
    ships_f = jnp.maximum(1.0, ships.astype(jnp.float32))
    ratio = jnp.log(ships_f) / jnp.log(jnp.float32(1000.0))
    ratio = jnp.clip(ratio, 0.0, 1.0)
    fast = 1.0 + (MAX_SPEED - 1.0) * ratio**1.5
    return jnp.where(ships_f <= 1.0, jnp.float32(1.0), fast)


def _point_to_segment_distance(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
    """Distance from point p to segment a-b (JAX port of geometry helper)."""
    d = b - a
    seg_len_sq = jnp.sum(d * d)
    t = jnp.where(
        seg_len_sq <= 1e-9,
        jnp.float32(0.0),
        jnp.clip(jnp.sum((p - a) * d) / jnp.maximum(seg_len_sq, 1e-9), 0.0, 1.0),
    )
    closest = a + t * d
    return jnp.sqrt(jnp.sum((p - closest) ** 2))


def _estimate_arrival(
    src_xy: jax.Array,
    tgt_xy: jax.Array,
    src_radius: jax.Array,
    tgt_radius: jax.Array,
    speed: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX port of estimate_arrival (-> angle, turns, sun_safe).

    Mirrors safe_angle_and_distance: launch from src + (src_radius + clearance)
    along the bearing, fly hit_distance = max(0, dist - (sr+clearance) - tr),
    and flag the straight shot sun-safe iff the launch->impact segment clears
    the sun. `speed` is precomputed (ships are constant across the loop).
    """
    delta = tgt_xy - src_xy
    angle = jnp.arctan2(delta[1], delta[0])
    direction = jnp.asarray([jnp.cos(angle), jnp.sin(angle)])
    clearance = src_radius + LAUNCH_CLEARANCE
    start = src_xy + direction * clearance
    full_dist = jnp.sqrt(jnp.sum(delta**2))
    hit_distance = jnp.maximum(0.0, full_dist - clearance - tgt_radius)
    end = start + direction * hit_distance
    sun_dist = _point_to_segment_distance(_CENTER, start, end)
    sun_safe = sun_dist >= (SUN_R + SUN_SAFETY)
    turns = jnp.maximum(1, jnp.ceil(hit_distance / speed).astype(jnp.int32))
    return angle, turns, sun_safe


def _predict_rotating(
    tgt_now_xy: jax.Array,
    tgt_init_xy: jax.Array,
    angular_velocity: jax.Array,
    turns: jax.Array,
) -> jax.Array:
    """JAX port of predict_planet_position for a rotating planet.

    Radius comes from the INITIAL position (matching the Python ref, which
    reads r from initial_by_id), the current angle from the CURRENT position,
    rotated by angular_velocity * turns about CENTER.
    """
    r = jnp.sqrt(jnp.sum((tgt_init_xy - _CENTER) ** 2))
    rel = tgt_now_xy - _CENTER
    cur_ang = jnp.arctan2(rel[1], rel[0])
    new_ang = cur_ang + angular_velocity * turns.astype(jnp.float32)
    return _CENTER + r * jnp.asarray([jnp.cos(new_ang), jnp.sin(new_ang)])


def intercept_angle_jax(
    src_xy: jax.Array,  # (2,) float32 — source planet center
    tgt_now_xy: jax.Array,  # (2,) float32 — target current position
    tgt_init_xy: jax.Array,  # (2,) float32 — target initial position (for radius)
    tgt_is_rotating: jax.Array,  # () bool — whether the target orbits
    ships: jax.Array,  # () int32 — fleet ship count (>=1 after caller clamp)
    angular_velocity: jax.Array,  # () float32
    src_radius: jax.Array = _DEFAULT_RADIUS,  # () float32
    tgt_radius: jax.Array = _DEFAULT_RADIUS,  # () float32
) -> jax.Array:
    """Aim angle that intercepts a rotating target (fixed-point loop).

    Returns the launch bearing. For a static target (`tgt_is_rotating` False)
    this equals the naive `atan2(target_now - source_now)` (no lead). For a
    rotating target it runs the 5-iteration fixed point: predict the target at
    the current ETA, recompute angle+ETA, repeat — exactly mirroring the
    Python `aim_with_prediction` fixed-point branch (within float32 + the
    fixed iteration count).

    Sun-unsafe / non-rotating cases collapse to the naive bearing (the
    `_search_safe_intercept` fallback is out of scope — see module docstring).
    """
    speed = _fleet_speed(ships)
    angle0, turns0, sun_safe0 = _estimate_arrival(
        src_xy, tgt_now_xy, src_radius, tgt_radius, speed
    )
    naive_angle = jnp.arctan2(tgt_now_xy[1] - src_xy[1], tgt_now_xy[0] - src_xy[0])

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        angle, turns, ok, _last_pos = carry
        pos = _predict_rotating(tgt_now_xy, tgt_init_xy, angular_velocity, turns)
        n_angle, n_turns, n_safe = _estimate_arrival(
            src_xy, pos, src_radius, tgt_radius, speed
        )
        # Stay sun-safe only while every step's straight shot clears the sun.
        n_ok = ok & n_safe
        return (n_angle, n_turns, n_ok, pos), None

    init = (angle0, turns0, sun_safe0, tgt_now_xy)
    (final_angle, _turns, final_ok, _pos), _ = jax.lax.scan(
        body, init, None, length=_INTERCEPT_ITERS
    )

    use_intercept = tgt_is_rotating & sun_safe0 & final_ok
    return jnp.where(use_intercept, final_angle, naive_angle)


__all__ = ["intercept_angle_jax"]
