"""Vectorized planet orbit prediction (vendor parity).

Vendor formula (`simulator/python/orbit_wars_vendor/orbit_wars.py` and
`bot/src/jax_env/step.py::_advance_planet_xy`):

    For each rotating planet (orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT):
        dx = initial_y - CENTER
        dy = initial_x - CENTER
        r          = sqrt(dx^2 + dy^2)
        init_angle = atan2(dy, dx)
        angle(t)   = init_angle + angular_velocity * t  (t = game_step)
        y(t)       = CENTER + r * cos(angle(t))
        x(t)       = CENTER + r * sin(angle(t))

    Static planets (orbital_radius + r >= ROTATION_RADIUS_LIMIT):
        stay at initial_xy.

    Comet planets (replaced via planet_initial_xy=(-99, -99) sentinel after
    a comet wave activates): we treat them as static-from-current-step for
    horizon prediction purposes (correct enough for capture planning — the
    actual jump per tick is hard to predict cheaply).

The function is pure-JAX, vmap-able over planet axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from orbit_wars_jax.constants import CENTER, ROTATION_RADIUS_LIMIT


def predict_planet_xy(
    initial_xy: jax.Array,  # (P, 2) — planet_initial_xy, layout (y, x) per vendor
    planet_radius: jax.Array,  # (P,)
    angular_velocity: jax.Array,  # () scalar
    game_step: jax.Array,  # () scalar int (current step at prediction time)
    dt: jax.Array,  # () scalar int (lookahead in turns; can be float for sub-step)
) -> jax.Array:
    """Return (P, 2) predicted (y, x) at `game_step + dt`.

    Static planets ignore `dt`. Rotating planets advance their angle by
    `angular_velocity * (game_step + dt)` from `init_angle`.

    Comet sentinel `(-99, -99)` is treated as static (no useful prediction).
    """
    init_y = initial_xy[:, 0]
    init_x = initial_xy[:, 1]
    dx = init_y - CENTER  # vendor mapping: dx uses y-coord (row 0)
    dy = init_x - CENTER  # vendor mapping: dy uses x-coord (row 1)
    r = jnp.sqrt(dx * dx + dy * dy + 1e-12)
    init_angle = jnp.arctan2(dy, dx)

    # Determine if planet rotates: orbital_radius + r_planet < ROTATION_RADIUS_LIMIT.
    is_rotating = (r + planet_radius) < jnp.float32(ROTATION_RADIUS_LIMIT)
    # Comet sentinel (initial_xy = (-99, -99)) → treat as static.
    is_comet_sentinel = (initial_xy[:, 0] < jnp.float32(-50.0)) & (
        initial_xy[:, 1] < jnp.float32(-50.0)
    )
    is_rotating = is_rotating & ~is_comet_sentinel

    t = (game_step.astype(jnp.float32)) + jnp.float32(dt)
    current_angle = init_angle + angular_velocity * t
    rot_y = CENTER + r * jnp.cos(current_angle)
    rot_x = CENTER + r * jnp.sin(current_angle)
    rot_xy = jnp.stack([rot_y, rot_x], axis=-1)  # (P, 2)

    return jnp.where(is_rotating[:, None], rot_xy, initial_xy)


def predict_planet_xy_at(
    state_initial_xy: jax.Array,  # (P, 2)
    state_radius: jax.Array,  # (P,)
    angular_velocity: jax.Array,  # ()
    target_step: jax.Array,  # () scalar (absolute game step)
) -> jax.Array:
    """Convenience: predict (y, x) at an absolute `target_step`."""
    return predict_planet_xy(
        state_initial_xy,
        state_radius,
        angular_velocity,
        game_step=jnp.int32(0),
        dt=target_step.astype(jnp.float32),
    )


__all__ = ["predict_planet_xy", "predict_planet_xy_at"]
