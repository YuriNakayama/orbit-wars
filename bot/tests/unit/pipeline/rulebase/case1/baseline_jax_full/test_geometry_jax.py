"""Vendor parity for predict_planet_xy (orbit math)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from orbit_wars_jax.constants import CENTER, ROTATION_RADIUS_LIMIT

from pipeline.rulebase.case1.baseline_jax_full.geometry_jax import predict_planet_xy


def _vendor_predict(
    init_y: float,
    init_x: float,
    radius: float,
    angular_velocity: float,
    game_step: int,
    dt: int,
) -> tuple[float, float]:
    """Vendor formula reference (matches bot/src/jax_env/step.py)."""
    dx = init_y - CENTER
    dy = init_x - CENTER
    r = math.sqrt(dx * dx + dy * dy + 1e-12)
    is_rotating = (r + radius) < ROTATION_RADIUS_LIMIT
    if not is_rotating:
        return init_y, init_x
    init_angle = math.atan2(dy, dx)
    cur_angle = init_angle + angular_velocity * (game_step + dt)
    y = CENTER + r * math.cos(cur_angle)
    x = CENTER + r * math.sin(cur_angle)
    return y, x


def test_predict_matches_vendor_for_rotating_planets() -> None:
    # 5 rotating planets at various orbital radii / angles.
    planets = [
        (55.0, 50.0, 2.0),  # init_y=55, init_x=50, r=5  → rotates
        (50.0, 60.0, 1.5),  # init_y=50, init_x=60, r=10 → rotates
        (70.0, 50.0, 3.0),  # r=20, rotates (20+3 < 50)
        (40.0, 60.0, 2.5),  # r=sqrt(100+100)≈14, rotates
        (50.0, 30.0, 1.0),  # r=20, rotates
    ]
    av = 0.03  # vendor uniform(0.025, 0.05)
    game_step = 7
    init_xy = jnp.array([[p[0], p[1]] for p in planets], dtype=jnp.float32)
    radius = jnp.array([p[2] for p in planets], dtype=jnp.float32)

    for dt in (0, 1, 5, 10, 20):
        out = predict_planet_xy(
            init_xy,
            radius,
            jnp.float32(av),
            game_step=jnp.int32(game_step),
            dt=jnp.float32(dt),
        )
        out_np = np.asarray(out)
        for i, (iy, ix, r) in enumerate(planets):
            exp_y, exp_x = _vendor_predict(iy, ix, r, av, game_step, dt)
            assert abs(out_np[i, 0] - exp_y) < 1e-3, (
                f"planet {i} dt={dt} y mismatch {out_np[i, 0]} vs {exp_y}"
            )
            assert abs(out_np[i, 1] - exp_x) < 1e-3


def test_static_planet_does_not_rotate() -> None:
    # Large orbital radius → static.
    # init at (90, 50): dx=40, dy=0, r=40, +planet_radius 15 → 55 >= 50 → static.
    init_xy = jnp.array([[90.0, 50.0]], dtype=jnp.float32)
    radius = jnp.array([15.0], dtype=jnp.float32)
    av = jnp.float32(0.04)
    for dt in (0, 1, 5, 10, 50):
        out = predict_planet_xy(
            init_xy, radius, av, game_step=jnp.int32(0), dt=jnp.float32(dt)
        )
        out_np = np.asarray(out)
        assert abs(out_np[0, 0] - 90.0) < 1e-5
        assert abs(out_np[0, 1] - 50.0) < 1e-5


def test_comet_sentinel_static() -> None:
    # Sentinel = (-99, -99) → treated static even if r < limit numerically.
    init_xy = jnp.array([[-99.0, -99.0]], dtype=jnp.float32)
    radius = jnp.array([3.0], dtype=jnp.float32)
    av = jnp.float32(0.04)
    out = predict_planet_xy(
        init_xy, radius, av, game_step=jnp.int32(5), dt=jnp.float32(10)
    )
    out_np = np.asarray(out)
    assert abs(out_np[0, 0] - (-99.0)) < 1e-5
    assert abs(out_np[0, 1] - (-99.0)) < 1e-5
