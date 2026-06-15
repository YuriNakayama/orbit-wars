"""Parity tests for the JAX orbital-intercept aim port.

Compares `intercept_angle_jax` (the rotating-planet fixed-point loop ported
into pure JAX for the training rollout) against the Python reference
`aim_with_prediction` from `policy/geometry.py`.

Scope (mirrors the port's v1 deviation):
  - Only the ROTATING-PLANET fixed-point branch is ported. Comet paths and the
    brute-force `_search_safe_intercept` fallback are out of scope; the JAX
    function falls back to the naive `atan2(target_now - source_now)` for
    static / sun-unsafe targets. So the parity cases here are chosen so the
    Python path takes the fixed-point loop (the straight `estimate_arrival`
    succeeds and the loop converges), NOT the search fallback.

Tolerance: 1e-2 rad — wider than the usual 1e-4 because the fixed point is run
for a fixed 5 iterations in float32 (the Python loop has an early-exit
convergence check, so for a slowly-converging case the two can differ by a
fraction of a degree). 1e-2 rad (~0.57°) comfortably covers that gap while
still proving the lead-angle (not the naive bearing) is being computed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.reinforce.case8.policy.aim_jax import intercept_angle_jax
from pipeline.reinforce.case8.policy.geometry import (
    CENTER_X,
    CENTER_Y,
    Planet,
    aim_with_prediction,
)

PARITY_TOL = 1e-2


def _orbiting_planet(angle_deg: float, radius_orbit: float, planet_id: int) -> Planet:
    """Build a planet on the orbit circle of radius `radius_orbit` about CENTER."""
    a = np.deg2rad(angle_deg)
    x = CENTER_X + radius_orbit * np.cos(a)
    y = CENTER_Y + radius_orbit * np.sin(a)
    # planet radius small enough that r + radius < ROTATION_LIMIT (=50) -> rotates.
    return Planet(
        id=planet_id,
        owner=-1,
        x=float(x),
        y=float(y),
        radius=2.0,
        ships=5,
        production=1,
    )


def _python_fixed_point_angle(
    src: Planet, target: Planet, ships: int, ang_vel: float
) -> float | None:
    """Reference angle from the fixed-point branch of aim_with_prediction.

    `initial_by_id` maps the target id to itself (the target's *current* pos is
    its initial pos in these single-step parity cases — the rollout threads the
    real initial_xy, but for parity we set ang_vel * turns rotation off a known
    base). comets empty so the planet branch is taken.
    """
    result = aim_with_prediction(
        src,
        target,
        ships,
        {target.id: target},
        ang_vel,
        [],
        set(),
    )
    if result is None:
        return None
    return float(result[0])


# Cases: (src, target, ships, ang_vel). Targets clearly orbit (small radius,
# inside ROTATION_LIMIT) and the straight shot is sun-safe, so the Python path
# runs the fixed-point loop AND hits its converged-return branch (verified the
# `aim_with_prediction` result is not the search fallback). Each has a real
# lead (0.06–0.28 rad) vs the naive bearing.
_CASES = [
    (Planet(0, 0, 80.0, 50.0, 2.0, 30, 1), _orbiting_planet(0.0, 20.0, 1), 30, 0.05),
    (Planet(0, 0, 80.0, 50.0, 2.0, 30, 1), _orbiting_planet(0.0, 25.0, 1), 30, 0.04),
    (Planet(0, 0, 85.0, 55.0, 2.0, 30, 1), _orbiting_planet(30.0, 22.0, 1), 30, 0.03),
    (Planet(0, 0, 75.0, 75.0, 2.0, 30, 1), _orbiting_planet(60.0, 24.0, 1), 30, 0.04),
    (Planet(0, 0, 50.0, 90.0, 2.0, 30, 1), _orbiting_planet(120.0, 26.0, 1), 30, 0.03),
    (Planet(0, 0, 90.0, 55.0, 2.0, 30, 1), _orbiting_planet(330.0, 23.0, 1), 30, 0.05),
]


def _jax_angle(src: Planet, target: Planet, ships: int, ang_vel: float) -> float:
    init_xy = jnp.asarray([target.x, target.y], dtype=jnp.float32)
    return float(
        intercept_angle_jax(
            src_xy=jnp.asarray([src.x, src.y], dtype=jnp.float32),
            tgt_now_xy=jnp.asarray([target.x, target.y], dtype=jnp.float32),
            tgt_init_xy=init_xy,
            tgt_is_rotating=jnp.asarray(True),
            ships=jnp.asarray(ships, dtype=jnp.int32),
            angular_velocity=jnp.asarray(ang_vel, dtype=jnp.float32),
        )
    )


@pytest.mark.parametrize("idx", range(len(_CASES)))
def test_rotating_intercept_matches_python_fixed_point(idx: int) -> None:
    src, target, ships, ang_vel = _CASES[idx]
    py = _python_fixed_point_angle(src, target, ships, ang_vel)
    assert py is not None, "case must hit the fixed-point branch (got search/None)"
    jx = _jax_angle(src, target, ships, ang_vel)
    # Compare on the circle to be robust to ±2π wrap.
    diff = np.arctan2(np.sin(jx - py), np.cos(jx - py))
    assert abs(diff) < PARITY_TOL, f"case {idx}: jax={jx} python={py} diff={diff}"


def test_lead_angle_is_not_the_naive_bearing() -> None:
    """At least one orbiting case must differ from the naive atan2 bearing.

    This proves the intercept lead is actually applied (else the port is a
    no-op rename of the existing naive path).
    """
    src, target, ships, ang_vel = _CASES[0]
    naive = np.arctan2(target.y - src.y, target.x - src.x)
    jx = _jax_angle(src, target, ships, ang_vel)
    diff = np.arctan2(np.sin(jx - naive), np.cos(jx - naive))
    assert abs(diff) > 1e-2, "expected a non-trivial lead vs the naive bearing"


def test_static_planet_equals_naive_atan2() -> None:
    src = Planet(0, 0, 80.0, 50.0, 2.0, 20, 1)
    # A static (non-rotating) target: tgt_is_rotating=False -> no lead.
    tgt_xy = (30.0, 40.0)
    jx = float(
        intercept_angle_jax(
            src_xy=jnp.asarray([src.x, src.y], dtype=jnp.float32),
            tgt_now_xy=jnp.asarray(tgt_xy, dtype=jnp.float32),
            tgt_init_xy=jnp.asarray(tgt_xy, dtype=jnp.float32),
            tgt_is_rotating=jnp.asarray(False),
            ships=jnp.asarray(20, dtype=jnp.int32),
            angular_velocity=jnp.asarray(0.05, dtype=jnp.float32),
        )
    )
    naive = float(np.arctan2(tgt_xy[1] - src.y, tgt_xy[0] - src.x))
    assert abs(jx - naive) < 1e-5


def test_vmap_is_jittable_over_a_batch() -> None:
    n = len(_CASES)
    src_xy = jnp.asarray([[c[0].x, c[0].y] for c in _CASES], dtype=jnp.float32)
    tgt_xy = jnp.asarray([[c[1].x, c[1].y] for c in _CASES], dtype=jnp.float32)
    ships = jnp.asarray([c[2] for c in _CASES], dtype=jnp.int32)
    ang_vel = jnp.asarray([c[3] for c in _CASES], dtype=jnp.float32)
    rotating = jnp.ones((n,), dtype=jnp.bool_)

    batched = jax.jit(jax.vmap(intercept_angle_jax, in_axes=(0, 0, 0, 0, 0, 0)))
    out = batched(src_xy, tgt_xy, tgt_xy, rotating, ships, ang_vel)
    out.block_until_ready()
    assert out.shape == (n,)
    # Each batched entry matches its scalar parity reference.
    for i, (src, target, sh, av) in enumerate(_CASES):
        py = _python_fixed_point_angle(src, target, sh, av)
        assert py is not None
        diff = np.arctan2(np.sin(float(out[i]) - py), np.cos(float(out[i]) - py))
        assert abs(diff) < PARITY_TOL, f"vmap case {i}: diff={diff}"
