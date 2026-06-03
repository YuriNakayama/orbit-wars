"""x64 parity: physics_jax (pure-prediction layer) vs baseline/core/physics.py.

Covers fleet_speed, is_static_planet, predict_planet_position, estimate_arrival.
The aim_with_prediction refinement loop is ported/tested separately.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline.core import physics as ppy
from pipeline.rulebase.case1.baseline.core.types import Planet
from pipeline.rulebase.case1.baseline_jax.core_jax import physics_jax as pjax

RTOL = 1e-9
ATOL = 1e-9


@pytest.mark.parametrize("ships", [0, 1, 2, 5, 10, 37, 100, 500, 999, 1000, 2000])
def test_fleet_speed_parity(ships: int) -> None:
    ref = ppy.fleet_speed(ships)
    got = float(pjax.fleet_speed(jnp.asarray(ships)))
    assert np.isclose(ref, got, rtol=RTOL, atol=ATOL), f"ships={ships}: {ref} vs {got}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_is_static_planet_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(128):
        x, y = rng.uniform(0, 100, 2)
        radius = rng.uniform(1, 5)
        p = Planet(id=0, owner=-1, x=x, y=y, radius=radius, ships=0, production=1)
        ref = ppy.is_static_planet(p)
        got = bool(
            pjax.is_static_planet(jnp.asarray(x), jnp.asarray(y), jnp.asarray(radius))
        )
        assert ref == got


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_predict_planet_position_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ang_vel = 0.035
    for _ in range(128):
        ix, iy = rng.uniform(0, 100, 2)
        irad = rng.uniform(1, 5)
        cx, cy = rng.uniform(0, 100, 2)
        turns = int(rng.integers(0, 50))
        cur = Planet(id=3, owner=-1, x=cx, y=cy, radius=irad, ships=0, production=1)
        init = Planet(id=3, owner=-1, x=ix, y=iy, radius=irad, ships=0, production=1)
        ref = ppy.predict_planet_position(cur, {3: init}, ang_vel, turns)
        gx, gy = pjax.predict_planet_position(
            jnp.asarray(cx),
            jnp.asarray(cy),
            jnp.asarray(ix),
            jnp.asarray(iy),
            jnp.asarray(irad),
            jnp.asarray(ang_vel),
            jnp.asarray(turns),
        )
        assert np.isclose(ref[0], float(gx), rtol=RTOL, atol=ATOL)
        assert np.isclose(ref[1], float(gy), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_estimate_arrival_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(256):
        sx, sy, tx, ty = rng.uniform(0, 100, 4)
        sr, tr = rng.uniform(1, 4, 2)
        ships = int(rng.integers(1, 300))
        ref = ppy.estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
        angle, turns, valid = pjax.estimate_arrival(
            *(jnp.asarray(v) for v in (sx, sy, sr, tx, ty, tr, ships))
        )
        if ref is None:
            assert not bool(valid)
        else:
            assert bool(valid)
            assert np.isclose(ref[0], float(angle), rtol=RTOL, atol=ATOL)
            assert int(ref[1]) == int(turns), f"turns {ref[1]} vs {int(turns)}"
