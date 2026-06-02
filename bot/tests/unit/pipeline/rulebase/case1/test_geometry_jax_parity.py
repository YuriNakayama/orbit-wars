"""x64 parity: geometry_jax vs baseline/core/geometry.py.

Bottom layer of the faithful JAX port. With jax_enable_x64 the JAX port must
match the Python original to float64 tolerance on every randomized input; this
isolates algorithm-port bugs from float32 precision drift (handled later at the
agent level). See docs/plans/rulebase-to-jax/07 (原則 2, 4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline.core import geometry as gpy
from pipeline.rulebase.case1.baseline_jax.core_jax import geometry_jax as gjax

jax.config.update("jax_enable_x64", True)

RTOL = 1e-9
ATOL = 1e-9


def _rng_points(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(n, 8)).astype(np.float64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dist_parity(seed: int) -> None:
    pts = _rng_points(seed, 64)
    for row in pts:
        ax, ay, bx, by = row[:4]
        py = gpy.dist(ax, ay, bx, by)
        jx = float(
            gjax.dist(
                jnp.asarray(ax), jnp.asarray(ay), jnp.asarray(bx), jnp.asarray(by)
            )
        )
        assert np.isclose(py, jx, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_point_to_segment_distance_parity(seed: int) -> None:
    pts = _rng_points(seed, 64)
    for row in pts:
        px, py_, x1, y1, x2, y2 = row[:6]
        ref = gpy.point_to_segment_distance(px, py_, x1, y1, x2, y2)
        got = float(
            gjax.point_to_segment_distance(
                *(jnp.asarray(v) for v in (px, py_, x1, y1, x2, y2))
            )
        )
        assert np.isclose(ref, got, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segment_hits_sun_parity(seed: int) -> None:
    pts = _rng_points(seed, 128)
    for row in pts:
        x1, y1, x2, y2 = row[:4]
        ref = gpy.segment_hits_sun(x1, y1, x2, y2)
        got = bool(gjax.segment_hits_sun(*(jnp.asarray(v) for v in (x1, y1, x2, y2))))
        assert ref == got


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_safe_angle_and_distance_parity(seed: int) -> None:
    pts = _rng_points(seed, 128)
    radii = np.random.default_rng(seed + 99).uniform(1.0, 4.0, size=(len(pts), 2))
    for row, (sr, tr) in zip(pts, radii, strict=True):
        sx, sy, tx, ty = row[:4]
        ref = gpy.safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
        angle, hit, valid = gjax.safe_angle_and_distance(
            *(jnp.asarray(v) for v in (sx, sy, sr, tx, ty, tr))
        )
        if ref is None:
            assert not bool(valid)
        else:
            assert bool(valid)
            assert np.isclose(ref[0], float(angle), rtol=RTOL, atol=ATOL)
            assert np.isclose(ref[1], float(hit), rtol=RTOL, atol=ATOL)


def test_geometry_jax_is_vmappable() -> None:
    """The port must vmap (the whole point of the JAX rewrite)."""
    xs = jnp.linspace(0.0, 100.0, 16)
    f = jax.vmap(lambda x: gjax.dist(x, x, 50.0, 50.0))
    out = f(xs)
    assert out.shape == (16,)
