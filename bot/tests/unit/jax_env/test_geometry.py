"""Parity tests: jax_env.geometry vs simulator/python/orbit_wars_vendor.

Cover degenerate cases (zero motion, tangent grazing, both moving) so any
re-implementation drift surfaces immediately.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Vendor sim lives under simulator/python; add it to sys.path.
_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "python"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.geometry import (  # noqa: E402
    distance,
    point_to_segment_distance,
    swept_pair_hit,
)
from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    distance as vendor_distance,
)
from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    point_to_segment_distance as vendor_p2s,
)
from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    swept_pair_hit as vendor_swept,
)


def _jx(*xs: float) -> jnp.ndarray:
    return jnp.asarray(xs, dtype=jnp.float32)


def test_distance_basic() -> None:
    a = _jx(0.0, 0.0)
    b = _jx(3.0, 4.0)
    assert math.isclose(float(distance(a, b)), 5.0, abs_tol=1e-6)
    assert math.isclose(
        float(distance(a, b)), vendor_distance((0.0, 0.0), (3.0, 4.0)), abs_tol=1e-6
    )


@pytest.mark.parametrize("seed", range(10))
def test_point_to_segment_distance_random(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        p = (rng.uniform(0, 100), rng.uniform(0, 100))
        v = (rng.uniform(0, 100), rng.uniform(0, 100))
        w = (rng.uniform(0, 100), rng.uniform(0, 100))
        expected = vendor_p2s(p, v, w)
        got = float(point_to_segment_distance(_jx(*p), _jx(*v), _jx(*w)))
        # float32 vs float64; relative tolerance covers larger magnitudes.
        assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-5), (
            p,
            v,
            w,
            got,
            expected,
        )


def test_point_to_segment_distance_degenerate() -> None:
    """v == w: vendor returns distance(p, v)."""
    p = (10.0, 10.0)
    v = (5.0, 5.0)
    w = (5.0, 5.0)
    got = float(point_to_segment_distance(_jx(*p), _jx(*v), _jx(*w)))
    expected = vendor_p2s(p, v, w)
    assert math.isclose(got, expected, abs_tol=1e-6)


def test_swept_pair_hit_no_motion_inside_radius() -> None:
    """Both segments are points; A == P0 within r -> hit."""
    A = B = _jx(50.0, 50.0)
    P0 = P1 = _jx(50.5, 50.0)
    r = 1.0
    assert bool(swept_pair_hit(A, B, P0, P1, jnp.float32(r)))
    assert vendor_swept((50.0, 50.0), (50.0, 50.0), (50.5, 50.0), (50.5, 50.0), r)


def test_swept_pair_hit_no_motion_outside_radius() -> None:
    A = B = _jx(0.0, 0.0)
    P0 = P1 = _jx(50.0, 0.0)
    r = 1.0
    assert not bool(swept_pair_hit(A, B, P0, P1, jnp.float32(r)))
    assert not vendor_swept((0.0, 0.0), (0.0, 0.0), (50.0, 0.0), (50.0, 0.0), r)


def test_swept_pair_hit_passing_through() -> None:
    """A moves through P0 region."""
    A = _jx(0.0, 0.0)
    B = _jx(10.0, 0.0)
    P0 = P1 = _jx(5.0, 0.5)  # static, very close to mid path
    r = 1.0
    assert bool(swept_pair_hit(A, B, P0, P1, jnp.float32(r)))
    assert vendor_swept((0.0, 0.0), (10.0, 0.0), (5.0, 0.5), (5.0, 0.5), r)


def test_swept_pair_hit_misses_after_t1() -> None:
    """Segments would intersect if extended, but only after t > 1."""
    A = _jx(0.0, 0.0)
    B = _jx(1.0, 0.0)
    P0 = P1 = _jx(10.0, 0.0)
    r = 0.5
    assert not bool(swept_pair_hit(A, B, P0, P1, jnp.float32(r)))
    assert not vendor_swept((0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (10.0, 0.0), r)


@pytest.mark.parametrize("seed", range(20))
def test_swept_pair_hit_random_parity(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(100):
        A = (rng.uniform(0, 100), rng.uniform(0, 100))
        B = (A[0] + rng.uniform(-5, 5), A[1] + rng.uniform(-5, 5))
        P0 = (rng.uniform(0, 100), rng.uniform(0, 100))
        P1 = (P0[0] + rng.uniform(-3, 3), P0[1] + rng.uniform(-3, 3))
        r = rng.uniform(0.5, 5.0)
        expected = vendor_swept(A, B, P0, P1, r)
        got = bool(swept_pair_hit(_jx(*A), _jx(*B), _jx(*P0), _jx(*P1), jnp.float32(r)))
        assert got == expected, (A, B, P0, P1, r, got, expected)
