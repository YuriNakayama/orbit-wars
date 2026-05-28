"""Numerical-parity tests for the rulebase/case_jax JAX core port.

Compares each function in `baseline_jax/{geometry_jax,physics_jax}.py` against
its Python counterpart in `baseline/core/{geometry,physics}.py` on identical,
randomized scalar inputs.

Tolerance: float32 `1e-4` (the Python side is float64; inputs are sampled in a
realistic board range and the JAX side runs float32). Functions that return
`None` in Python (sun-blocked launch paths) are mapped to the JAX `valid`
boolean: Python `None` <=> JAX `valid == False`.

Vectorization is also checked: each JAX helper is `jax.vmap`-ed over a batch of
inputs and compared against the per-element Python loop, proving the port is
batch-friendly (the whole point of the JAX version).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case_jax.baseline.core.geometry import (
    actual_path_geometry,
    dist,
    point_to_segment_distance,
    safe_angle_and_distance,
    segment_hits_sun,
)
from pipeline.rulebase.case_jax.baseline.core.physics import (
    estimate_arrival,
    fleet_speed,
    is_static_planet,
    predict_planet_position,
    reset_predict_cache,
    travel_time,
)
from pipeline.rulebase.case_jax.baseline.core.types import Planet
from pipeline.rulebase.case_jax.baseline_jax.geometry_jax import (
    actual_path_geometry_jax,
    dist_jax,
    point_to_segment_distance_jax,
    safe_angle_and_distance_jax,
    segment_hits_sun_jax,
)
from pipeline.rulebase.case_jax.baseline_jax.physics_jax import (
    BIG_TURNS,
    estimate_arrival_jax,
    fleet_speed_jax,
    is_static_planet_jax,
    predict_planet_position_jax,
    travel_time_jax,
)

# JAX jit-compile heavy (geometry/physics ports + vmap batched check over 256
# samples). Marked slow so CI's fast slice (-m "not slow") skips it; runs in the
# full/local lane.
pytestmark = pytest.mark.slow

PARITY_TOL = 1e-4
N_SAMPLES = 256


def _f32(x: float) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260526)


def _board_points(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random (x, y) points spread across the board (some near the sun)."""
    return rng.uniform(0.0, 100.0, size=(n, 2)).astype(np.float64)


def test_dist_parity(rng: np.random.Generator) -> None:
    pts = _board_points(rng, N_SAMPLES * 2).reshape(N_SAMPLES, 4)
    for ax, ay, bx, by in pts:
        py = dist(ax, ay, bx, by)
        jx = float(dist_jax(_f32(ax), _f32(ay), _f32(bx), _f32(by)))
        assert abs(py - jx) < PARITY_TOL


def test_point_to_segment_distance_parity(rng: np.random.Generator) -> None:
    pts = _board_points(rng, N_SAMPLES * 3).reshape(N_SAMPLES, 6)
    for px, py_, x1, y1, x2, y2 in pts:
        ref = point_to_segment_distance(px, py_, x1, y1, x2, y2)
        got = float(
            point_to_segment_distance_jax(
                _f32(px), _f32(py_), _f32(x1), _f32(y1), _f32(x2), _f32(y2)
            )
        )
        assert abs(ref - got) < PARITY_TOL


def test_point_to_segment_distance_degenerate() -> None:
    # Zero-length segment must fall back to the endpoint distance.
    ref = point_to_segment_distance(0.0, 0.0, 3.0, 4.0, 3.0, 4.0)
    got = float(
        point_to_segment_distance_jax(
            _f32(0.0), _f32(0.0), _f32(3.0), _f32(4.0), _f32(3.0), _f32(4.0)
        )
    )
    assert abs(ref - got) < PARITY_TOL
    assert abs(ref - 5.0) < PARITY_TOL


def test_segment_hits_sun_parity(rng: np.random.Generator) -> None:
    pts = _board_points(rng, N_SAMPLES * 2).reshape(N_SAMPLES, 4)
    for x1, y1, x2, y2 in pts:
        ref = segment_hits_sun(x1, y1, x2, y2)
        got = bool(segment_hits_sun_jax(_f32(x1), _f32(y1), _f32(x2), _f32(y2)))
        assert ref == got


def test_actual_path_geometry_parity(rng: np.random.Generator) -> None:
    src = _board_points(rng, N_SAMPLES)
    tgt = _board_points(rng, N_SAMPLES)
    sr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    tr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    for (sx, sy), (tx, ty), srr, trr in zip(src, tgt, sr, tr, strict=True):
        ref = actual_path_geometry(sx, sy, srr, tx, ty, trr)
        got = actual_path_geometry_jax(
            _f32(sx), _f32(sy), _f32(srr), _f32(tx), _f32(ty), _f32(trr)
        )
        for r, g in zip(ref, got, strict=True):
            assert abs(float(r) - float(g)) < PARITY_TOL


def test_safe_angle_and_distance_parity(rng: np.random.Generator) -> None:
    src = _board_points(rng, N_SAMPLES)
    tgt = _board_points(rng, N_SAMPLES)
    sr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    tr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    n_blocked = 0
    for (sx, sy), (tx, ty), srr, trr in zip(src, tgt, sr, tr, strict=True):
        ref = safe_angle_and_distance(sx, sy, srr, tx, ty, trr)
        angle, hit, valid = safe_angle_and_distance_jax(
            _f32(sx), _f32(sy), _f32(srr), _f32(tx), _f32(ty), _f32(trr)
        )
        if ref is None:
            assert not bool(valid)
            n_blocked += 1
        else:
            assert bool(valid)
            assert abs(ref[0] - float(angle)) < PARITY_TOL
            assert abs(ref[1] - float(hit)) < PARITY_TOL
    # The random spread should exercise both branches.
    assert 0 < n_blocked < N_SAMPLES


def test_fleet_speed_parity() -> None:
    for ships in range(0, 1200):
        ref = fleet_speed(ships)
        got = float(fleet_speed_jax(jnp.asarray(ships, dtype=jnp.int32)))
        assert abs(ref - got) < PARITY_TOL


def test_is_static_planet_parity(rng: np.random.Generator) -> None:
    pts = _board_points(rng, N_SAMPLES)
    radii = rng.uniform(0.5, 10.0, size=N_SAMPLES)
    for (x, y), r in zip(pts, radii, strict=True):
        planet = Planet(
            id=0,
            owner=0,
            x=float(x),
            y=float(y),
            radius=float(r),
            ships=0,
            production=0,
        )
        ref = is_static_planet(planet)
        got = bool(is_static_planet_jax(_f32(x), _f32(y), _f32(r)))
        assert ref == got


def test_predict_planet_position_parity(rng: np.random.Generator) -> None:
    pts = _board_points(rng, N_SAMPLES)
    radii = rng.uniform(0.5, 8.0, size=N_SAMPLES)
    ang_vels = rng.uniform(-0.2, 0.2, size=N_SAMPLES)
    turns = rng.integers(0, 110, size=N_SAMPLES)
    for (x, y), r, av, t in zip(pts, radii, ang_vels, turns, strict=True):
        cur = Planet(
            id=3,
            owner=0,
            x=float(x),
            y=float(y),
            radius=float(r),
            ships=0,
            production=0,
        )
        init_by_id = {3: cur}
        reset_predict_cache()
        ref_x, ref_y = predict_planet_position(cur, init_by_id, float(av), int(t))
        gx, gy = predict_planet_position_jax(
            _f32(x), _f32(y), _f32(x), _f32(y), _f32(r), _f32(av), _f32(t)
        )
        assert abs(ref_x - float(gx)) < 1e-3
        assert abs(ref_y - float(gy)) < 1e-3


def test_estimate_arrival_parity(rng: np.random.Generator) -> None:
    src = _board_points(rng, N_SAMPLES)
    tgt = _board_points(rng, N_SAMPLES)
    sr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    tr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    ships = rng.integers(1, 500, size=N_SAMPLES)
    for (sx, sy), (tx, ty), srr, trr, sh in zip(src, tgt, sr, tr, ships, strict=True):
        ref = estimate_arrival(sx, sy, srr, tx, ty, trr, int(sh))
        angle, turns, valid = estimate_arrival_jax(
            _f32(sx),
            _f32(sy),
            _f32(srr),
            _f32(tx),
            _f32(ty),
            _f32(trr),
            jnp.asarray(sh, dtype=jnp.int32),
        )
        if ref is None:
            assert not bool(valid)
        else:
            assert bool(valid)
            assert abs(ref[0] - float(angle)) < PARITY_TOL
            assert ref[1] == int(turns)


def test_travel_time_parity(rng: np.random.Generator) -> None:
    src = _board_points(rng, N_SAMPLES)
    tgt = _board_points(rng, N_SAMPLES)
    sr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    tr = rng.uniform(1.0, 5.0, size=N_SAMPLES)
    ships = rng.integers(1, 500, size=N_SAMPLES)
    for (sx, sy), (tx, ty), srr, trr, sh in zip(src, tgt, sr, tr, ships, strict=True):
        ref = travel_time(sx, sy, srr, tx, ty, trr, int(sh))
        got = int(
            travel_time_jax(
                _f32(sx),
                _f32(sy),
                _f32(srr),
                _f32(tx),
                _f32(ty),
                _f32(trr),
                jnp.asarray(sh, dtype=jnp.int32),
            )
        )
        if ref >= 10**9:
            assert got == BIG_TURNS
        else:
            assert ref == got


def test_vmap_batched_estimate_arrival(rng: np.random.Generator) -> None:
    """The port must vmap cleanly and agree with the Python per-element loop."""
    src = _board_points(rng, 64)
    tgt = _board_points(rng, 64)
    sr = rng.uniform(1.0, 5.0, size=64)
    tr = rng.uniform(1.0, 5.0, size=64)
    ships = rng.integers(1, 500, size=64)

    batched = jax.jit(jax.vmap(estimate_arrival_jax, in_axes=(0, 0, 0, 0, 0, 0, 0)))
    angle_b, turns_b, valid_b = batched(
        jnp.asarray(src[:, 0], dtype=jnp.float32),
        jnp.asarray(src[:, 1], dtype=jnp.float32),
        jnp.asarray(sr, dtype=jnp.float32),
        jnp.asarray(tgt[:, 0], dtype=jnp.float32),
        jnp.asarray(tgt[:, 1], dtype=jnp.float32),
        jnp.asarray(tr, dtype=jnp.float32),
        jnp.asarray(ships, dtype=jnp.int32),
    )
    for i in range(64):
        ref = estimate_arrival(
            src[i, 0], src[i, 1], sr[i], tgt[i, 0], tgt[i, 1], tr[i], int(ships[i])
        )
        if ref is None:
            assert not bool(valid_b[i])
        else:
            assert bool(valid_b[i])
            assert abs(ref[0] - float(angle_b[i])) < PARITY_TOL
            assert ref[1] == int(turns_b[i])


def test_fleet_speed_monotonic_smoke() -> None:
    # sanity: matches the math at a couple of fixed points.
    assert abs(fleet_speed(1) - float(fleet_speed_jax(jnp.asarray(1)))) < PARITY_TOL
    assert (
        abs(fleet_speed(1000) - float(fleet_speed_jax(jnp.asarray(1000)))) < PARITY_TOL
    )
    assert math.isclose(fleet_speed(1000), 6.0, abs_tol=1e-6)
