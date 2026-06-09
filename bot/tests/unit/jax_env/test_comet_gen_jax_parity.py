"""Structural + distributional parity for JAX-native comet generation.

`orbit_wars_jax.comet_gen_jax.generate_comets_jax(key, planet_buf, angular_velocity)`
replaces the host `comet_gen.py` (random.Random + Python rejection loops + 5000-pt
dense sampling + arc-length resample). NOT byte-equal to the vendor RNG path
(intentionally — see comet_gen_jax docstring). Verifies STRUCTURE + the vendor's
own validity constraints, not byte-equality:

  (a) output shapes/dtypes match the EnvState comet fields;
  (b) jit + vmap run;
  (c) for every successfully-generated comet (path_len > 0):
        - path_len in [5, MAX_COMET_PATH_LEN]  (vendor 5 <= visible <= 40);
        - all visible points are inside the board [0, BOARD_SIZE];
        - all visible points are outside the sun (dist to CENTER >= SUN+COMET_R);
        - the 4 quadrant copies are the documented symmetry of quadrant 0;
        - initial_ships in [1, 99].
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "jax"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.comet_gen_jax import generate_comets_jax  # noqa: E402
from orbit_wars_jax.constants import (  # noqa: E402
    BOARD_SIZE,
    COMET_RADIUS,
    MAX_COMET_PATH_LEN,
    MAX_COMETS,
    SUN_RADIUS,
)
from orbit_wars_jax.planet_gen_jax import generate_planets_jax  # noqa: E402


def _gen(seed: int):
    k = jax.random.PRNGKey(seed)
    kp, kc = jax.random.split(k)
    buf = generate_planets_jax(kp)
    av = 0.035
    return generate_comets_jax(kc, buf, av)


def test_comet_jax_shapes() -> None:
    paths, path_len, initial_ships = _gen(0)
    assert paths.shape == (MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2)
    assert path_len.shape == (MAX_COMETS,)
    assert initial_ships.shape == (MAX_COMETS,)
    assert paths.dtype == np.float32
    assert path_len.dtype == np.int32
    assert initial_ships.dtype == np.int32


def test_comet_jax_jit_and_vmap() -> None:
    keys = jax.random.split(jax.random.PRNGKey(3), 4)

    def one(k):
        kp, kc = jax.random.split(k)
        buf = generate_planets_jax(kp)
        return generate_comets_jax(kc, buf, 0.035)

    batched = jax.vmap(one)(keys)
    paths, path_len, ships = batched
    assert paths.shape == (4, MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2)
    assert path_len.shape == (4, MAX_COMETS)


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11])
def test_comet_jax_validity(seed: int) -> None:
    paths, path_len, initial_ships = (np.asarray(x) for x in _gen(seed))

    for c in range(MAX_COMETS):
        n = int(path_len[c])
        if n == 0:
            continue  # comet not generated this slot — allowed
        assert 5 <= n <= MAX_COMET_PATH_LEN, f"comet {c}: path_len {n} out of [5,40]"
        assert 1 <= int(initial_ships[c]) <= 99, f"comet {c}: ships out of [1,99]"

        # quadrant 0 visible points (stored as (y, x) per vendor format).
        q0 = paths[c, 0, :n]  # (n, 2) = (y, x)
        ys, xs = q0[:, 0], q0[:, 1]
        # inside board
        assert (xs >= -1e-3).all() and (xs <= BOARD_SIZE + 1e-3).all()
        assert (ys >= -1e-3).all() and (ys <= BOARD_SIZE + 1e-3).all()
        # outside the sun
        d_center = np.hypot(ys - BOARD_SIZE / 2, xs - BOARD_SIZE / 2)
        assert (d_center >= SUN_RADIUS + COMET_RADIUS - 1e-2).all(), (
            f"comet {c}: a visible point is inside the sun"
        )

        # symmetry of the 4 quadrant copies (vendor rows 91-96).
        # q0 = [y, x]; q1 = [BOARD-x, y]; q2 = [x, BOARD-y]; q3 = [BOARD-y, BOARD-x]
        # where (x, y) is the original cartesian (note q0 stores (y, x)).
        x_orig, y_orig = xs, ys  # q0 stored (y,x) so xs=x_orig? see below
        # q0 row is [y, x] meaning row[0]=y_cart? Actually vendor: paths[0]=[[y,x]..]
        # built from `visible` which holds (x, y). So q0 = (y, x): q0[:,0]=y, q0[:,1]=x.
        # Reconstruct cartesian (x, y):
        y_cart = q0[:, 0]
        x_cart = q0[:, 1]
        q1 = paths[c, 1, :n]  # [BOARD-x, y]
        q2 = paths[c, 2, :n]  # [x, BOARD-y]
        q3 = paths[c, 3, :n]  # [BOARD-y, BOARD-x]
        assert np.allclose(q1[:, 0], BOARD_SIZE - x_cart, atol=1e-2)
        assert np.allclose(q1[:, 1], y_cart, atol=1e-2)
        assert np.allclose(q2[:, 0], x_cart, atol=1e-2)
        assert np.allclose(q2[:, 1], BOARD_SIZE - y_cart, atol=1e-2)
        assert np.allclose(q3[:, 0], BOARD_SIZE - y_cart, atol=1e-2)
        assert np.allclose(q3[:, 1], BOARD_SIZE - x_cart, atol=1e-2)


def test_comet_jax_deterministic() -> None:
    a = _gen(42)
    b = _gen(42)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))
