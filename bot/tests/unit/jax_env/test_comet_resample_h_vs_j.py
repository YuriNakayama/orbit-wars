"""Observation 5-1: RNG-INDEPENDENT logic parity, H (host) vs J (JAX) comets.

The comet *path geometry* — dense ellipse sampling → arc-length resample →
contiguous in-board visible window — depends ONLY on the ellipse params
(e, a, phi), not on the RNG stream. So for a FIXED (e, a, phi) we can compare the
host implementation (`comet_gen.generate_comet_paths`) against the JAX one
(`comet_gen_jax._resample_by_arclen` + `_extract_visible`) by VALUE, within float
tolerance. This directly validates the resample/visible-extraction logic that the
distributional tests can only check structurally.

Drive H with a fake rng returning the chosen (e, a, phi) and run it with an empty
planet set so the first candidate is accepted (no overlap rejection), then
compare H's quadrant-0 visible path to J's.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "jax"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.comet_gen import generate_comet_paths  # noqa: E402  (H)
from orbit_wars_jax.comet_gen_jax import (  # noqa: E402  (J)
    _ellipse_dense,
    _extract_visible,
    _resample_by_arclen,
)


class _FakeRng:
    """Returns a fixed (e, a, phi) on successive uniform() calls.

    H's generate_comet_paths draws e=uniform(0.75,0.93), a=uniform(60,150),
    phi=uniform(pi/6,pi/3) in that order, then deterministic geometry. We feed
    the exact values and accept any subsequent calls (there are none on the
    geometry path before the first candidate is built).
    """

    def __init__(self, e: float, a: float, phi: float) -> None:
        self._vals = [e, a, phi]
        self._i = 0

    def uniform(self, lo: float, hi: float) -> float:
        v = self._vals[self._i]
        self._i += 1
        return v


# (e, a, phi) triples chosen so the perihelion check passes and the arc crosses
# the board (so a visible window of length 5..40 exists).
_PARAMS = [
    (0.80, 90.0, math.pi / 4),
    (0.85, 110.0, math.pi / 5),
    (0.78, 75.0, math.pi / 4.5),
    (0.90, 130.0, math.pi / 3.5),
]


@pytest.mark.parametrize("e,a,phi", _PARAMS)
def test_resample_visible_h_vs_j(e: float, a: float, phi: float) -> None:
    # ---- H: drive generate_comet_paths with fixed (e,a,phi), no planets ----
    # empty planet list → no overlap rejection → first candidate accepted.
    h_paths = generate_comet_paths(
        initial_planets=[],
        angular_velocity=0.035,
        spawn_step=50,
        comet_planet_ids=None,
        rng=_FakeRng(e, a, phi),
    )
    assert h_paths is not None, (
        f"H rejected (e={e},a={a},phi={phi}) — pick other params"
    )
    # H quadrant 0 = [[y, x] for (x,y) in visible]; recover cartesian (x, y).
    h_q0 = np.asarray(h_paths[0], dtype=np.float64)  # (Lh, 2) = (y, x)
    h_y = h_q0[:, 0]
    h_x = h_q0[:, 1]
    lh = len(h_q0)

    # ---- J: same (e,a,phi) through the JAX resample + visible extraction ----
    x, y = _ellipse_dense(jnp.float32(e), jnp.float32(a), jnp.float32(phi))
    rx, ry, rvalid = _resample_by_arclen(x, y)
    vx, vy, vlen = _extract_visible(rx, ry, rvalid)
    lj = int(vlen)
    j_x = np.asarray(vx)[:lj]
    j_y = np.asarray(vy)[:lj]

    # length must match (same geometry, same resample spacing).
    assert lj == lh, f"visible length mismatch H={lh} J={lj} (e={e},a={a},phi={phi})"

    # point-by-point match within float tolerance. J uses float32 dense sampling
    # vs H float64, so allow a small band (the resample picks discrete dense
    # indices; a sub-step float drift can shift a chosen index by 1 → looser tol).
    assert np.allclose(j_x, h_x, atol=0.5), (
        f"x mismatch (e={e},a={a},phi={phi}): max|d|={np.max(np.abs(j_x - h_x)):.4f}"
    )
    assert np.allclose(j_y, h_y, atol=0.5), (
        f"y mismatch (e={e},a={a},phi={phi}): max|d|={np.max(np.abs(j_y - h_y)):.4f}"
    )


@pytest.mark.parametrize("e,a,phi", _PARAMS)
def test_dense_ellipse_h_vs_j(e: float, a: float, phi: float) -> None:
    """The dense ellipse sampling itself (pre-resample) matches H's formula.

    H builds dense points with the same parametric ellipse; recompute a few
    sample points the H way and compare to J's _ellipse_dense.
    """
    x_j, y_j = (
        np.asarray(z)
        for z in _ellipse_dense(jnp.float32(e), jnp.float32(a), jnp.float32(phi))
    )
    # H formula (comet_gen rows 53-65), float64.
    CENTER = 50.0
    num = 5000
    b = a * math.sqrt(1 - e**2)
    c_val = a * e
    for i in (0, 1234, 2500, 4999):
        t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
        ex = c_val + a * math.cos(t)
        ey = b * math.sin(t)
        hx = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
        hy = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
        assert abs(x_j[i] - hx) < 1e-2, f"dense x[{i}] mismatch {x_j[i]} vs {hx}"
        assert abs(y_j[i] - hy) < 1e-2, f"dense y[{i}] mismatch {y_j[i]} vs {hy}"
