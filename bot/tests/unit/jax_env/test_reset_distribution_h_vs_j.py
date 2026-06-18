"""Observation 3: distributional equivalence, H (host reset) vs J (JAX reset).

J abandons vendor RNG byte-parity, so per-seed values differ. What MUST hold is
that J draws from the SAME distribution of valid layouts as H: similar planet
counts, comet success rates, angular_velocity range, and spatial spread. These
are statistical (not per-seed) comparisons over many independent seeds.

Tolerances are loose on purpose — the two RNG streams are unrelated, so we check
that the summary statistics land in the same ballpark, not that they're equal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "jax"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.constants import (  # noqa: E402
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
)
from orbit_wars_jax.reset import reset as reset_host  # noqa: E402  (H)
from orbit_wars_jax.reset_jax import reset_jax  # noqa: E402  (J)

_N = 96  # seeds per implementation


def _host_stats() -> dict[str, np.ndarray]:
    counts, comet_ok, ang, orb = [], [], [], []
    for seed in range(_N):
        s = reset_host(seed=seed, num_agents=2)
        v = np.asarray(s.planet_valid)
        counts.append(int(v.sum()))
        comet_ok.append(int((np.asarray(s.comet_path_len) > 0).sum()))
        ang.append(float(s.angular_velocity))
        xy = np.asarray(s.planet_xy)[v]
        orb.append(np.hypot(xy[:, 0] - 50.0, xy[:, 1] - 50.0))
    return {
        "counts": np.array(counts),
        "comet_ok": np.array(comet_ok),
        "ang": np.array(ang),
        "orb": np.concatenate(orb),
    }


def _jax_stats() -> dict[str, np.ndarray]:
    keys = jax.random.split(jax.random.PRNGKey(2024), _N)
    batched = jax.vmap(lambda k: reset_jax(k, num_agents=2))(keys)
    v = np.asarray(batched.planet_valid)  # (N, P)
    counts = v.sum(axis=1)
    comet_ok = (np.asarray(batched.comet_path_len) > 0).sum(axis=1)
    ang = np.asarray(batched.angular_velocity)
    xy = np.asarray(batched.planet_xy)
    orb = []
    for g in range(_N):
        p = xy[g, v[g]]
        orb.append(np.hypot(p[:, 0] - 50.0, p[:, 1] - 50.0))
    return {
        "counts": counts,
        "comet_ok": comet_ok,
        "ang": ang,
        "orb": np.concatenate(orb),
    }


def test_planet_count_distribution() -> None:
    """3-1: planet counts land in the same band; means within ~1 group (4)."""
    h = _host_stats()["counts"]
    j = _jax_stats()["counts"]
    lo, hi = MIN_PLANET_GROUPS * 4, MAX_PLANET_GROUPS * 4
    # both stay within (a slightly relaxed) vendor band.
    assert h.min() >= lo - 4 and h.max() <= hi
    assert j.min() >= lo - 4 and j.max() <= hi
    # means within one group of each other.
    assert abs(float(h.mean()) - float(j.mean())) <= 4.0, (
        f"planet-count mean H={h.mean():.1f} J={j.mean():.1f}"
    )


def test_comet_success_distribution() -> None:
    """3-2: comet generation success rate (of 5 slots) is comparable."""
    h = _host_stats()["comet_ok"]
    j = _jax_stats()["comet_ok"]
    # mean number of comets successfully generated per game (out of 5).
    assert abs(float(h.mean()) - float(j.mean())) <= 1.5, (
        f"comet-success mean H={h.mean():.2f} J={j.mean():.2f}"
    )
    # both should usually generate at least some comets.
    assert j.mean() >= 1.0, f"J comet success too low: {j.mean():.2f}"


def test_angular_velocity_distribution() -> None:
    """3-4: angular_velocity uniform(0.025, 0.05) for both."""
    h = _host_stats()["ang"]
    j = _jax_stats()["ang"]
    for a in (h, j):
        assert a.min() >= 0.025 - 1e-6
        assert a.max() <= 0.05 + 1e-6
    assert abs(float(h.mean()) - float(j.mean())) <= 0.005


def test_orbital_radius_distribution() -> None:
    """3-3: spatial spread (orbital radius of planets) is comparable."""
    h = _host_stats()["orb"]
    j = _jax_stats()["orb"]
    # compare mean + std of the orbital-radius distribution loosely.
    assert abs(float(h.mean()) - float(j.mean())) <= 8.0, (
        f"orbital mean H={h.mean():.1f} J={j.mean():.1f}"
    )
    assert abs(float(h.std()) - float(j.std())) <= 8.0, (
        f"orbital std H={h.std():.1f} J={j.std():.1f}"
    )
