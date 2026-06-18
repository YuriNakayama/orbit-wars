"""Distributional + structural parity for the JAX-native reset.

`orbit_wars_jax.reset_jax.reset_jax(key, num_agents)` replaces the host-side
`reset.py` (random.Random + numpy + rejection-sampling Python loops) with a
jit/vmap-friendly version (jax.random + lax.while_loop + fixed buffers). It is
NOT byte-equal to the vendor RNG path (that parity is intentionally abandoned —
the reinforce/case8 PFSP rollout only needs self-consistent valid states, not
exact vendor planet layouts). So this test verifies STRUCTURE + DISTRIBUTION,
not byte-equality:

  (a) reset_jax(key) produces an EnvState that passes validate_state;
  (b) jax.jit(reset_jax) and jax.vmap(reset_jax) over split keys both run and
      yield batched valid states;
  (c) distributional sanity over many keys: planet counts in the vendor range,
      home planets owned 0..num_agents-1 with ships==10, no overlap among valid
      planets (modulo max_attempts fallback), comets within MAX_COMETS and
      path_len <= MAX_COMET_PATH_LEN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "jax"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.constants import (  # noqa: E402
    BOARD_SIZE,
    MAX_COMET_PATH_LEN,
    MAX_COMETS,
    MAX_PLANET_GROUPS,
    MAX_PLANETS,
    MIN_PLANET_GROUPS,
    PLANET_CLEARANCE,
)
from orbit_wars_jax.reset_jax import reset_jax  # noqa: E402
from orbit_wars_jax.state import validate_state  # noqa: E402


@pytest.mark.parametrize("num_agents", [2, 4])
def test_reset_jax_validate_state(num_agents: int) -> None:
    """(a) A single reset_jax output passes the shape/dtype contract."""
    state = reset_jax(jax.random.PRNGKey(0), num_agents=num_agents)
    validate_state(state)  # raises on shape/dtype mismatch


def test_reset_jax_jit_runs() -> None:
    """(b1) jit-compiles and runs (num_agents static)."""
    jitted = jax.jit(reset_jax, static_argnums=(1,))
    state = jitted(jax.random.PRNGKey(1), 2)
    validate_state(state)
    state.planet_valid.block_until_ready()


def test_reset_jax_vmap_batches() -> None:
    """(b2) vmap over split keys yields a batched EnvState with a leading axis."""
    keys = jax.random.split(jax.random.PRNGKey(2), 8)
    batched = jax.vmap(lambda k: reset_jax(k, num_agents=2))(keys)
    assert batched.planet_valid.shape == (8, MAX_PLANETS)
    assert batched.planet_xy.shape == (8, MAX_PLANETS, 2)
    # Each game in the batch must have at least the minimum home/static planets.
    counts = np.asarray(batched.planet_valid.sum(axis=1))
    assert (counts > 0).all()


@pytest.mark.parametrize("num_agents", [2, 4])
def test_reset_jax_distribution(num_agents: int) -> None:
    """(c) Distributional / structural sanity over many seeds."""
    n = 64
    keys = jax.random.split(jax.random.PRNGKey(7), n)
    batched = jax.vmap(lambda k: reset_jax(k, num_agents=num_agents))(keys)

    valid = np.asarray(batched.planet_valid)  # (n, MAX_PLANETS) bool
    owner = np.asarray(batched.planet_owner)  # (n, MAX_PLANETS)
    ships = np.asarray(batched.planet_ships)
    xy = np.asarray(batched.planet_xy)  # (n, MAX_PLANETS, 2)
    radius = np.asarray(batched.planet_radius)

    counts = valid.sum(axis=1)
    # Planet count: groups of 4. Allow the max_attempts fallback to undershoot a
    # little but stay within a sane band around the vendor range.
    assert counts.max() <= MAX_PLANETS
    assert counts.min() >= MIN_PLANET_GROUPS * 4 - 4  # tolerate one short group
    # mean should sit inside the vendor group range, loosely.
    assert MIN_PLANET_GROUPS * 4 - 4 <= counts.mean() <= MAX_PLANET_GROUPS * 4

    # Home planets: each game has exactly `num_agents` owned planets, owners are
    # the distinct seats 0..num_agents-1, each with ships == 10.
    for g in range(n):
        owned = (owner[g] >= 0) & valid[g]
        owned_seats = np.sort(owner[g][owned])
        assert owned_seats.tolist() == list(range(num_agents)), (
            f"game {g}: owned seats {owned_seats.tolist()} != {list(range(num_agents))}"
        )
        assert (ships[g][owned] == 10).all(), f"game {g}: home ships != 10"

    # In-bounds: valid planets are inside the board.
    for g in range(n):
        v = valid[g]
        px = xy[g, v, 1]
        py = xy[g, v, 0]
        assert (px >= 0).all() and (px <= BOARD_SIZE).all()
        assert (py >= 0).all() and (py <= BOARD_SIZE).all()

    # No overlap among valid planets: pairwise center distance >= r_i + r_j +
    # PLANET_CLEARANCE. The rejection sampler enforces this; allow a tiny float
    # epsilon. (max_attempts fallback never *adds* an overlapping planet, so this
    # must hold for every accepted planet.)
    eps = 1e-3
    for g in range(n):
        idx = np.where(valid[g])[0]
        p = xy[g, idx]  # (k, 2) as (y, x)
        r = radius[g, idx]
        k = len(idx)
        for i in range(k):
            for j in range(i + 1, k):
                d = float(np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]))
                assert d >= r[i] + r[j] + PLANET_CLEARANCE - eps, (
                    f"game {g}: planets {i},{j} overlap d={d:.3f}"
                )

    # Comets: within ceilings.
    plen = np.asarray(batched.comet_path_len)  # (n, MAX_COMETS)
    assert plen.shape == (n, MAX_COMETS)
    assert (plen <= MAX_COMET_PATH_LEN).all()
    assert (plen >= 0).all()


def test_reset_jax_deterministic_per_key() -> None:
    """Same key → identical state (pure function)."""
    k = jax.random.PRNGKey(123)
    a = reset_jax(k, num_agents=2)
    b = reset_jax(k, num_agents=2)
    assert bool(jnp.array_equal(a.planet_xy, b.planet_xy))
    assert bool(jnp.array_equal(a.planet_owner, b.planet_owner))
    assert bool(jnp.array_equal(a.comet_path_len, b.comet_path_len))
