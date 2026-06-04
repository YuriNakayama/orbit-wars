"""x64 parity: worldmodel_jax.resolve_arrival_event vs Python.

Combat resolution: aggregate incoming ships per owner, top-2, survivor logic.
The JAX port takes a fixed [NUM_PLAYERS] per-owner ship vector; the Python
original takes a list of (eta, owner, ships) — we build matching inputs.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline.core import world_model as wpy
from pipeline.rulebase.case1.baseline_jax.core_jax import worldmodel_jax as wjax

NUM_PLAYERS = wjax.NUM_PLAYERS


def _by_owner_vec(arrivals: list[tuple[int, int, int]]) -> jnp.ndarray:
    vec = np.zeros(NUM_PLAYERS, dtype=np.float64)
    for _eta, owner, ships in arrivals:
        vec[owner] += ships
    return jnp.asarray(vec)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_resolve_arrival_event_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mismatches: list[Any] = []
    n = 400
    for _ in range(n):
        owner = int(rng.integers(-1, NUM_PLAYERS))  # -1 neutral or a player
        garrison = float(rng.integers(0, 60))
        k = int(rng.integers(0, 5))  # number of incoming fleets
        arrivals = [
            (1, int(rng.integers(0, NUM_PLAYERS)), int(rng.integers(1, 40)))
            for _ in range(k)
        ]
        ref_owner, ref_g = wpy.resolve_arrival_event(owner, garrison, arrivals)

        jo, jg = wjax.resolve_arrival_event(
            jnp.asarray(owner), jnp.asarray(garrison), _by_owner_vec(arrivals)
        )
        if int(ref_owner) != int(jo) or not np.isclose(ref_g, float(jg), atol=1e-9):
            mismatches.append(
                (owner, garrison, arrivals, (ref_owner, ref_g), (int(jo), float(jg)))
            )

    assert not mismatches, f"seed={seed}: {len(mismatches)}/{n}: {mismatches[:5]}"


from pipeline.rulebase.case1.baseline.core.types import Planet  # noqa: E402

# Marked slow: x64 JAX recompilation makes these ~1-3s each; excluded from
# the 5-min CI Bot budget (matches case2 parity convention). Run via dev/test-bot -m slow.
pytestmark = pytest.mark.slow

HORIZON = 110
MAX_SHIPS = 60


def _pad_arrivals(arrivals: Any, max_arr: int = 64) -> tuple[Any, ...]:
    eta = np.full(max_arr, 10**9, dtype=np.int64)
    own = np.zeros(max_arr, dtype=np.int64)
    shp = np.zeros(max_arr, dtype=np.float64)
    for i, (e, o, s) in enumerate(arrivals[:max_arr]):
        eta[i], own[i], shp[i] = e, o, s
    return jnp.asarray(eta), jnp.asarray(own), jnp.asarray(shp)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_keep_needed_parity(seed: int) -> None:
    rng = np.random.default_rng(seed + 50)
    mismatches: list[Any] = []
    n = 120
    for _ in range(n):
        player = 0
        ships = int(rng.integers(5, MAX_SHIPS))
        production = int(rng.integers(1, 6))
        p = Planet(
            id=7, owner=player, x=30, y=30, radius=2, ships=ships, production=production
        )
        # build random enemy arrivals within horizon
        k = int(rng.integers(0, 4))
        arrivals = [
            (
                int(rng.integers(1, HORIZON + 1)),
                int(rng.integers(1, NUM_PLAYERS)),
                int(rng.integers(1, 40)),
            )
            for _ in range(k)
        ]
        ref = wpy.simulate_planet_timeline(p, arrivals, player, HORIZON)["keep_needed"]

        eta, own, shp = _pad_arrivals(arrivals)
        got = int(
            wjax.keep_needed(
                jnp.asarray(player),
                jnp.asarray(ships),
                jnp.asarray(production),
                jnp.asarray(player),
                eta,
                own,
                shp,
                HORIZON,
                MAX_SHIPS,
            )
        )
        if int(ref) != got:
            mismatches.append((ships, production, arrivals, int(ref), got))

    assert not mismatches, f"seed={seed}: {len(mismatches)}/{n}: {mismatches[:5]}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threatened_info_parity(seed: int) -> None:
    """holds_full + fall_turn match simulate_planet_timeline (random arrivals)."""
    rng = np.random.default_rng(seed + 700)
    mism: list[Any] = []
    n = 150
    for _ in range(n):
        ships = int(rng.integers(5, 50))
        prod = int(rng.integers(1, 6))
        p = Planet(id=0, owner=0, x=30, y=30, radius=2, ships=ships, production=prod)
        k = int(rng.integers(0, 4))
        arrivals = [
            (
                int(rng.integers(1, HORIZON + 1)),
                int(rng.integers(1, NUM_PLAYERS)),
                int(rng.integers(1, 40)),
            )
            for _ in range(k)
        ]
        tl = wpy.simulate_planet_timeline(p, arrivals, 0, HORIZON)
        eta, own, shp = _pad_arrivals(arrivals)
        hf, ft, _df = wjax.threatened_info(
            jnp.asarray(0),
            jnp.asarray(ships),
            jnp.asarray(prod),
            jnp.asarray(0),
            eta,
            own,
            shp,
            HORIZON,
        )
        ref_ft = tl["fall_turn"] if tl["fall_turn"] is not None else -1
        if bool(hf) != tl["holds_full"] or int(ft) != ref_ft:
            mism.append(
                (
                    ships,
                    prod,
                    arrivals,
                    tl["holds_full"],
                    tl["fall_turn"],
                    bool(hf),
                    int(ft),
                )
            )
    assert not mism, f"seed={seed}: {len(mism)}/{n}: {mism[:3]}"
