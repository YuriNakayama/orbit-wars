"""Numerical-parity tests for `baseline_jax/timeline_jax.py`.

Compares the four pure timeline/world-model helpers against their Python
counterparts in `baseline/core/world_model.py` on identical, randomized inputs:

* `resolve_arrival_event`     -> `resolve_arrival_event_jax`
* `simulate_planet_timeline`  -> `simulate_planet_timeline_jax`
* `state_at_timeline`         -> `state_at_timeline_jax`
* `indirect_wealth`           -> `indirect_wealth_jax`

Tolerance: integer / owner fields match EXACTLY; float fields within float32
`1e-4`. The JAX port uses fixed-shape padded arrays + valid masks; the host
pre-buckets arrivals by turn into `(HORIZON+1, MAX_ARRIVALS_PER_TURN)` tables.

A `jax.vmap` batched test proves the port vmaps cleanly across many planets and
agrees with the per-element Python loop.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case8.baseline.core.types import Planet
from pipeline.rulebase.case8.baseline.core.world_model import (
    indirect_wealth,
    resolve_arrival_event,
    simulate_planet_timeline,
    state_at_timeline,
)
from pipeline.rulebase.case8.baseline_jax.timeline_jax import (
    MAX_ARRIVALS_PER_TURN,
    MAX_PLANETS,
    TimelineResult,
    indirect_wealth_jax,
    resolve_arrival_event_jax,
    simulate_planet_timeline_jax,
    state_at_timeline_jax,
)

pytestmark = pytest.mark.slow

PARITY_TOL = 1e-4
HORIZON = 110
N_SAMPLES = 200


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260529)


def _random_arrivals(
    rng: np.random.Generator, max_eta: int, n_max: int
) -> list[tuple[int, int, int]]:
    """A random arrival list (eta in 1..max_eta, owner in {0,1,2,3}, ships 0..200)."""
    n = int(rng.integers(0, n_max + 1))
    arrivals: list[tuple[int, int, int]] = []
    for _ in range(n):
        eta = int(rng.integers(1, max_eta + 1))
        owner = int(rng.integers(0, 4))
        ships = int(rng.integers(0, 201))
        arrivals.append((eta, owner, ships))
    return arrivals


def _bucket_arrivals(
    arrivals: list[tuple[int, int, int]], horizon: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Host-side pre-bucketing of arrivals into (H+1, K) owner/ship/valid tables.

    Mirrors the Python `normalize_arrivals` + `by_turn` grouping: ships<=0 and
    eta>horizon dropped, eta = max(1, ceil(turns)). Overflow beyond K is dropped
    (asserted not to happen for the test's n_max).
    """
    owner = np.full((horizon + 1, MAX_ARRIVALS_PER_TURN), -1, dtype=np.int32)
    ships = np.zeros((horizon + 1, MAX_ARRIVALS_PER_TURN), dtype=np.int32)
    valid = np.zeros((horizon + 1, MAX_ARRIVALS_PER_TURN), dtype=bool)
    counts = np.zeros(horizon + 1, dtype=np.int32)
    for turns, own, shp in arrivals:
        if shp <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        slot = counts[eta]
        assert slot < MAX_ARRIVALS_PER_TURN, "arrival bucket overflow in test"
        owner[eta, slot] = own
        ships[eta, slot] = int(shp)
        valid[eta, slot] = True
        counts[eta] += 1
    return jnp.asarray(owner), jnp.asarray(ships), jnp.asarray(valid)


def _arr_arrays(
    arrivals: list[tuple[int, int, int]],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack a single same-turn arrival group into (K,) owner/ship/valid arrays."""
    owner = np.full(MAX_ARRIVALS_PER_TURN, -1, dtype=np.int32)
    ships = np.zeros(MAX_ARRIVALS_PER_TURN, dtype=np.int32)
    valid = np.zeros(MAX_ARRIVALS_PER_TURN, dtype=bool)
    for i, (_, own, shp) in enumerate(arrivals):
        assert i < MAX_ARRIVALS_PER_TURN
        owner[i] = own
        ships[i] = int(shp)
        valid[i] = True
    return jnp.asarray(owner), jnp.asarray(ships), jnp.asarray(valid)


# --------------------------------------------------------------------------- #
# 1. resolve_arrival_event                                                    #
# --------------------------------------------------------------------------- #


def test_resolve_arrival_event_parity(rng: np.random.Generator) -> None:
    max_div = 0.0
    for _ in range(N_SAMPLES):
        owner = int(rng.integers(-1, 4))
        garrison = float(rng.integers(0, 201))
        n = int(rng.integers(0, 6))
        arrivals = [
            (1, int(rng.integers(0, 4)), int(rng.integers(0, 201))) for _ in range(n)
        ]
        ref_owner, ref_garr = resolve_arrival_event(owner, garrison, arrivals)

        arr_owner, arr_ships, arr_valid = _arr_arrays(arrivals)
        got_owner, got_garr = resolve_arrival_event_jax(
            jnp.int32(owner),
            jnp.float32(garrison),
            arr_owner,
            arr_ships,
            arr_valid,
        )
        assert int(got_owner) == ref_owner
        max_div = max(max_div, abs(float(got_garr) - ref_garr))
    assert max_div < PARITY_TOL, f"resolve max float divergence {max_div}"


def test_resolve_arrival_event_tie_zero_survivor() -> None:
    # Two attackers tied at 30 each -> survivor_ships == 0, so nothing lands and
    # the defender (owner 2) keeps the planet at its current garrison.
    arrivals = [(1, 0, 30), (1, 1, 30)]
    ref_owner, ref_garr = resolve_arrival_event(2, 50.0, arrivals)
    arr_owner, arr_ships, arr_valid = _arr_arrays(arrivals)
    got_owner, got_garr = resolve_arrival_event_jax(
        jnp.int32(2), jnp.float32(50.0), arr_owner, arr_ships, arr_valid
    )
    assert ref_owner == 2 and ref_garr == 50.0
    assert int(got_owner) == ref_owner
    assert abs(float(got_garr) - ref_garr) < PARITY_TOL


def test_resolve_arrival_event_tie_then_neutral() -> None:
    # Tie between the two TOP owners but a third (smaller) makes the survivor
    # owner -1 only when top != second; here top==second so survivor is 0 ships.
    # Use an unequal top to exercise the survivor=top-second neutral-defender path.
    arrivals = [(1, 0, 40), (1, 1, 25)]
    ref_owner, ref_garr = resolve_arrival_event(-1, 0.0, arrivals)
    arr_owner, arr_ships, arr_valid = _arr_arrays(arrivals)
    got_owner, got_garr = resolve_arrival_event_jax(
        jnp.int32(-1), jnp.float32(0.0), arr_owner, arr_ships, arr_valid
    )
    # neutral defender (owner -1), attacker 0 wins with 40-25=15 survivors ->
    # garrison 0 - 15 = -15 < 0 -> owner flips to 0, ships 15.
    assert ref_owner == 0 and ref_garr == 15.0
    assert int(got_owner) == ref_owner
    assert abs(float(got_garr) - ref_garr) < PARITY_TOL


def test_resolve_arrival_event_capture() -> None:
    # Attacker overwhelms a defended planet -> owner flips, ships = overflow.
    arrivals = [(1, 0, 100), (1, 1, 10)]
    ref_owner, ref_garr = resolve_arrival_event(1, 5.0, arrivals)
    arr_owner, arr_ships, arr_valid = _arr_arrays(arrivals)
    got_owner, got_garr = resolve_arrival_event_jax(
        jnp.int32(1), jnp.float32(5.0), arr_owner, arr_ships, arr_valid
    )
    assert int(got_owner) == ref_owner
    assert abs(float(got_garr) - ref_garr) < PARITY_TOL


# --------------------------------------------------------------------------- #
# 2. simulate_planet_timeline                                                 #
# --------------------------------------------------------------------------- #


def _random_planet(rng: np.random.Generator, owner: int) -> Planet:
    return Planet(
        id=0,
        owner=owner,
        x=float(rng.uniform(0, 100)),
        y=float(rng.uniform(0, 100)),
        radius=float(rng.uniform(1, 5)),
        ships=int(rng.integers(0, 201)),
        production=int(rng.integers(0, 6)),
    )


def _check_timeline(
    planet: Planet,
    arrivals: list[tuple[int, int, int]],
    player: int,
) -> float:
    ref = simulate_planet_timeline(planet, arrivals, player, HORIZON)
    tarr_owner, tarr_ships, tarr_valid = _bucket_arrivals(arrivals, HORIZON)
    got: TimelineResult = simulate_planet_timeline_jax(
        jnp.int32(planet.owner),
        jnp.float32(planet.ships),
        jnp.float32(planet.production),
        tarr_owner,
        tarr_ships,
        tarr_valid,
        jnp.int32(player),
        HORIZON,
    )

    # owner_at / ships_at exact (owner) + float (ships) at sampled turns.
    max_div = 0.0
    sample_turns = [0, 1, 2, 5, 17, 40, 73, 109, 110]
    owner_at_g = np.asarray(got.owner_at)
    ships_at_g = np.asarray(got.ships_at)
    for t in sample_turns:
        ref_owner = ref["owner_at"][t]
        ref_ships = ref["ships_at"][t]
        assert int(owner_at_g[t]) == ref_owner, (
            f"owner@{t}: {owner_at_g[t]}!={ref_owner}"
        )
        max_div = max(max_div, abs(float(ships_at_g[t]) - ref_ships))

    assert int(got.keep_needed) == ref["keep_needed"], (
        f"keep_needed {int(got.keep_needed)} != {ref['keep_needed']}"
    )
    assert int(got.min_owned) == ref["min_owned"], (
        f"min_owned {int(got.min_owned)} != {ref['min_owned']}"
    )
    ref_first = -1 if ref["first_enemy"] is None else ref["first_enemy"]
    ref_fall = -1 if ref["fall_turn"] is None else ref["fall_turn"]
    assert int(got.first_enemy) == ref_first
    assert int(got.fall_turn) == ref_fall
    assert bool(got.holds_full) == ref["holds_full"]
    assert int(got.horizon) == ref["horizon"]
    return max_div


def test_simulate_timeline_owned_parity(rng: np.random.Generator) -> None:
    """planet.owner == player (exercises keep_needed / min_owned / holds_full)."""
    max_div = 0.0
    for _ in range(N_SAMPLES):
        planet = _random_planet(rng, owner=1)
        arrivals = _random_arrivals(rng, HORIZON + 10, n_max=10)
        max_div = max(max_div, _check_timeline(planet, arrivals, player=1))
    assert max_div < PARITY_TOL, f"timeline owned max ships divergence {max_div}"


def test_simulate_timeline_neutral_parity(rng: np.random.Generator) -> None:
    """planet.owner == -1 (neutral planet, no production, keep_needed == 0)."""
    max_div = 0.0
    for _ in range(N_SAMPLES):
        planet = _random_planet(rng, owner=-1)
        arrivals = _random_arrivals(rng, HORIZON + 10, n_max=10)
        max_div = max(max_div, _check_timeline(planet, arrivals, player=1))
    assert max_div < PARITY_TOL, f"timeline neutral max ships divergence {max_div}"


def test_simulate_timeline_enemy_parity(rng: np.random.Generator) -> None:
    """planet.owner is an enemy (owner != player and != -1)."""
    max_div = 0.0
    for _ in range(N_SAMPLES):
        planet = _random_planet(rng, owner=2)
        arrivals = _random_arrivals(rng, HORIZON + 10, n_max=10)
        max_div = max(max_div, _check_timeline(planet, arrivals, player=1))
    assert max_div < PARITY_TOL, f"timeline enemy max ships divergence {max_div}"


def test_simulate_timeline_no_arrivals(rng: np.random.Generator) -> None:
    """A quiet planet still produces the right owner_at/ships_at ramp."""
    planet = Planet(id=0, owner=1, x=20.0, y=20.0, radius=2.0, ships=10, production=3)
    _check_timeline(planet, [], player=1)


# --------------------------------------------------------------------------- #
# 3. state_at_timeline                                                        #
# --------------------------------------------------------------------------- #


def test_state_at_timeline_parity(rng: np.random.Generator) -> None:
    max_div = 0.0
    for _ in range(N_SAMPLES):
        planet = _random_planet(rng, owner=1)
        arrivals = _random_arrivals(rng, HORIZON + 10, n_max=8)
        timeline = simulate_planet_timeline(planet, arrivals, 1, HORIZON)
        tarr_owner, tarr_ships, tarr_valid = _bucket_arrivals(arrivals, HORIZON)
        got: TimelineResult = simulate_planet_timeline_jax(
            jnp.int32(planet.owner),
            jnp.float32(planet.ships),
            jnp.float32(planet.production),
            tarr_owner,
            tarr_ships,
            tarr_valid,
            jnp.int32(1),
            HORIZON,
        )
        for at in [-3.0, 0.0, 0.4, 1.0, 1.5, 13.7, 109.9, 110.0, 250.0]:
            ref_owner, ref_ships = state_at_timeline(timeline, at)
            g_owner, g_ships = state_at_timeline_jax(
                got.owner_at, got.ships_at, jnp.int32(HORIZON), jnp.float32(at)
            )
            assert int(g_owner) == ref_owner, f"state owner@{at}"
            max_div = max(max_div, abs(float(g_ships) - ref_ships))
    assert max_div < PARITY_TOL, f"state_at max ships divergence {max_div}"


# --------------------------------------------------------------------------- #
# 4. indirect_wealth                                                          #
# --------------------------------------------------------------------------- #


def _pad_planets(
    planets: list[Planet],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad a planet list to MAX_PLANETS: xy, owner, prod, valid mask."""
    xs = np.zeros(MAX_PLANETS, dtype=np.float32)
    ys = np.zeros(MAX_PLANETS, dtype=np.float32)
    owners = np.full(MAX_PLANETS, -1, dtype=np.int32)
    prods = np.zeros(MAX_PLANETS, dtype=np.float32)
    valid = np.zeros(MAX_PLANETS, dtype=bool)
    for i, p in enumerate(planets):
        xs[i] = p.x
        ys[i] = p.y
        owners[i] = p.owner
        prods[i] = p.production
        valid[i] = True
    return (
        jnp.asarray(xs),
        jnp.asarray(ys),
        jnp.asarray(owners),
        jnp.asarray(prods),
        jnp.asarray(valid),
    )


def _random_planet_field(rng: np.random.Generator, n: int) -> list[Planet]:
    return [
        Planet(
            id=i,
            owner=int(rng.integers(-1, 3)),
            x=float(rng.uniform(0, 100)),
            y=float(rng.uniform(0, 100)),
            radius=float(rng.uniform(1, 5)),
            ships=int(rng.integers(0, 201)),
            production=int(rng.integers(0, 6)),
        )
        for i in range(n)
    ]


def test_indirect_wealth_parity(rng: np.random.Generator) -> None:
    max_div = 0.0
    for _ in range(N_SAMPLES):
        n = int(rng.integers(2, 12))
        planets = _random_planet_field(rng, n)
        player = 1
        xs, ys, owners, prods, valid = _pad_planets(planets)
        for focal in range(n):
            ref = indirect_wealth(planets[focal], planets, player)
            got = float(
                indirect_wealth_jax(
                    jnp.int32(focal), xs, ys, owners, prods, valid, jnp.int32(player)
                )
            )
            max_div = max(max_div, abs(ref - got))
    assert max_div < PARITY_TOL, f"indirect_wealth max divergence {max_div}"


# --------------------------------------------------------------------------- #
# vmap batched parity (the whole point of the port)                           #
# --------------------------------------------------------------------------- #


def test_vmap_simulate_timeline(rng: np.random.Generator) -> None:
    """vmap simulate_planet_timeline_jax over many planets == per-element Python."""
    n = 32
    planets = [_random_planet(rng, owner=int(rng.integers(-1, 3))) for _ in range(n)]
    arrival_lists = [_random_arrivals(rng, HORIZON + 10, n_max=8) for _ in range(n)]
    player = 1

    p_owner = jnp.asarray([p.owner for p in planets], dtype=jnp.int32)
    p_ships = jnp.asarray([float(p.ships) for p in planets], dtype=jnp.float32)
    p_prod = jnp.asarray([float(p.production) for p in planets], dtype=jnp.float32)
    buckets = [_bucket_arrivals(a, HORIZON) for a in arrival_lists]
    tarr_owner = jnp.stack([b[0] for b in buckets])
    tarr_ships = jnp.stack([b[1] for b in buckets])
    tarr_valid = jnp.stack([b[2] for b in buckets])

    batched = jax.jit(
        jax.vmap(
            simulate_planet_timeline_jax,
            in_axes=(0, 0, 0, 0, 0, 0, None, None),
        ),
        static_argnums=(7,),
    )
    got: TimelineResult = batched(
        p_owner,
        p_ships,
        p_prod,
        tarr_owner,
        tarr_ships,
        tarr_valid,
        jnp.int32(player),
        HORIZON,
    )

    owner_at_g = np.asarray(got.owner_at)
    ships_at_g = np.asarray(got.ships_at)
    keep_g = np.asarray(got.keep_needed)
    min_g = np.asarray(got.min_owned)
    first_g = np.asarray(got.first_enemy)
    fall_g = np.asarray(got.fall_turn)
    holds_g = np.asarray(got.holds_full)

    max_div = 0.0
    for i in range(n):
        ref = simulate_planet_timeline(planets[i], arrival_lists[i], player, HORIZON)
        for t in [0, 1, 3, 40, 110]:
            assert int(owner_at_g[i, t]) == ref["owner_at"][t]
            max_div = max(max_div, abs(float(ships_at_g[i, t]) - ref["ships_at"][t]))
        assert int(keep_g[i]) == ref["keep_needed"]
        assert int(min_g[i]) == ref["min_owned"]
        ref_first = -1 if ref["first_enemy"] is None else ref["first_enemy"]
        ref_fall = -1 if ref["fall_turn"] is None else ref["fall_turn"]
        assert int(first_g[i]) == ref_first
        assert int(fall_g[i]) == ref_fall
        assert bool(holds_g[i]) == ref["holds_full"]
    assert max_div < PARITY_TOL, f"vmap timeline max ships divergence {max_div}"


def test_vmap_indirect_wealth(rng: np.random.Generator) -> None:
    """vmap indirect_wealth_jax over all focal indices == per-element Python."""
    n = 10
    planets = _random_planet_field(rng, n)
    player = 1
    xs, ys, owners, prods, valid = _pad_planets(planets)

    batched = jax.jit(
        jax.vmap(
            indirect_wealth_jax,
            in_axes=(0, None, None, None, None, None, None),
        )
    )
    focal_idx = jnp.arange(n, dtype=jnp.int32)
    got = np.asarray(
        batched(focal_idx, xs, ys, owners, prods, valid, jnp.int32(player))
    )

    max_div = 0.0
    for i in range(n):
        ref = indirect_wealth(planets[i], planets, player)
        max_div = max(max_div, abs(ref - float(got[i])))
    assert max_div < PARITY_TOL, f"vmap indirect_wealth max divergence {max_div}"
