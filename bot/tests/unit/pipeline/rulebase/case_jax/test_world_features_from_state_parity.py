"""Parity: pure-JAX `build_world_features_from_state` vs the host featurizer.

`build_world_features_from_state(state, seat)` computes the SAME `WorldFeatures`
as `build_world_features(state_to_obs(state, player=seat))`, but entirely from a
fixed-shape `EnvState` with no host-side Python parsing or arrival loop, so it is
`jax.jit`- and `jax.vmap`-able. This test advances `orbit_wars_jax.reset` /
`step` a few turns across several seeds (so fleets and comets appear) and asserts
EVERY `WorldFeatures` field matches the host build:

* owner / int / bool fields EXACT,
* float fields within float32 `1e-4`.

Arrival ledger: the raw `arr_owner / arr_ships / arr_valid` tables are compared
EXACTLY (same dropped-overflow behavior — fleets are scattered in slot order,
which equals obs order, with a per-(planet, turn) counter, so even the within-turn
slot order matches the host's `normalize_arrivals` stable-by-eta bucketing). As a
belt-and-braces invariant we ALSO feed both tables through
`simulate_planet_timeline_jax` and confirm the per-planet timeline agrees — this
is the order-invariant property the downstream agent actually relies on.

A 4-state `jax.vmap` batch confirms the function vmaps and agrees with the
per-state host loop.

Relied-on config flags (dead branches for this agent):
* OPPONENT_MODEL_ENABLED == False -> predicted_arrivals empty
* LOOKAHEAD_ENABLED      == False -> no probe injection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case_jax.baseline_jax.timeline_jax import (
    simulate_planet_timeline_jax,
)
from pipeline.rulebase.case_jax.baseline_jax.world_features import (
    WorldFeatures,
    build_world_features,
    build_world_features_from_state,
)

pytestmark = pytest.mark.slow

PARITY_TOL = 1e-4
SEAT = 0


def _states() -> list[EnvState]:
    """Realistic EnvStates at several steps across a couple of seeds."""
    states: list[EnvState] = []
    for seed in (0, 1, 7):
        state = reset(seed=seed, num_agents=2)
        for stp in range(0, 70):
            state, _r, term = step(state, empty_actions())
            if stp in (5, 12, 55, 60):
                states.append(state)
            if bool(term):
                break
    state4 = reset(seed=3, num_agents=4)
    for _ in range(8):
        state4, _r, _t = step(state4, empty_actions())
    states.append(state4)
    return states


# Per-field comparison helpers ------------------------------------------------

_INT_FIELDS = (
    "planet_id",
    "owner",
    "ships",
    "prod",
    "comet_life",
    "reserve",
    "available",
    "comet_path_index",
    "comet_path_len",
    "plan_max_turns",
    "other_path_index",
    "other_path_len",
    "other_planet_id",
    "step",
    "player",
    "num_players",
    "remaining_steps",
    "my_total",
    "enemy_total",
    "max_enemy_strength",
    "my_prod",
    "enemy_prod",
    "static_neutral_count",
    "owner_strength",
    "horizon",
    "arr_owner",
    "arr_ships",
)
_BOOL_FIELDS = (
    "planet_valid",
    "is_static",
    "is_comet",
    "is_early",
    "is_opening",
    "is_late",
    "is_very_late",
    "is_four_player",
    "is_behind",
    "is_ahead",
    "is_dominating",
    "is_finishing",
    "arr_valid",
)
_FLOAT_FIELDS = (
    "xy",
    "radius",
    "indirect_wealth",
    "initial_xy",
    "initial_radius",
    "reaction_my_t",
    "reaction_enemy_t",
    "comet_path",
    "other_paths",
    "comet_radius",
    "ang_vel",
    "domination",
    "attack_margin_mult",
)


def _assert_features_equal(host: WorldFeatures, got: WorldFeatures) -> float:
    """Assert every field matches; return the max float divergence seen."""
    max_div = 0.0
    for name in _INT_FIELDS:
        a = np.asarray(getattr(host, name))
        b = np.asarray(getattr(got, name))
        assert np.array_equal(a, b), f"int field {name} mismatch"
    for name in _BOOL_FIELDS:
        a = np.asarray(getattr(host, name)).astype(bool)
        b = np.asarray(getattr(got, name)).astype(bool)
        assert np.array_equal(a, b), f"bool field {name} mismatch"
    for name in _FLOAT_FIELDS:
        a = np.asarray(getattr(host, name)).astype(np.float64)
        b = np.asarray(getattr(got, name)).astype(np.float64)
        # Reaction sentinels (BIG_REACTION 1e9) compare as "both large".
        if name in ("reaction_my_t", "reaction_enemy_t"):
            big = a >= 1e9
            assert np.array_equal(big, b >= 1e9), f"{name} sentinel mask mismatch"
            small = ~big
            div = float(np.max(np.abs(a[small] - b[small]))) if small.any() else 0.0
        else:
            div = float(np.max(np.abs(a - b))) if a.size else 0.0
        max_div = max(max_div, div)
    return max_div


def test_from_state_full_parity() -> None:
    max_div = 0.0
    for state in _states():
        obs = state_to_obs(state, player=SEAT)
        host = build_world_features(obs)
        got = build_world_features_from_state(state, SEAT)
        max_div = max(max_div, _assert_features_equal(host, got))
    assert max_div < PARITY_TOL, f"float divergence {max_div}"


def test_from_state_arrival_ledger_exact_and_timeline_invariant() -> None:
    """Raw arrival tables match EXACTLY, and the resulting timeline agrees too."""
    for state in _states():
        obs = state_to_obs(state, player=SEAT)
        host = build_world_features(obs)
        got = build_world_features_from_state(state, SEAT)

        # Raw ledger EXACT (owner / ships / valid).
        assert np.array_equal(np.asarray(host.arr_owner), np.asarray(got.arr_owner))
        assert np.array_equal(np.asarray(host.arr_ships), np.asarray(got.arr_ships))
        assert np.array_equal(np.asarray(host.arr_valid), np.asarray(got.arr_valid))

        # Order-invariant downstream check: the per-planet timeline.
        valid = np.asarray(host.planet_valid)
        owner = np.asarray(host.owner)
        ships = np.asarray(host.ships)
        prod = np.asarray(host.prod)
        horizon = int(host.horizon)
        for i in range(len(valid)):
            if not bool(valid[i]):
                continue
            tl_h = simulate_planet_timeline_jax(
                jnp.int32(int(owner[i])),
                jnp.float32(float(ships[i])),
                jnp.float32(float(prod[i])),
                host.arr_owner[i],
                host.arr_ships[i],
                host.arr_valid[i],
                jnp.int32(SEAT),
                horizon,
            )
            tl_g = simulate_planet_timeline_jax(
                jnp.int32(int(owner[i])),
                jnp.float32(float(ships[i])),
                jnp.float32(float(prod[i])),
                got.arr_owner[i],
                got.arr_ships[i],
                got.arr_valid[i],
                jnp.int32(SEAT),
                horizon,
            )
            assert int(tl_h.keep_needed) == int(tl_g.keep_needed)
            assert int(tl_h.min_owned) == int(tl_g.min_owned)
            assert int(tl_h.fall_turn) == int(tl_g.fall_turn)
            assert int(tl_h.first_enemy) == int(tl_g.first_enemy)
            assert np.array_equal(np.asarray(tl_h.owner_at), np.asarray(tl_g.owner_at))


def test_from_state_jits() -> None:
    """`build_world_features_from_state` is jax.jit-able (seat static)."""
    state = _states()[0]
    jitted = jax.jit(build_world_features_from_state, static_argnums=1)
    got = jitted(state, SEAT)
    host = build_world_features(state_to_obs(state, player=SEAT))
    assert int(got.my_total) == int(host.my_total)
    assert np.array_equal(np.asarray(got.arr_owner), np.asarray(host.arr_owner))


def test_from_state_vmaps_over_batch() -> None:
    """`jax.vmap` over a batch of 4 EnvStates agrees with the per-state host loop."""
    states = _states()[:4]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    vmapped = jax.vmap(build_world_features_from_state, in_axes=(0, None))(
        batched, SEAT
    )

    max_div = 0.0
    for b, state in enumerate(states):
        host = build_world_features(state_to_obs(state, player=SEAT))
        # Slice batch element b out of the vmapped WorldFeatures.
        sliced = jax.tree_util.tree_map(lambda arr: arr[b], vmapped)  # noqa: B023
        max_div = max(max_div, _assert_features_equal(host, sliced))
    assert max_div < PARITY_TOL, f"vmap float divergence {max_div}"
