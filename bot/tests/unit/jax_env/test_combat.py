"""Unit tests for combat.resolve_combat."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_env.combat import resolve_combat
from jax_env.constants import (
    MAX_COMETS,
    MAX_COMET_PATH_LEN,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
)
from jax_env.state import EnvState


def _empty_state(
    *,
    planet_owner: list[int],
    planet_ships: list[int],
    fleets: list[tuple[int, int, int]],  # (owner, ships, hit_planet_slot)
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Build a minimal EnvState for combat unit tests."""
    n_planets = len(planet_owner)
    p_owner = np.full(MAX_PLANETS, -1, dtype=np.int32)
    p_ships = np.zeros(MAX_PLANETS, dtype=np.int32)
    p_valid = np.zeros(MAX_PLANETS, dtype=bool)
    for i in range(n_planets):
        p_owner[i] = planet_owner[i]
        p_ships[i] = planet_ships[i]
        p_valid[i] = True

    f_owner = np.full(MAX_FLEETS, -1, dtype=np.int32)
    f_ships = np.zeros(MAX_FLEETS, dtype=np.int32)
    f_valid = np.zeros(MAX_FLEETS, dtype=bool)
    hit_slot = np.full(MAX_FLEETS, -1, dtype=np.int32)
    hit_mask = np.zeros(MAX_FLEETS, dtype=bool)
    for i, (o, s, slot) in enumerate(fleets):
        f_owner[i] = o
        f_ships[i] = s
        f_valid[i] = True
        hit_slot[i] = slot
        hit_mask[i] = True

    state = EnvState(
        planet_id=jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        planet_owner=jnp.asarray(p_owner),
        planet_xy=jnp.zeros((MAX_PLANETS, 2), dtype=jnp.float32),
        planet_radius=jnp.ones(MAX_PLANETS, dtype=jnp.float32),
        planet_ships=jnp.asarray(p_ships),
        planet_prod=jnp.zeros(MAX_PLANETS, dtype=jnp.int32),
        planet_initial_xy=jnp.zeros((MAX_PLANETS, 2), dtype=jnp.float32),
        planet_valid=jnp.asarray(p_valid),
        planet_is_comet=jnp.zeros(MAX_PLANETS, dtype=bool),
        planet_is_rotating=jnp.zeros(MAX_PLANETS, dtype=bool),
        fleet_owner=jnp.asarray(f_owner),
        fleet_xy=jnp.zeros((MAX_FLEETS, 2), dtype=jnp.float32),
        fleet_angle=jnp.zeros(MAX_FLEETS, dtype=jnp.float32),
        fleet_ships=jnp.asarray(f_ships),
        fleet_from_pid=jnp.full(MAX_FLEETS, -1, dtype=jnp.int32),
        fleet_valid=jnp.asarray(f_valid),
        comet_paths=jnp.zeros(
            (MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2), dtype=jnp.float32
        ),
        comet_path_len=jnp.zeros(MAX_COMETS, dtype=jnp.int32),
        comet_spawn_step=jnp.zeros(MAX_COMETS, dtype=jnp.int32),
        comet_path_index=jnp.full(MAX_COMETS, -1, dtype=jnp.int32),
        comet_planet_slot=jnp.full((MAX_COMETS, 4), -1, dtype=jnp.int32),
        comet_initial_ships=jnp.zeros(MAX_COMETS, dtype=jnp.int32),
        comet_active=jnp.zeros(MAX_COMETS, dtype=bool),
        angular_velocity=jnp.asarray(0.03, dtype=jnp.float32),
        step=jnp.asarray(0, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        next_planet_id=jnp.asarray(0, dtype=jnp.int32),
        num_agents=jnp.asarray(2, dtype=jnp.int32),
        terminated=jnp.asarray(False),
        rewards=jnp.zeros(NUM_AGENTS_MAX, dtype=jnp.float32),
    )
    return state, jnp.asarray(hit_slot), jnp.asarray(hit_mask)


def test_single_attacker_overwhelms_neutral() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[-1],
        planet_ships=[5],
        fleets=[(0, 20, 0)],
    )
    out = resolve_combat(state, slot, mask)
    # survivor = 20 - 0 = 20 (only attacker), planet owner=-1, ships=5
    # different owner: 5 - 20 = -15 -> flip owner=0, ships=15
    assert int(out.planet_owner[0]) == 0
    assert int(out.planet_ships[0]) == 15


def test_tied_attackers_cancel() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[-1],
        planet_ships=[3],
        fleets=[(0, 10, 0), (1, 10, 0)],
    )
    out = resolve_combat(state, slot, mask)
    # tied -> survivor_ships=0 -> planet untouched
    assert int(out.planet_owner[0]) == -1
    assert int(out.planet_ships[0]) == 3


def test_attacker_and_reinforcement() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[0],
        planet_ships=[5],
        fleets=[(0, 8, 0), (1, 3, 0)],
    )
    out = resolve_combat(state, slot, mask)
    # top1=agent0:8, top2=agent1:3, survivor=5 owner=0
    # same owner -> ships += 5 -> 10
    assert int(out.planet_owner[0]) == 0
    assert int(out.planet_ships[0]) == 10


def test_defender_survives() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[0],
        planet_ships=[100],
        fleets=[(1, 30, 0)],
    )
    out = resolve_combat(state, slot, mask)
    # survivor owner=1, ships=30; planet owner=0 different
    # 100 - 30 = 70 (no flip)
    assert int(out.planet_owner[0]) == 0
    assert int(out.planet_ships[0]) == 70


def test_two_planets_independent() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[-1, 0],
        planet_ships=[5, 100],
        fleets=[(0, 20, 0), (1, 30, 1)],
    )
    out = resolve_combat(state, slot, mask)
    assert int(out.planet_owner[0]) == 0
    assert int(out.planet_ships[0]) == 15
    assert int(out.planet_owner[1]) == 0
    assert int(out.planet_ships[1]) == 70


def test_fleet_marked_invalid_after_combat() -> None:
    state, slot, mask = _empty_state(
        planet_owner=[-1],
        planet_ships=[5],
        fleets=[(0, 20, 0)],
    )
    out = resolve_combat(state, slot, mask)
    assert bool(out.fleet_valid[0]) is False
