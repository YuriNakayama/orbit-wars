"""CPU-side reset path. Constructs an `EnvState` for a given seed.

Mirrors the vendor `interpreter` initialization block (rows 343-395) so the
RNG path is byte-equal. Heavy lifting is done in numpy; final arrays are
converted to ``jnp`` only at the very end. This is not in the jit path.
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np

from .comet_gen import precompute_all_comets
from .constants import (
    CENTER,
    MAX_COMETS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
    ROTATION_RADIUS_LIMIT,
)
from .planet_gen import generate_planets
from .state import EnvState, validate_state


def reset(seed: int, num_agents: int = 2) -> EnvState:
    """Construct the initial `EnvState` for ``seed``.

    Vendor parity: the same ``seed`` produces the same planet placement,
    angular velocity, home assignment, and pre-computed comets.
    """
    if num_agents not in (2, 4):
        raise ValueError(f"num_agents must be 2 or 4, got {num_agents}")

    init_rng = random.Random(seed)
    angular_velocity = init_rng.uniform(0.025, 0.05)
    planets_raw = generate_planets(init_rng)

    # Home assignment — vendor rows 369-382.
    num_groups = len(planets_raw) // 4
    if num_groups > 0:
        home_group = init_rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_agents == 2:
            planets_raw[base][1] = 0
            planets_raw[base][5] = 10
            planets_raw[base + 3][1] = 1
            planets_raw[base + 3][5] = 10
        else:  # num_agents == 4
            for j in range(4):
                planets_raw[base + j][1] = j
                planets_raw[base + j][5] = 10

    n = len(planets_raw)
    if n > MAX_PLANETS:
        raise ValueError(
            f"seed={seed} produced {n} planets, exceeds MAX_PLANETS={MAX_PLANETS}"
        )

    # Build fixed-shape planet arrays.
    planet_id = np.full(MAX_PLANETS, -1, dtype=np.int32)
    planet_owner = np.full(MAX_PLANETS, -1, dtype=np.int32)
    planet_xy = np.zeros((MAX_PLANETS, 2), dtype=np.float32)
    planet_radius = np.zeros(MAX_PLANETS, dtype=np.float32)
    planet_ships = np.zeros(MAX_PLANETS, dtype=np.int32)
    planet_prod = np.zeros(MAX_PLANETS, dtype=np.int32)
    planet_initial_xy = np.zeros((MAX_PLANETS, 2), dtype=np.float32)
    planet_valid = np.zeros(MAX_PLANETS, dtype=bool)
    planet_is_comet = np.zeros(MAX_PLANETS, dtype=bool)
    planet_is_rotating = np.zeros(MAX_PLANETS, dtype=bool)

    # Vendor row layout: planet[i] = [id, owner, y, x, radius, ships, prod].
    # Our planet_xy stores (y, x) — same physical convention as vendor.
    for slot, p in enumerate(planets_raw):
        pid = int(p[0])
        owner = int(p[1])
        y = float(p[2])
        x = float(p[3])
        r = float(p[4])
        ships = int(p[5])
        prod = int(p[6])
        planet_id[slot] = pid
        planet_owner[slot] = owner
        planet_xy[slot, 0] = y
        planet_xy[slot, 1] = x
        planet_radius[slot] = r
        planet_ships[slot] = ships
        planet_prod[slot] = prod
        planet_initial_xy[slot, 0] = y
        planet_initial_xy[slot, 1] = x
        planet_valid[slot] = True
        # is_rotating: orbital_radius + r < ROTATION_RADIUS_LIMIT, computed
        # from initial position relative to (CENTER, CENTER).
        orbital = math.sqrt((y - CENTER) ** 2 + (x - CENTER) ** 2)
        planet_is_rotating[slot] = (orbital + r) < ROTATION_RADIUS_LIMIT

    # Pre-compute all comets.
    paths_arr, path_lens, initial_ships = precompute_all_comets(
        planets_raw, angular_velocity, seed
    )

    # Build empty fleet arrays.
    fleet_owner = np.full(MAX_FLEETS, -1, dtype=np.int32)
    fleet_xy = np.zeros((MAX_FLEETS, 2), dtype=np.float32)
    fleet_angle = np.zeros(MAX_FLEETS, dtype=np.float32)
    fleet_ships = np.zeros(MAX_FLEETS, dtype=np.int32)
    fleet_from_pid = np.full(MAX_FLEETS, -1, dtype=np.int32)
    fleet_valid = np.zeros(MAX_FLEETS, dtype=bool)

    # Comet bookkeeping.
    comet_spawn_step = np.array([50, 150, 250, 350, 450], dtype=np.int32)
    comet_path_index = np.full(MAX_COMETS, -1, dtype=np.int32)
    comet_planet_slot = np.full((MAX_COMETS, 4), -1, dtype=np.int32)
    comet_active = np.zeros(MAX_COMETS, dtype=bool)

    next_planet_id = max((int(p[0]) for p in planets_raw), default=-1) + 1

    state = EnvState(
        planet_id=jnp.asarray(planet_id),
        planet_owner=jnp.asarray(planet_owner),
        planet_xy=jnp.asarray(planet_xy),
        planet_radius=jnp.asarray(planet_radius),
        planet_ships=jnp.asarray(planet_ships),
        planet_prod=jnp.asarray(planet_prod),
        planet_initial_xy=jnp.asarray(planet_initial_xy),
        planet_valid=jnp.asarray(planet_valid),
        planet_is_comet=jnp.asarray(planet_is_comet),
        planet_is_rotating=jnp.asarray(planet_is_rotating),
        fleet_owner=jnp.asarray(fleet_owner),
        fleet_xy=jnp.asarray(fleet_xy),
        fleet_angle=jnp.asarray(fleet_angle),
        fleet_ships=jnp.asarray(fleet_ships),
        fleet_from_pid=jnp.asarray(fleet_from_pid),
        fleet_valid=jnp.asarray(fleet_valid),
        comet_paths=jnp.asarray(paths_arr),
        comet_path_len=jnp.asarray(path_lens),
        comet_spawn_step=jnp.asarray(comet_spawn_step),
        comet_path_index=jnp.asarray(comet_path_index),
        comet_planet_slot=jnp.asarray(comet_planet_slot),
        comet_initial_ships=jnp.asarray(initial_ships),
        comet_active=jnp.asarray(comet_active),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float32),
        step=jnp.asarray(0, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        next_planet_id=jnp.asarray(next_planet_id, dtype=jnp.int32),
        num_agents=jnp.asarray(num_agents, dtype=jnp.int32),
        terminated=jnp.asarray(False),
        rewards=jnp.zeros(NUM_AGENTS_MAX, dtype=jnp.float32),
    )

    if __debug__:
        validate_state(state)
    return state
