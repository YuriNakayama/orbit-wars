"""JAX-native reset (jit/vmap-friendly) for GPU rollouts.

Mirrors the STRATEGY of `reset.py` but runs entirely under `jax.jit` / `jax.vmap`
so the per-episode initial states can be built on-device inside the rollout
graph (eliminating the host-side `[reset(seed+i) for i in range(N)]` loop that
left the GPU idle). Uses `planet_gen_jax.generate_planets_jax` (lax.while_loop
rejection sampling into fixed buffers).

Intentional deviations from `reset.py` (the old host reset stays for vendor
parity):
  * RNG: `random.Random(seed)` → `jax.random` (vendor byte-stream parity
    abandoned; layouts differ but are self-consistent valid states — see
    planet_gen_jax docstring).
  * num_agents is a STATIC argument (2 or 4).
  * comets ARE generated (JAX-native, via comet_gen_jax.generate_comets_jax):
    ellipse rejection + arc-length resample + sun/static/orbiting overlap, all
    in lax.while_loop + vmap. Layouts differ from vendor (RNG parity abandoned)
    but satisfy the same validity constraints.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .comet_gen_jax import generate_comets_jax
from .constants import (
    MAX_COMETS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
)
from .planet_gen_jax import generate_planets_jax
from .state import EnvState

_ANG_VEL_MIN = 0.025
_ANG_VEL_MAX = 0.05


def reset_jax(key: jax.Array, num_agents: int = 2) -> EnvState:
    """Build an initial `EnvState` for `key` (jit/vmap-friendly).

    Same pytree (fields/shapes/dtypes) as `reset.py`'s output and passes
    `validate_state`. `num_agents` is static (2 or 4).
    """
    if num_agents not in (2, 4):
        raise ValueError(f"num_agents must be 2 or 4, got {num_agents}")

    key, k_ang, k_home, k_comet = jax.random.split(key, 4)
    angular_velocity = jax.random.uniform(
        k_ang, (), minval=_ANG_VEL_MIN, maxval=_ANG_VEL_MAX
    )

    buf = generate_planets_jax(key)
    count = buf.n_valid  # int32 scalar — number of valid planets (multiple of 4)

    # ---- Home assignment (vendor rows 369-382) --------------------------------
    # num_groups = count // 4; pick home_group in [0, num_groups-1]; base = hg*4.
    # 2 players: seats on base (slot offset 0) and base+3. 4 players: base+0..3.
    num_groups = jnp.maximum(count // 4, 1)
    home_group = jax.random.randint(k_home, (), 0, num_groups)
    base = home_group * 4

    owner = jnp.full(MAX_PLANETS, -1, dtype=jnp.int32)
    ships = buf.ships
    if num_agents == 2:
        s0 = base
        s1 = base + 3
        owner = owner.at[s0].set(0).at[s1].set(1)
        ships = ships.at[s0].set(10).at[s1].set(10)
    else:  # num_agents == 4
        idx = base + jnp.arange(4)
        owner = owner.at[idx].set(jnp.arange(4, dtype=jnp.int32))
        ships = ships.at[idx].set(10)
    # Only valid slots keep their owner (defensive: home_group always valid).
    owner = jnp.where(buf.valid, owner, -1)

    # ---- planet arrays --------------------------------------------------------
    slot = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    planet_id = jnp.where(buf.valid, slot, -1).astype(jnp.int32)
    planet_xy = buf.xy
    planet_initial_xy = buf.xy
    planet_radius = buf.radius
    planet_orbital = jnp.sqrt((buf.xy[:, 0] - 50.0) ** 2 + (buf.xy[:, 1] - 50.0) ** 2)
    planet_is_rotating = jnp.where(
        buf.valid, (planet_orbital + buf.radius) < 50.0, False
    )

    # ---- fleets (empty) -------------------------------------------------------
    fleet_owner = jnp.full(MAX_FLEETS, -1, dtype=jnp.int32)
    fleet_xy = jnp.zeros((MAX_FLEETS, 2), dtype=jnp.float32)
    fleet_angle = jnp.zeros(MAX_FLEETS, dtype=jnp.float32)
    fleet_ships = jnp.zeros(MAX_FLEETS, dtype=jnp.int32)
    fleet_from_pid = jnp.full(MAX_FLEETS, -1, dtype=jnp.int32)
    fleet_valid = jnp.zeros(MAX_FLEETS, dtype=jnp.bool_)

    # ---- comets (JAX-native generation via generate_comets_jax) ---------------
    comet_paths, comet_path_len, comet_initial_ships = generate_comets_jax(
        k_comet, buf, angular_velocity
    )
    comet_spawn_step = jnp.array([50, 150, 250, 350, 450], dtype=jnp.int32)
    comet_path_index = jnp.full(MAX_COMETS, -1, dtype=jnp.int32)
    comet_planet_slot = jnp.full((MAX_COMETS, 4), -1, dtype=jnp.int32)
    comet_active = jnp.zeros(MAX_COMETS, dtype=jnp.bool_)

    next_planet_id = count  # ids 0..count-1 used; next free id = count

    return EnvState(
        planet_id=planet_id,
        planet_owner=owner,
        planet_xy=planet_xy,
        planet_radius=planet_radius,
        planet_ships=ships.astype(jnp.int32),
        planet_prod=buf.prod.astype(jnp.int32),
        planet_initial_xy=planet_initial_xy,
        planet_valid=buf.valid,
        planet_is_comet=jnp.zeros(MAX_PLANETS, dtype=jnp.bool_),
        planet_is_rotating=planet_is_rotating,
        fleet_owner=fleet_owner,
        fleet_xy=fleet_xy,
        fleet_angle=fleet_angle,
        fleet_ships=fleet_ships,
        fleet_from_pid=fleet_from_pid,
        fleet_valid=fleet_valid,
        comet_paths=comet_paths,
        comet_path_len=comet_path_len,
        comet_spawn_step=comet_spawn_step,
        comet_path_index=comet_path_index,
        comet_planet_slot=comet_planet_slot,
        comet_initial_ships=comet_initial_ships,
        comet_active=comet_active,
        angular_velocity=angular_velocity.astype(jnp.float32),
        step=jnp.asarray(0, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        next_planet_id=next_planet_id.astype(jnp.int32),
        num_agents=jnp.asarray(num_agents, dtype=jnp.int32),
        terminated=jnp.asarray(False),
        rewards=jnp.zeros(NUM_AGENTS_MAX, dtype=jnp.float32),
    )
