"""EnvState pytree — fixed-shape, JAX-friendly representation.

Registered as a pytree so it can flow through `jax.jit` / `jax.vmap` as a single
argument. Use `EnvState.replace(**overrides)` to update a field; the underlying
arrays must keep the shape and dtype below.

`*_valid` mask arrays mark which slots are in use. Slot order is not stable
across steps (slots are recycled on fleet removal / comet expiration).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .constants import (
    MAX_COMETS,
    MAX_COMET_PATH_LEN,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
)


class EnvState(NamedTuple):
    """Fixed-shape Orbit Wars state.

    Field shapes are documented inline; all leaves are jnp arrays so the whole
    struct is a single pytree (NamedTuple is registered as a pytree by JAX).
    """

    # ---- planets (length MAX_PLANETS) ------------------------------------
    planet_id: jax.Array  # int32[MAX_PLANETS]    — vendor id; -1 = unused
    planet_owner: jax.Array  # int32[MAX_PLANETS] — -1 = neutral
    planet_xy: jax.Array  # float32[MAX_PLANETS, 2]
    planet_radius: jax.Array  # float32[MAX_PLANETS]
    planet_ships: jax.Array  # int32[MAX_PLANETS]
    planet_prod: jax.Array  # int32[MAX_PLANETS]
    planet_initial_xy: jax.Array  # float32[MAX_PLANETS, 2] — for orbital math
    planet_valid: jax.Array  # bool[MAX_PLANETS]
    planet_is_comet: jax.Array  # bool[MAX_PLANETS]
    # Pre-computed at reset: True iff the planet rotates (initial_orbital + r < limit).
    planet_is_rotating: jax.Array  # bool[MAX_PLANETS]

    # ---- fleets (length MAX_FLEETS) --------------------------------------
    fleet_owner: jax.Array  # int32[MAX_FLEETS]
    fleet_xy: jax.Array  # float32[MAX_FLEETS, 2]
    fleet_angle: jax.Array  # float32[MAX_FLEETS]
    fleet_ships: jax.Array  # int32[MAX_FLEETS]
    fleet_from_pid: jax.Array  # int32[MAX_FLEETS]
    fleet_valid: jax.Array  # bool[MAX_FLEETS]

    # ---- comets (length MAX_COMETS, 4 quadrant copies each) --------------
    # comet_paths[c, q, t, :] = (x, y) at step (spawn_step + t) for quadrant q.
    comet_paths: jax.Array  # float32[MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2]
    # comet_path_len[c] = actual length (<= MAX_COMET_PATH_LEN). 0 = unused.
    comet_path_len: jax.Array  # int32[MAX_COMETS]
    # spawn_step at which this comet should activate. 0 = unused slot.
    comet_spawn_step: jax.Array  # int32[MAX_COMETS]
    # path_index = current position along the path (-1 = not yet active or expired).
    comet_path_index: jax.Array  # int32[MAX_COMETS]
    # planet slot index for each (comet, quadrant) pair. -1 = unused.
    comet_planet_slot: jax.Array  # int32[MAX_COMETS, 4]
    # initial ships when the comet activates (set at reset, used at activation).
    comet_initial_ships: jax.Array  # int32[MAX_COMETS]
    # True once the comet has been activated (planets added to the planet table).
    comet_active: jax.Array  # bool[MAX_COMETS]

    # ---- global ----------------------------------------------------------
    angular_velocity: jax.Array  # float32 scalar
    step: jax.Array  # int32 scalar
    next_fleet_id: jax.Array  # int32 scalar — vendor parity, debug only
    next_planet_id: jax.Array  # int32 scalar — for new comet planets
    num_agents: jax.Array  # int32 scalar — 2 or 4
    terminated: jax.Array  # bool scalar
    rewards: jax.Array  # float32[NUM_AGENTS_MAX] — slots [num_agents:] are 0

    def replace(self, **overrides: Any) -> "EnvState":
        return self._replace(**overrides)


# ---------------------------------------------------------------------------
# Shape / dtype contract — used by tests and the reset path to validate
# initial state construction.
# ---------------------------------------------------------------------------
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "planet_id": (MAX_PLANETS,),
    "planet_owner": (MAX_PLANETS,),
    "planet_xy": (MAX_PLANETS, 2),
    "planet_radius": (MAX_PLANETS,),
    "planet_ships": (MAX_PLANETS,),
    "planet_prod": (MAX_PLANETS,),
    "planet_initial_xy": (MAX_PLANETS, 2),
    "planet_valid": (MAX_PLANETS,),
    "planet_is_comet": (MAX_PLANETS,),
    "planet_is_rotating": (MAX_PLANETS,),
    "fleet_owner": (MAX_FLEETS,),
    "fleet_xy": (MAX_FLEETS, 2),
    "fleet_angle": (MAX_FLEETS,),
    "fleet_ships": (MAX_FLEETS,),
    "fleet_from_pid": (MAX_FLEETS,),
    "fleet_valid": (MAX_FLEETS,),
    "comet_paths": (MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2),
    "comet_path_len": (MAX_COMETS,),
    "comet_spawn_step": (MAX_COMETS,),
    "comet_path_index": (MAX_COMETS,),
    "comet_planet_slot": (MAX_COMETS, 4),
    "comet_initial_ships": (MAX_COMETS,),
    "comet_active": (MAX_COMETS,),
    "angular_velocity": (),
    "step": (),
    "next_fleet_id": (),
    "next_planet_id": (),
    "num_agents": (),
    "terminated": (),
    "rewards": (NUM_AGENTS_MAX,),
}

EXPECTED_DTYPES: dict[str, Any] = {
    "planet_id": jnp.int32,
    "planet_owner": jnp.int32,
    "planet_xy": jnp.float32,
    "planet_radius": jnp.float32,
    "planet_ships": jnp.int32,
    "planet_prod": jnp.int32,
    "planet_initial_xy": jnp.float32,
    "planet_valid": jnp.bool_,
    "planet_is_comet": jnp.bool_,
    "planet_is_rotating": jnp.bool_,
    "fleet_owner": jnp.int32,
    "fleet_xy": jnp.float32,
    "fleet_angle": jnp.float32,
    "fleet_ships": jnp.int32,
    "fleet_from_pid": jnp.int32,
    "fleet_valid": jnp.bool_,
    "comet_paths": jnp.float32,
    "comet_path_len": jnp.int32,
    "comet_spawn_step": jnp.int32,
    "comet_path_index": jnp.int32,
    "comet_planet_slot": jnp.int32,
    "comet_initial_ships": jnp.int32,
    "comet_active": jnp.bool_,
    "angular_velocity": jnp.float32,
    "step": jnp.int32,
    "next_fleet_id": jnp.int32,
    "next_planet_id": jnp.int32,
    "num_agents": jnp.int32,
    "terminated": jnp.bool_,
    "rewards": jnp.float32,
}


def validate_state(state: EnvState) -> None:
    """Raise if any field has the wrong shape or dtype. For tests / reset."""
    for name, expected in EXPECTED_SHAPES.items():
        arr = getattr(state, name)
        if arr.shape != expected:
            raise AssertionError(
                f"EnvState.{name} shape={arr.shape}, expected {expected}"
            )
        if arr.dtype != EXPECTED_DTYPES[name]:
            raise AssertionError(
                f"EnvState.{name} dtype={arr.dtype}, expected {EXPECTED_DTYPES[name]}"
            )
