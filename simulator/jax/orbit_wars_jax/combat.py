"""Combat resolution — pure JAX, vmap/jit friendly.

Mirrors vendor `interpreter` rows 616-655: for each planet, sum incoming fleet
ships per agent, compute (top1 - top2) survivor, then apply against the planet.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import MAX_FLEETS, MAX_PLANETS, NUM_AGENTS_MAX
from .state import EnvState


def resolve_combat(
    state: EnvState,
    fleet_hit_planet_slot: jax.Array,
    fleet_hit_mask: jax.Array,
) -> EnvState:
    """Apply per-planet combat outcome.

    Args:
        state: current EnvState
        fleet_hit_planet_slot: int32[MAX_FLEETS] — slot index of planet hit by
            each fleet, or -1 if no hit.
        fleet_hit_mask: bool[MAX_FLEETS] — True iff fleet actually collided.
    """
    # Per (planet, agent) ship total: build (MAX_FLEETS, MAX_PLANETS, NUM_AGENTS_MAX)
    # mask and reduce.
    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)  # (P,)
    agent_idx = jnp.arange(NUM_AGENTS_MAX, dtype=jnp.int32)  # (A,)

    # (F, P): fleet hit this planet
    fp_mask = (
        fleet_hit_mask[:, None]
        & (fleet_hit_planet_slot[:, None] == planet_idx[None, :])
        & state.fleet_valid[:, None]
    )
    # (F, A): fleet owner == agent
    fa_mask = state.fleet_owner[:, None] == agent_idx[None, :]
    # Combine into (F, P, A) and weight by fleet_ships → reduce over F.
    weighted = (
        fp_mask[:, :, None].astype(jnp.int32)
        * fa_mask[:, None, :].astype(jnp.int32)
        * state.fleet_ships[:, None, None]
    )  # int32[F, P, A]
    planet_agent_ships = jnp.sum(weighted, axis=0)  # int32[P, A]

    # Top1 / Top2 per planet (across agents).
    sorted_desc = jnp.sort(planet_agent_ships, axis=-1)[:, ::-1]  # (P, A) descending
    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1]
    top_owner = jnp.argmax(planet_agent_ships, axis=-1).astype(jnp.int32)  # (P,)

    tie = top1 == top2
    survivor_ships_raw = jnp.where(tie, 0, top1 - top2)  # int32 (P,)
    has_attackers = top1 > 0
    survivor_owner = jnp.where(
        has_attackers & (~tie),
        top_owner,
        jnp.int32(-1),
    )

    # Apply per planet:
    #   if planet_owner == survivor_owner: planet_ships += survivor_ships
    #   else:                              planet_ships -= survivor_ships
    #                                      if < 0 -> flip owner, abs
    owner = state.planet_owner  # (P,)
    ships = state.planet_ships  # (P,)

    same_owner = owner == survivor_owner
    add_case_ships = ships + survivor_ships_raw
    sub_result = ships - survivor_ships_raw
    flip = sub_result < 0
    sub_case_ships = jnp.where(flip, -sub_result, sub_result)
    sub_case_owner = jnp.where(flip, survivor_owner, owner)

    # Skip when survivor_ships == 0 (tied attackers or no attackers).
    has_survivor = survivor_ships_raw > 0
    new_ships = jnp.where(
        has_survivor,
        jnp.where(same_owner, add_case_ships, sub_case_ships),
        ships,
    )
    new_owner = jnp.where(
        has_survivor,
        jnp.where(same_owner, owner, sub_case_owner),
        owner,
    )

    # Only valid planets can be mutated.
    new_ships = jnp.where(state.planet_valid, new_ships, ships)
    new_owner = jnp.where(state.planet_valid, new_owner, owner)

    # Mark fleets that collided as invalid (vendor removes them after combat
    # for both planet hits and out-of-bounds / sun crosses — the caller passes
    # an aggregate `fleet_hit_mask` that may also include those).
    new_fleet_valid = state.fleet_valid & (~fleet_hit_mask)

    return state._replace(
        planet_ships=new_ships,
        planet_owner=new_owner,
        fleet_valid=new_fleet_valid,
    )


__all__ = ["resolve_combat"]


# Static asserts for shape (helps catch refactors).
assert MAX_PLANETS > 0
assert MAX_FLEETS > 0
