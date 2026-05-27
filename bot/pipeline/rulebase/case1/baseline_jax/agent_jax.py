"""JAX rule-based agent — lightweight opponent for PPO training.

Strategy (single-pass, jit-friendly):
  1. Compute per-source surplus = ships - reserve, where reserve covers ~3
     turns of self-production (a coarse approximation of the Python
     `WorldModel` defense buffer).
  2. Build the (S, T) pair-score matrix between my planets (S) and all
     non-self planets (T). score favours nearby, high-production, easy-to-
     capture targets.
  3. Per-source argmax target. Ships sent = clip(need, 1, surplus) where
     `need = ceil(target.ships + travel_time * target.prod) + 1` (rough
     parity-of-arrival approximation, ignoring orbit prediction).
  4. Mask out invalid emissions (no surplus / no valid target / target is
     self / ships<=0) → sentinel -1 so jax_env.step ignores them.

This produces up to MAX_PLANETS launches per call, exactly matching
`MAX_LAUNCHES_PER_AGENT` in `jax_env.step`. The function is pure and jit-
able: it takes an `EnvState` and returns the action tensor shape expected
by `env.step` (with the other agents' rows untouched — the caller fills
those before calling `step`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from orbit_wars_jax.constants import MAX_PLANETS
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT

# Vendor constants — replicated to avoid pulling in the Python sim.
# These match `simulator/python/orbit_wars_vendor/orbit_wars.py`.
SUN_RADIUS: float = 10.0
PRODUCTION_TURNS_RESERVE: int = 3
SCORE_PROD_WEIGHT: float = 8.0
NEED_BUFFER: int = 1


def _fleet_speed(ships: jax.Array) -> jax.Array:
    """Vendor: speed = max(0.5, 2.0 - 0.05 * sqrt(ships))."""
    return jnp.maximum(0.5, 2.0 - 0.05 * jnp.sqrt(jnp.maximum(1.0, ships)))


def compute_actions_jax(state: EnvState, seat: int) -> jax.Array:
    """Return the (MAX_LAUNCHES_PER_AGENT, 3) launch tensor for one seat.

    Caller is responsible for stitching this row into the full
    `(NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3)` action tensor.

    Columns: (from_planet_id, angle_rad, ships). Sentinel `-1` in column 0
    suppresses the launch.
    """
    pid = state.planet_id  # (P,)
    owner = state.planet_owner  # (P,)
    xy = state.planet_xy  # (P, 2)
    ships = state.planet_ships.astype(jnp.float32)  # (P,)
    prod = state.planet_prod.astype(jnp.float32)  # (P,)
    valid = state.planet_valid  # (P,)

    seat_i32 = jnp.int32(seat)
    is_mine = valid & (owner == seat_i32)  # (P,)
    is_target = valid & (owner != seat_i32)  # (P,) — enemy + neutral

    # ---- reserve: cover ~3 turns of self-production (rough defense). ----
    reserve = jnp.ceil(prod * jnp.float32(PRODUCTION_TURNS_RESERVE)).astype(jnp.float32)
    reserve = jnp.minimum(reserve, ships)
    surplus = jnp.where(is_mine, jnp.maximum(0.0, ships - reserve), 0.0)  # (P,)

    # ---- pairwise (src x tgt) distance + travel time -------------------
    src_xy = xy[:, None, :]  # (P, 1, 2)
    tgt_xy = xy[None, :, :]  # (1, P, 2)
    dxdy = tgt_xy - src_xy  # (P, P, 2)
    dx = dxdy[..., 0]
    dy = dxdy[..., 1]
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)  # (P, P)

    # Travel time uses surplus-based speed (over-approximation: use sqrt
    # of average surplus as ships count proxy). For the per-pair speed we
    # estimate by surplus[src] floored at 1.
    src_ships_proxy = jnp.maximum(1.0, surplus)[:, None]  # (P, 1)
    speed = _fleet_speed(src_ships_proxy)  # (P, 1)
    travel_turns = dist / speed  # (P, P)

    # ---- need to capture: ships_target + travel * prod_target + buffer -
    tgt_ships = ships[None, :]  # (1, P)
    tgt_prod = prod[None, :]  # (1, P)
    raw_need = jnp.ceil(tgt_ships + travel_turns * tgt_prod).astype(jnp.float32)
    need = raw_need + jnp.float32(NEED_BUFFER)

    # ---- pair score ----------------------------------------------------
    # Prefer high-production targets within reach. Subtract need so easy
    # captures rank above costly ones; divide by dist+1 to bias near.
    base_score = (tgt_prod * jnp.float32(SCORE_PROD_WEIGHT) - need) / (dist + 1.0)

    # mask invalid pairs.
    pair_valid = (
        is_mine[:, None]  # src is mine
        & is_target[None, :]  # tgt is enemy/neutral
        & (need <= surplus[:, None])  # we can afford the capture
        & (~jnp.eye(MAX_PLANETS, dtype=jnp.bool_))  # not self
    )
    pair_score = jnp.where(pair_valid, base_score, jnp.float32(-1e9))

    # ---- per-source argmax target --------------------------------------
    tgt_idx = jnp.argmax(pair_score, axis=1)  # (P,)
    row_has_valid = jnp.any(pair_valid, axis=1)  # (P,)

    # Gather chosen target metadata. (P,) each.
    chosen_need = jnp.take_along_axis(need, tgt_idx[:, None], axis=1).squeeze(-1)
    chosen_dx = jnp.take_along_axis(dx, tgt_idx[:, None], axis=1).squeeze(-1)
    chosen_dy = jnp.take_along_axis(dy, tgt_idx[:, None], axis=1).squeeze(-1)
    angle = jnp.arctan2(chosen_dy, chosen_dx)  # (P,)

    # Ships sent: enough to capture, capped by surplus.
    ships_to_send = jnp.clip(chosen_need, 1.0, surplus).astype(jnp.int32)  # (P,)

    can_fire = is_mine & row_has_valid & (ships_to_send > 0)

    # ---- pack into (MAX_LAUNCHES_PER_AGENT, 3) tensor ------------------
    # MAX_LAUNCHES_PER_AGENT == MAX_PLANETS so 1-to-1.
    from_pid = jnp.where(can_fire, pid, jnp.int32(-1)).astype(jnp.float32)
    angle_col = jnp.where(can_fire, angle, jnp.float32(0.0))
    ships_col = jnp.where(can_fire, ships_to_send, jnp.int32(0)).astype(jnp.float32)

    actions = jnp.stack([from_pid, angle_col, ships_col], axis=-1)  # (P, 3)
    assert actions.shape == (MAX_LAUNCHES_PER_AGENT, 3)
    return actions


__all__ = ["compute_actions_jax"]
