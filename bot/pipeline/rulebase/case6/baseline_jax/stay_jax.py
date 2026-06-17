"""case6 STAY burst-hold, JAX port (mirror of baseline/missions/stay.py).

case6 adds a STAY pass that, *before* missions are collected, holds a source
planet's ships for one turn when waiting lets the next launch arrive EARLIER
(higher fleet speed from accumulated ships). The held ships are subtracted from
`source_attack_left`, suppressing that source's launches this turn.

Only the BURST motive is ported: `STAY_DEFENSE_ENABLED = False` in production,
so defense holds never fire. Burst is a pure function of the current board
(positions / ships / production), so it fires even on a fresh turn — unlike
case9's ANTI_PING_PONG which needs accumulated history.

Per `_build_burst_holds` / `_best_burst_target`:
  usable = available[src]  (= ships - reserve, already in WorldFeatures)
  skip if usable < STAY_BURST_MIN_SHIPS
  ships_next = usable + production
  skip if fleet_speed(ships_next) <= fleet_speed(usable)
  for each non-friendly target:
    eta_now  = travel_time(src, tgt, usable)        # current positions
    eta_next = travel_time(src, tgt, ships_next)
    eligible if eta_now <= STAY_BURST_MAX_TARGET_TURNS
               and (eta_now - (eta_next + 1)) >= STAY_BURST_MIN_GAIN
  if any target eligible -> hold ALL `usable` ships (available[src] -> 0)

The `consecutive_holds` cap (STAY_BURST_MAX_HOLD_TURNS) and the `doomed_candidates`
exclusion are cross-turn / timeline concerns handled on the host (see agent_jax);
this device-side pass computes the single-turn burst eligibility mask.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics_jax import fleet_speed_jax, travel_time_jax

STAY_BURST_MIN_GAIN: int = 1
STAY_BURST_MIN_SHIPS: int = 8
STAY_BURST_MAX_TARGET_TURNS: int = 30
BIG_TURNS_THRESH: int = 10**8


def burst_held_mask(
    xy: jax.Array,  # float32[P, 2]
    radius: jax.Array,  # float32[P]
    owner: jax.Array,  # int32[P] (-1 neutral)
    planet_valid: jax.Array,  # bool[P]
    available: jax.Array,  # int32[P] (= ships - reserve for mine, else 0)
    prod: jax.Array,  # int32[P]
    player: int,
) -> jax.Array:
    """Return bool[P]: True where the source's launches are burst-held this turn."""
    p = owner.shape[0]
    xs = xy[:, 0]
    ys = xy[:, 1]
    is_mine = planet_valid & (owner == player)

    usable = available  # available is already max(0, ships - reserve) for mine
    ships_next = usable + jnp.maximum(0, prod)
    speed_now = fleet_speed_jax(usable)
    speed_next = fleet_speed_jax(ships_next)

    src_ok = is_mine & (usable >= STAY_BURST_MIN_SHIPS) & (speed_next > speed_now)

    def per_target(s_idx: jax.Array, t_idx: jax.Array) -> jax.Array:
        not_self = s_idx != t_idx
        not_friendly = planet_valid[t_idx] & (owner[t_idx] != player)
        eta_now = travel_time_jax(
            xs[s_idx],
            ys[s_idx],
            radius[s_idx],
            xs[t_idx],
            ys[t_idx],
            radius[t_idx],
            usable[s_idx],
        )
        eta_next = travel_time_jax(
            xs[s_idx],
            ys[s_idx],
            radius[s_idx],
            xs[t_idx],
            ys[t_idx],
            radius[t_idx],
            ships_next[s_idx],
        )
        reachable = (eta_now < BIG_TURNS_THRESH) & (eta_next < BIG_TURNS_THRESH)
        within = eta_now <= STAY_BURST_MAX_TARGET_TURNS
        gain = eta_now - (eta_next + 1)
        return (
            not_self & not_friendly & reachable & within & (gain >= STAY_BURST_MIN_GAIN)
        )

    idx = jnp.arange(p, dtype=jnp.int32)

    def per_src(s_idx: jax.Array) -> jax.Array:
        any_target = jnp.any(jax.vmap(lambda t: per_target(s_idx, t))(idx))
        return src_ok[s_idx] & any_target

    return jax.vmap(per_src)(idx)
