"""Faithful JAX agent assembly (capture-single-source slice).

Composes the parity-verified core_jax pieces into compute_actions_jax. This
FIRST slice implements the single-source capture path (option_collector's
capture mission + the single-source resolver), which PoC1 showed dominates
turn-0 and opening play. Reinforce / swarm / crash / followup / evac are added
in later slices.

Pipeline (mirrors option_collector + plan_moves for capture/single):
  per (src, tgt): aim_with_prediction → turns → need = ceil(proj_ships)+1 →
  opening_filter veto → send_cap = min(avail, preferred_send) → value → score.
  Then: per source pick best-scoring affordable target (send_cap >= need); emit
  [src_pid, angle, send] with send sized like the resolver.

Non-comet only (comet targets skipped this slice). Uses available = ships -
keep_needed reserve (keep_needed needs arrivals; at no-fleet turns reserve=0).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from orbit_wars_jax.constants import MAX_PLANETS
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT

from . import featurize_jax as fz
from . import missions_jax as mj
from . import worldmodel_jax as wm
from .aim_jax import aim_with_prediction

PARTIAL_SOURCE_MIN_SHIPS = 6
_RESERVE_HORIZON = 110
# keep_needed candidate cap. Bounds the parallel survival search; planets with
# more ships fall back conservatively (rare early/mid; refine if it bites).
_RESERVE_MAX_SHIPS = 80


def compute_actions_jax(state: EnvState, seat: int) -> jax.Array:
    """Capture-single-source slice. Returns (MAX_LAUNCHES_PER_AGENT, 3)."""
    seat_i = jnp.int32(seat)
    pid = state.planet_id
    owner = state.planet_owner
    xy = state.planet_xy
    ix = state.planet_initial_xy
    radius = state.planet_radius.astype(jnp.float_)
    ships = state.planet_ships.astype(jnp.float_)
    prod = state.planet_prod.astype(jnp.float_)
    valid = state.planet_valid
    is_comet = state.planet_is_comet
    ang_vel = state.angular_velocity.astype(jnp.float_)

    is_mine = valid & (owner == seat_i)
    is_enemy = valid & (owner != seat_i) & (owner != -1)
    is_target = valid & (owner != seat_i) & ~is_comet  # enemy + neutral, non-comet

    px, py = xy[:, 0], xy[:, 1]
    ixx, ixy = ix[:, 0], ix[:, 1]

    # available = ships - keep_needed reserve. Build per-fleet arrival ledger
    # (target slot + eta) then keep_needed per my-planet (no-arrival → reserve 0).
    led_slot, led_eta, led_owner = fz.build_arrival_ledger(
        state.fleet_xy[:, 0],
        state.fleet_xy[:, 1],
        state.fleet_angle,
        state.fleet_ships.astype(jnp.float_),
        state.fleet_owner,
        state.fleet_valid,
        px,
        py,
        radius,
        valid,
    )
    led_ships = state.fleet_ships.astype(jnp.float_)
    pslot = jnp.arange(MAX_PLANETS)

    def reserve_for(slot: jax.Array) -> jax.Array:
        return wm.compute_reserve_per_planet(
            owner[slot],
            state.planet_ships[slot],
            state.planet_prod[slot],
            seat_i,
            led_slot,
            led_eta,
            led_owner,
            led_ships,
            slot,
            _RESERVE_HORIZON,
            _RESERVE_MAX_SHIPS,
        )

    reserve = jax.vmap(reserve_for)(pslot).astype(jnp.float_)
    available = jnp.where(is_mine, jnp.maximum(0.0, ships - reserve), 0.0)

    is_static_arr = ~state.planet_is_rotating  # precomputed at reset
    is_opening = state.step < 80
    is_early = state.step < 40
    is_late = (500 - state.step) < 60
    num_owners = jnp.sum(
        jnp.array([jnp.any(valid & (owner == k)) for k in range(4)])
    ) + jnp.sum(jnp.array([0]))  # players present; fleets ignored (turn snapshot)
    is_four_player = num_owners >= 4
    remaining = jnp.maximum(1, 500 - state.step)
    static_neutral_count = jnp.sum(is_target & (owner == -1) & is_static_arr)

    # modes (turn-snapshot approximation: strengths from planets only).
    my_total = jnp.sum(jnp.where(is_mine, ships, 0.0))
    enemy_total = jnp.sum(jnp.where(is_enemy, ships, 0.0))
    domination = (my_total - enemy_total) / jnp.maximum(1.0, my_total + enemy_total)
    is_behind = domination < -0.15  # BEHIND_DOMINATION (approx; refine later)
    is_ahead = domination > 0.15
    is_dominating = is_ahead
    is_finishing = jnp.asarray(False)  # step>100 + prod ratio; opening slice skips
    attack_margin_mult = (
        1.0 + jnp.where(is_ahead, 0.0, 0.0) - jnp.where(is_behind, 0.0, 0.0)
    )  # AHEAD/BEHIND bonuses are 0 in opening; refined later

    # per-target reaction times (vectorized)
    def reaction(
        tx: jax.Array, ty: jax.Array, tr: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return fz.reaction_times(tx, ty, tr, px, py, radius, ships, is_mine, is_enemy)

    my_t, en_t = jax.vmap(reaction)(px, py, radius)

    # per-pair (src, tgt) aim + score
    def pair(
        src_i: jax.Array, tgt_i: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        sx, sy, sr = px[src_i], py[src_i], radius[src_i]
        tx, ty, tr = px[tgt_i], py[tgt_i], radius[tgt_i]
        tox, toy = ixx[tgt_i], ixy[tgt_i]
        avail = available[src_i]
        t_owner = owner[tgt_i]
        t_prod = prod[tgt_i]
        t_ships = ships[tgt_i]
        # need = ceil(target ships)+1 (no-commitment projection at this slice)
        need = jnp.where(t_owner == seat_i, 0, jnp.ceil(t_ships).astype(jnp.int32) + 1)
        angle, turns, _ix, _iy, aim_ok = aim_with_prediction(
            sx,
            sy,
            sr,
            tx,
            ty,
            tox,
            toy,
            radius[tgt_i],
            tr,
            jnp.maximum(
                1.0,
                jnp.minimum(avail, jnp.maximum(PARTIAL_SOURCE_MIN_SHIPS, t_ships + 1)),
            ),
            ang_vel,
        )
        veto = mj.opening_filter(
            t_owner,
            t_prod,
            is_static_arr[tgt_i],
            turns,
            need,
            avail.astype(jnp.int32),
            my_t[tgt_i],
            en_t[tgt_i],
            is_opening,
            is_four_player,
        )
        send_cap = jnp.minimum(
            avail.astype(jnp.int32),
            mj.preferred_send(
                t_owner,
                t_prod,
                need,
                turns,
                avail.astype(jnp.int32),
                is_static_arr[tgt_i],
                my_t[tgt_i],
                en_t[tgt_i],
                is_four_player,
                attack_margin_mult,
                is_finishing,
                seat_i,
            ),
        )
        value = mj.target_value(
            t_owner,
            t_prod,
            t_ships,
            is_static_arr[tgt_i],
            jnp.asarray(0.0),
            turns,  # indirect=0 slice
            my_t[tgt_i],
            en_t[tgt_i],
            remaining,
            is_opening,
            is_early,
            is_late,
            jnp.asarray(0.0),
            is_finishing,
            is_behind,
            is_dominating,
            seat_i,
        )
        expected_send = jnp.maximum(need, jnp.minimum(send_cap, send_cap)).astype(
            jnp.float_
        )
        raw = value / (expected_send + turns * mj.ATTACK_COST_TURN_WEIGHT + 1.0)
        score = mj.apply_score_modifiers(
            raw,
            t_owner,
            is_static_arr[tgt_i],
            is_early,
            is_four_player,
            static_neutral_count,
        )
        affordable = (
            is_mine[src_i]
            & is_target[tgt_i]
            & aim_ok
            & (need > 0)
            & ~veto
            & (send_cap >= need)
            & (send_cap >= 1)
            & (value > 0)
            & (src_i != tgt_i)
        )
        return score, angle, send_cap, affordable

    idx = jnp.arange(MAX_PLANETS)
    src_grid, tgt_grid = jnp.meshgrid(idx, idx, indexing="ij")
    score, angle, send_cap, ok = jax.vmap(jax.vmap(pair))(src_grid, tgt_grid)

    masked = jnp.where(ok, score, -jnp.inf)
    best_tgt = jnp.argmax(masked, axis=1)  # per source
    row_ok = jnp.any(ok, axis=1)

    chosen_angle = jnp.take_along_axis(angle, best_tgt[:, None], axis=1).squeeze(-1)
    chosen_send = jnp.take_along_axis(send_cap, best_tgt[:, None], axis=1).squeeze(-1)

    can_fire = is_mine & row_ok & (chosen_send >= 1)
    from_pid = jnp.where(can_fire, pid, -1).astype(jnp.float_)
    angle_col = jnp.where(can_fire, chosen_angle, 0.0)
    ships_col = jnp.where(can_fire, chosen_send, 0).astype(jnp.float_)
    actions = jnp.stack([from_pid, angle_col, ships_col], axis=-1)
    assert actions.shape == (MAX_LAUNCHES_PER_AGENT, 3)
    return actions


from functools import partial  # noqa: E402


@partial(jax.jit, static_argnames=("seat",))
def compute_actions_jax_jit(state: EnvState, seat: int) -> jax.Array:
    """jit-compiled entry (seat static). Use this in self-play/vmap rollouts."""
    return compute_actions_jax(state, seat)
