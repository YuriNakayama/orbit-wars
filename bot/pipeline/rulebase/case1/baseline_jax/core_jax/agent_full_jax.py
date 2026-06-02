"""Faithful JAX agent assembly (capture missions + score-sorted resolver).

Composes the parity-verified core_jax pieces into compute_actions_jax:
  per (src, tgt): aim_with_prediction → turns → need → opening_filter veto →
  send_cap = min(avail, preferred_send) → value → score. Then flatten all
  (src,tgt) options, sort by -score, and lax.scan accumulating spent[src] +
  committed[tgt] (the mission_resolver / planned_commitments loop, PoC2). One
  launch per source.

available = ships - keep_needed reserve (arrival ledger from in-flight fleets).
Non-comet targets only; reinforce / swarm / crash / followup / evac TODO.
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        sx, sy, sr = px[src_i], py[src_i], radius[src_i]
        tx, ty, tr = px[tgt_i], py[tgt_i], radius[tgt_i]
        tox, toy = ixx[tgt_i], ixy[tgt_i]
        avail = available[src_i]
        t_owner = owner[tgt_i]
        t_prod = prod[tgt_i]
        t_ships = ships[tgt_i]
        # arrivals targeting THIS planet (in-flight fleets), for projection.
        tgt_arr = led_slot == tgt_i
        a_eta = jnp.where(tgt_arr, led_eta, 10**9)
        a_own = jnp.clip(led_owner, 0, wm.NUM_PLAYERS - 1)
        a_shp = jnp.where(tgt_arr, led_ships, 0.0)
        # rough aim first (need a turns estimate to project to arrival)
        rough_ships = jnp.maximum(
            1.0, jnp.minimum(avail, jnp.maximum(PARTIAL_SOURCE_MIN_SHIPS, t_ships + 1))
        )
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
            rough_ships,
            ang_vel,
        )
        # need projects the target to `turns` accounting for in-flight fleets.
        # owner==me at arrival → need 0 (don't re-attack what's already inbound).
        need = wm.ships_needed_to_capture(
            t_owner.astype(jnp.int32),
            t_ships,
            t_prod,
            seat_i,
            a_eta,
            a_own,
            a_shp,
            turns,
            _RESERVE_HORIZON,
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
        eligible = (
            is_mine[src_i]
            & is_target[tgt_i]
            & aim_ok
            & ~veto
            & (value > 0)
            & (src_i != tgt_i)
        )
        return score, angle, send_cap, need, eligible

    idx = jnp.arange(MAX_PLANETS)
    src_grid, tgt_grid = jnp.meshgrid(idx, idx, indexing="ij")
    score, angle, send_cap, need_grid, elig = jax.vmap(jax.vmap(pair))(
        src_grid, tgt_grid
    )

    # Flatten all (src,tgt) capture options and sort by -score (mission_resolver
    # order). Then lax.scan accumulating spent[P] and committed[P] (ships
    # committed per target). This mirrors plan_moves' sorted-mission loop +
    # source_attack_left / planned_commitments (PoC2 structure).
    flat = MAX_PLANETS * MAX_PLANETS
    f_score = score.reshape(flat)
    f_angle = angle.reshape(flat)
    f_sendcap = send_cap.reshape(flat)
    f_need = need_grid.reshape(flat)
    f_elig = elig.reshape(flat)
    f_src = src_grid.reshape(flat)
    f_tgt = tgt_grid.reshape(flat)
    f_pid = pid[f_src]

    keyed = jnp.where(f_elig, f_score, -jnp.inf)
    order = jnp.argsort(-keyed)  # descending score

    def resolve_step(
        carry: tuple[jax.Array, jax.Array, jax.Array], oi: jax.Array
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]
    ]:
        spent, committed, out = carry
        src = f_src[oi]
        tgt = f_tgt[oi]
        avail_now = available[src] - spent[src]
        need_now = jnp.maximum(0, f_need[oi] - committed[tgt])
        send = jnp.minimum(f_sendcap[oi], avail_now.astype(jnp.int32))
        fire = f_elig[oi] & (need_now > 0) & (send >= need_now) & (send >= 1)
        send = jnp.where(fire, send, 0)
        spent = spent.at[src].add(jnp.where(fire, send, 0))
        committed = committed.at[tgt].add(jnp.where(fire, send, 0))
        # record one launch per source: only emit if this source hasn't fired.
        emit = fire & (out[src] < 0)
        new_out_pid = jnp.where(emit, f_pid[oi], out[src])
        out = out.at[src].set(new_out_pid)
        return (spent, committed, out), (
            jnp.where(emit, src, -1),
            jnp.where(emit, f_angle[oi], 0.0),
            jnp.where(emit, send, 0),
        )

    init = (
        jnp.zeros(MAX_PLANETS, jnp.float_),
        jnp.zeros(MAX_PLANETS, jnp.float_),
        jnp.full(MAX_PLANETS, -1.0),
    )
    (_spent, _committed, _out), (emit_src, emit_angle, emit_send) = jax.lax.scan(
        resolve_step, init, order
    )

    # Collapse per-source emissions (one per source) into the action rows.
    # For each source slot, find its emitted angle/send (last emit wins, but
    # emit only fires once per source so it's unique).
    src_fired = jnp.zeros(MAX_PLANETS, jnp.bool_)
    src_angle = jnp.zeros(MAX_PLANETS, jnp.float_)
    src_send = jnp.zeros(MAX_PLANETS, jnp.float_)

    def collapse(
        carry: tuple[jax.Array, jax.Array, jax.Array], k: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        fired, ang, snd = carry
        s = emit_src[k]
        valid_emit = s >= 0
        fired = jnp.where(valid_emit, fired.at[s].set(True), fired)
        ang = jnp.where(valid_emit, ang.at[s].set(emit_angle[k]), ang)
        snd = jnp.where(valid_emit, snd.at[s].set(emit_send[k].astype(jnp.float_)), snd)
        return (fired, ang, snd), None

    (src_fired, src_angle, src_send), _ = jax.lax.scan(
        collapse, (src_fired, src_angle, src_send), jnp.arange(flat)
    )

    can_fire = is_mine & src_fired & (src_send >= 1)
    from_pid = jnp.where(can_fire, pid, -1).astype(jnp.float_)
    angle_col = jnp.where(can_fire, src_angle, 0.0)
    ships_col = jnp.where(can_fire, src_send, 0.0)
    actions = jnp.stack([from_pid, angle_col, ships_col], axis=-1)
    assert actions.shape == (MAX_LAUNCHES_PER_AGENT, 3)
    return actions


from functools import partial  # noqa: E402


@partial(jax.jit, static_argnames=("seat",))
def compute_actions_jax_jit(state: EnvState, seat: int) -> jax.Array:
    """jit-compiled entry (seat static). Use this in self-play/vmap rollouts."""
    return compute_actions_jax(state, seat)
