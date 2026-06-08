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
from . import safety_jax as sf
from . import worldmodel_jax as wm
from .aim_jax import aim_with_prediction

PARTIAL_SOURCE_MIN_SHIPS = 6
_RESERVE_HORIZON = 110
# keep_needed candidate cap. Bounds the parallel survival search; planets with
# more ships fall back conservatively (rare early/mid; refine if it bites).
_RESERVE_MAX_SHIPS = 80
_REINFORCE_MIN_PRODUCTION = 2
_REINFORCE_MIN_FUTURE_TURNS = 40
_REINFORCE_MAX_TRAVEL_TURNS = 22
_REINFORCE_MAX_SOURCE_FRACTION = 0.75
_REINFORCE_SAFETY_MARGIN = 2
_REINFORCE_VALUE_MULT = 1.35
_ATTACK_COST_TURN_WEIGHT = 0.55
_FOLLOWUP_MIN_SHIPS = 8
# build_modes thresholds/bonuses (strategy_helpers.build_modes)
_AHEAD_DOMINATION = 0.18
_BEHIND_DOMINATION = -0.2
_AHEAD_ATTACK_MARGIN_BONUS = 0.08
_BEHIND_ATTACK_MARGIN_PENALTY = 0.05
_FINISHING_DOMINATION = 0.35
_FINISHING_ATTACK_MARGIN_BONUS = 0.08
_FINISHING_PROD_RATIO = 1.25


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

    # Reinforcement targets: my planets that will FALL within the horizon
    # (threatened_info.holds_full == False), with prod >= REINFORCE_MIN_PRODUCTION
    # and enough remaining steps. fall_turn/deficit feed the reinforce missions.
    remaining_steps = jnp.maximum(1, 500 - state.step)

    def threat_for(slot: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        t_arr = led_slot == slot
        a_eta = jnp.where(t_arr, led_eta, 10**9)
        a_own = jnp.clip(led_owner, 0, wm.NUM_PLAYERS - 1)
        a_shp = jnp.where(t_arr, led_ships, 0.0)
        return wm.threatened_info(
            owner[slot],
            state.planet_ships[slot],
            state.planet_prod[slot],
            seat_i,
            a_eta,
            a_own,
            a_shp,
            _RESERVE_HORIZON,
        )

    holds_full, fall_turn, deficit_hint = jax.vmap(threat_for)(pslot)
    is_threatened = (
        is_mine
        & ~holds_full
        & (fall_turn >= 0)
        & (state.planet_prod >= _REINFORCE_MIN_PRODUCTION)
        & (remaining_steps >= _REINFORCE_MIN_FUTURE_TURNS)
    )

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

    # modes (build_modes): owner_strength = planet ships + in-flight fleet ships
    # (matches
    # WorldModel.owner_strength). modes mirror strategy_helpers.build_modes
    # verbatim (thresholds + bonuses) — previously the bonuses were stubbed 0,
    # which over-sent by 1-3 ships mid/late game (ships-only-diff).
    fl_owner = state.fleet_owner
    fl_ships = state.fleet_ships.astype(jnp.float_)
    fl_valid = state.fleet_valid
    fl_mine = fl_valid & (fl_owner == seat_i)
    fl_enemy = fl_valid & (fl_owner != seat_i) & (fl_owner != -1)
    my_total = jnp.sum(jnp.where(is_mine, ships, 0.0)) + jnp.sum(
        jnp.where(fl_mine, fl_ships, 0.0)
    )
    enemy_total = jnp.sum(jnp.where(is_enemy, ships, 0.0)) + jnp.sum(
        jnp.where(fl_enemy, fl_ships, 0.0)
    )

    # max single-enemy strength (planets+fleets) for is_dominating.
    def _owner_strength(k: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(valid & (owner == k), ships, 0.0)) + jnp.sum(
            jnp.where(fl_valid & (fl_owner == k), fl_ships, 0.0)
        )

    enemy_strengths = jnp.array(
        [
            jnp.where(jnp.int32(k) == seat_i, 0.0, _owner_strength(jnp.int32(k)))
            for k in range(4)
        ]
    )
    max_enemy_strength = jnp.max(enemy_strengths)
    my_prod = jnp.sum(jnp.where(is_mine, prod, 0.0))
    enemy_prod = jnp.sum(jnp.where(is_enemy, prod, 0.0))

    domination = (my_total - enemy_total) / jnp.maximum(1.0, my_total + enemy_total)
    is_behind = domination < _BEHIND_DOMINATION
    is_ahead = domination > _AHEAD_DOMINATION
    is_dominating = is_ahead | (
        (max_enemy_strength > 0) & (my_total > max_enemy_strength * 1.25)
    )
    is_finishing = (
        (domination > _FINISHING_DOMINATION)
        & (my_prod > enemy_prod * _FINISHING_PROD_RATIO)
        & (state.step > 100)
    )
    attack_margin_mult = (
        1.0
        + jnp.where(is_ahead, _AHEAD_ATTACK_MARGIN_BONUS, 0.0)
        - jnp.where(is_behind, _BEHIND_ATTACK_MARGIN_PENALTY, 0.0)
        + jnp.where(is_finishing, _FINISHING_ATTACK_MARGIN_BONUS, 0.0)
    )

    # per-target reaction times (vectorized)
    def reaction(
        tx: jax.Array, ty: jax.Array, tr: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return fz.reaction_times(tx, ty, tr, px, py, radius, ships, is_mine, is_enemy)

    my_t, en_t = jax.vmap(reaction)(px, py, radius)
    indirect = fz.indirect_wealth(px, py, state.planet_prod, owner, valid, seat_i)

    # per-pair (src, tgt) aim + score
    def pair(
        src_i: jax.Array, tgt_i: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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
        angle, turns, aim_ix, aim_iy, aim_ok0 = aim_with_prediction(
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
        # plan_shot guards: full-trajectory sun safety + intercept tolerance
        # (the post-aim checks WorldModel.plan_shot applies; over-fire fix).
        aim_ok = sf.plan_shot_ok(
            sx,
            sy,
            sr,
            tx,
            ty,
            tox,
            toy,
            radius[tgt_i],
            tr,
            angle,
            turns,
            aim_ix,
            aim_iy,
            rough_ships,
            ang_vel,
            aim_ok0,
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
            indirect[tgt_i],
            turns,
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
        cap_eligible = (
            is_mine[src_i]
            & is_target[tgt_i]
            & aim_ok
            & ~veto
            & (value > 0)
            & (src_i != tgt_i)
        )

        # ---- reinforce alternative: tgt is a threatened OWN planet ----------
        # send a friendly fleet to defend. source_cap caps at a fraction of src
        # ships; need = deficit_hint; send = need + margin (mirrors
        # build_reinforcement_missions, single best source per target).
        # source budget at option-gen = full inventory (resolver's spent[] caps
        # actual sends during the scan via avail_now).
        source_cap = jnp.floor(ships[src_i] * _REINFORCE_MAX_SOURCE_FRACTION).astype(
            jnp.int32
        )
        r_need = deficit_hint[tgt_i]
        r_send = jnp.minimum(source_cap, r_need + _REINFORCE_SAFETY_MARGIN)
        r_value = (
            mj.target_value(
                owner[tgt_i],
                prod[tgt_i],
                ships[tgt_i],
                is_static_arr[tgt_i],
                indirect[tgt_i],
                turns,
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
            * _REINFORCE_VALUE_MULT
        )
        r_score = r_value / (r_send + turns * _ATTACK_COST_TURN_WEIGHT + 1.0)
        r_eligible = (
            is_mine[src_i]
            & is_threatened[tgt_i]
            & aim_ok
            & (src_i != tgt_i)
            & (turns <= _REINFORCE_MAX_TRAVEL_TURNS)
            & (turns <= fall_turn[tgt_i])
            & (r_need > 0)
            & (r_send >= r_need)
            & (r_value > 0)
        )
        # prefer capture when both somehow apply (disjoint in practice: target is
        # either mine-threatened or enemy/neutral).
        use_cap = cap_eligible
        out_score = jnp.where(use_cap, score, r_score)
        out_send = jnp.where(use_cap, send_cap, r_send)
        out_need = jnp.where(use_cap, need, r_need)
        eligible = cap_eligible | r_eligible
        is_reinf = r_eligible & ~cap_eligible
        return out_score, angle, out_send, out_need, eligible, is_reinf

    idx = jnp.arange(MAX_PLANETS)
    src_grid, tgt_grid = jnp.meshgrid(idx, idx, indexing="ij")
    score, angle, send_cap, need_grid, elig, is_reinf_g = jax.vmap(jax.vmap(pair))(
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
    f_reinf = is_reinf_g.reshape(flat)
    f_src = src_grid.reshape(flat)
    f_tgt = tgt_grid.reshape(flat)

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
        # capture uses attack budget (available = ships - reserve); reinforce
        # uses source_inventory_left (ships - spent, reserve not subtracted —
        # mirrors process_single_source_mission's reinforce branch).
        avail_now = jnp.where(
            f_reinf[oi], ships[src] - spent[src], available[src] - spent[src]
        )
        # Single-source semantics (process_single_source_mission): this source
        # must INDEPENDENTLY satisfy `need` (send_limit < missing → skip). We do
        # NOT let other launches' commitments reduce need below the single-source
        # threshold (that over-fired: a tiny source "finishing" a target). Track
        # committed only to suppress a 2nd source piling onto an already-covered
        # target this turn (need - committed <= 0 → already handled).
        need_now = f_need[oi]
        already = committed[tgt] >= need_now
        send = jnp.minimum(f_sendcap[oi], avail_now.astype(jnp.int32))
        # `out` now counts launches already emitted by this source (0/1/2).
        # 1st launch = main mission; 2nd = followup (allowed only if the source
        # still has >= FOLLOWUP_MIN_SHIPS attack budget left, mirroring
        # emit_followup_moves). Capture-only for the 2nd (reinforce stays 1st).
        count = out[src]
        is_followup = count >= 1
        followup_ok = (~is_followup) | (
            (available[src] - spent[src]) >= _FOLLOWUP_MIN_SHIPS
        )
        fire = (
            f_elig[oi]
            & (need_now > 0)
            & (send >= need_now)
            & (send >= 1)
            & ~already
            & (count < 2)
            & followup_ok
            & ~(is_followup & f_reinf[oi])  # followup is capture-only
        )
        send = jnp.where(fire, send, 0)
        spent = spent.at[src].add(jnp.where(fire, send, 0))
        committed = committed.at[tgt].add(jnp.where(fire, send, 0))
        out = out.at[src].add(jnp.where(fire, 1, 0))
        return (spent, committed, out), (
            jnp.where(fire, src, -1),
            jnp.where(fire, f_angle[oi], 0.0),
            jnp.where(fire, send, 0),
        )

    # Bind carry float dtype to `ships` (== jnp.float_ at the active x64 config)
    # so the scan carry init and body agree under both float32 and x64. A bare
    # `jnp.float_` here was baked to float32 at import while the body computes in
    # float64 under x64, tripping lax.scan's carry-dtype equality check.
    _f = ships.dtype
    init = (
        jnp.zeros(MAX_PLANETS, _f),  # spent[src]
        jnp.zeros(MAX_PLANETS, _f),  # committed[tgt]
        jnp.zeros(MAX_PLANETS, jnp.int32),  # out[src] = launch count
    )
    (_spent, _committed, _out), (emit_src, emit_angle, emit_send) = jax.lax.scan(
        resolve_step, init, order
    )

    # Pack all fired emissions (a source may fire twice: main + followup) into
    # the first free action slots, in score order.
    out_pid = jnp.full(MAX_LAUNCHES_PER_AGENT, -1.0, dtype=_f)
    out_ang = jnp.zeros(MAX_LAUNCHES_PER_AGENT, _f)
    out_snd = jnp.zeros(MAX_LAUNCHES_PER_AGENT, _f)

    def pack(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], k: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        op, oa, os, slot = carry
        s = emit_src[k]
        fired = (s >= 0) & (slot < MAX_LAUNCHES_PER_AGENT)
        wslot = jnp.clip(slot, 0, MAX_LAUNCHES_PER_AGENT - 1)
        op = jnp.where(fired, op.at[wslot].set(pid[s].astype(jnp.float_)), op)
        oa = jnp.where(fired, oa.at[wslot].set(emit_angle[k]), oa)
        os = jnp.where(fired, os.at[wslot].set(emit_send[k].astype(jnp.float_)), os)
        return (op, oa, os, slot + jnp.where(fired, 1, 0)), None

    (out_pid, out_ang, out_snd, _slot), _ = jax.lax.scan(
        pack, (out_pid, out_ang, out_snd, jnp.int32(0)), jnp.arange(flat)
    )

    actions = jnp.stack([out_pid, out_ang, out_snd], axis=-1)
    assert actions.shape == (MAX_LAUNCHES_PER_AGENT, 3)
    return actions


from functools import partial  # noqa: E402


@partial(jax.jit, static_argnames=("seat",))
def compute_actions_jax_jit(state: EnvState, seat: int) -> jax.Array:
    """jit-compiled entry (seat static). Use this in self-play/vmap rollouts."""
    return compute_actions_jax(state, seat)
