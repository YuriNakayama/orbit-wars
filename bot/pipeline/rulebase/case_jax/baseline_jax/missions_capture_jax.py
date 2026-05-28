# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Fixed-shape (P_src, P_tgt) CAPTURE + SNIPE mission candidate grids.

`build_capture_grid` / `build_snipe_grid` mirror
`baseline/missions/build_capture_and_snipe_missions` /
`baseline/missions/build_snipe_mission` over the full source x target grid,
emitting one per-cell candidate that the swarm builder later consumes as
`source_options_by_target`.

Collection-time simplification (verified in `capture.py` + `plan_moves`)
-----------------------------------------------------------------------
`build_capture_and_snipe_missions` runs *before* the greedy commit loop, so
`planned_commitments` / `spent_total` are EMPTY:

* `src_available = world.available[src.id]` (the WorldFeatures.available field,
  no spent subtraction).
* `ships_needed_to_capture(target, turns, {})` reduces to the BASE-timeline
  projection: `state_at_timeline_jax` on the target's base `owner_at`/`ships_at`
  at `cutoff = max(1, ceil(turns))`, `need = owner==player ? 0 : ceil(ships)+1`.

The base timeline is `simulate_planet_timeline_jax` over each planet's full
arrival ledger (`features.arr_*`), computed once and gathered per target.

Each cell runs `plan_shot_jax` twice (rough_ships, then send_guess); both calls
are `vmap`-ed over the grid so the cost is two vmapped plan_shot kernels.

The `opening_filter` (strategy_helpers) is ported here as a pure mask because it
is only used by the mission builders. Every Python branch is maskable: it reads
`is_opening`, `owner == -1`, comet, `is_static`, reaction times, the
`SAFE_OPENING_*` window, and the `is_four_player` branch vs the
`ROTATING_OPENING_*` fallback.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    ATTACK_COST_TURN_WEIGHT,
    COMET_MAX_CHASE_TURNS,
    FOUR_PLAYER_ROTATING_REACTION_GAP,
    FOUR_PLAYER_ROTATING_SEND_RATIO,
    FOUR_PLAYER_ROTATING_TURN_LIMIT,
    HORIZON,
    PARTIAL_SOURCE_MIN_SHIPS,
    ROTATING_OPENING_LOW_PROD,
    ROTATING_OPENING_MAX_TURNS,
    SAFE_NEUTRAL_MARGIN,
    SAFE_OPENING_PROD_THRESHOLD,
    SAFE_OPENING_TURN_LIMIT,
    SNIPE_COST_TURN_WEIGHT,
    VERY_LATE_CAPTURE_BUFFER,
)
from .plan_shot_jax import plan_shot_jax
from .scoring_jax import (
    MISSION_CAPTURE,
    MISSION_SNIPE,
    ModesArrays,
    preferred_send_jax,
    score_attack_jax,
)
from .timeline_jax import (
    MAX_PLANETS,
    simulate_planet_timeline_jax,
    state_at_timeline_jax,
)
from .world_features import WorldFeatures

# Number of distinct enemy ETAs `build_snipe_mission` probes (`enemy_etas[:3]`).
SNIPE_MAX_ETAS: int = 3

# Snipe probe ship-count slack over the target garrison (`target.ships + 8`).
_SNIPE_PROBE_SLACK: int = 8


class CaptureGrid(NamedTuple):
    """Per-(P_src, P_tgt) capture-mission candidate fields.

    `valid` marks a cell where the Python builder produced a ShotOption.
    `is_single` additionally marks `send_cap >= needed` (Python emits a "single"
    Mission). The remaining fields are the ShotOption-equivalent payload the
    swarm builder reuses as `source_options_by_target`.
    """

    valid: jax.Array  # bool[P_src, P_tgt]
    is_single: jax.Array  # bool[P_src, P_tgt]
    score: jax.Array  # float32[P_src, P_tgt]
    angle: jax.Array  # float32[P_src, P_tgt]
    turns: jax.Array  # int32[P_src, P_tgt]
    needed: jax.Array  # int32[P_src, P_tgt]
    send_cap: jax.Array  # int32[P_src, P_tgt]


class SnipeGrid(NamedTuple):
    """Per-(P_src, P_tgt) snipe-mission candidate fields.

    Mirrors `build_snipe_mission`: a neutral target whose inbound enemy wave is
    synced against. `turns` is the option aim turns; `sync_turn` is the mission
    turns (`max(turns, enemy_eta)`). `needed` is the synced capture cost.
    """

    valid: jax.Array  # bool[P_src, P_tgt]
    score: jax.Array  # float32[P_src, P_tgt]
    angle: jax.Array  # float32[P_src, P_tgt]
    turns: jax.Array  # int32[P_src, P_tgt] (aim turns)
    sync_turn: jax.Array  # int32[P_src, P_tgt] (mission turns)
    needed: jax.Array  # int32[P_src, P_tgt]


def _base_timelines(features: WorldFeatures) -> tuple[jax.Array, jax.Array]:
    """Per-planet base `(owner_at, ships_at)` via `simulate_planet_timeline_jax`.

    Reproduces `WorldModel.base_timeline` with EMPTY planned_commitments (the
    collection-time state). Returns `(owner_at, ships_at)` stacked over planets:
    `owner_at[P, H+1]`, `ships_at[P, H+1]`.
    """

    def one(
        p_owner: jax.Array,
        p_ships: jax.Array,
        p_prod: jax.Array,
        t_owner: jax.Array,
        t_ships: jax.Array,
        t_valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        tl = simulate_planet_timeline_jax(
            p_owner,
            p_ships.astype(jnp.float32),
            p_prod.astype(jnp.float32),
            t_owner,
            t_ships,
            t_valid,
            features.player,
            HORIZON,
        )
        return tl.owner_at, tl.ships_at

    return jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0))(
        features.owner,
        features.ships,
        features.prod,
        features.arr_owner,
        features.arr_ships,
        features.arr_valid,
    )


def _base_need(
    owner_at: jax.Array,
    ships_at: jax.Array,
    arrival_turns: jax.Array,
    player: jax.Array,
) -> jax.Array:
    """Base `ships_needed_to_capture` (empty commitments) at `arrival_turns`.

    `cutoff = max(1, ceil(turns))`; project owner/ships via state_at_timeline;
    `need = owner==player ? 0 : ceil(ships)+1`. Returns int32.
    """
    cutoff = jnp.maximum(jnp.int32(1), jnp.ceil(arrival_turns).astype(jnp.int32))
    owner_t, ships_t = state_at_timeline_jax(
        owner_at, ships_at, jnp.int32(HORIZON), cutoff.astype(jnp.float32)
    )
    need = jnp.where(
        owner_t == player, jnp.int32(0), jnp.ceil(ships_t).astype(jnp.int32) + 1
    )
    return need.astype(jnp.int32)


def opening_filter_jax(
    target_idx: jax.Array,
    arrival_turns: jax.Array,
    needed: jax.Array,
    src_available: jax.Array,
    features: WorldFeatures,
) -> jax.Array:
    """Mirror `strategy_helpers.opening_filter`. Returns True to REJECT the cell.

    Every Python early-return is a boolean override; the final fallthrough is the
    `ROTATING_OPENING_*` rule (non-four-player) or the four-player affordability
    rule. The function returns False (keep) for non-opening / non-neutral /
    comet / static targets and the SAFE_OPENING window.
    """
    i = target_idx
    owner = features.owner[i]
    prod = features.prod[i]
    is_static = features.is_static[i]
    is_comet = features.is_comet[i]
    my_t = features.reaction_my_t[i]
    enemy_t = features.reaction_enemy_t[i]
    is_opening = features.is_opening
    is_four = features.is_four_player

    at = arrival_turns
    reaction_gap = enemy_t - my_t

    # Python guards that short-circuit to False (keep the cell).
    not_applicable = jnp.logical_not(is_opening) | (owner != -1) | is_comet | is_static

    safe_window = (
        (prod >= SAFE_OPENING_PROD_THRESHOLD)
        & (at <= SAFE_OPENING_TURN_LIMIT)
        & (reaction_gap >= SAFE_NEUTRAL_MARGIN)
    )

    # Four-player branch.
    affordable_cap = jnp.maximum(
        PARTIAL_SOURCE_MIN_SHIPS,
        jnp.floor(
            src_available.astype(jnp.float32) * FOUR_PLAYER_ROTATING_SEND_RATIO
        ).astype(jnp.int32),
    )
    affordable = needed <= affordable_cap
    four_keep = (
        affordable
        & (at <= FOUR_PLAYER_ROTATING_TURN_LIMIT)
        & (reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP)
    )
    # Four-player: keep iff four_keep else reject.
    four_reject = jnp.logical_not(four_keep)

    # Non-four-player rotating-opening fallback.
    rotating_reject = (at > ROTATING_OPENING_MAX_TURNS) | (
        prod <= ROTATING_OPENING_LOW_PROD
    )

    reject_after_safe = jnp.where(is_four, four_reject, rotating_reject)
    reject = jnp.where(safe_window, jnp.bool_(False), reject_after_safe)
    return jnp.where(not_applicable, jnp.bool_(False), reject)


def _plan_shot_cell(
    features: WorldFeatures,
    src_idx: jax.Array,
    tgt_idx: jax.Array,
    ships: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """`plan_shot_jax` for one (src, target) cell using WorldFeatures arrays."""
    s = src_idx
    t = tgt_idx
    return plan_shot_jax(
        features.xy[s, 0],
        features.xy[s, 1],
        features.radius[s],
        features.xy[t, 0],
        features.xy[t, 1],
        features.initial_xy[t, 0],
        features.initial_xy[t, 1],
        features.initial_radius[t],
        features.radius[t],
        ships,
        features.ang_vel,
        features.plan_max_turns[t],
        features.comet_path[t],
        features.comet_path_index[t],
        features.comet_path_len[t],
        features.planet_id[t],
        features.is_comet[t],
        features.comet_life[t],
        features.other_paths,
        features.other_path_index,
        features.other_path_len,
        features.other_planet_id,
        features.comet_radius,
    )


def _capture_cell(
    features: WorldFeatures,
    modes: ModesArrays,
    owner_at: jax.Array,
    ships_at: jax.Array,
    src_idx: jax.Array,
    tgt_idx: jax.Array,
) -> CaptureGrid:
    """Mirror `build_capture_and_snipe_missions`'s capture body for one cell."""
    s = src_idx
    t = tgt_idx
    player = features.player

    src_mine = features.planet_valid[s] & (features.owner[s] == player)
    src_available = features.available[s]
    tgt_valid = features.planet_valid[t]
    tgt_not_mine = features.owner[t] != player
    distinct = s != t
    cell_ok = src_mine & (src_available > 0) & tgt_valid & tgt_not_mine & distinct

    is_comet = features.is_comet[t]
    comet_life = features.comet_life[t]
    is_very_late = features.is_very_late
    remaining = features.remaining_steps
    tgt_owner_at = owner_at[t]
    tgt_ships_at = ships_at[t]

    # rough_ships = max(1, min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, ships+1)))
    rough_ships = jnp.maximum(
        jnp.int32(1),
        jnp.minimum(
            src_available,
            jnp.maximum(PARTIAL_SOURCE_MIN_SHIPS, features.ships[t] + 1),
        ),
    )
    r_angle, r_turns, _rix, _riy, r_valid = _plan_shot_cell(features, s, t, rough_ships)

    very_late_reject_rough = is_very_late & (
        r_turns > remaining - VERY_LATE_CAPTURE_BUFFER
    )
    comet_reject_rough = is_comet & (
        (r_turns >= comet_life) | (r_turns > COMET_MAX_CHASE_TURNS)
    )

    rough_needed = _base_need(tgt_owner_at, tgt_ships_at, r_turns, player)
    opening_reject_rough = opening_filter_jax(
        t, r_turns, rough_needed, src_available, features
    )

    stage1_ok = (
        cell_ok
        & r_valid
        & jnp.logical_not(very_late_reject_rough)
        & jnp.logical_not(comet_reject_rough)
        & (rough_needed > 0)
        & jnp.logical_not(opening_reject_rough)
    )

    # send_guess = preferred_send(target, rough_needed, rough_turns, src_available)
    send_guess = preferred_send_jax(
        t, rough_needed, r_turns, src_available, features, modes
    )
    aim_ships = jnp.maximum(jnp.int32(1), send_guess)
    angle, turns, _ix, _iy, aim_valid = _plan_shot_cell(features, s, t, aim_ships)

    very_late_reject = is_very_late & (turns > remaining - VERY_LATE_CAPTURE_BUFFER)
    comet_reject = is_comet & ((turns >= comet_life) | (turns > COMET_MAX_CHASE_TURNS))

    needed = _base_need(tgt_owner_at, tgt_ships_at, turns, player)
    opening_reject = opening_filter_jax(t, turns, needed, src_available, features)

    send_cap = src_available
    send_cap_reject = (send_cap < 1) | (
        (send_cap < needed) & (send_cap < PARTIAL_SOURCE_MIN_SHIPS)
    )

    # expected_send = max(needed, min(send_cap, preferred_send(needed, turns)))
    pref_send = preferred_send_jax(t, needed, turns, send_cap, features, modes)
    expected_send = jnp.maximum(needed, jnp.minimum(send_cap, pref_send))

    score = score_attack_jax(
        t,
        expected_send,
        turns,
        MISSION_CAPTURE,
        jnp.float32(ATTACK_COST_TURN_WEIGHT),
        features,
        modes,
    )

    valid = (
        stage1_ok
        & aim_valid
        & jnp.logical_not(very_late_reject)
        & jnp.logical_not(comet_reject)
        & (needed > 0)
        & jnp.logical_not(opening_reject)
        & jnp.logical_not(send_cap_reject)
        & (score > 0.0)
    )

    is_single = valid & (send_cap >= needed)

    return CaptureGrid(
        valid=valid,
        is_single=is_single,
        score=jnp.where(valid, score, 0.0).astype(jnp.float32),
        angle=jnp.where(valid, angle, 0.0).astype(jnp.float32),
        turns=jnp.where(valid, turns, 0).astype(jnp.int32),
        needed=jnp.where(valid, needed, 0).astype(jnp.int32),
        send_cap=jnp.where(valid, send_cap, 0).astype(jnp.int32),
    )


def build_capture_grid(features: WorldFeatures, modes: ModesArrays) -> CaptureGrid:
    """Build the `(MAX_PLANETS, MAX_PLANETS)` capture-mission candidate grid."""
    owner_at, ships_at = _base_timelines(features)
    idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    def per_src(s: jax.Array) -> CaptureGrid:
        def per_tgt(t: jax.Array) -> CaptureGrid:
            return _capture_cell(features, modes, owner_at, ships_at, s, t)

        return jax.vmap(per_tgt)(idx)

    return jax.vmap(per_src)(idx)


def _snipe_enemy_etas(
    features: WorldFeatures, tgt_idx: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Up to SNIPE_MAX_ETAS smallest DISTINCT enemy ETAs for the target.

    Mirrors `build_snipe_mission`'s
    `sorted({ceil(eta) for eta,owner,ships in arrivals if enemy and ships>0})`,
    using the bucketed per-turn arrival ledger (already `ceil`-ed & filtered to
    `ships > 0` by `normalize_arrivals` at bucketing time). The ledger turn index
    IS the ceil(eta), so distinct enemy turns are exactly the turns whose slice
    has an enemy owner. Returns `(etas[SNIPE_MAX_ETAS], eta_valid[SNIPE_MAX_ETAS])`.
    """
    t = tgt_idx
    player = features.player
    a_owner = features.arr_owner[t]  # (H+1, K)
    a_valid = features.arr_valid[t]
    enemy = a_valid & (a_owner != -1) & (a_owner != player)
    # A turn has an enemy ETA iff any slot in its row is an enemy arrival.
    turn_has_enemy = jnp.any(enemy, axis=1)  # (H+1,)
    turns_axis = jnp.arange(HORIZON + 1, dtype=jnp.int32)
    # Distinct ETAs sorted ascending: select the turns flagged, pad with a big
    # sentinel so the smallest SNIPE_MAX_ETAS land first after a sort.
    big = jnp.int32(HORIZON + 100)
    keyed = jnp.where(turn_has_enemy, turns_axis, big)
    sorted_turns = jnp.sort(keyed)
    etas = sorted_turns[:SNIPE_MAX_ETAS]
    eta_valid = etas < big
    return etas.astype(jnp.int32), eta_valid


def _snipe_cell(
    features: WorldFeatures,
    modes: ModesArrays,
    owner_at: jax.Array,
    ships_at: jax.Array,
    src_idx: jax.Array,
    tgt_idx: jax.Array,
) -> SnipeGrid:
    """Mirror `build_snipe_mission` for one (src, target) cell.

    Probes the target with `probe = min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS,
    ships+8))`, then walks `enemy_etas[:3]` and returns the FIRST eta that passes
    every gate (Python `return`s on the first match).
    """
    s = src_idx
    t = tgt_idx
    player = features.player

    src_mine = features.planet_valid[s] & (features.owner[s] == player)
    src_available = features.available[s]
    tgt_valid = features.planet_valid[t]
    is_neutral = features.owner[t] == -1
    distinct = s != t
    cell_ok = src_mine & (src_available > 0) & tgt_valid & is_neutral & distinct

    is_comet = features.is_comet[t]
    comet_life = features.comet_life[t]
    tgt_owner_at = owner_at[t]
    tgt_ships_at = ships_at[t]

    etas, eta_valid = _snipe_enemy_etas(features, t)
    any_eta = jnp.any(eta_valid)

    probe = jnp.minimum(
        src_available,
        jnp.maximum(PARTIAL_SOURCE_MIN_SHIPS, features.ships[t] + _SNIPE_PROBE_SLACK),
    )
    _ra, rough_turns, _rix, _riy, rough_valid = _plan_shot_cell(features, s, t, probe)

    base_ok = cell_ok & any_eta & rough_valid

    # Walk enemy_etas[:3]; pick the FIRST that passes all gates.
    SnipeCarry = tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ]

    def per_eta(
        carry: SnipeCarry,
        xs: tuple[jax.Array, jax.Array],
    ) -> tuple[SnipeCarry, None]:
        done, b_valid, b_score, b_angle, b_turns, b_sync, b_need = carry
        enemy_eta, e_ok = xs

        consider = base_ok & e_ok & jnp.logical_not(done)
        # `if abs(rough[1] - enemy_eta) > 1: continue`
        rough_close = jnp.abs(rough_turns - enemy_eta) <= 1

        sync_turn = jnp.maximum(rough_turns, enemy_eta)
        comet_reject = is_comet & (
            (sync_turn >= comet_life) | (sync_turn > COMET_MAX_CHASE_TURNS)
        )

        need0 = _base_need(tgt_owner_at, tgt_ships_at, sync_turn, player)
        need0_reject = (need0 <= 0) | (need0 > src_available)

        _fa, turns, _fix, _fiy, final_valid = _plan_shot_cell(
            features, s, t, jnp.maximum(jnp.int32(1), need0)
        )
        turns_close = jnp.abs(turns - enemy_eta) <= 1

        sync_turn2 = jnp.maximum(turns, enemy_eta)
        need = _base_need(tgt_owner_at, tgt_ships_at, sync_turn2, player)
        need_reject = (need <= 0) | (need > src_available)

        score = score_attack_jax(
            t,
            need,
            sync_turn2,
            MISSION_SNIPE,
            jnp.float32(SNIPE_COST_TURN_WEIGHT),
            features,
            modes,
        )

        match = (
            consider
            & rough_close
            & jnp.logical_not(comet_reject)
            & jnp.logical_not(need0_reject)
            & final_valid
            & turns_close
            & jnp.logical_not(need_reject)
            & (score > 0.0)
        )

        new_done = done | match
        return (
            new_done,
            jnp.where(match, jnp.bool_(True), b_valid),
            jnp.where(match, score, b_score),
            jnp.where(match, _fa, b_angle),
            jnp.where(match, turns, b_turns),
            jnp.where(match, sync_turn2, b_sync),
            jnp.where(match, need, b_need),
        ), None

    init: SnipeCarry = (
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
    )
    (_done, valid, score, angle, turns, sync_turn, needed), _ = jax.lax.scan(
        per_eta, init, (etas, eta_valid)
    )

    return SnipeGrid(
        valid=valid,
        score=jnp.where(valid, score, 0.0).astype(jnp.float32),
        angle=jnp.where(valid, angle, 0.0).astype(jnp.float32),
        turns=jnp.where(valid, turns, 0).astype(jnp.int32),
        sync_turn=jnp.where(valid, sync_turn, 0).astype(jnp.int32),
        needed=jnp.where(valid, needed, 0).astype(jnp.int32),
    )


def build_snipe_grid(features: WorldFeatures, modes: ModesArrays) -> SnipeGrid:
    """Build the `(MAX_PLANETS, MAX_PLANETS)` snipe-mission candidate grid."""
    owner_at, ships_at = _base_timelines(features)
    idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    def per_src(s: jax.Array) -> SnipeGrid:
        def per_tgt(t: jax.Array) -> SnipeGrid:
            return _snipe_cell(features, modes, owner_at, ships_at, s, t)

        return jax.vmap(per_tgt)(idx)

    return jax.vmap(per_src)(idx)


__all__ = [
    "CaptureGrid",
    "SnipeGrid",
    "build_capture_grid",
    "build_snipe_grid",
    "opening_filter_jax",
]
