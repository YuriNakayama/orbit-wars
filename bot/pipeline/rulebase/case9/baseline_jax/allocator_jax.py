# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the greedy mission-commit loop in `baseline/strategy.py`.

`plan_moves` collects missions, sorts by `-score`, then processes them in order,
threading a `planned_commitments` ledger + `spent_total` per source through the
loop (each mission's viability depends on prior commits). This is the one
genuinely order-dependent stage of the agent — the part the old per-source-argmax
approximation dropped (and lost 0W-10L for). We reproduce it faithfully as
`jnp.argsort(-score)` + a `lax.scan` over the sorted missions whose carry is the
evolving ledger; vmap-ing across games keeps the per-game scan sequential while
running thousands of games in parallel on GPU.

This module handles the SINGLE-source mission family first (capture "single" +
snipe + reinforce + crash_exploit + harass share `_process_single_source_mission`).
Multi-source (swarm) commitment is layered on in a sibling step.

`ships_needed_to_capture(target, turn, commitments)` (via `projected_state`) is
recomputed each scan step from the carry's commit ledger: we add the target's
accumulated `(eta, player-owned, ships)` commits to its base arrival table and
re-run `simulate_planet_timeline_jax` to `cutoff = max(1, ceil(turn))`, then read
`state_at_timeline`. `need = 0` if the projected owner is us, else `ceil(ships)+1`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    ATTACK_COST_TURN_WEIGHT,
    FOLLOWUP_MIN_SHIPS,
    HORIZON,
    LATE_CAPTURE_BUFFER,
    LOW_VALUE_COMET_PRODUCTION,
    PARTIAL_SOURCE_MIN_SHIPS,
    REINFORCE_SAFETY_MARGIN,
)
from .missions_capture_jax import _plan_shot_cell, opening_filter_jax
from .scoring_jax import (
    MISSION_CAPTURE,
    ModesArrays,
    preferred_send_jax,
    score_attack_jax,
)
from .timeline_jax import (
    MAX_PLANETS,
    resolve_arrival_event_jax,
)
from .world_features import WorldFeatures

# Max launches the move buffer holds — one per (src, mission) at most; bounded by
# the candidate count. MAX_PLANETS sources * a few missions each; cap at the
# kaggle launch limit (== MAX_PLANETS per agent in jax_env). Capture single
# emits at most one move per source per accepted mission; we cap the buffer at
# MAX_MOVES and the final inventory cap trims.
MAX_MOVES: int = MAX_PLANETS

# Identity-preserving GPU speedup: the mission scan walks `argsort(-score)` over
# the full candidate table, but at most ~10 candidates are ever valid (measured)
# and argsort puts them all first (invalid cells score -inf). The truncated tail
# is -inf no-ops that never modify the carry, so top-K scanning is byte-identical
# (case1: regression 2/2 + 40/40 states). GPU win: per-turn cost is the SEQUENTIAL
# kernel-launch chain; cutting 4608->64 steps measured 15-18x (RTX4090).
MAX_ALLOC_CANDIDATES: int = 64

# Max commitments tracked per target planet (distinct accepted missions hitting
# one target). Generous: at most one per source.
MAX_COMMIT: int = MAX_PLANETS

# Mission-kind codes for the single-source scan (subset of strategy kinds).
KIND_SINGLE: int = 0
KIND_SNIPE: int = 1
KIND_REINFORCE: int = 2
KIND_CRASH: int = 3
KIND_HARASS: int = 4


class SingleMissionTable(NamedTuple):
    """Flattened, fixed-length single-source mission candidates (pre-sort).

    Every field is `(N,)` where N is the padded candidate count. `valid` masks
    real candidates; invalid rows sort to the bottom (score -inf) and are skipped.
    """

    valid: jax.Array  # bool[N]
    score: jax.Array  # float32[N]
    kind: jax.Array  # int32[N]  (KIND_*)
    src_slot: jax.Array  # int32[N]  index into planet arrays
    target_slot: jax.Array  # int32[N]
    angle: jax.Array  # float32[N]
    turns: jax.Array  # int32[N]  (arrival_turn / option.turns)
    send_cap: jax.Array  # int32[N]  (min(left, option.send_cap) handled in scan)


class AllocResult(NamedTuple):
    """Output of the greedy scan: a padded move buffer + final spent ledger."""

    move_src: jax.Array  # int32[MAX_MOVES]  planet id (-1 == empty slot)
    move_angle: jax.Array  # float32[MAX_MOVES]
    move_ships: jax.Array  # int32[MAX_MOVES]
    move_count: jax.Array  # int32 scalar


class AllocCarry(NamedTuple):
    """Evolving ledger threaded through every greedy scan step.

    The mission scan and the followup pass share this exact carry: the followup
    pass starts from the carry the mission scan ended with, so later sources see
    earlier commits (the SAME `planned_commitments` + `spent_total` semantics the
    Python `plan_moves` loop + `emit_followup_moves` share).
    """

    commit_eta: jax.Array  # int32[MAX_PLANETS, MAX_COMMIT]  per-target commit ETAs
    commit_ships: jax.Array  # int32[MAX_PLANETS, MAX_COMMIT]
    commit_count: jax.Array  # int32[MAX_PLANETS]
    spent: jax.Array  # int32[MAX_PLANETS]  per-source spent_total
    move_src: jax.Array  # int32[MAX_MOVES]  planet id (-1 == empty slot)
    move_angle: jax.Array  # float32[MAX_MOVES]
    move_ships: jax.Array  # int32[MAX_MOVES]
    move_count: jax.Array  # int32 scalar


def _empty_carry() -> AllocCarry:
    """Fresh carry: empty commit ledger, zero spend, empty move buffer."""
    return AllocCarry(
        commit_eta=jnp.zeros((MAX_PLANETS, MAX_COMMIT), dtype=jnp.int32),
        commit_ships=jnp.zeros((MAX_PLANETS, MAX_COMMIT), dtype=jnp.int32),
        commit_count=jnp.zeros((MAX_PLANETS,), dtype=jnp.int32),
        spent=jnp.zeros((MAX_PLANETS,), dtype=jnp.int32),
        move_src=jnp.full((MAX_MOVES,), -1, dtype=jnp.int32),
        move_angle=jnp.zeros((MAX_MOVES,), dtype=jnp.float32),
        move_ships=jnp.zeros((MAX_MOVES,), dtype=jnp.int32),
        move_count=jnp.int32(0),
    )


def _carry_to_result(carry: AllocCarry) -> AllocResult:
    """Project the move buffer out of a carry."""
    return AllocResult(
        move_src=carry.move_src,
        move_angle=carry.move_angle,
        move_ships=carry.move_ships,
        move_count=carry.move_count,
    )


def _need_with_commits(
    features: WorldFeatures,
    target_slot: jax.Array,
    cutoff: jax.Array,
    commit_eta: jax.Array,
    commit_ships: jax.Array,
) -> jax.Array:
    """`ships_needed_to_capture(target, cutoff, commitments)` with the carry ledger.

    Builds the target's augmented per-turn arrival table (base arrivals +
    accumulated player-owned commits, both restricted to `eta <= cutoff` by the
    timeline horizon), simulates to `cutoff`, reads the projected state, and
    returns `0` if we own it else `ceil(ships) + 1`. Mirrors `projected_state` +
    `ships_needed_to_capture`.

    `cutoff` is `max(1, ceil(arrival_turn))`. We simulate over the full HORIZON
    table but read state at `cutoff`, which is equivalent to the Python
    `simulate_planet_timeline(..., cutoff)` + `state_at_timeline(cutoff)` because
    only arrivals with `eta <= cutoff` affect the state at `cutoff`.
    """
    player = features.player
    base_owner = features.arr_owner[target_slot]  # (H+1, K)
    base_ships = features.arr_ships[target_slot]
    base_valid = features.arr_valid[target_slot]

    # Append commits into spare slots of each turn row. Commits are player-owned
    # arrivals at integer eta. Build a (H+1, MAX_COMMIT) commit overlay then
    # concatenate along the arrival axis.
    h1 = HORIZON + 1
    turn_ids = jnp.arange(h1, dtype=jnp.int32)[:, None]  # (H+1, 1)
    # commit_eta/commit_ships: (MAX_COMMIT,). A commit lands on row eta.
    eta_row = commit_eta[None, :]  # (1, MAX_COMMIT)
    on_turn = (eta_row == turn_ids) & (commit_ships[None, :] > 0)  # (H+1, MAX_COMMIT)
    c_owner = jnp.where(on_turn, player, jnp.int32(-1))  # (H+1, MAX_COMMIT)
    c_ships = jnp.where(on_turn, commit_ships[None, :], jnp.int32(0))
    c_valid = on_turn

    aug_owner = jnp.concatenate([base_owner, c_owner], axis=1)  # (H+1, K+MAX_COMMIT)
    aug_ships = jnp.concatenate([base_ships, c_ships], axis=1)
    aug_valid = jnp.concatenate([base_valid, c_valid], axis=1)

    # Forward-simulate the target to read (owner, ships) at cutoff. We inline a
    # lightweight forward pass (no keep_needed) since we only need the state.
    p_owner = features.owner[target_slot]
    p_prod = features.prod[target_slot].astype(jnp.float32)
    init_garr = jnp.maximum(0.0, features.ships[target_slot].astype(jnp.float32))

    def body(
        carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        owner, garrison = carry
        g_owner, g_ships, g_valid = xs
        produced = jnp.where(owner != -1, garrison + p_prod, garrison)
        has_group = jnp.any(g_valid)
        res_owner, res_garr = resolve_arrival_event_jax(
            owner, produced, g_owner, g_ships, g_valid
        )
        new_owner = jnp.where(has_group, res_owner, owner).astype(jnp.int32)
        new_garr = jnp.where(has_group, res_garr, produced)
        return (new_owner, new_garr), (new_owner, jnp.maximum(0.0, new_garr))

    (_o, _g), (owner_seq, ships_seq) = jax.lax.scan(
        body,
        (p_owner, init_garr),
        (aug_owner[1:], aug_ships[1:], aug_valid[1:]),
    )
    owner_at = jnp.concatenate([p_owner[None], owner_seq])  # (H+1,)
    ships_at = jnp.concatenate([init_garr[None].astype(jnp.float32), ships_seq])

    t = jnp.clip(cutoff, 0, HORIZON)
    owner_t = owner_at[t]
    ships_t = ships_at[t]
    need = jnp.where(
        owner_t == player,
        jnp.int32(0),
        jnp.ceil(ships_t).astype(jnp.int32) + 1,
    )
    return need


def _commit_move(
    carry: AllocCarry,
    ok: jax.Array,
    src_pid: jax.Array,
    angle: jax.Array,
    sent: jax.Array,
    tgt: jax.Array,
    src: jax.Array,
    eta: jax.Array,
) -> AllocCarry:
    """Masked commit: append a move + bump spent[src] + extend target's ledger.

    Shared by the mission scan and the followup pass — both mirror `append_move`
    + `planned_commitments[target].append((eta, player, sent))`.
    """
    new_spent = carry.spent.at[src].add(jnp.where(ok, sent, 0))

    m_slot = jnp.where(ok, carry.move_count, MAX_MOVES - 1)
    new_move_src = carry.move_src.at[m_slot].set(
        jnp.where(ok, src_pid, carry.move_src[m_slot])
    )
    new_move_angle = carry.move_angle.at[m_slot].set(
        jnp.where(ok, angle, carry.move_angle[m_slot])
    )
    new_move_ships = carry.move_ships.at[m_slot].set(
        jnp.where(ok, sent, carry.move_ships[m_slot])
    )
    new_move_count = carry.move_count + jnp.where(ok, jnp.int32(1), jnp.int32(0))

    c_slot = jnp.where(ok, carry.commit_count[tgt], MAX_COMMIT - 1)
    new_commit_eta = carry.commit_eta.at[tgt, c_slot].set(
        jnp.where(ok, eta, carry.commit_eta[tgt, c_slot])
    )
    new_commit_ships = carry.commit_ships.at[tgt, c_slot].set(
        jnp.where(ok, sent, carry.commit_ships[tgt, c_slot])
    )
    new_commit_count = carry.commit_count.at[tgt].add(
        jnp.where(ok, jnp.int32(1), jnp.int32(0))
    )
    return AllocCarry(
        commit_eta=new_commit_eta,
        commit_ships=new_commit_ships,
        commit_count=new_commit_count,
        spent=new_spent,
        move_src=new_move_src,
        move_angle=new_move_angle,
        move_ships=new_move_ships,
        move_count=new_move_count,
    )


def _run_mission_scan(
    table: SingleMissionTable,
    init_carry: AllocCarry,
    features: WorldFeatures,
    modes: ModesArrays,
) -> AllocCarry:
    """Greedy commit of single-source missions over `argsort(-score)`.

    Starts from `init_carry` and returns the ending carry (commit ledger + spent
    + move buffer). Factored out so the followup pass can continue from the carry
    this scan ends with, exactly as Python's `emit_followup_moves` runs after the
    mission loop sharing the same `planned_commitments` / `spent_total`.
    """
    eff_score = jnp.where(table.valid, table.score, jnp.float32(-jnp.inf))
    order = jnp.argsort(-eff_score)[:MAX_ALLOC_CANDIDATES]  # (K,) descending

    available = features.available  # (P,) defense-adjusted inventory
    ships_arr = features.ships  # (P,) raw inventory (source_inventory_left base)

    def step(carry: AllocCarry, idx: jax.Array) -> tuple[AllocCarry, None]:
        spent = carry.spent

        valid = table.valid[idx]
        kind = table.kind[idx]
        src = table.src_slot[idx]
        tgt = table.target_slot[idx]
        angle = table.angle[idx]
        turns = table.turns[idx]
        cap = table.send_cap[idx]

        is_reinforce = kind == KIND_REINFORCE
        # left = source_inventory_left (reinforce) or source_attack_left (else).
        inv_left = jnp.maximum(0, ships_arr[src] - spent[src])
        atk_left = jnp.maximum(0, available[src] - spent[src])
        left = jnp.where(is_reinforce, inv_left, atk_left)

        cutoff = jnp.maximum(1, jnp.ceil(turns.astype(jnp.float32)).astype(jnp.int32))
        # missing: reinforce uses reinforcement_needed_for (not ported here yet) —
        # for the capture/single family we use ships_needed_to_capture.
        missing = _need_with_commits(
            features, tgt, cutoff, carry.commit_eta[tgt], carry.commit_ships[tgt]
        )

        send_limit = jnp.minimum(left, cap)

        # send by kind:
        #  snipe/crash/harass -> missing
        #  reinforce          -> min(send_limit, missing + REINFORCE_SAFETY_MARGIN)
        #  single (capture)   -> min(send_limit, max(missing, preferred_send))
        pref = preferred_send_jax(tgt, missing, turns, send_limit, features, modes)
        send_single = jnp.minimum(send_limit, jnp.maximum(missing, pref))
        send_reinforce = jnp.minimum(send_limit, missing + REINFORCE_SAFETY_MARGIN)
        send_exact = missing
        send = jnp.where(
            is_reinforce,
            send_reinforce,
            jnp.where(kind == KIND_SINGLE, send_single, send_exact),
        )

        # append_move clamps to source_inventory_left and updates spent.
        cur_inv_left = jnp.maximum(0, ships_arr[src] - spent[src])
        sent = jnp.minimum(send, cur_inv_left)

        # acceptance gates (mirror _process_single_source_mission early returns):
        ok = (
            valid
            & (left > 0)
            & (missing > 0)
            & (send_limit >= missing)
            & (send >= missing)
            & (sent >= missing)
            & (sent >= 1)
            & (carry.move_count < MAX_MOVES)
        )

        new_carry = _commit_move(
            carry, ok, features.planet_id[src], angle, sent, tgt, src, turns
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(step, init_carry, order)
    return final_carry


def run_single_source_allocator(
    table: SingleMissionTable,
    features: WorldFeatures,
    modes: ModesArrays,
) -> AllocResult:
    """Greedy commit of single-source missions, mirroring `plan_moves`'s loop.

    Sorts candidates by `-score` and scans, threading commit + spent ledgers.
    Returns a padded move buffer (pre inventory-cap; the cap is applied in a
    final pure pass by the caller / a sibling helper).
    """
    final_carry = _run_mission_scan(table, _empty_carry(), features, modes)
    return _carry_to_result(final_carry)


def _followup_score_row(
    features: WorldFeatures,
    modes: ModesArrays,
    carry: AllocCarry,
    src: jax.Array,
    src_left: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-target scores for one followup source (mirrors followup's inner loop).

    Returns `(valid[P], score[P], turns[P], send[P])` over all target slots for
    the given `src`, using `src_left = source_attack_left(src)` at followup time
    (the carry's `available - spent`) and followup's OWN filters:
    `LOW_VALUE_COMET_PRODUCTION` comet skip, the `LATE_CAPTURE_BUFFER` late gate
    (NOT the very-late / comet-chase gate of capture), `opening_filter`,
    `rough_needed > 0`, `preferred_send >= rough_needed`, `score > 0`.

    `send` is the rough-stage `preferred_send` candidate; the chosen target is
    re-planned in the scan body, so this only drives the per-source argmax.
    """
    player = features.player
    src_mine = features.planet_valid[src] & (features.owner[src] == player)
    tgt_axis = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    def per_tgt(
        t: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        tgt_valid = features.planet_valid[t]
        tgt_not_mine = features.owner[t] != player
        distinct = src != t
        low_comet = features.is_comet[t] & (
            features.prod[t] <= LOW_VALUE_COMET_PRODUCTION
        )
        cell_ok = (
            src_mine
            & tgt_valid
            & tgt_not_mine
            & distinct
            & jnp.logical_not(low_comet)
            & (src_left >= FOLLOWUP_MIN_SHIPS)
        )

        # rough_ships = max(1, min(src_left, max(PARTIAL_SOURCE_MIN_SHIPS, ships+1)))
        rough_ships = jnp.maximum(
            jnp.int32(1),
            jnp.minimum(
                src_left,
                jnp.maximum(PARTIAL_SOURCE_MIN_SHIPS, features.ships[t] + 1),
            ),
        )
        angle, turns, _ix, _iy, aim_valid = _plan_shot_cell(
            features, src, t, rough_ships
        )

        # if world.is_late and est_turns > remaining - LATE_CAPTURE_BUFFER: continue
        late_reject = features.is_late & (
            turns > features.remaining_steps - LATE_CAPTURE_BUFFER
        )

        cutoff = jnp.maximum(1, jnp.ceil(turns.astype(jnp.float32)).astype(jnp.int32))
        rough_needed = _need_with_commits(
            features, t, cutoff, carry.commit_eta[t], carry.commit_ships[t]
        )
        opening_reject = opening_filter_jax(t, turns, rough_needed, src_left, features)

        send = preferred_send_jax(t, rough_needed, turns, src_left, features, modes)

        score = score_attack_jax(
            t,
            send,
            turns,
            MISSION_CAPTURE,
            jnp.float32(ATTACK_COST_TURN_WEIGHT),
            features,
            modes,
        )

        valid = (
            cell_ok
            & aim_valid
            & jnp.logical_not(late_reject)
            & (rough_needed > 0)
            & jnp.logical_not(opening_reject)
            & (send >= rough_needed)
            & (score > 0.0)
        )
        return valid, score, turns, send

    return jax.vmap(per_tgt)(tgt_axis)


def run_followup_pass(
    carry: AllocCarry,
    features: WorldFeatures,
    modes: ModesArrays,
) -> AllocCarry:
    """Followup attack pass: leftover ships -> opportunistic captures.

    Mirrors `emit_followup_moves`: a SECOND scan over source slots IN ORDER that
    continues from the mission scan's ending carry. For each source with
    `source_attack_left(src) >= FOLLOWUP_MIN_SHIPS`, picks the per-source argmax
    capture target (vmap over targets), re-plans with the chosen send, recomputes
    `missing` from the carry's commit ledger, and commits. SEQUENTIAL over
    sources so later sources see earlier followup commits.

    The whole pass is a no-op when `world.is_very_late` (the Python early return).
    """
    not_very_late = jnp.logical_not(features.is_very_late)
    available = features.available
    ships_arr = features.ships
    src_axis = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    def step(carry: AllocCarry, src: jax.Array) -> tuple[AllocCarry, None]:
        src_left = jnp.maximum(0, available[src] - carry.spent[src])
        gate = not_very_late & (src_left >= FOLLOWUP_MIN_SHIPS)

        valid_row, score_row, _turns_row, send_row = _followup_score_row(
            features, modes, carry, src, src_left
        )
        masked_score = jnp.where(valid_row, score_row, jnp.float32(-jnp.inf))
        best_t = jnp.argmax(masked_score)
        any_target = jnp.any(valid_row) & gate

        # --- re-plan with the chosen target (Python's second plan_shot) ---
        # Python re-plans with the rough-stage `send` of the chosen target, then
        # recomputes `missing` from aim.turns and the live commit ledger.
        send0 = jnp.maximum(jnp.int32(1), send_row[best_t])
        angle, turns, _ix, _iy, aim_valid = _plan_shot_cell(
            features, src, best_t, send0
        )
        cutoff = jnp.maximum(1, jnp.ceil(turns.astype(jnp.float32)).astype(jnp.int32))
        missing = _need_with_commits(
            features,
            best_t,
            cutoff,
            carry.commit_eta[best_t],
            carry.commit_ships[best_t],
        )

        # src_left = source_attack_left(src) (re-read; spent unchanged within step)
        pref2 = preferred_send_jax(best_t, missing, turns, src_left, features, modes)
        send = jnp.minimum(src_left, jnp.maximum(missing, pref2))

        # append_move clamps to source_inventory_left and updates spent.
        cur_inv_left = jnp.maximum(0, ships_arr[src] - carry.spent[src])
        sent = jnp.minimum(send, cur_inv_left)

        ok = (
            any_target
            & aim_valid
            & (missing > 0)
            & (send >= missing)
            & (sent >= missing)
            & (sent >= 1)
            & (carry.move_count < MAX_MOVES)
        )

        new_carry = _commit_move(
            carry, ok, features.planet_id[src], angle, sent, best_t, src, turns
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(step, carry, src_axis)
    return final_carry


def run_mission_and_followup(
    table: SingleMissionTable,
    features: WorldFeatures,
    modes: ModesArrays,
) -> AllocResult:
    """Mission scan THEN followup pass, sharing one carry (mirrors plan_moves).

    `plan_moves` runs the greedy mission-commit loop, then `emit_followup_moves`
    reuses the same `planned_commitments` / `spent_total`. Here the followup pass
    continues from the carry the mission scan ended with, so it reads the
    post-commit `source_attack_left` and recomputes `missing` off the live ledger.
    """
    mission_carry = _run_mission_scan(table, _empty_carry(), features, modes)
    final_carry = run_followup_pass(mission_carry, features, modes)
    return _carry_to_result(final_carry)


__all__ = [
    "KIND_CRASH",
    "KIND_HARASS",
    "KIND_REINFORCE",
    "KIND_SINGLE",
    "KIND_SNIPE",
    "MAX_COMMIT",
    "MAX_MOVES",
    "AllocCarry",
    "AllocResult",
    "SingleMissionTable",
    "run_followup_pass",
    "run_mission_and_followup",
    "run_single_source_allocator",
]
