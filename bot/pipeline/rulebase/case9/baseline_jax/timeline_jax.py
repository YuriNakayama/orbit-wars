# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the world-model timeline helpers in `baseline/core/world_model.py`.

Ported (pure, jit/vmap-friendly):

* `resolve_arrival_event`     -> `resolve_arrival_event_jax`
* `simulate_planet_timeline`  -> `simulate_planet_timeline_jax`
* `state_at_timeline`         -> `state_at_timeline_jax`
* `indirect_wealth`           -> `indirect_wealth_jax`

Fixed-shape contract
--------------------
The ragged Python structures are flattened to fixed-shape padded arrays + valid
masks on the host, then fed to the traced functions:

* Same-turn arrivals (`resolve_arrival_event`) become `arr_owner[K]`,
  `arr_ships[K]`, `arr_valid[K]` with `K == MAX_ARRIVALS_PER_TURN`.
* A planet's full arrival ledger (`simulate_planet_timeline`) becomes a per-turn
  table `tarr_owner[H+1, K]`, `tarr_ships[H+1, K]`, `tarr_valid[H+1, K]`; the host
  applies `normalize_arrivals` (drop `ships<=0` and `eta>horizon`,
  `eta = max(1, ceil(turns))`) when bucketing. Index 0 is unused (turns run 1..H).
* `indirect_wealth` takes the focal planet index plus `MAX_PLANETS`-padded
  per-planet field arrays + a `valid` mask.

Attacker owners are 0..3; the planet/garrison owner may also be -1 (neutral).
The Python `by_owner` aggregation keys on whatever attacker owner appears; in
practice game fleets are owned 0..3, so a `-1` *attacker* never occurs. We
segment-sum only over owners 0..3 and ignore any `arr_owner == -1` slot, which
is exactly the realistic input domain (documented deviation: a literal `-1`
attacker is treated as "no arrival").

`keep_needed` is monotone in `keep` (more reserve ships can only help the planet
survive), so we replace the Python binary search with a fixed-iteration binary
search via `lax.fori_loop` over `KEEP_SEARCH_ITERS` steps, guarding each step on
`lo < hi` so converged iterations are no-ops. This is bit-for-bit equivalent to
the Python `while lo < hi` loop for any `ships < 2**KEEP_SEARCH_ITERS`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    INDIRECT_ENEMY_WEIGHT,
    INDIRECT_FRIENDLY_WEIGHT,
    INDIRECT_NEUTRAL_WEIGHT,
)
from .geometry_jax import dist_jax

# Fixed shapes. MAX_ARRIVALS_PER_TURN bounds how many fleets can land on a single
# planet on a single turn; 16 is well above any realistic Orbit Wars board (a
# handful of fleets per planet per turn). MAX_PLANETS=48 matches the rest of the
# JAX port.
MAX_PLANETS: int = 48
MAX_ARRIVALS_PER_TURN: int = 16

# Number of fixed binary-search iterations for keep_needed. ships < 2**20 ~ 1e6
# always converges; the realistic max is ~hundreds.
KEEP_SEARCH_ITERS: int = 20

# Attacker owners are exactly {0, 1, 2, 3}.
_OWNERS = jnp.arange(4, dtype=jnp.int32)


class TimelineResult(NamedTuple):
    """JAX mirror of `simulate_planet_timeline`'s dict result.

    `owner_at` / `ships_at` are indexed by turn 0..horizon (length HORIZON+1).
    `first_enemy` / `fall_turn` use -1 for the Python `None`.
    """

    owner_at: jax.Array  # int32[H+1]
    ships_at: jax.Array  # float32[H+1]
    keep_needed: jax.Array  # int32
    min_owned: jax.Array  # int32
    first_enemy: jax.Array  # int32 (-1 == None)
    fall_turn: jax.Array  # int32 (-1 == None)
    holds_full: jax.Array  # bool
    horizon: jax.Array  # int32


def resolve_arrival_event_jax(
    owner: jax.Array,
    garrison: jax.Array,
    arr_owner: jax.Array,
    arr_ships: jax.Array,
    arr_valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mirror `resolve_arrival_event`. Returns `(owner, garrison)`.

    `arr_owner[K]` / `arr_ships[K]` / `arr_valid[K]` are a same-turn arrival
    group. Empty group (no valid slot) returns `(owner, max(0, garrison))`.
    """
    ships_f = arr_ships.astype(jnp.float32)
    valid = arr_valid & (arr_owner >= 0)

    # Per-owner ship sums over owners 0..3. Absent owners get sum -1 so they sort
    # below any real (>=0) total and never form a tie at a positive count.
    def owner_sum(o: jax.Array) -> jax.Array:
        sel = valid & (arr_owner == o)
        present = jnp.any(sel)
        total = jnp.sum(jnp.where(sel, ships_f, 0.0))
        return jnp.where(present, total, -1.0)

    sums = jax.vmap(owner_sum)(_OWNERS)  # (4,) float32, absent == -1
    present = sums >= 0.0
    n_present = jnp.sum(present.astype(jnp.int32))
    any_arrival = n_present > 0

    top_idx = jnp.argmax(sums)
    top_owner = _OWNERS[top_idx]
    top_ships = sums[top_idx]
    # second-highest: mask out the top slot, take the max of the rest.
    masked = sums.at[top_idx].set(-1.0)
    second_ships = jnp.max(masked)

    multi = n_present >= 2
    tie = multi & (top_ships == second_ships)
    survivor_owner = jnp.where(tie, jnp.int32(-1), top_owner)
    survivor_ships = jnp.where(
        tie, 0.0, jnp.where(multi, top_ships - second_ships, top_ships)
    )

    has_survivor = survivor_ships > 0.0

    # owner == survivor_owner: reinforce. else: subtract; flip if it goes negative.
    same = owner == survivor_owner
    reinforced = garrison + survivor_ships
    after_attack = garrison - survivor_ships
    flipped = after_attack < 0.0
    attacked_owner = jnp.where(flipped, survivor_owner, owner)
    attacked_garr = jnp.where(flipped, -after_attack, after_attack)

    resolved_owner = jnp.where(same, owner, attacked_owner)
    resolved_garr = jnp.where(same, reinforced, attacked_garr)

    # No arrivals or no positive survivor -> (owner, max(0, garrison)).
    keep_owner = owner
    keep_garr = jnp.maximum(0.0, garrison)
    apply = any_arrival & has_survivor
    out_owner = jnp.where(apply, resolved_owner, keep_owner)
    out_garr = jnp.where(apply, resolved_garr, keep_garr)
    return out_owner.astype(jnp.int32), out_garr.astype(jnp.float32)


def _forward_scan(
    p_owner: jax.Array,
    init_garrison: jax.Array,
    p_prod: jax.Array,
    tarr_owner: jax.Array,
    tarr_ships: jax.Array,
    tarr_valid: jax.Array,
    player: jax.Array,
    horizon: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward timeline pass via lax.scan over turns 1..horizon.

    Returns `(owner_at, ships_at, min_owned, first_enemy, fall_turn)` where
    owner_at/ships_at are length horizon+1 (index 0 == initial state).
    """
    turns = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    group_owner = tarr_owner[1:]  # (horizon, K)
    group_ships = tarr_ships[1:]
    group_valid = tarr_valid[1:]

    init_owner = p_owner
    init_garr = init_garrison
    init_min = jnp.where(init_owner == player, init_garr, 0.0)

    Carry = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

    def body(
        carry: Carry, xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[Carry, tuple[jax.Array, jax.Array]]:
        owner, garrison, min_owned, first_enemy, fall_turn = carry
        turn, g_owner, g_ships, g_valid = xs

        produced = jnp.where(owner != -1, garrison + p_prod, garrison)

        prev_owner = owner
        has_group = jnp.any(g_valid)
        # first_enemy: prev owner == player, none seen yet, and a non-(-1, player)
        # attacker is in this group.
        enemy_in_group = jnp.any(g_valid & (g_owner != -1) & (g_owner != player))
        set_first = (
            has_group & (prev_owner == player) & (first_enemy < 0) & enemy_in_group
        )
        new_first = jnp.where(set_first, turn, first_enemy)

        res_owner, res_garr = resolve_arrival_event_jax(
            owner, produced, g_owner, g_ships, g_valid
        )
        new_owner = jnp.where(has_group, res_owner, owner)
        new_garr = jnp.where(has_group, res_garr, produced)

        set_fall = (
            has_group & (prev_owner == player) & (new_owner != player) & (fall_turn < 0)
        )
        new_fall = jnp.where(set_fall, turn, fall_turn)

        ships_out = jnp.maximum(0.0, new_garr)
        new_min = jnp.where(
            new_owner == player, jnp.minimum(min_owned, new_garr), min_owned
        )
        return (
            new_owner.astype(jnp.int32),
            new_garr,
            new_min,
            new_first,
            new_fall,
        ), (new_owner.astype(jnp.int32), ships_out)

    init_carry: Carry = (
        init_owner,
        init_garr,
        init_min,
        jnp.int32(-1),
        jnp.int32(-1),
    )
    (_o, _g, min_owned, first_enemy, fall_turn), (owner_seq, ships_seq) = jax.lax.scan(
        body, init_carry, (turns, group_owner, group_ships, group_valid)
    )

    owner_at = jnp.concatenate([init_owner[None].astype(jnp.int32), owner_seq])
    ships_at = jnp.concatenate(
        [jnp.maximum(0.0, init_garr)[None].astype(jnp.float32), ships_seq]
    )
    return owner_at, ships_at, min_owned, first_enemy, fall_turn


def _survives_with_keep(
    p_owner: jax.Array,
    keep: jax.Array,
    p_prod: jax.Array,
    tarr_owner: jax.Array,
    tarr_ships: jax.Array,
    tarr_valid: jax.Array,
    player: jax.Array,
    horizon: int,
) -> jax.Array:
    """Mirror `survives_with_keep`: does the planet stay `player` for all arrivals?

    Returns a bool. The Python early-returns False on the first arrival that
    leaves the planet non-player; we track a sticky `failed` flag instead.
    """
    group_owner = tarr_owner[1:]
    group_ships = tarr_ships[1:]
    group_valid = tarr_valid[1:]

    Carry = tuple[jax.Array, jax.Array, jax.Array]

    def body(
        carry: Carry, xs: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[Carry, None]:
        owner, garrison, failed = carry
        g_owner, g_ships, g_valid = xs
        produced = jnp.where(owner != -1, garrison + p_prod, garrison)
        has_group = jnp.any(g_valid)
        res_owner, res_garr = resolve_arrival_event_jax(
            owner, produced, g_owner, g_ships, g_valid
        )
        new_owner = jnp.where(has_group, res_owner, owner)
        new_garr = jnp.where(has_group, res_garr, produced)
        new_failed = failed | (has_group & (new_owner != player))
        return (new_owner.astype(jnp.int32), new_garr, new_failed), None

    init_carry: Carry = (p_owner, keep.astype(jnp.float32), jnp.bool_(False))
    (_o, _g, failed), _ = jax.lax.scan(
        body, init_carry, (group_owner, group_ships, group_valid)
    )
    return jnp.logical_not(failed)


def _keep_needed(
    p_owner: jax.Array,
    p_ships: jax.Array,
    p_prod: jax.Array,
    tarr_owner: jax.Array,
    tarr_ships: jax.Array,
    tarr_valid: jax.Array,
    player: jax.Array,
    horizon: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute `(keep_needed, holds_full)` for an owned planet.

    Fixed-iteration binary search replacing the Python `while lo < hi`.
    """
    ships_i = jnp.maximum(0, jnp.round(p_ships).astype(jnp.int32))

    def survives(keep: jax.Array) -> jax.Array:
        return _survives_with_keep(
            p_owner,
            keep,
            p_prod,
            tarr_owner,
            tarr_ships,
            tarr_valid,
            player,
            horizon,
        )

    full_holds = survives(ships_i)

    def search_body(
        _i: int, lohi: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lo, hi = lohi
        active = lo < hi
        mid = (lo + hi) // 2
        ok = survives(mid)
        new_lo = jnp.where(active & jnp.logical_not(ok), mid + 1, lo)
        new_hi = jnp.where(active & ok, mid, hi)
        return new_lo, new_hi

    lo, _hi = jax.lax.fori_loop(
        0, KEEP_SEARCH_ITERS, search_body, (jnp.int32(0), ships_i)
    )
    keep_when_holds = lo
    keep = jnp.where(full_holds, keep_when_holds, ships_i)
    return keep.astype(jnp.int32), full_holds


def simulate_planet_timeline_jax(
    p_owner: jax.Array,
    p_ships: jax.Array,
    p_prod: jax.Array,
    tarr_owner: jax.Array,
    tarr_ships: jax.Array,
    tarr_valid: jax.Array,
    player: jax.Array,
    horizon: int,
) -> TimelineResult:
    """Mirror `simulate_planet_timeline`. `horizon` is a static Python int.

    `tarr_*` are `(horizon+1, MAX_ARRIVALS_PER_TURN)` per-turn arrival tables
    (host pre-bucketed via `normalize_arrivals`). Index 0 is unused.
    """
    init_garrison = jnp.maximum(0.0, p_ships.astype(jnp.float32))

    owner_at, ships_at, min_owned_raw, first_enemy, fall_turn = _forward_scan(
        p_owner,
        init_garrison,
        p_prod,
        tarr_owner,
        tarr_ships,
        tarr_valid,
        player,
        horizon,
    )

    is_owned = p_owner == player
    keep_needed, holds_full = _keep_needed(
        p_owner,
        p_ships,
        p_prod,
        tarr_owner,
        tarr_ships,
        tarr_valid,
        player,
        horizon,
    )
    keep_needed = jnp.where(is_owned, keep_needed, jnp.int32(0))
    holds_full = jnp.where(is_owned, holds_full, jnp.bool_(True))

    min_owned = jnp.where(
        is_owned,
        jnp.maximum(0, jnp.floor(min_owned_raw).astype(jnp.int32)),
        jnp.int32(0),
    )

    return TimelineResult(
        owner_at=owner_at,
        ships_at=ships_at,
        keep_needed=keep_needed.astype(jnp.int32),
        min_owned=min_owned.astype(jnp.int32),
        first_enemy=first_enemy.astype(jnp.int32),
        fall_turn=fall_turn.astype(jnp.int32),
        holds_full=holds_full,
        horizon=jnp.int32(horizon),
    )


def state_at_timeline_jax(
    owner_at: jax.Array,
    ships_at: jax.Array,
    horizon: jax.Array,
    arrival_turn: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mirror `state_at_timeline`: gather owner/ships at clamped ceil(arrival_turn)."""
    turn = jnp.ceil(arrival_turn).astype(jnp.int32)
    turn = jnp.maximum(0, turn)
    turn = jnp.minimum(turn, horizon)
    owner = owner_at[turn]
    ships = jnp.maximum(0.0, ships_at[turn])
    return owner.astype(jnp.int32), ships.astype(jnp.float32)


def indirect_wealth_jax(
    focal_idx: jax.Array,
    xs: jax.Array,
    ys: jax.Array,
    owners: jax.Array,
    prods: jax.Array,
    valid: jax.Array,
    player: jax.Array,
) -> jax.Array:
    """Mirror `indirect_wealth`: sum of distance-discounted production of others.

    `focal_idx` indexes into the `MAX_PLANETS`-padded arrays. The self-skip is by
    array index (Python skips `other.id == planet.id`); padded slots are dropped
    via `valid`. Distances < 1 are skipped (Python guard).
    """
    fx = xs[focal_idx]
    fy = ys[focal_idx]
    idx = jnp.arange(xs.shape[0])

    d = dist_jax(fx, fy, xs, ys)
    is_self = idx == focal_idx
    usable = valid & jnp.logical_not(is_self) & (d >= 1.0)

    factor = prods / (d + 12.0)
    friendly = owners == player
    neutral = owners == -1
    weight = jnp.where(
        friendly,
        INDIRECT_FRIENDLY_WEIGHT,
        jnp.where(neutral, INDIRECT_NEUTRAL_WEIGHT, INDIRECT_ENEMY_WEIGHT),
    )
    contrib = jnp.where(usable, factor * weight, 0.0)
    return jnp.sum(contrib).astype(jnp.float32)


__all__ = [
    "KEEP_SEARCH_ITERS",
    "MAX_ARRIVALS_PER_TURN",
    "MAX_PLANETS",
    "TimelineResult",
    "indirect_wealth_jax",
    "resolve_arrival_event_jax",
    "simulate_planet_timeline_jax",
    "state_at_timeline_jax",
]
