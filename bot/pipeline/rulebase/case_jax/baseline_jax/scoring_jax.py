# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the scoring helpers in `baseline/strategy_helpers.py`.

Pure, `jit`/`vmap`-friendly translations operating on `WorldFeatures` arrays.
Every Python `if` on a per-target quantity becomes `jnp.where`; the mission
string becomes an int constant (`MISSION_*`). The functions are per-target
(`target_idx` selects a planet slot) so they `vmap` over a `(P_tgt,)` axis; the
`cost` / `turns` / `src_available` args come from the caller and `vmap` too.

Ported:
* `is_safe_neutral`        -> `is_safe_neutral_jax`
* `is_contested_neutral`   -> `is_contested_neutral_jax`
* `target_value`           -> `target_value_jax`     (OM block omitted, see below)
* `apply_score_modifiers`  -> `apply_score_modifiers_jax`
* `score_attack`           -> `score_attack_jax`
* `preferred_send`         -> `preferred_send_jax`

Dead branches (NOT ported, verified in `core/config.py`)
-------------------------------------------------------
* `OPPONENT_MODEL_ENABLED == False` -> the whole `if OPPONENT_MODEL_ENABLED`
  block in `target_value` is unreachable (predicted_arrivals / threat empty).
* `COMET_NPV_ENABLED == False`      -> comet value uses the flat
  `COMET_VALUE_MULT` (the NPV short/long-life branch is dead).
* `FINISHING_TIE_GUARD == False`    -> `_avoid_tie_total` never runs, so
  `preferred_send` skips it. The flag is read so the port stays correct if it is
  ever re-enabled (a guarded `jnp.where(False, ...)` would be needed then; for
  now the dead path is omitted and the flag value asserted == False).

`build_modes` is resolved on the host into `WorldFeatures` scalars; the per-mode
booleans/floats are repacked into `ModesArrays` so the scoring ops receive them
as traced arrays.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    BEHIND_ROTATING_NEUTRAL_VALUE_MULT,
    COMET_MARGIN_RELIEF,
    COMET_VALUE_MULT,
    CONTESTED_NEUTRAL_MARGIN,
    CONTESTED_NEUTRAL_VALUE_MULT,
    CONTESTED_TARGET_MARGIN,
    CRASH_EXPLOIT_VALUE_MULT,
    DENSE_ROTATING_NEUTRAL_SCORE_MULT,
    DENSE_STATIC_NEUTRAL_COUNT,
    EARLY_NEUTRAL_VALUE_MULT,
    EARLY_STATIC_NEUTRAL_SCORE_MULT,
    ELIMINATION_BONUS,
    FINISHING_HOSTILE_SEND_BONUS,
    FINISHING_HOSTILE_VALUE_MULT,
    FINISHING_TIE_GUARD,
    FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT,
    FOUR_PLAYER_TARGET_MARGIN,
    FULL_COMMIT_FRACTION,
    FULL_COMMIT_THRESHOLD_SHIPS,
    HOSTILE_MARGIN_BASE,
    HOSTILE_MARGIN_CAP,
    HOSTILE_MARGIN_PROD_WEIGHT,
    HOSTILE_TARGET_VALUE_MULT,
    INDIRECT_VALUE_SCALE,
    LATE_IMMEDIATE_SHIP_VALUE,
    LONG_TRAVEL_MARGIN_CAP,
    LONG_TRAVEL_MARGIN_DIVISOR,
    LONG_TRAVEL_MARGIN_START,
    NEUTRAL_MARGIN_BASE,
    NEUTRAL_MARGIN_CAP,
    NEUTRAL_MARGIN_PROD_WEIGHT,
    OPENING_HOSTILE_TARGET_VALUE_MULT,
    REINFORCE_VALUE_MULT,
    ROTATING_OPENING_VALUE_MULT,
    SAFE_NEUTRAL_MARGIN,
    SAFE_NEUTRAL_VALUE_MULT,
    SNIPE_SCORE_MULT,
    SNIPE_VALUE_MULT,
    STATIC_HOSTILE_VALUE_MULT,
    STATIC_NEUTRAL_VALUE_MULT,
    STATIC_TARGET_MARGIN,
    STATIC_TARGET_SCORE_MULT,
    SWARM_SCORE_MULT,
    SWARM_VALUE_MULT,
    WEAK_ENEMY_THRESHOLD,
)
from .world_features import WorldFeatures

# Mission encodings. "capture"/"single" share the same (no special) value path.
MISSION_CAPTURE: int = 0
MISSION_SNIPE: int = 1
MISSION_SWARM: int = 2
MISSION_REINFORCE: int = 3
MISSION_CRASH_EXPLOIT: int = 4
MISSION_HARASS: int = 5


class ModesArrays(NamedTuple):
    """`build_modes` result as traced arrays (host-resolved into WorldFeatures)."""

    domination: jax.Array  # float32
    is_behind: jax.Array  # bool
    is_ahead: jax.Array  # bool
    is_dominating: jax.Array  # bool
    is_finishing: jax.Array  # bool
    attack_margin_mult: jax.Array  # float32


def is_safe_neutral_jax(
    owner: jax.Array, my_t: jax.Array, enemy_t: jax.Array
) -> jax.Array:
    """Mirror `is_safe_neutral`: neutral target the player can reach first."""
    is_neutral = owner == -1
    return is_neutral & (my_t <= enemy_t - SAFE_NEUTRAL_MARGIN)


def is_contested_neutral_jax(
    owner: jax.Array, my_t: jax.Array, enemy_t: jax.Array
) -> jax.Array:
    """Mirror `is_contested_neutral`: neutral with near-equal reaction times."""
    is_neutral = owner == -1
    return is_neutral & (jnp.abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN)


def target_value_jax(
    target_idx: jax.Array,
    arrival_turns: jax.Array,
    mission: int,
    features: WorldFeatures,
    modes: ModesArrays,
) -> jax.Array:
    """Mirror `target_value` (OM block omitted; COMET_NPV flat).

    `mission` is a static Python int (`MISSION_*`); `arrival_turns` is traced.
    Returns the float value; the comet `turns_profit <= 0` early-return is -1.0.
    """
    i = target_idx
    player = features.player
    owner = features.owner[i]
    prod = features.prod[i].astype(jnp.float32)
    ship_count = features.ships[i].astype(jnp.float32)
    is_static = features.is_static[i]
    is_comet = features.is_comet[i]
    comet_life = features.comet_life[i].astype(jnp.float32)
    wealth = features.indirect_wealth[i]
    my_t = features.reaction_my_t[i]
    enemy_t = features.reaction_enemy_t[i]
    at = arrival_turns.astype(jnp.float32)

    remaining = features.remaining_steps.astype(jnp.float32)
    is_opening = features.is_opening
    is_early = features.is_early
    is_late = features.is_late
    is_enemy = (owner != -1) & (owner != player)
    is_neutral = owner == -1

    # turns_profit (comet shortens it; <=0 -> short-circuit to -1.0)
    turns_profit = jnp.maximum(1.0, remaining - at)
    comet_profit = jnp.maximum(0.0, jnp.minimum(turns_profit, comet_life - at))
    turns_profit = jnp.where(is_comet, comet_profit, turns_profit)
    comet_dead = is_comet & (turns_profit <= 0.0)

    value = prod * turns_profit
    value = value + wealth * turns_profit * INDIRECT_VALUE_SCALE

    static_mult = jnp.where(
        owner == -1, STATIC_NEUTRAL_VALUE_MULT, STATIC_HOSTILE_VALUE_MULT
    )
    rotating_mult = jnp.where(is_opening, ROTATING_OPENING_VALUE_MULT, 1.0)
    value = value * jnp.where(is_static, static_mult, rotating_mult)

    hostile_mult = jnp.where(
        is_opening, OPENING_HOSTILE_TARGET_VALUE_MULT, HOSTILE_TARGET_VALUE_MULT
    )
    value = jnp.where(is_enemy, value * hostile_mult, value)

    safe = is_safe_neutral_jax(owner, my_t, enemy_t)
    contested = is_contested_neutral_jax(owner, my_t, enemy_t)
    neutral_mult = jnp.where(
        safe,
        SAFE_NEUTRAL_VALUE_MULT,
        jnp.where(contested, CONTESTED_NEUTRAL_VALUE_MULT, 1.0),
    )
    value = jnp.where(is_neutral, value * neutral_mult, value)
    value = jnp.where(is_neutral & is_early, value * EARLY_NEUTRAL_VALUE_MULT, value)

    value = jnp.where(is_comet, value * COMET_VALUE_MULT, value)

    mission_mult = 1.0
    if mission == MISSION_SNIPE:
        mission_mult = SNIPE_VALUE_MULT
    elif mission == MISSION_SWARM:
        mission_mult = SWARM_VALUE_MULT
    elif mission == MISSION_REINFORCE:
        mission_mult = REINFORCE_VALUE_MULT
    elif mission == MISSION_CRASH_EXPLOIT:
        mission_mult = CRASH_EXPLOIT_VALUE_MULT
    value = value * mission_mult

    # late-game additive ship value + elimination bonus
    late_ship = jnp.maximum(0.0, ship_count) * LATE_IMMEDIATE_SHIP_VALUE
    enemy_strength = features.owner_strength[jnp.clip(owner, 0, 3)].astype(jnp.float32)
    weak = is_enemy & (enemy_strength <= WEAK_ENEMY_THRESHOLD)
    elim = jnp.where(weak, ELIMINATION_BONUS, 0.0)
    value = jnp.where(is_late, value + late_ship + elim, value)

    value = jnp.where(
        modes.is_finishing & is_enemy, value * FINISHING_HOSTILE_VALUE_MULT, value
    )
    value = jnp.where(
        modes.is_behind & is_neutral & jnp.logical_not(is_static),
        value * BEHIND_ROTATING_NEUTRAL_VALUE_MULT,
        value,
    )
    value = jnp.where(modes.is_behind & is_neutral & safe, value * 1.08, value)
    value = jnp.where(modes.is_dominating & is_neutral & contested, value * 0.92, value)

    return jnp.where(comet_dead, -1.0, value).astype(jnp.float32)


def apply_score_modifiers_jax(
    base_score: jax.Array,
    target_idx: jax.Array,
    mission: int,
    features: WorldFeatures,
) -> jax.Array:
    """Mirror `apply_score_modifiers`: static / early-static / dense modifiers."""
    i = target_idx
    owner = features.owner[i]
    is_static = features.is_static[i]
    is_neutral = owner == -1
    is_early = features.is_early
    is_four = features.is_four_player
    dense = features.static_neutral_count >= DENSE_STATIC_NEUTRAL_COUNT

    score = base_score
    score = jnp.where(is_static, score * STATIC_TARGET_SCORE_MULT, score)
    score = jnp.where(
        is_early & is_neutral & is_static,
        score * EARLY_STATIC_NEUTRAL_SCORE_MULT,
        score,
    )
    rotating_neutral = is_neutral & jnp.logical_not(is_static)
    score = jnp.where(
        is_four & rotating_neutral,
        score * FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT,
        score,
    )
    score = jnp.where(
        dense & rotating_neutral, score * DENSE_ROTATING_NEUTRAL_SCORE_MULT, score
    )
    if mission == MISSION_SNIPE:
        score = score * SNIPE_SCORE_MULT
    elif mission == MISSION_SWARM:
        score = score * SWARM_SCORE_MULT
    return score.astype(jnp.float32)


def score_attack_jax(
    target_idx: jax.Array,
    cost: jax.Array,
    turns: jax.Array,
    mission: int,
    turn_weight: jax.Array,
    features: WorldFeatures,
    modes: ModesArrays,
) -> jax.Array:
    """Mirror `score_attack`: value / (cost + turns*weight + 1), then modifiers."""
    value = target_value_jax(target_idx, turns, mission, features, modes)
    denom = cost.astype(jnp.float32) + turns.astype(jnp.float32) * turn_weight + 1.0
    base = value / denom
    scored = apply_score_modifiers_jax(base, target_idx, mission, features)
    return jnp.where(value <= 0.0, 0.0, scored).astype(jnp.float32)


def preferred_send_jax(
    target_idx: jax.Array,
    base_needed: jax.Array,
    arrival_turns: jax.Array,
    src_available: jax.Array,
    features: WorldFeatures,
    modes: ModesArrays,
) -> jax.Array:
    """Mirror `preferred_send`: send amount with margins / full-commit snap.

    `_avoid_tie_total` is omitted because `FINISHING_TIE_GUARD == False`. The
    flag is asserted at import time so re-enabling it fails loudly here rather
    than silently diverging.
    """
    i = target_idx
    player = features.player
    owner = features.owner[i]
    prod = features.prod[i]
    is_static = features.is_static[i]
    is_comet = features.is_comet[i]
    my_t = features.reaction_my_t[i]
    enemy_t = features.reaction_enemy_t[i]
    is_enemy = (owner != -1) & (owner != player)
    is_neutral = owner == -1
    at = arrival_turns

    base_needed_f = base_needed.astype(jnp.float32)
    send = jnp.maximum(
        base_needed,
        jnp.ceil(base_needed_f * modes.attack_margin_mult).astype(jnp.int32),
    )

    neutral_margin = jnp.minimum(
        NEUTRAL_MARGIN_CAP, NEUTRAL_MARGIN_BASE + prod * NEUTRAL_MARGIN_PROD_WEIGHT
    )
    hostile_margin = jnp.minimum(
        HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + prod * HOSTILE_MARGIN_PROD_WEIGHT
    )
    margin = jnp.where(is_neutral, neutral_margin, hostile_margin)

    margin = jnp.where(is_static, margin + STATIC_TARGET_MARGIN, margin)
    contested = is_contested_neutral_jax(owner, my_t, enemy_t)
    margin = jnp.where(contested, margin + CONTESTED_TARGET_MARGIN, margin)
    margin = jnp.where(
        features.is_four_player, margin + FOUR_PLAYER_TARGET_MARGIN, margin
    )

    long_travel = jnp.minimum(LONG_TRAVEL_MARGIN_CAP, at // LONG_TRAVEL_MARGIN_DIVISOR)
    margin = jnp.where(at > LONG_TRAVEL_MARGIN_START, margin + long_travel, margin)

    margin = jnp.where(is_comet, jnp.maximum(0, margin - COMET_MARGIN_RELIEF), margin)
    margin = jnp.where(
        modes.is_finishing & is_enemy, margin + FINISHING_HOSTILE_SEND_BONUS, margin
    )

    total = send + margin
    # FINISHING_TIE_GUARD == False -> _avoid_tie_total skipped.
    total = jnp.minimum(src_available, total)

    full_commit = (src_available >= FULL_COMMIT_THRESHOLD_SHIPS) & (
        total
        >= jnp.floor(src_available.astype(jnp.float32) * FULL_COMMIT_FRACTION).astype(
            jnp.int32
        )
    )
    total = jnp.where(full_commit, src_available, total)
    return total.astype(jnp.int32)


# Fail loudly if the dead-branch assumptions stop holding.
assert not FINISHING_TIE_GUARD, (
    "preferred_send_jax omits _avoid_tie_total under FINISHING_TIE_GUARD==False"
)


__all__ = [
    "MISSION_CAPTURE",
    "MISSION_CRASH_EXPLOIT",
    "MISSION_HARASS",
    "MISSION_REINFORCE",
    "MISSION_SNIPE",
    "MISSION_SWARM",
    "ModesArrays",
    "apply_score_modifiers_jax",
    "is_contested_neutral_jax",
    "is_safe_neutral_jax",
    "preferred_send_jax",
    "score_attack_jax",
    "target_value_jax",
]
