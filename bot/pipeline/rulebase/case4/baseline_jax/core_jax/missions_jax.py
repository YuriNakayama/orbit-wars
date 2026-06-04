"""JAX-native port of the mission score chain (baseline/strategy_helpers.py).

Covers opening_filter, target_value, preferred_send, apply_score_modifiers for
NON-COMET targets. All pure arithmetic over per-target scalars + world-level
scalars + a `modes` struct; faithful constant-for-constant to the Python.

Comet-specific multipliers (COMET_VALUE_MULT, comet life clamping) are handled
upstream where comet paths are host-resolved; here `is_comet` is assumed False.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Arr = jax.Array

# strategy_helpers / config constants (verbatim).
SAFE_NEUTRAL_MARGIN = 2
PARTIAL_SOURCE_MIN_SHIPS = 16
SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.62
FOUR_PLAYER_ROTATING_TURN_LIMIT = 10

INDIRECT_VALUE_SCALE = 0.15
STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.55
ROTATING_OPENING_VALUE_MULT = 0.95
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.45
HOSTILE_TARGET_VALUE_MULT = 1.85
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92  # from config
FINISHING_HOSTILE_VALUE_MULT = 1.15  # from config

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 3
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
FINISHING_HOSTILE_SEND_BONUS = 3  # from config

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.84
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.90
ATTACK_COST_TURN_WEIGHT = 0.55


def opening_filter(
    owner: Arr,
    production: Arr,
    is_static: Arr,
    arrival_turns: Arr,
    needed: Arr,
    src_available: Arr,
    my_t: Arr,
    enemy_t: Arr,
    is_opening: Arr,
    is_four_player: Arr,
) -> Arr:
    """True = veto this target. Mirrors strategy_helpers.opening_filter (non-comet)."""
    not_opening = ~is_opening
    not_neutral = owner != -1
    early_skip = not_opening | not_neutral | is_static  # → return False

    reaction_gap = enemy_t - my_t
    safe_release = (
        (production >= SAFE_OPENING_PROD_THRESHOLD)
        & (arrival_turns <= SAFE_OPENING_TURN_LIMIT)
        & (reaction_gap >= SAFE_NEUTRAL_MARGIN)
    )

    # four-player branch
    affordable = needed <= jnp.maximum(
        PARTIAL_SOURCE_MIN_SHIPS,
        (src_available * FOUR_PLAYER_ROTATING_SEND_RATIO).astype(jnp.int32),
    )
    fp_release = (
        affordable
        & (arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT)
        & (reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP)
    )
    fp_veto = ~fp_release  # four-player: return True unless released

    # standard (2-player) branch
    std_veto = (arrival_turns > ROTATING_OPENING_MAX_TURNS) | (
        production <= ROTATING_OPENING_LOW_PROD
    )

    branch_veto = jnp.where(is_four_player, fp_veto, std_veto)
    veto = jnp.where(safe_release, False, branch_veto)
    return jnp.where(early_skip, False, veto)


def target_value(
    owner: Arr,
    production: Arr,
    ships: Arr,
    is_static: Arr,
    indirect_wealth: Arr,
    arrival_turns: Arr,
    my_t: Arr,
    enemy_t: Arr,
    remaining_steps: Arr,
    is_opening: Arr,
    is_early: Arr,
    is_late: Arr,
    enemy_owner_strength: Arr,
    is_finishing: Arr,
    is_behind: Arr,
    is_dominating: Arr,
    player: Arr,
) -> Arr:
    """Mission='capture' target value (non-comet). Mirrors strategy_helpers."""
    turns_profit = jnp.maximum(1, remaining_steps - arrival_turns).astype(jnp.float_)
    value = production.astype(jnp.float_) * turns_profit
    value = value + indirect_wealth * turns_profit * INDIRECT_VALUE_SCALE

    is_neutral = owner == -1
    is_enemy = (owner != -1) & (owner != player)

    # static vs rotating multiplier
    static_mult = jnp.where(
        is_neutral, STATIC_NEUTRAL_VALUE_MULT, STATIC_HOSTILE_VALUE_MULT
    )
    rot_mult = jnp.where(is_opening, ROTATING_OPENING_VALUE_MULT, 1.0)
    value = value * jnp.where(is_static, static_mult, rot_mult)

    # hostile target multiplier
    hostile_mult = jnp.where(
        is_opening, OPENING_HOSTILE_TARGET_VALUE_MULT, HOSTILE_TARGET_VALUE_MULT
    )
    value = jnp.where(is_enemy, value * hostile_mult, value)

    # neutral safe/contested/early
    safe_n = is_neutral & (my_t <= enemy_t - SAFE_NEUTRAL_MARGIN)
    cont_n = is_neutral & (jnp.abs(my_t - enemy_t) <= 2)  # CONTESTED_NEUTRAL_MARGIN
    value = jnp.where(safe_n, value * SAFE_NEUTRAL_VALUE_MULT, value)
    value = jnp.where(cont_n & ~safe_n, value * CONTESTED_NEUTRAL_VALUE_MULT, value)
    value = jnp.where(is_neutral & is_early, value * EARLY_NEUTRAL_VALUE_MULT, value)

    # late game
    late_add = jnp.maximum(0, ships).astype(jnp.float_) * LATE_IMMEDIATE_SHIP_VALUE
    value = jnp.where(is_late, value + late_add, value)
    weak_enemy = is_enemy & (enemy_owner_strength <= WEAK_ENEMY_THRESHOLD)
    value = jnp.where(is_late & weak_enemy, value + ELIMINATION_BONUS, value)

    # modes
    value = jnp.where(
        is_finishing & is_enemy, value * FINISHING_HOSTILE_VALUE_MULT, value
    )
    value = jnp.where(
        is_behind & is_neutral & ~is_static,
        value * BEHIND_ROTATING_NEUTRAL_VALUE_MULT,
        value,
    )
    value = jnp.where(is_behind & safe_n, value * 1.08, value)
    value = jnp.where(is_dominating & cont_n, value * 0.92, value)
    return value


def preferred_send(
    owner: Arr,
    production: Arr,
    base_needed: Arr,
    arrival_turns: Arr,
    src_available: Arr,
    is_static: Arr,
    my_t: Arr,
    enemy_t: Arr,
    is_four_player: Arr,
    attack_margin_mult: Arr,
    is_finishing: Arr,
    player: Arr,
) -> Arr:
    """Mirrors strategy_helpers.preferred_send (non-comet)."""
    base = base_needed.astype(jnp.float_)
    send = jnp.maximum(base, jnp.ceil(base * attack_margin_mult))
    is_neutral = owner == -1
    is_enemy = (owner != -1) & (owner != player)

    neutral_margin = jnp.minimum(
        NEUTRAL_MARGIN_CAP,
        NEUTRAL_MARGIN_BASE + production * NEUTRAL_MARGIN_PROD_WEIGHT,
    )
    hostile_margin = jnp.minimum(
        HOSTILE_MARGIN_CAP,
        HOSTILE_MARGIN_BASE + production * HOSTILE_MARGIN_PROD_WEIGHT,
    )
    margin = jnp.where(is_neutral, neutral_margin, hostile_margin).astype(jnp.float_)
    margin = margin + jnp.where(is_static, STATIC_TARGET_MARGIN, 0)
    cont_n = is_neutral & (jnp.abs(my_t - enemy_t) <= 2)
    margin = margin + jnp.where(cont_n, CONTESTED_TARGET_MARGIN, 0)
    margin = margin + jnp.where(is_four_player, FOUR_PLAYER_TARGET_MARGIN, 0)
    long_travel = jnp.where(
        arrival_turns > LONG_TRAVEL_MARGIN_START,
        jnp.minimum(
            LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR
        ),
        0,
    )
    margin = margin + long_travel.astype(jnp.float_)
    margin = margin + jnp.where(
        is_finishing & is_enemy, FINISHING_HOSTILE_SEND_BONUS, 0
    )
    return jnp.minimum(src_available.astype(jnp.float_), send + margin).astype(
        jnp.int32
    )


def apply_score_modifiers(
    base_score: Arr,
    owner: Arr,
    is_static: Arr,
    is_early: Arr,
    is_four_player: Arr,
    static_neutral_count: Arr,
) -> Arr:
    """Mirrors strategy_helpers.apply_score_modifiers (mission='capture')."""
    is_neutral = owner == -1
    score = base_score
    score = jnp.where(is_static, score * STATIC_TARGET_SCORE_MULT, score)
    score = jnp.where(
        is_early & is_neutral & is_static,
        score * EARLY_STATIC_NEUTRAL_SCORE_MULT,
        score,
    )
    score = jnp.where(
        is_four_player & is_neutral & ~is_static,
        score * FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT,
        score,
    )
    dense = static_neutral_count >= DENSE_STATIC_NEUTRAL_COUNT
    score = jnp.where(
        dense & is_neutral & ~is_static,
        score * DENSE_ROTATING_NEUTRAL_SCORE_MULT,
        score,
    )
    return score
