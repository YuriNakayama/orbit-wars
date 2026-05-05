# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Scoring and filtering helpers shared between plan_moves and mission builders."""

from __future__ import annotations

import math
from typing import Any

from .core.config import (
    AHEAD_ATTACK_MARGIN_BONUS,
    AHEAD_DOMINATION,
    BEHIND_ATTACK_MARGIN_PENALTY,
    BEHIND_DOMINATION,
    BEHIND_ROTATING_NEUTRAL_VALUE_MULT,
    COMET_MARGIN_RELIEF,
    COMET_NPV_ENABLED,
    COMET_NPV_LONGLIFE_BONUS,
    COMET_NPV_LONGLIFE_TURNS,
    COMET_NPV_MIN_USABLE_TURNS,
    COMET_NPV_SHORTLIFE_FLOOR,
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
    FINISHING_ATTACK_MARGIN_BONUS,
    FINISHING_DOMINATION,
    FINISHING_HOSTILE_SEND_BONUS,
    FINISHING_HOSTILE_VALUE_MULT,
    FINISHING_PROD_RATIO,
    FINISHING_TIE_GUARD,
    FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT,
    FOUR_PLAYER_ROTATING_REACTION_GAP,
    FOUR_PLAYER_ROTATING_SEND_RATIO,
    FOUR_PLAYER_ROTATING_TURN_LIMIT,
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
    OM_LAUNCH_SOURCE_BONUS,
    OM_PREFERENCE_BONUS,
    OM_PROACTIVE_THREAT_BOOST,
    OPENING_HOSTILE_TARGET_VALUE_MULT,
    OPPONENT_MODEL_ENABLED,
    PARTIAL_SOURCE_MIN_SHIPS,
    REINFORCE_VALUE_MULT,
    ROTATING_OPENING_LOW_PROD,
    ROTATING_OPENING_MAX_TURNS,
    ROTATING_OPENING_VALUE_MULT,
    SAFE_NEUTRAL_MARGIN,
    SAFE_NEUTRAL_VALUE_MULT,
    SAFE_OPENING_PROD_THRESHOLD,
    SAFE_OPENING_TURN_LIMIT,
    SNIPE_SCORE_MULT,
    SNIPE_VALUE_MULT,
    STATIC_HOSTILE_VALUE_MULT,
    STATIC_NEUTRAL_VALUE_MULT,
    STATIC_TARGET_MARGIN,
    STATIC_TARGET_SCORE_MULT,
    SWARM_SCORE_MULT,
    SWARM_VALUE_MULT,
    THRASH_FILTER_ENABLED,
    THRASH_REPEAT_COMMIT_LIMIT,
    THRASH_SCORE_MULT,
    THRASH_WINDOW,
    WEAK_ENEMY_THRESHOLD,
)
from .core.types import Planet
from .core.world_model import WorldModel


def planet_distance(first: Planet, second: Planet) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def build_modes(world: WorldModel) -> dict[str, Any]:
    domination = (world.my_total - world.enemy_total) / max(
        1, world.my_total + world.enemy_total
    )
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0
        and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 100
    )

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def is_safe_neutral(target: Planet, world: WorldModel) -> bool:
    if target.owner != -1:
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target: Planet, world: WorldModel) -> bool:
    if target.owner != -1:
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def opening_filter(
    target: Planet,
    arrival_turns: int,
    needed: int,
    src_available: int,
    world: WorldModel,
) -> bool:
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if world.is_static(target.id):
        return False

    my_t, enemy_t = world.reaction_times(target.id)
    reaction_gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and reaction_gap >= SAFE_NEUTRAL_MARGIN
    ):
        return False

    if world.is_four_player:
        affordable = needed <= max(
            PARTIAL_SOURCE_MIN_SHIPS,
            int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO),
        )
        if (
            affordable
            and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT
            and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True

    return (
        arrival_turns > ROTATING_OPENING_MAX_TURNS
        or target.production <= ROTATING_OPENING_LOW_PROD
    )


def target_value(
    target: Planet,
    arrival_turns: int,
    mission: str,
    world: WorldModel,
    modes: dict[str, Any],
) -> float:
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value: float = float(target.production * turns_profit)
    value += world.indirect_wealth_map[target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= (
            STATIC_NEUTRAL_VALUE_MULT
            if target.owner == -1
            else STATIC_HOSTILE_VALUE_MULT
        )
    else:
        value *= ROTATING_OPENING_VALUE_MULT if world.is_opening else 1.0

    if target.owner not in (-1, world.player):
        value *= (
            OPENING_HOSTILE_TARGET_VALUE_MULT
            if world.is_opening
            else HOSTILE_TARGET_VALUE_MULT
        )

    if target.owner == -1:
        if is_safe_neutral(target, world):
            value *= SAFE_NEUTRAL_VALUE_MULT
        elif is_contested_neutral(target, world):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT

    if target.id in world.comet_ids:
        if COMET_NPV_ENABLED:
            life = world.comet_life(target.id)
            usable = max(0, life - arrival_turns)
            if usable < COMET_NPV_MIN_USABLE_TURNS:
                ratio = usable / max(1, COMET_NPV_MIN_USABLE_TURNS)
                shortlife_mult = (
                    COMET_NPV_SHORTLIFE_FLOOR
                    + (1.0 - COMET_NPV_SHORTLIFE_FLOOR) * ratio
                )
                value *= COMET_VALUE_MULT * shortlife_mult
            elif usable >= COMET_NPV_LONGLIFE_TURNS:
                value *= COMET_VALUE_MULT * COMET_NPV_LONGLIFE_BONUS
            else:
                value *= COMET_VALUE_MULT
        else:
            value *= COMET_VALUE_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_VALUE_MULT

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if target.owner not in (-1, world.player):
            enemy_strength = world.owner_strength.get(target.owner, 0)
            if enemy_strength <= WEAK_ENEMY_THRESHOLD:
                value += ELIMINATION_BONUS

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, world):
        value *= 1.08
    if (
        modes["is_dominating"]
        and target.owner == -1
        and is_contested_neutral(target, world)
    ):
        value *= 0.92

    if OPPONENT_MODEL_ENABLED:
        if (
            target.owner not in (-1, world.player)
            and world.predicted_arrivals
            and any(
                target.id == ev_target for ev_target in world.predicted_arrivals.keys()
            )
        ):
            # enemy owner with active launch pattern → snipe priority
            value *= 1.0 + OM_LAUNCH_SOURCE_BONUS
        if mission == "reinforce" and world.opponent_threat_score:
            threat = world.opponent_threat_score.get(target.id, 0.0)
            if threat > 0:
                value *= 1.0 + OM_PROACTIVE_THREAT_BOOST
        if target.owner not in (-1, world.player) and world.opponent_threat_score:
            # enemy whose preference aims at our planets → offense priority
            per_enemy = max(
                (
                    world.opponent_threat_score.get(my.id, 0.0)
                    for my in world.my_planets
                ),
                default=0.0,
            )
            if per_enemy > 0.3:
                value *= 1.0 + OM_PREFERENCE_BONUS

    return value


def preferred_send(
    target: Planet,
    base_needed: int,
    arrival_turns: int,
    src_available: int,
    world: WorldModel,
    modes: dict[str, Any],
) -> int:
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, world):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(
            LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR
        )
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    total = send + margin
    if (
        FINISHING_TIE_GUARD
        and modes["is_finishing"]
        and target.owner not in (-1, world.player)
    ):
        total = _avoid_tie_total(total, target, arrival_turns, world)
    total = min(src_available, total)
    if src_available >= FULL_COMMIT_THRESHOLD_SHIPS and total >= int(
        src_available * FULL_COMMIT_FRACTION
    ):
        total = src_available
    return total


def _avoid_tie_total(
    total: int, target: Planet, arrival_turns: int, world: WorldModel
) -> int:
    arrivals = world.arrivals_by_planet.get(target.id, [])
    eta_window = 1
    collisions = [
        ships
        for eta, owner, ships in arrivals
        if owner not in (-1, world.player)
        and abs(eta - arrival_turns) <= eta_window
        and ships > 0
    ]
    if not collisions:
        return total
    for enemy_ships in collisions:
        if total == enemy_ships:
            return total + 1
    return total


def apply_score_modifiers(
    base_score: float, target: Planet, mission: str, world: WorldModel
) -> float:
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (
        len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
        and target.owner == -1
        and not world.is_static(target.id)
    ):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    if THRASH_FILTER_ENABLED and mission in ("capture", "snipe", "swarm"):
        step = world.step
        lost_at = world.recently_lost.get(target.id)
        commits = world.mission_commits.get(target.id, ())
        commit_count = sum(1 for t in commits if step - t <= THRASH_WINDOW)
        is_thrash = (
            lost_at is not None and step - lost_at <= THRASH_WINDOW
        ) or commit_count >= THRASH_REPEAT_COMMIT_LIMIT
        if is_thrash:
            score *= THRASH_SCORE_MULT
    return score


def score_attack(
    target: Planet,
    cost: int,
    turns: int,
    mission: str,
    turn_weight: float,
    world: WorldModel,
    modes: dict[str, Any],
) -> float:
    """value / (cost + turns*weight + 1) の集約スコア式。mission 別 modifier を適用。"""
    value = target_value(target, turns, mission, world, modes)
    if value <= 0:
        return 0.0
    base = value / (cost + turns * turn_weight + 1.0)
    return apply_score_modifiers(base, target, mission, world)
