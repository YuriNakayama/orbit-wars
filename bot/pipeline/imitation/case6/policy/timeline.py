"""Per-planet × per-turn ship-prediction timeline (rulebase case6 から portage).

featurizer から呼ばれ、各 planet について敵 fleet 到着順に owner / ships の推移を
シミュレートし、loss_3turn / ttf / min_owned / surplus / fall_predicted /
keep_needed の 6 集約値を返す。

オリジナル: bot/pipeline/rulebase/case6/baseline/core/world_model.py
(`simulate_planet_timeline` / `normalize_arrivals` / `resolve_arrival_event`)。
cross-case import 禁止 (.claude/rules/bot/pipeline.md) のため case5 内にコピーする。

Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
Licensed under Apache License 2.0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

DEFAULT_HORIZON = 30


@dataclass(frozen=True)
class TimelinePlanet:
    """timeline simulation 用 minimal Planet."""

    id: int
    owner: int
    ships: int
    production: int


def resolve_arrival_event(
    owner: int, garrison: float, arrivals: list[tuple[int, int, int]]
) -> tuple[int, float]:
    by_owner: dict[int, int] = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(
    arrivals: list[tuple[float, int, int]], horizon: int
) -> list[tuple[int, int, int]]:
    events: list[tuple[int, int, int]] = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(
    planet: TimelinePlanet,
    arrivals: list[tuple[float, int, int]],
    player: int,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """planet ごとの owner / ships 推移を turn=0..horizon で計算して集約値を返す。

    Returns:
        owner_at: {turn: owner_player_id (-1=neutral)}
        ships_at: {turn: float ships}
        keep_needed: defense として残すべき最小自軍 ships (二分探索)
        min_owned: 自所有期間中の最小 ships (自所有でなければ 0)
        first_enemy: 自所有 → 敵到来の最早 turn (None なら無事)
        fall_turn: 自所有 → 敵所有に陥落した turn (None なら陥落なし)
        holds_full: 全 ships 保持しても陥落するか (False なら陥落確定)
        horizon: シミュレーション上限 turn
    """
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at: dict[int, int] = {0: owner}
    ships_at: dict[int, float] = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy: int | None = None
    fall_turn: int | None = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep: int) -> bool:
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(
                        sim_owner, sim_garrison, group
                    )
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned)))
        if planet.owner == player
        else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def summarize_timeline(
    timeline: dict[str, Any],
    *,
    short_window: int = 3,
) -> dict[str, float]:
    """featurizer 用 6 集約値を timeline から計算する。

    Returns:
        loss_3turn: 自所有 planet なら turn=0 → turn=short_window で失う ships 量 (float)
        ttf_norm: time-to-fall を horizon で割った値 (1.0 = 陥落なし、0.0 = 即陥落)
        min_owned: timeline['min_owned']
        surplus: turn=horizon で残る ships (自所有なら ships_at[horizon]、敵所有なら 0)
        fall_predicted: 1.0 if fall_turn is not None else 0.0
        keep_needed: timeline['keep_needed']
    """
    horizon = int(timeline.get("horizon", DEFAULT_HORIZON))
    fall_turn = timeline.get("fall_turn")
    ships_at = timeline.get("ships_at", {})
    owner_at = timeline.get("owner_at", {})

    s0 = float(ships_at.get(0, 0.0))
    s_w = float(ships_at.get(min(short_window, horizon), s0))
    loss_3turn = (
        max(0.0, s0 - s_w)
        if owner_at.get(0, -1) == owner_at.get(min(short_window, horizon), -1)
        else max(0.0, s0)
    )

    if fall_turn is None:
        ttf_norm = 1.0
    else:
        ttf_norm = max(0.0, min(1.0, float(fall_turn) / max(1, horizon)))

    s_horizon = float(ships_at.get(horizon, 0.0))
    own_h = owner_at.get(horizon, -1)
    surplus = s_horizon if own_h == owner_at.get(0, -1) else 0.0

    return {
        "loss_3turn": loss_3turn,
        "ttf_norm": ttf_norm,
        "min_owned": float(timeline.get("min_owned", 0)),
        "surplus": surplus,
        "fall_predicted": 1.0 if fall_turn is not None else 0.0,
        "keep_needed": float(timeline.get("keep_needed", 0)),
    }
