"""Opponent modeling: launch detection, preference learning, future-launch prediction.

All functions are pure. Per-player rolling history is kept as a small list of
per-turn snapshots and a deque of detected launches. No module-level mutation
happens here; the caller owns the history dict and passes it in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .core.config import (
    OM_HISTORY_WINDOW,
    OM_MAX_HISTORY_ENTRIES,
    OM_MIN_LAUNCH_SHIPS,
    OM_MIN_LAUNCH_STOCK,
    OM_PREDICT_HORIZON,
    TOTAL_STEPS,
)
from .core.physics import fleet_speed
from .core.types import Fleet, Planet


@dataclass
class LaunchEvent:
    step: int
    from_pid: int
    owner: int
    ships: int
    target_pid: int | None
    target_class: (
        str  # "my_planet" | "neutral_high" | "neutral_low" | "enemy" | "unknown"
    )


@dataclass
class TurnSnapshot:
    step: int
    planet_ships: dict[int, int]
    planet_owner: dict[int, int]
    planet_prod: dict[int, int]
    fleet_ids: set[int]


@dataclass
class OMState:
    """Per-game rolling state. Owned by the caller (module-level in agent.py)."""

    last_snapshot: TurnSnapshot | None = None
    launches: list[LaunchEvent] = field(default_factory=list)
    # cumulative target-class counts per enemy owner, for preference learning
    pref_counts: dict[int, dict[str, float]] = field(default_factory=dict)


def make_snapshot(
    step: int, planets: list[Planet], fleets: list[Fleet]
) -> TurnSnapshot:
    return TurnSnapshot(
        step=step,
        planet_ships={p.id: int(p.ships) for p in planets},
        planet_owner={p.id: int(p.owner) for p in planets},
        planet_prod={p.id: int(p.production) for p in planets},
        fleet_ids={int(f.id) for f in fleets},
    )


def _classify_target(target: Planet | None, viewer_player: int) -> str:
    if target is None:
        return "unknown"
    if target.owner == viewer_player:
        return "my_planet"
    if target.owner == -1:
        return "neutral_high" if target.production >= 2 else "neutral_low"
    return "enemy"


def _infer_fleet_target(
    fleet: Fleet, planets: list[Planet], horizon: int
) -> Planet | None:
    """Closest planet that the fleet's ray will intersect within horizon turns."""
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    best: Planet | None = None
    best_turns = 1e9
    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= horizon and turns < best_turns:
            best_turns = turns
            best = planet
    return best


def detect_launches(
    prev: TurnSnapshot,
    curr_step: int,
    curr_planets: list[Planet],
    curr_fleets: list[Fleet],
    viewer_player: int,
) -> list[LaunchEvent]:
    """Detect enemy launches between prev and curr.

    A launch is inferred when an enemy-owned planet whose owner did not change
    lost more ships than can be explained by production + combat (we avoid the
    combat case by only considering planets whose owner did NOT change).
    Target is inferred from the newest fleet originating at that planet.
    """
    if prev.step >= curr_step:
        return []

    events: list[LaunchEvent] = []
    curr_ships = {p.id: int(p.ships) for p in curr_planets}
    curr_owner = {p.id: int(p.owner) for p in curr_planets}
    curr_prod = {p.id: int(p.production) for p in curr_planets}
    curr_by_id = {p.id: p for p in curr_planets}

    new_fleets = [f for f in curr_fleets if int(f.id) not in prev.fleet_ids]
    new_fleets_by_src: dict[int, list[Fleet]] = {}
    for fleet in new_fleets:
        new_fleets_by_src.setdefault(int(fleet.from_planet_id), []).append(fleet)

    for pid, prev_ships in prev.planet_ships.items():
        if pid not in curr_ships:
            continue
        if curr_owner[pid] != prev.planet_owner.get(pid):
            continue  # owner changed → combat ambiguity, skip
        owner = curr_owner[pid]
        if owner in (-1, viewer_player):
            continue
        prod = prev.planet_prod.get(pid, curr_prod[pid])
        expected = prev_ships + prod
        delta = expected - curr_ships[pid]
        if delta < OM_MIN_LAUNCH_SHIPS:
            continue

        # split detected delta across new fleets originating here if present;
        # otherwise we still record a launch with unknown target.
        origin_fleets = new_fleets_by_src.get(pid, [])
        if origin_fleets:
            for fleet in origin_fleets:
                target = _infer_fleet_target(fleet, curr_planets, OM_PREDICT_HORIZON)
                target_pid = target.id if target is not None else None
                tclass = _classify_target(target, viewer_player)
                events.append(
                    LaunchEvent(
                        step=curr_step - 1,
                        from_pid=pid,
                        owner=owner,
                        ships=int(fleet.ships),
                        target_pid=target_pid,
                        target_class=tclass,
                    )
                )
        else:
            # fleet already despawned (rare, combat on arrival same turn) or
            # detection noise — still count the launch but target unknown.
            target_pid = None
            tclass = "unknown"
            # fall back: guess target as nearest enemy/neutral planet
            src = curr_by_id.get(pid)
            if src is not None:
                target_guess = _nearest_non_self(src, curr_planets, owner)
                if target_guess is not None:
                    target_pid = target_guess.id
                    tclass = _classify_target(target_guess, viewer_player)
            events.append(
                LaunchEvent(
                    step=curr_step - 1,
                    from_pid=pid,
                    owner=owner,
                    ships=int(delta),
                    target_pid=target_pid,
                    target_class=tclass,
                )
            )
    return events


def _nearest_non_self(src: Planet, planets: list[Planet], owner: int) -> Planet | None:
    best: Planet | None = None
    best_d = 1e18
    for planet in planets:
        if planet.id == src.id or planet.owner == owner:
            continue
        d = (planet.x - src.x) ** 2 + (planet.y - src.y) ** 2
        if d < best_d:
            best_d = d
            best = planet
    return best


def update_preferences(
    pref_counts: dict[int, dict[str, float]],
    events: list[LaunchEvent],
    decay: float = 0.9,
) -> None:
    """Exponential moving count of target classes per enemy owner."""
    # decay existing
    for owner_map in pref_counts.values():
        for k in list(owner_map.keys()):
            owner_map[k] *= decay
    for ev in events:
        owner_map = pref_counts.setdefault(ev.owner, {})
        owner_map[ev.target_class] = owner_map.get(ev.target_class, 0.0) + float(
            ev.ships
        )


def preference_distribution(
    pref_counts: dict[int, dict[str, float]], owner: int
) -> dict[str, float]:
    """Normalized probability of each target class for a given enemy."""
    counts = pref_counts.get(owner)
    if not counts:
        return {}
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def compute_launch_rate(
    events: list[LaunchEvent], current_step: int, window: int = OM_HISTORY_WINDOW
) -> dict[int, float]:
    """Average ships launched per turn per source planet in the last `window` turns."""
    if not events:
        return {}
    low = current_step - window
    by_src: dict[int, int] = {}
    for ev in events:
        if ev.step < low:
            continue
        by_src[ev.from_pid] = by_src.get(ev.from_pid, 0) + int(ev.ships)
    if not by_src:
        return {}
    return {pid: ships / float(window) for pid, ships in by_src.items()}


def compute_threat_score(
    my_planets: list[Planet],
    enemy_planets: list[Planet],
    reaction_lookup: Any,
    pref_counts: dict[int, dict[str, float]],
) -> dict[int, float]:
    """
    threat_score[p.id] ∝ Σ_enemy P(my_planet | enemy) × 1 / max(1, ETA(enemy -> p))

    `reaction_lookup(target_pid)` must return (my_reaction, enemy_reaction) or
    we use planetary ETA via travel_time; here we accept a callable.
    """
    if not enemy_planets:
        return {p.id: 0.0 for p in my_planets}

    per_enemy_my_share: dict[int, float] = {}
    for enemy in enemy_planets:
        dist = preference_distribution(pref_counts, enemy.owner)
        per_enemy_my_share[enemy.owner] = dist.get("my_planet", 0.0)

    score: dict[int, float] = {}
    for p in my_planets:
        total = 0.0
        for enemy in enemy_planets:
            share = per_enemy_my_share.get(enemy.owner, 0.0)
            if share <= 0:
                continue
            dx = p.x - enemy.x
            dy = p.y - enemy.y
            eta = max(1.0, math.sqrt(dx * dx + dy * dy) / 3.0)
            total += share / eta
        score[p.id] = total
    return score


def predict_future_arrivals(
    enemy_planets: list[Planet],
    my_planets: list[Planet],
    launch_rate: dict[int, float],
    pref_counts: dict[int, dict[str, float]],
    horizon: int = OM_PREDICT_HORIZON,
) -> dict[int, list[tuple[int, int, int]]]:
    """Predict 1-turn-ahead launches from each enemy planet.

    Returns: dict[my_planet_id → [(eta, enemy_owner, ships)]]
    We assume each enemy planet with recent launches will send `launch_rate` ships
    next turn toward its most-likely target class. We bias toward nearest
    my_planet when "my_planet" preference is high.
    """
    predictions: dict[int, list[tuple[int, int, int]]] = {}
    if not my_planets or not enemy_planets:
        return predictions
    for enemy in enemy_planets:
        rate = launch_rate.get(enemy.id, 0.0)
        if rate <= 0 or enemy.ships < OM_MIN_LAUNCH_STOCK:
            continue
        dist = preference_distribution(pref_counts, enemy.owner)
        my_share = dist.get("my_planet", 0.0)
        if my_share <= 0.0:
            continue
        # pick the nearest my_planet as predicted target
        target = _nearest_non_self(enemy, my_planets, enemy.owner)
        if target is None:
            continue
        dx = target.x - enemy.x
        dy = target.y - enemy.y
        eta = max(1, int(math.ceil(math.sqrt(dx * dx + dy * dy) / 3.0)))
        if eta > horizon:
            continue
        ships_predicted = int(math.ceil(rate * my_share))
        if ships_predicted <= 0:
            continue
        predictions.setdefault(target.id, []).append(
            (eta, enemy.owner, ships_predicted)
        )
    return predictions


def trim_history(state: OMState) -> None:
    """Cap launches list and pref_counts to avoid unbounded growth."""
    if len(state.launches) > OM_MAX_HISTORY_ENTRIES:
        del state.launches[: len(state.launches) - OM_MAX_HISTORY_ENTRIES]


def is_new_game(prev: TurnSnapshot | None, curr_step: int) -> bool:
    """Detect game reset (step reset to 0 or went backwards)."""
    if prev is None:
        return True
    if curr_step <= 0:
        return True
    if curr_step < prev.step:
        return True
    if curr_step - prev.step > TOTAL_STEPS // 2:
        return True
    return False
