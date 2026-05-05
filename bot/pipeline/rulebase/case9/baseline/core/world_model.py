# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""World model: arrival ledger, timeline simulation, defense buffers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .config import (
    ANTI_PING_PONG_ENABLED,
    CRASH_EXPLOIT_ENABLED,
    CRASH_EXPLOIT_ETA_WINDOW,
    CRASH_EXPLOIT_MIN_TOTAL_SHIPS,
    DOOMED_EVAC_TURN_LIMIT,
    DOOMED_MIN_SHIPS,
    DYNAMIC_PROACTIVE_HORIZON_ENABLED,
    DYNAMIC_PROACTIVE_HORIZON_MAX,
    DYNAMIC_PROACTIVE_HORIZON_PROD_CAP,
    DYNAMIC_PROACTIVE_HORIZON_PROD_STEP,
    EARLY_TURN_LIMIT,
    HORIZON,
    INDIRECT_ENEMY_WEIGHT,
    INDIRECT_FRIENDLY_WEIGHT,
    INDIRECT_NEUTRAL_WEIGHT,
    LATE_REMAINING_TURNS,
    LAUNCH_CLEARANCE,
    MULTI_ENEMY_PROACTIVE_HORIZON,
    MULTI_ENEMY_PROACTIVE_RATIO,
    MULTI_ENEMY_STACK_WINDOW,
    OPENING_TURN_LIMIT,
    PROACTIVE_DEFENSE_HORIZON,
    PROACTIVE_DEFENSE_RATIO,
    REINFORCE_ENABLED,
    REINFORCE_MIN_DEFICIT,
    REINFORCE_MIN_FUTURE_TURNS,
    REINFORCE_MIN_PRODUCTION,
    TOTAL_STEPS,
    VERY_LATE_REMAINING_TURNS,
)
from .geometry import dist
from .physics import (
    aim_with_prediction,
    comet_remaining_life,
    fleet_speed,
    is_static_planet,
    travel_time,
)
from .safety import (
    fleet_crosses_other_comet,
    intercept_holds_within_tolerance,
    is_trajectory_sun_safe,
    target_reachable_before_comet_expiry,
)
from .types import Fleet, Planet


def fleet_target_planet(
    fleet: Fleet, planets: list[Planet]
) -> tuple[Planet | None, int | None]:
    best_planet: Planet | None = None
    best_time: float = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

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
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(
    fleets: list[Fleet], planets: list[Planet]
) -> dict[int, list[tuple[int, int, int]]]:
    arrivals_by_planet: dict[int, list[tuple[int, int, int]]] = {
        planet.id: [] for planet in planets
    }
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None or eta is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


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
    arrivals: list[tuple[int, int, int]], horizon: int
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
    planet: Planet,
    arrivals: list[tuple[int, int, int]],
    player: int,
    horizon: int,
) -> dict[str, Any]:
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
        "min_owned": (
            max(0, int(math.floor(min_owned))) if planet.owner == player else 0
        ),
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(
    timeline: dict[str, Any], arrival_turn: float
) -> tuple[int, float]:
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets: list[Planet], fleets: list[Fleet]) -> int:
    owners: set[int] = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px: float, py: float, planets: list[Planet]) -> float:
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_wealth(planet: Planet, planets: list[Planet], player: int) -> float:
    wealth = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            wealth += factor * INDIRECT_FRIENDLY_WEIGHT
        elif other.owner == -1:
            wealth += factor * INDIRECT_NEUTRAL_WEIGHT
        else:
            wealth += factor * INDIRECT_ENEMY_WEIGHT
    return wealth


def detect_enemy_crashes(
    arrivals_by_planet: dict[int, list[tuple[int, int, int]]],
    player: int,
    eta_window: int,
) -> list[dict[str, Any]]:
    crashes: list[dict[str, Any]] = []
    for target_id, arrivals in arrivals_by_planet.items():
        enemy_events = [
            (eta, owner, ships)
            for eta, owner, ships in arrivals
            if owner not in (-1, player) and ships > 0
        ]
        if len(enemy_events) < 2:
            continue

        by_owner: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for eta, owner, ships in enemy_events:
            by_owner[owner].append((eta, ships))
        if len(by_owner) < 2:
            continue

        enemy_events.sort(key=lambda item: item[0])
        matched = False
        for i in range(len(enemy_events)):
            if matched:
                break
            for j in range(i + 1, len(enemy_events)):
                eta_a, owner_a, ships_a = enemy_events[i]
                eta_b, owner_b, ships_b = enemy_events[j]
                if owner_a == owner_b:
                    continue
                if abs(eta_a - eta_b) > eta_window:
                    continue
                total = ships_a + ships_b
                if total < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crash_turn = max(eta_a, eta_b)
                crashes.append(
                    {
                        "target_id": target_id,
                        "crash_turn": crash_turn,
                        "total_enemy_ships": total,
                        "contributors": (
                            (owner_a, ships_a),
                            (owner_b, ships_b),
                        ),
                    }
                )
                matched = True
                break
    return crashes


class WorldModel:
    def __init__(
        self,
        player: int,
        step: int,
        planets: list[Planet],
        fleets: list[Fleet],
        initial_by_id: dict[int, Planet],
        ang_vel: float,
        comets: list[dict[str, Any]],
        comet_ids: set[int],
        predicted_arrivals: dict[int, list[tuple[int, int, int]]] | None = None,
        opponent_threat_score: dict[int, float] | None = None,
        recent_dispatches: dict[tuple[int, int], int] | None = None,
    ) -> None:
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids: set[int] = set(comet_ids)
        self.predicted_arrivals: dict[int, list[tuple[int, int, int]]] = (
            predicted_arrivals or {}
        )
        self.opponent_threat_score: dict[int, float] = opponent_threat_score or {}
        self.recent_dispatches: dict[tuple[int, int], int] = recent_dispatches or {}

        self.planet_by_id: dict[int, Planet] = {planet.id: planet for planet in planets}
        self.my_planets: list[Planet] = [
            planet for planet in planets if planet.owner == player
        ]
        self.enemy_planets: list[Planet] = [
            planet for planet in planets if planet.owner not in (-1, player)
        ]
        self.neutral_planets: list[Planet] = [
            planet for planet in planets if planet.owner == -1
        ]
        self.static_neutral_planets: list[Planet] = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength: dict[int, int] = defaultdict(int)
        self.owner_production: dict[int, int] = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(
            strength
            for owner, strength in self.owner_strength.items()
            if owner != player
        )
        self.max_enemy_strength = max(
            (
                strength
                for owner, strength in self.owner_strength.items()
                if owner != player
            ),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        use_predictions = bool(self.predicted_arrivals)
        self.base_timeline: dict[int, dict[str, Any]] = {
            planet.id: simulate_planet_timeline(
                planet,
                (
                    self.arrivals_by_planet[planet.id]
                    + self.predicted_arrivals.get(planet.id, [])
                    if use_predictions
                    else self.arrivals_by_planet[planet.id]
                ),
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.indirect_wealth_map: dict[int, float] = {
            planet.id: indirect_wealth(planet, planets, player) for planet in planets
        }
        self.reaction_cache: dict[int, tuple[int, int]] = {}
        self.base_need_cache: dict[tuple[int, int], int] = {}

        (
            self.reserve,
            self.available,
            self.doomed_candidates,
            self.threatened_candidates,
        ) = self._compute_defense_buffers()

        if CRASH_EXPLOIT_ENABLED and self.is_four_player:
            self.enemy_crashes = detect_enemy_crashes(
                self.arrivals_by_planet,
                player,
                CRASH_EXPLOIT_ETA_WINDOW,
            )
        else:
            self.enemy_crashes = []

    def _dynamic_proactive_horizon(self, planet: Planet) -> tuple[int, int]:
        if not DYNAMIC_PROACTIVE_HORIZON_ENABLED:
            return MULTI_ENEMY_PROACTIVE_HORIZON, PROACTIVE_DEFENSE_HORIZON
        prod_boost = (
            min(
                DYNAMIC_PROACTIVE_HORIZON_PROD_CAP,
                max(0, int(planet.production) - 1),
            )
            * DYNAMIC_PROACTIVE_HORIZON_PROD_STEP
        )
        stack_hz = min(
            DYNAMIC_PROACTIVE_HORIZON_MAX,
            MULTI_ENEMY_PROACTIVE_HORIZON + prod_boost,
        )
        legacy_hz = min(
            DYNAMIC_PROACTIVE_HORIZON_MAX,
            PROACTIVE_DEFENSE_HORIZON + prod_boost,
        )
        return stack_hz, legacy_hz

    def _multi_enemy_proactive_keep(self, planet: Planet) -> int:
        if not self.enemy_planets:
            return 0

        stack_horizon, legacy_horizon = self._dynamic_proactive_horizon(planet)

        threats: list[tuple[int, int]] = []
        for enemy in self.enemy_planets:
            eta = travel_time(
                enemy.x,
                enemy.y,
                enemy.radius,
                planet.x,
                planet.y,
                planet.radius,
                max(1, enemy.ships),
            )
            if eta > stack_horizon:
                continue
            threats.append((eta, int(enemy.ships)))
        if not threats:
            return 0

        threats.sort()
        best_stacked = 0
        left = 0
        running = 0
        for right in range(len(threats)):
            running += threats[right][1]
            while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
                running -= threats[left][1]
                left += 1
            best_stacked = max(best_stacked, running)

        proactive = int(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO)

        legacy = 0
        for eta, ships in threats:
            if eta <= legacy_horizon:
                legacy = max(legacy, int(ships * PROACTIVE_DEFENSE_RATIO))
        return max(proactive, legacy)

    def _compute_defense_buffers(
        self,
    ) -> tuple[dict[int, int], dict[int, int], set[int], dict[int, dict[str, Any]]]:
        reserve: dict[int, int] = {}
        available: dict[int, int] = {}
        doomed_candidates: set[int] = set()
        threatened_candidates: dict[int, dict[str, Any]] = {}

        for planet in self.my_planets:
            timeline = self.base_timeline[planet.id]
            exact_keep = timeline["keep_needed"]

            proactive_keep = self._multi_enemy_proactive_keep(planet)

            reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
            available[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

            if not timeline["holds_full"] and timeline["fall_turn"] is not None:
                fall_turn = timeline["fall_turn"]
                if (
                    fall_turn <= DOOMED_EVAC_TURN_LIMIT
                    and planet.ships >= DOOMED_MIN_SHIPS
                ):
                    doomed_candidates.add(planet.id)

                if (
                    REINFORCE_ENABLED
                    and planet.production >= REINFORCE_MIN_PRODUCTION
                    and self.remaining_steps >= REINFORCE_MIN_FUTURE_TURNS
                ):
                    ships_at = timeline["ships_at"]
                    owner_at = timeline["owner_at"]
                    deficit_hint = 0
                    for turn in range(1, fall_turn + 1):
                        if owner_at.get(turn) != self.player:
                            deficit_hint = max(
                                deficit_hint,
                                int(math.ceil(ships_at.get(turn, 0))) + 1,
                            )
                            break
                    if (
                        ANTI_PING_PONG_ENABLED
                        and deficit_hint < REINFORCE_MIN_DEFICIT
                    ):
                        continue
                    threatened_candidates[planet.id] = {
                        "fall_turn": fall_turn,
                        "deficit_hint": max(1, deficit_hint),
                    }

        return reserve, available, doomed_candidates, threatened_candidates

    def is_static(self, planet_id: int) -> bool:
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id: int) -> int:
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id: int, spent_total: dict[int, int]) -> int:
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def source_attack_left(self, source_id: int, spent_total: dict[int, int]) -> int:
        return max(0, self.available.get(source_id, 0) - spent_total[source_id])

    def plan_shot(
        self, src_id: int, target_id: int, ships: int
    ) -> tuple[float, int, float, float] | None:
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        aim = aim_with_prediction(
            src,
            target,
            ships,
            self.initial_by_id,
            self.ang_vel,
            self.comets,
            self.comet_ids,
        )
        if aim is None:
            return None
        angle, turns, ix, iy = aim
        clearance = src.radius + LAUNCH_CLEARANCE
        launch_x = src.x + math.cos(angle) * clearance
        launch_y = src.y + math.sin(angle) * clearance
        if not is_trajectory_sun_safe(launch_x, launch_y, angle, turns, ships):
            return None
        if not intercept_holds_within_tolerance(
            target=target,
            predicted_turns=turns,
            predicted_pos=(ix, iy),
            initial_by_id=self.initial_by_id,
            ang_vel=self.ang_vel,
            comets=self.comets,
            comet_ids=self.comet_ids,
        ):
            return None
        if not target_reachable_before_comet_expiry(target_id, turns, self.comets):
            return None
        if fleet_crosses_other_comet(
            launch_x=launch_x,
            launch_y=launch_y,
            angle=angle,
            turns=turns,
            ships=ships,
            current_step=self.step,
            comets=self.comets,
            exclude_planet_id=target_id,
        ):
            return None
        return aim

    def reaction_times(self, target_id: int) -> tuple[int, int]:
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = min(
            (
                travel_time(
                    planet.x,
                    planet.y,
                    planet.radius,
                    target.x,
                    target.y,
                    target.radius,
                    max(1, planet.ships),
                )
                for planet in self.my_planets
            ),
            default=10**9,
        )
        enemy_t = min(
            (
                travel_time(
                    planet.x,
                    planet.y,
                    planet.radius,
                    target.x,
                    target.y,
                    target.radius,
                    max(1, planet.ships),
                )
                for planet in self.enemy_planets
            ),
            default=10**9,
        )
        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(
        self,
        target_id: int,
        arrival_turn: float,
        planned_commitments: dict[int, list[tuple[int, int, int]]] | None = None,
        extra_arrivals: tuple[tuple[int, int, int], ...] = (),
    ) -> tuple[int, float]:
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def ships_needed_to_capture(
        self,
        target_id: int,
        arrival_turn: float,
        planned_commitments: dict[int, list[tuple[int, int, int]]] | None = None,
        extra_arrivals: tuple[tuple[int, int, int], ...] = (),
    ) -> int:
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        cache_key: tuple[int, int] | None = None
        if not planned_commitments.get(target_id) and not extra_arrivals:
            cache_key = (target_id, cutoff)
            if cache_key in self.base_need_cache:
                return self.base_need_cache[cache_key]

        owner_t, ships_t = self.projected_state(
            target_id,
            cutoff,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )
        if owner_t == self.player:
            need = 0
        else:
            need = int(math.ceil(ships_t)) + 1

        if cache_key is not None:
            self.base_need_cache[cache_key] = need
        return need

    def reinforcement_needed_for(
        self,
        planet_id: int,
        arrival_turn: float,
        planned_commitments: dict[int, list[tuple[int, int, int]]] | None = None,
    ) -> int:
        planned_commitments = planned_commitments or {}
        arrival_turn_i = max(1, int(math.ceil(arrival_turn)))
        planet = self.planet_by_id[planet_id]
        if planet.owner != self.player:
            return self.ships_needed_to_capture(
                planet_id, arrival_turn_i, planned_commitments
            )

        arrivals = list(self.arrivals_by_planet.get(planet_id, []))
        for item in planned_commitments.get(planet_id, []):
            arrivals.append(item)

        horizon = max(arrival_turn_i + 5, self.base_timeline[planet_id]["horizon"])
        timeline = simulate_planet_timeline(planet, arrivals, self.player, horizon)

        lookahead_end = min(horizon, arrival_turn_i + 20)
        worst_deficit = 0
        for turn in range(arrival_turn_i, lookahead_end + 1):
            owner = timeline["owner_at"].get(turn)
            ships = timeline["ships_at"].get(turn, 0)
            if owner != self.player:
                worst_deficit = max(worst_deficit, int(math.ceil(ships)) + 1)
        return worst_deficit
