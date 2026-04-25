# Adapted from "orbit-star-wars-lb-max-1224" by Roman Tamrazov
# https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224
# Licensed under Apache License 2.0
"""WorldModel + strategy + plan_moves for baseline_v5.

Phase A imported the entire LB1224 notebook here verbatim. Phase B extracted
pure-function blocks (config, types, physics, world helpers) into
`baseline/core/`. The remaining code — WorldModel (stateful, with caches),
the policy/score/plan layer, and the agent entrypoint — stays in this file
because it forms one tightly-coupled strategy unit.
"""

import math
import time
from collections import defaultdict

from .core.config import (
    AHEAD_ATTACK_MARGIN_BONUS,
    AHEAD_DOMINATION,
    ATTACK_COST_TURN_WEIGHT,
    BEHIND_ATTACK_MARGIN_PENALTY,
    BEHIND_DOMINATION,
    BEHIND_ROTATING_NEUTRAL_VALUE_MULT,
    BLOOD_IN_WATER_VALUE_MULT,
    COMET_MARGIN_RELIEF,
    COMET_MAX_CHASE_TURNS,
    COMET_VALUE_MULT,
    CONTESTED_NEUTRAL_MARGIN,
    CONTESTED_NEUTRAL_VALUE_MULT,
    CONTESTED_TARGET_MARGIN,
    CRASH_EXPLOIT_ENABLED,
    CRASH_EXPLOIT_ETA_WINDOW,
    CRASH_EXPLOIT_MIN_TOTAL_SHIPS,
    CRASH_EXPLOIT_POST_CRASH_DELAY,
    CRASH_EXPLOIT_SCORE_MULT,
    CRASH_EXPLOIT_VALUE_MULT,
    DEFENSE_COST_TURN_WEIGHT,
    DEFENSE_FRONTIER_SCORE_MULT,
    DEFENSE_LOOKAHEAD_TURNS,
    DEFENSE_SEND_MARGIN_BASE,
    DEFENSE_SEND_MARGIN_PROD_WEIGHT,
    DEFENSE_SHIP_VALUE,
    DENSE_ROTATING_NEUTRAL_SCORE_MULT,
    DENSE_STATIC_NEUTRAL_COUNT,
    DOOMED_EVAC_TURN_LIMIT,
    DOOMED_MIN_SHIPS,
    EARLY_NEUTRAL_VALUE_MULT,
    EARLY_STATIC_NEUTRAL_SCORE_MULT,
    EARLY_TURN_LIMIT,
    ELIMINATION_BONUS,
    ELIMINATION_PROD_BONUS,
    EXPOSED_PLANET_VALUE_MULT,
    FFA_ELIMINATION_SHIPS,
    FINISHING_ATTACK_MARGIN_BONUS,
    FINISHING_DOMINATION,
    FINISHING_HOSTILE_SEND_BONUS,
    FINISHING_HOSTILE_VALUE_MULT,
    FINISHING_PROD_RATIO,
    FOLLOWUP_MIN_SHIPS,
    FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT,
    FOUR_PLAYER_ROTATING_REACTION_GAP,
    FOUR_PLAYER_ROTATING_SEND_RATIO,
    FOUR_PLAYER_ROTATING_TURN_LIMIT,
    FOUR_PLAYER_TARGET_MARGIN,
    HEAVY_PHASE_MIN_TIME,
    HEAVY_ROUTE_PLANET_LIMIT,
    HORIZON,
    HOSTILE_MARGIN_BASE,
    HOSTILE_MARGIN_CAP,
    HOSTILE_MARGIN_PROD_WEIGHT,
    HOSTILE_SWARM_ETA_TOLERANCE,
    HOSTILE_TARGET_VALUE_MULT,
    INDIRECT_ENEMY_WEIGHT,
    INDIRECT_FRIENDLY_WEIGHT,
    INDIRECT_NEUTRAL_WEIGHT,
    INDIRECT_VALUE_SCALE,
    LATE_CAPTURE_BUFFER,
    LATE_IMMEDIATE_SHIP_VALUE,
    LATE_REMAINING_TURNS,
    LET_THEM_FIGHT_PENALTY,
    LONG_TRAVEL_MARGIN_CAP,
    LONG_TRAVEL_MARGIN_DIVISOR,
    LONG_TRAVEL_MARGIN_START,
    LOW_VALUE_COMET_PRODUCTION,
    MULTI_ENEMY_PROACTIVE_HORIZON,
    MULTI_ENEMY_PROACTIVE_RATIO,
    MULTI_ENEMY_STACK_WINDOW,
    MULTI_SOURCE_ETA_TOLERANCE,
    MULTI_SOURCE_PLAN_PENALTY,
    MULTI_SOURCE_TOP_K,
    NEUTRAL_MARGIN_BASE,
    NEUTRAL_MARGIN_CAP,
    NEUTRAL_MARGIN_PROD_WEIGHT,
    OPENING_HOSTILE_TARGET_VALUE_MULT,
    OPENING_TURN_LIMIT,
    OPTIONAL_PHASE_MIN_TIME,
    PARTIAL_SOURCE_MIN_SHIPS,
    PROACTIVE_DEFENSE_HORIZON,
    PROACTIVE_DEFENSE_RATIO,
    PROACTIVE_ENEMY_TOP_K,
    REACTION_SOURCE_TOP_K_ENEMY,
    REACTION_SOURCE_TOP_K_MY,
    REAR_DISTANCE_RATIO,
    REAR_MAX_TRAVEL_TURNS,
    REAR_SEND_MIN_SHIPS,
    REAR_SEND_RATIO_FOUR_PLAYER,
    REAR_SEND_RATIO_TWO_PLAYER,
    REAR_SOURCE_MIN_SHIPS,
    REAR_STAGE_PROGRESS,
    RECAPTURE_COST_TURN_WEIGHT,
    RECAPTURE_FRONTIER_MULT,
    RECAPTURE_IMMEDIATE_WEIGHT,
    RECAPTURE_LOOKAHEAD_TURNS,
    RECAPTURE_PRODUCTION_WEIGHT,
    RECAPTURE_VALUE_MULT,
    REINFORCE_COST_TURN_WEIGHT,
    REINFORCE_ENABLED,
    REINFORCE_HOLD_LOOKAHEAD,
    REINFORCE_MAX_SOURCE_FRACTION,
    REINFORCE_MAX_TRAVEL_TURNS,
    REINFORCE_MIN_FUTURE_TURNS,
    REINFORCE_MIN_PRODUCTION,
    REINFORCE_SAFETY_MARGIN,
    REINFORCE_VALUE_MULT,
    ROTATING_OPENING_LOW_PROD,
    ROTATING_OPENING_MAX_TURNS,
    ROTATING_OPENING_VALUE_MULT,
    SAFE_NEUTRAL_MARGIN,
    SAFE_NEUTRAL_VALUE_MULT,
    SAFE_OPENING_PROD_THRESHOLD,
    SAFE_OPENING_TURN_LIMIT,
    SNIPE_COST_TURN_WEIGHT,
    SNIPE_SCORE_MULT,
    SNIPE_VALUE_MULT,
    SOFT_ACT_DEADLINE,
    STATIC_HOSTILE_VALUE_MULT,
    STATIC_NEUTRAL_VALUE_MULT,
    STATIC_TARGET_MARGIN,
    STATIC_TARGET_SCORE_MULT,
    SWARM_MIN_PARTICIPANT_SHIPS,
    SWARM_SCORE_MULT,
    SWARM_VALUE_MULT,
    THREE_SOURCE_ETA_TOLERANCE,
    THREE_SOURCE_MIN_TARGET_SHIPS,
    THREE_SOURCE_PLAN_PENALTY,
    THREE_SOURCE_SWARM_ENABLED,
    TOTAL_STEPS,
    TOTAL_WAR_REMAINING_TURNS,
    VERY_LATE_CAPTURE_BUFFER,
    VERY_LATE_REMAINING_TURNS,
    WEAK_ENEMY_THRESHOLD,
)
from .core.types import Fleet, Mission, Planet, ShotOption

__all__ = [
    "Fleet",
    "Mission",
    "Planet",
    "ShotOption",
    "agent",
    "build_world",
]


from .core.physics import (  # noqa: E402
    aim_with_prediction,
    comet_remaining_life,
    is_static_planet,
)

# ============================================================
# World Model
# ============================================================
from .core.world_helpers import (  # noqa: E402
    build_arrival_ledger,
    count_players,
    detect_enemy_fights_at_neutrals,
    detect_exposed_enemy_planets,
    indirect_features,
    nearest_distance_to_set,
    simulate_planet_timeline,
    state_at_timeline,
)


class WorldModel:
    def __init__(
        self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids
    ):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {p.id: p for p in planets}
        self.my_planets = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]
        self.static_neutral_planets = [
            p for p in self.neutral_planets if is_static_planet(p)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_total_war = self.remaining_steps < TOTAL_WAR_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.owner_strength[p.owner] += int(p.ships)
                self.owner_production[p.owner] += int(p.production)
        for f in fleets:
            self.owner_strength[f.owner] += int(f.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(v for k, v in self.owner_strength.items() if k != player)
        self.max_enemy_strength = max(
            (v for k, v in self.owner_strength.items() if k != player), default=0
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            v for k, v in self.owner_production.items() if k != player
        )

        enemy_owners = [k for k in self.owner_strength if k != player]
        if enemy_owners:
            self.weakest_enemy = min(enemy_owners, key=lambda o: self.owner_strength[o])
            self.weakest_enemy_strength = self.owner_strength[self.weakest_enemy]
            self.weakest_enemy_prod = self.owner_production.get(self.weakest_enemy, 0)
        else:
            self.weakest_enemy = None
            self.weakest_enemy_strength = 0
            self.weakest_enemy_prod = 0

        self.blood_in_water_owners = {
            o for o in enemy_owners if self.owner_strength[o] <= FFA_ELIMINATION_SHIPS
        }

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        self.enemy_fights = detect_enemy_fights_at_neutrals(
            self.arrivals_by_planet, player
        )

        self.base_timeline = {
            p.id: simulate_planet_timeline(
                p, self.arrivals_by_planet[p.id], player, HORIZON
            )
            for p in planets
        }
        self.keep_needed_map = {
            p.id: self.base_timeline[p.id]["keep_needed"] for p in planets
        }
        self.fall_turn_map = {
            p.id: self.base_timeline[p.id]["fall_turn"] for p in planets
        }
        self.holds_full_map = {
            p.id: self.base_timeline[p.id]["holds_full"] for p in planets
        }
        self.indirect_feature_map = {
            p.id: indirect_features(p, planets, player) for p in planets
        }
        self.exposed_planet_ids = detect_exposed_enemy_planets(
            fleets, self.enemy_planets
        )

        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(p.ships) for p in planets) + sum(
            int(f.ships) for f in fleets
        )
        self.total_production = sum(int(p.production) for p in planets)

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        ships = int(ships)
        key = (src_id, target_id, ships)
        if key in self.shot_cache:
            return self.shot_cache[key]
        src = self.planet_by_id[src_id]
        tgt = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src,
            tgt,
            ships,
            self.initial_by_id,
            self.ang_vel,
            self.comets,
            self.comet_ids,
        )
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        source_cap = max(1, int(source_cap))
        nh = tuple(int(math.ceil(h)) for h in hints if h is not None)
        cache_key = (src_id, target_id, source_cap, nh)
        if cache_key in self.probe_candidate_cache:
            return self.probe_candidate_cache[cache_key]
        tgt = self.planet_by_id[target_id]
        ts = max(1, int(math.ceil(tgt.ships)))
        values = set(range(1, min(6, source_cap) + 1))
        values.update(
            {
                source_cap,
                max(1, source_cap // 2),
                max(1, source_cap // 3),
                min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
                min(source_cap, ts + 1),
                min(source_cap, ts + 2),
                min(source_cap, ts + 4),
                min(source_cap, ts + 8),
            }
        )
        for h in nh:
            base = max(1, min(source_cap, h))
            for delta in (-2, -1, 0, 1, 2):
                c = base + delta
                if 1 <= c <= source_cap:
                    values.add(c)
        result = sorted(values)
        self.probe_candidate_cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        cache_key = (
            src_id,
            target_id,
            max(1, int(source_cap)),
            tuple(hints),
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        )
        if cache_key in self.best_probe_cache:
            return self.best_probe_cache[cache_key]
        best = best_key = None
        for ships in self.probe_ship_candidates(
            src_id, target_id, source_cap, hints=hints
        ):
            aim = self.plan_shot(src_id, target_id, ships)
            if aim is None:
                continue
            angle, turns, d2t, pt = aim
            if min_turn is not None and turns < min_turn:
                continue
            if max_turn is not None and turns > max_turn:
                continue
            if (
                anchor_turn is not None
                and max_anchor_diff is not None
                and abs(turns - anchor_turn) > max_anchor_diff
            ):
                continue
            key = (
                (abs(turns - anchor_turn), turns, ships)
                if anchor_turn is not None
                else (turns, ships)
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, d2t, pt))
        self.best_probe_cache[cache_key] = best
        return best

    def projected_state(
        self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()
    ):
        pc = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not pc.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)
        arrivals = [
            x for x in self.arrivals_by_planet.get(target_id, []) if x[0] <= cutoff
        ]
        arrivals.extend(x for x in pc.get(target_id, []) if x[0] <= cutoff)
        arrivals.extend(x for x in extra_arrivals if x[0] <= cutoff)
        tgt = self.planet_by_id[target_id]
        return state_at_timeline(
            simulate_planet_timeline(tgt, arrivals, self.player, cutoff), cutoff
        )

    def projected_timeline(
        self, target_id, horizon, planned_commitments=None, extra_arrivals=()
    ):
        pc = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            x for x in self.arrivals_by_planet.get(target_id, []) if x[0] <= horizon
        ]
        arrivals.extend(x for x in pc.get(target_id, []) if x[0] <= horizon)
        arrivals.extend(x for x in extra_arrivals if x[0] <= horizon)
        return simulate_planet_timeline(
            self.planet_by_id[target_id], arrivals, self.player, horizon
        )

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        pc = planned_commitments or {}
        tl = (
            self.projected_timeline(target_id, horizon, planned_commitments=pc)
            if pc.get(target_id)
            else self.base_timeline[target_id]
        )
        return {
            "keep_needed": tl["keep_needed"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        return max(
            32,
            int(
                self.total_visible_ships
                + self.total_production * max(2, eval_turn + 2)
                + 32
            ),
        )

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        pc = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = (
            eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        )
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        ne = tuple(
            (max(1, int(math.ceil(t))), o, int(s))
            for t, o, s in extra_arrivals
            if s > 0 and max(1, int(math.ceil(t))) <= eval_turn
        )

        cache_key = None
        if arrival_turn == eval_turn and not pc.get(target_id) and not ne:
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, _ = self.projected_state(
            target_id, eval_turn, planned_commitments=pc, extra_arrivals=ne
        )
        if owner_before == attacker_owner:
            if cache_key:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            o, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=pc,
                extra_arrivals=ne + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return o == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            _, ships_before = self.projected_state(
                target_id, eval_turn, planned_commitments=pc, extra_arrivals=ne
            )
            hi = max(1, int(math.ceil(ships_before)) + 1)
            sc = self._ownership_search_cap(eval_turn)
            while hi <= sc and not owns_at(hi):
                hi *= 2
            if hi > sc and not owns_at(sc):
                return sc + 1
            hi = min(hi, sc)

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1
        if cache_key:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        pc = planned_commitments or {}
        tgt = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))
        if tgt.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=pc,
                upper_bound=upper_bound,
            )

        def holds_with(ships):
            tl = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=pc,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            return all(
                tl["owner_at"].get(t) == self.player
                for t in range(arrival_turn, hold_until + 1)
            )

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with(hi):
                return hi + 1
        else:
            hi = 1
            sc = self._ownership_search_cap(hold_until)
            while hi <= sc and not holds_with(hi):
                hi *= 2
            if hi > sc and not holds_with(sc):
                return sc + 1
            hi = min(hi, sc)
        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


# ============================================================
# Strategy helpers
# ============================================================


def planet_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def nearest_sources_to_target(target, sources, top_k):
    if not top_k or len(sources) <= top_k:
        return sources
    return sorted(
        sources, key=lambda s: (planet_distance(s, target), -int(s.ships), s.id)
    )[:top_k]


def min_legal_reaction_time(target, sources, world):
    best = 10**9
    for src in sources:
        seed = world.best_probe_aim(src.id, target.id, max(1, int(src.ships)))
        if seed:
            best = min(best, seed[1][1])
    return best


def policy_reaction_times(target_id, policy):
    return policy["reaction_time_map"].get(target_id, (10**9, 10**9))


def candidate_time_valid(target, turns, world, remaining_buffer):
    if turns > world.remaining_steps - remaining_buffer:
        return False
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        if turns >= life or turns > COMET_MAX_CHASE_TURNS:
            return False
    return True


def stacked_enemy_proactive_keep(planet, world):
    threats = []
    for enemy in world.enemy_planets:
        seed = world.best_probe_aim(enemy.id, planet.id, max(1, int(enemy.ships)))
        if seed is None:
            continue
        eta = seed[1][1]
        if eta > MULTI_ENEMY_PROACTIVE_HORIZON:
            continue
        threats.append((eta, int(enemy.ships)))
    if not threats:
        return 0
    threats.sort()
    best = left = running = 0
    for right in range(len(threats)):
        running += threats[right][1]
        while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
            running -= threats[left][1]
            left += 1
        best = max(best, running)
    return int(best * MULTI_ENEMY_PROACTIVE_RATIO)


def swarm_eta_tolerance(options, target, world):
    if len(options) >= 3:
        return THREE_SOURCE_ETA_TOLERANCE
    if target.owner not in (-1, world.player):
        return HOSTILE_SWARM_ETA_TOLERANCE
    return MULTI_SOURCE_ETA_TOLERANCE


def detect_enemy_crashes(world):
    crashes = []
    for target_id, arrivals in world.arrivals_by_planet.items():
        evs = sorted(
            [
                (int(math.ceil(e)), o, int(s))
                for e, o, s in arrivals
                if o not in (-1, world.player) and s > 0
            ]
        )
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                ea, oa, sa = evs[i]
                eb, ob, sb = evs[j]
                if oa == ob or abs(ea - eb) > CRASH_EXPLOIT_ETA_WINDOW:
                    break
                if sa + sb < CRASH_EXPLOIT_MIN_TOTAL_SHIPS:
                    continue
                crashes.append(
                    {
                        "target_id": target_id,
                        "crash_turn": max(ea, eb),
                        "owners": (oa, ob),
                        "ships": (sa, sb),
                    }
                )
    return crashes


def build_policy_state(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    indirect_wealth_map = {
        tid: (
            f * INDIRECT_FRIENDLY_WEIGHT
            + n * INDIRECT_NEUTRAL_WEIGHT
            + e * INDIRECT_ENEMY_WEIGHT
        )
        for tid, (f, n, e) in world.indirect_feature_map.items()
    }

    reserve = {}
    attack_budget = {}
    reaction_time_map = {}

    for target in world.planets:
        if expired() or target.owner == world.player:
            continue
        my_s = nearest_sources_to_target(
            target, world.my_planets, REACTION_SOURCE_TOP_K_MY
        )
        en_s = nearest_sources_to_target(
            target, world.enemy_planets, REACTION_SOURCE_TOP_K_ENEMY
        )
        reaction_time_map[target.id] = (
            min_legal_reaction_time(target, my_s, world),
            min_legal_reaction_time(target, en_s, world),
        )

    for planet in world.my_planets:
        if expired():
            break
        exact_keep = world.keep_needed_map.get(planet.id, 0)
        proactive_keep = 0
        for enemy in nearest_sources_to_target(
            planet, world.enemy_planets, PROACTIVE_ENEMY_TOP_K
        ):
            aim = world.plan_shot(enemy.id, planet.id, max(1, int(enemy.ships)))
            if aim and aim[1] <= PROACTIVE_DEFENSE_HORIZON:
                proactive_keep = max(
                    proactive_keep, int(enemy.ships * PROACTIVE_DEFENSE_RATIO)
                )
        proactive_keep = max(
            proactive_keep, stacked_enemy_proactive_keep(planet, world)
        )
        if world.is_total_war:
            exact_keep = min(exact_keep, max(1, exact_keep // 2))
            proactive_keep = min(proactive_keep, max(1, proactive_keep // 2))
        reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
        attack_budget[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

    return {
        "indirect_wealth_map": indirect_wealth_map,
        "reserve": reserve,
        "attack_budget": attack_budget,
        "reaction_time_map": reaction_time_map,
    }


def build_modes(world):
    dom = (world.my_total - world.enemy_total) / max(
        1, world.my_total + world.enemy_total
    )
    is_behind = dom < BEHIND_DOMINATION
    is_ahead = dom > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        world.max_enemy_strength > 0
        and world.my_total > world.max_enemy_strength * 1.25
    )
    is_finishing = (
        dom > FINISHING_DOMINATION
        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
        and world.step > 80
    )
    margin = (
        1.0
        + (AHEAD_ATTACK_MARGIN_BONUS if is_ahead else 0)
        - (BEHIND_ATTACK_MARGIN_PENALTY if is_behind else 0)
        + (FINISHING_ATTACK_MARGIN_BONUS if is_finishing else 0)
    )
    return {
        "domination": dom,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": margin,
    }


def is_safe_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def is_contested_neutral(target, policy):
    if target.owner != -1:
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def opening_filter(target, arrival_turns, needed, src_available, world, policy):
    if not world.is_opening or target.owner != -1 or target.id in world.comet_ids:
        return False
    if world.is_static(target.id):
        return False
    my_t, enemy_t = policy_reaction_times(target.id, policy)
    gap = enemy_t - my_t
    if (
        target.production >= SAFE_OPENING_PROD_THRESHOLD
        and arrival_turns <= SAFE_OPENING_TURN_LIMIT
        and gap >= SAFE_NEUTRAL_MARGIN
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
            and gap >= FOUR_PLAYER_ROTATING_REACTION_GAP
        ):
            return False
        return True
    return (
        arrival_turns > ROTATING_OPENING_MAX_TURNS
        or target.production <= ROTATING_OPENING_LOW_PROD
    )


def target_value(target, arrival_turns, mission, world, modes, policy):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += (
        policy["indirect_wealth_map"].get(target.id, 0.0)
        * turns_profit
        * INDIRECT_VALUE_SCALE
    )

    if world.is_static(target.id):
        value *= (
            STATIC_NEUTRAL_VALUE_MULT
            if target.owner == -1
            else STATIC_HOSTILE_VALUE_MULT
        )
    elif world.is_opening:
        value *= ROTATING_OPENING_VALUE_MULT

    if target.owner not in (-1, world.player):
        value *= (
            OPENING_HOSTILE_TARGET_VALUE_MULT
            if world.is_opening
            else HOSTILE_TARGET_VALUE_MULT
        )

    if target.owner == -1:
        if is_safe_neutral(target, policy):
            value *= SAFE_NEUTRAL_VALUE_MULT
        elif is_contested_neutral(target, policy):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT

    mission_mult = {
        "snipe": SNIPE_VALUE_MULT,
        "swarm": SWARM_VALUE_MULT,
        "reinforce": REINFORCE_VALUE_MULT,
        "crash_exploit": CRASH_EXPLOIT_VALUE_MULT,
    }
    value *= mission_mult.get(mission, 1.0)

    if target.owner in world.blood_in_water_owners:
        value *= BLOOD_IN_WATER_VALUE_MULT
        value += ELIMINATION_BONUS + target.production * ELIMINATION_PROD_BONUS

    if target.id in world.exposed_planet_ids:
        value *= EXPOSED_PLANET_VALUE_MULT

    if (
        world.is_four_player
        and target.owner == world.weakest_enemy
        and target.owner is not None
    ):
        value *= 1.20

    if world.is_four_player and target.id in world.enemy_fights:
        value *= LET_THEM_FIGHT_PENALTY

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if target.owner not in (-1, world.player):
            if world.owner_strength.get(target.owner, 0) <= WEAK_ENEMY_THRESHOLD:
                value += ELIMINATION_BONUS

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and is_safe_neutral(target, policy):
        value *= 1.10
    if (
        modes["is_dominating"]
        and target.owner == -1
        and is_contested_neutral(target, policy)
    ):
        value *= 0.90

    return value


def reinforce_value(target, hold_until, world, policy):
    saved = max(1, world.remaining_steps - hold_until)
    value = target.production * saved + max(0, target.ships) * DEFENSE_SHIP_VALUE
    if (
        world.enemy_planets
        and nearest_distance_to_set(target.x, target.y, world.enemy_planets) < 22
    ):
        value *= DEFENSE_FRONTIER_SCORE_MULT
    value += (
        policy["indirect_wealth_map"].get(target.id, 0.0)
        * saved
        * INDIRECT_VALUE_SCALE
        * 0.35
    )
    return value * REINFORCE_VALUE_MULT


def preferred_send(
    target, base_needed, arrival_turns, src_available, world, modes, policy
):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    if target.owner == -1:
        margin = min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin = min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if is_contested_neutral(target, policy):
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
    if target.id in world.exposed_planet_ids:
        margin = max(0, margin - 2)
    return min(src_available, send + margin)


def apply_score_modifiers(base_score, target, mission, world):
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
    score *= {
        "snipe": SNIPE_SCORE_MULT,
        "swarm": SWARM_SCORE_MULT,
        "crash_exploit": CRASH_EXPLOIT_SCORE_MULT,
    }.get(mission, 1.0)
    if target.id in world.exposed_planet_ids:
        score *= 1.30
    if target.owner in world.blood_in_water_owners:
        score *= 1.28
    if world.is_four_player and target.owner == world.weakest_enemy:
        score *= 1.18
    return score


def settle_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    modes,
    policy,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tolerance=None,
    max_iter=4,
):
    if src_cap < 1:
        return None
    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda t: t)
    if anchor_tolerance is None:
        anchor_tolerance = 1 if mission == "snipe" else None
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        if send in tested:
            return tested[send]
        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None
        angle, turns, _, _ = aim
        if (
            mission == "crash_exploit"
            and anchor_turn is not None
            and turns < anchor_turn
        ):
            tested[send] = None
            return None
        eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if eval_turn < turns:
            tested[send] = None
            return None
        need = world.min_ships_to_own_by(
            target.id,
            eval_turn,
            world.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None
        if mission in ("snipe", "crash_exploit"):
            desired = need
        elif mission == "rescue":
            desired = min(
                src_cap,
                max(
                    need,
                    need
                    + DEFENSE_SEND_MARGIN_BASE
                    + target.production * DEFENSE_SEND_MARGIN_PROD_WEIGHT,
                ),
            )
        else:
            desired = min(
                src_cap,
                max(
                    need,
                    preferred_send(target, need, turns, src_cap, world, modes, policy),
                ),
            )
        result = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial = sorted(
        world.probe_ship_candidates(src.id, target.id, src_cap, hints=(seed_hint,)),
        key=lambda s: (abs(s - seed_hint), s),
    )
    current_send = None
    for seed in initial:
        r = evaluate(seed)
        if r is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(r[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break
    if current_send is None:
        return None

    for _ in range(max_iter):
        r = evaluate(current_send)
        if r is None:
            break
        angle, turns, eval_turn, need, actual_send, desired = r
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            if mission == "rescue" and turns > eval_turn:
                return None
            return angle, turns, eval_turn, need, actual_send
        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidates = sorted(
        [s for s in tested_order if tested.get(s) is not None],
        key=lambda s: (
            0
            if mission != "snipe" or anchor_turn is None
            else abs(tested[s][1] - anchor_turn),
            abs(s - seed_hint),
            tested[s][1],
            s,
        ),
    )
    for s in dict.fromkeys(candidates):
        r = tested.get(s)
        if r is None:
            continue
        angle, turns, eval_turn, need, actual_send, _ = r
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        if mission == "rescue" and turns > eval_turn:
            continue
        return angle, turns, eval_turn, need, actual_send
    return None


def settle_reinforce_plan(
    src,
    target,
    src_cap,
    send_guess,
    world,
    planned_commitments,
    hold_until,
    max_arrival_turn,
    max_iter=4,
):
    if src_cap < 1:
        return None
    seed_hint = max(1, min(src_cap, int(send_guess)))
    tested = {}
    tested_order = []

    def evaluate(send):
        send = max(1, min(src_cap, int(send)))
        if send in tested:
            return tested[send]
        aim = world.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None
        angle, turns, _, _ = aim
        if turns > max_arrival_turn:
            tested[send] = None
            return None
        need = world.reinforcement_needed_to_hold_until(
            target.id,
            turns,
            hold_until,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None
        desired = min(src_cap, need + REINFORCE_SAFETY_MARGIN)
        result = (angle, turns, hold_until, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial = sorted(
        world.probe_ship_candidates(src.id, target.id, src_cap, hints=(seed_hint,)),
        key=lambda s: (abs(s - seed_hint), s),
    )
    current_send = None
    for seed in initial:
        if evaluate(seed) is not None:
            current_send = seed
            break
    if current_send is None:
        return None

    for _ in range(max_iter):
        r = evaluate(current_send)
        if r is None:
            break
        angle, turns, eval_turn, need, actual_send, desired = r
        if desired == actual_send:
            return angle, turns, eval_turn, need, actual_send
        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    for s in dict.fromkeys(tested_order):
        r = tested.get(s)
        if r and r[4] >= r[3] and r[1] <= max_arrival_turn:
            return r[0], r[1], r[2], r[3], r[4]
    return None


def build_snipe_mission(
    src, target, src_available, world, planned_commitments, modes, policy
):
    if target.owner != -1:
        return None
    enemy_etas = sorted(
        {
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        }
    )
    if not enemy_etas:
        return None
    best = None
    for enemy_eta in enemy_etas[:3]:
        seed = world.best_probe_aim(
            src.id,
            target.id,
            src_available,
            hints=(int(target.ships) + 1, int(target.ships) + 8),
            anchor_turn=enemy_eta,
            max_anchor_diff=1,
        )
        if seed is None:
            continue
        probe, rough = seed
        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            if (
                sync_turn >= world.comet_life(target.id)
                or sync_turn > COMET_MAX_CHASE_TURNS
            ):
                continue
        plan = settle_plan(
            src,
            target,
            src_available,
            probe,
            world,
            planned_commitments,
            modes,
            policy,
            mission="snipe",
            eval_turn_fn=lambda t, ee=enemy_eta: max(t, ee),
            anchor_turn=enemy_eta,
        )
        if plan is None:
            continue
        angle, turns, sync_turn, need, send_pref = plan
        if target.id in world.comet_ids and sync_turn >= world.comet_life(target.id):
            continue
        value = target_value(target, sync_turn, "snipe", world, modes, policy)
        if value <= 0:
            continue
        score = apply_score_modifiers(
            value / (send_pref + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
            target,
            "snipe",
            world,
        )
        option = ShotOption(
            score=score,
            src_id=src.id,
            target_id=target.id,
            angle=angle,
            turns=turns,
            needed=need,
            send_cap=send_pref,
            mission="snipe",
            anchor_turn=enemy_eta,
        )
        m = Mission(
            kind="snipe",
            score=score,
            target_id=target.id,
            turns=sync_turn,
            options=[option],
        )
        if best is None or m.score > best.score:
            best = m
    return best


def build_rescue_missions(world, policy, planned_commitments, modes):
    missions = []
    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue
        for src in world.my_planets:
            if src.id == target.id:
                continue
            sa = policy["attack_budget"].get(src.id, 0)
            if sa < PARTIAL_SOURCE_MIN_SHIPS:
                continue
            seed = world.best_probe_aim(
                src.id,
                target.id,
                sa,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                max_turn=fall_turn,
            )
            if seed is None:
                continue
            plan = settle_plan(
                src,
                target,
                sa,
                seed[0],
                world,
                planned_commitments,
                modes,
                policy,
                mission="rescue",
                eval_turn_fn=lambda t, ft=fall_turn: ft,
                anchor_turn=fall_turn,
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            saved = max(1, world.remaining_steps - fall_turn)
            value = (
                target.production * saved + max(0, target.ships) * DEFENSE_SHIP_VALUE
            )
            if (
                world.enemy_planets
                and nearest_distance_to_set(target.x, target.y, world.enemy_planets)
                < 22
            ):
                value *= DEFENSE_FRONTIER_SCORE_MULT
            score = value / (send_pref + turns * DEFENSE_COST_TURN_WEIGHT + 1.0)
            missions.append(
                Mission(
                    kind="rescue",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[
                        ShotOption(
                            score=score,
                            src_id=src.id,
                            target_id=target.id,
                            angle=angle,
                            turns=turns,
                            needed=need,
                            send_cap=send_pref,
                            mission="rescue",
                            anchor_turn=fall_turn,
                        )
                    ],
                )
            )
    return missions


def build_recapture_missions(world, policy, planned_commitments, modes):
    missions = []
    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or fall_turn > DEFENSE_LOOKAHEAD_TURNS:
            continue
        for src in world.my_planets:
            if src.id == target.id:
                continue
            sa = policy["attack_budget"].get(src.id, 0)
            if sa < PARTIAL_SOURCE_MIN_SHIPS:
                continue
            seed = world.best_probe_aim(
                src.id,
                target.id,
                sa,
                hints=(target.production + DEFENSE_SEND_MARGIN_BASE + 2,),
                min_turn=fall_turn + 1,
                max_turn=fall_turn + RECAPTURE_LOOKAHEAD_TURNS,
            )
            if seed is None:
                continue
            plan = settle_plan(
                src,
                target,
                sa,
                seed[0],
                world,
                planned_commitments,
                modes,
                policy,
                mission="capture",
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            if turns <= fall_turn or turns - fall_turn > RECAPTURE_LOOKAHEAD_TURNS:
                continue
            saved = max(1, world.remaining_steps - turns)
            value = (
                RECAPTURE_PRODUCTION_WEIGHT * target.production * saved
                + RECAPTURE_IMMEDIATE_WEIGHT * max(0, target.ships)
            )
            if (
                world.enemy_planets
                and nearest_distance_to_set(target.x, target.y, world.enemy_planets)
                < 22
            ):
                value *= RECAPTURE_FRONTIER_MULT
            value *= RECAPTURE_VALUE_MULT
            score = value / (send_pref + turns * RECAPTURE_COST_TURN_WEIGHT + 1.0)
            missions.append(
                Mission(
                    kind="recapture",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[
                        ShotOption(
                            score=score,
                            src_id=src.id,
                            target_id=target.id,
                            angle=angle,
                            turns=turns,
                            needed=need,
                            send_cap=send_pref,
                            mission="recapture",
                            anchor_turn=fall_turn,
                        )
                    ],
                )
            )
    return missions


def build_reinforce_missions(
    world, policy, planned_commitments, modes, inventory_left_fn
):
    if not REINFORCE_ENABLED or world.remaining_steps < REINFORCE_MIN_FUTURE_TURNS:
        return []
    missions = []
    for target in world.my_planets:
        fall_turn = world.fall_turn_map.get(target.id)
        if fall_turn is None or target.production < REINFORCE_MIN_PRODUCTION:
            continue
        hold_until = min(HORIZON, fall_turn + REINFORCE_HOLD_LOOKAHEAD)
        max_arrival_turn = min(fall_turn, REINFORCE_MAX_TRAVEL_TURNS)
        for src in world.my_planets:
            if src.id == target.id:
                continue
            budget = inventory_left_fn(src.id)
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue
            seed = world.best_probe_aim(
                src.id,
                target.id,
                source_cap,
                hints=(target.production + REINFORCE_SAFETY_MARGIN + 2,),
                max_turn=max_arrival_turn,
            )
            if seed is None:
                continue
            plan = settle_reinforce_plan(
                src,
                target,
                source_cap,
                seed[0],
                world,
                planned_commitments,
                hold_until,
                max_arrival_turn,
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            value = reinforce_value(target, hold_until, world, policy)
            score = value / (send_pref + turns * REINFORCE_COST_TURN_WEIGHT + 1.0)
            missions.append(
                Mission(
                    kind="reinforce",
                    score=score,
                    target_id=target.id,
                    turns=fall_turn,
                    options=[
                        ShotOption(
                            score=score,
                            src_id=src.id,
                            target_id=target.id,
                            angle=angle,
                            turns=turns,
                            needed=need,
                            send_cap=send_pref,
                            mission="reinforce",
                            anchor_turn=hold_until,
                        )
                    ],
                )
            )
    return missions


def build_crash_exploit_missions(world, policy, planned_commitments, modes):
    if not CRASH_EXPLOIT_ENABLED or not world.is_four_player:
        return []
    missions = []
    for crash in detect_enemy_crashes(world):
        target = world.planet_by_id[crash["target_id"]]
        if target.owner == world.player:
            continue
        desired_arrival = crash["crash_turn"] + CRASH_EXPLOIT_POST_CRASH_DELAY
        for src in world.my_planets:
            sa = policy["attack_budget"].get(src.id, 0)
            if sa < PARTIAL_SOURCE_MIN_SHIPS:
                continue
            seed = world.best_probe_aim(
                src.id,
                target.id,
                sa,
                hints=(12, int(target.ships) + 1),
                anchor_turn=desired_arrival,
                max_anchor_diff=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if seed is None:
                continue
            plan = settle_plan(
                src,
                target,
                sa,
                seed[0],
                world,
                planned_commitments,
                modes,
                policy,
                mission="crash_exploit",
                eval_turn_fn=lambda t, da=desired_arrival: max(t, da),
                anchor_turn=desired_arrival,
                anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            if not candidate_time_valid(target, turns, world, LATE_CAPTURE_BUFFER):
                continue
            value = target_value(target, turns, "crash_exploit", world, modes, policy)
            if value <= 0:
                continue
            score = apply_score_modifiers(
                value / (send_pref + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                target,
                "crash_exploit",
                world,
            )
            missions.append(
                Mission(
                    kind="crash_exploit",
                    score=score,
                    target_id=target.id,
                    turns=turns,
                    options=[
                        ShotOption(
                            score=score,
                            src_id=src.id,
                            target_id=target.id,
                            angle=angle,
                            turns=turns,
                            needed=need,
                            send_cap=send_pref,
                            mission="crash_exploit",
                            anchor_turn=desired_arrival,
                        )
                    ],
                )
            )
    return missions


def plan_moves(world, deadline=None):
    def expired():
        return deadline is not None and time.perf_counter() > deadline

    def time_left():
        return (deadline - time.perf_counter()) if deadline is not None else 10**9

    def allow_heavy():
        return (
            time_left() > HEAVY_PHASE_MIN_TIME
            and len(world.planets) <= HEAVY_ROUTE_PLANET_LIMIT
        )

    def allow_optional():
        return time_left() > OPTIONAL_PHASE_MIN_TIME

    modes = build_modes(world)
    policy = build_policy_state(world, deadline=deadline)
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def inv_left(sid):
        return world.source_inventory_left(sid, spent_total)

    def atk_left(sid):
        return max(0, policy["attack_budget"].get(sid, 0) - spent_total[sid])

    def append_move(src_id, angle, ships):
        send = min(int(ships), inv_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    def finalize():
        final, used = [], defaultdict(int)
        for src_id, angle, ships in moves:
            src = world.planet_by_id[src_id]
            send = min(int(ships), int(src.ships) - used[src_id])
            if send >= 1:
                final.append([src_id, float(angle), int(send)])
                used[src_id] += send
        return final

    def compute_live_doomed():
        doomed = set()
        for p in world.my_planets:
            st = world.hold_status(
                p.id,
                planned_commitments=planned_commitments,
                horizon=DOOMED_EVAC_TURN_LIMIT,
            )
            if (
                not st["holds_full"]
                and st["fall_turn"] is not None
                and st["fall_turn"] <= DOOMED_EVAC_TURN_LIMIT
                and inv_left(p.id) >= DOOMED_MIN_SHIPS
            ):
                doomed.add(p.id)
        return doomed

    def time_ok(target, turns, needed, src_cap):
        buf = VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER
        if not candidate_time_valid(target, turns, world, buf):
            return False
        if opening_filter(target, turns, needed, src_cap, world, policy):
            return False
        return True

    # ── Defense ─────────────────────────────────────────────
    if allow_heavy():
        missions.extend(
            build_reinforce_missions(
                world, policy, planned_commitments, modes, inv_left
            )
        )
    missions.extend(build_rescue_missions(world, policy, planned_commitments, modes))
    missions.extend(build_recapture_missions(world, policy, planned_commitments, modes))

    # ── Candidate generation ─────────────────────────────────
    for src in world.my_planets:
        if expired():
            return finalize()
        sa = atk_left(src.id)
        if sa <= 0:
            continue
        for target in world.planets:
            if expired():
                return finalize()
            if target.id == src.id or target.owner == world.player:
                continue

            seed = world.best_probe_aim(
                src.id, target.id, sa, hints=(int(target.ships) + 1,)
            )
            if seed is None:
                continue
            _, rough_aim = seed
            rough_turns = rough_aim[1]
            buf = (
                VERY_LATE_CAPTURE_BUFFER if world.is_very_late else LATE_CAPTURE_BUFFER
            )
            if not candidate_time_valid(target, rough_turns, world, buf):
                continue

            global_needed = world.min_ships_to_own_at(
                target.id,
                rough_turns,
                world.player,
                planned_commitments=planned_commitments,
            )
            if global_needed <= 0:
                continue
            if opening_filter(target, rough_turns, global_needed, sa, world, policy):
                continue

            partial_cap = min(
                sa,
                preferred_send(
                    target, global_needed, rough_turns, sa, world, modes, policy
                ),
            )
            if partial_cap >= SWARM_MIN_PARTICIPANT_SHIPS:
                ps = world.best_probe_aim(
                    src.id,
                    target.id,
                    partial_cap,
                    hints=(partial_cap, global_needed, int(target.ships) + 1),
                )
                if ps is not None:
                    _, pa = ps
                    p_ang, p_turns = pa[0], pa[1]
                    if time_ok(target, p_turns, global_needed, sa):
                        pv = target_value(
                            target, p_turns, "swarm", world, modes, policy
                        )
                        if pv > 0:
                            ps_score = apply_score_modifiers(
                                pv
                                / (
                                    partial_cap
                                    + p_turns * ATTACK_COST_TURN_WEIGHT
                                    + 1.0
                                ),
                                target,
                                "swarm",
                                world,
                            )
                            source_options_by_target[target.id].append(
                                ShotOption(
                                    score=ps_score,
                                    src_id=src.id,
                                    target_id=target.id,
                                    angle=p_ang,
                                    turns=p_turns,
                                    needed=global_needed,
                                    send_cap=partial_cap,
                                    mission="swarm",
                                )
                            )

            if global_needed <= sa:
                send_guess = preferred_send(
                    target, global_needed, rough_turns, sa, world, modes, policy
                )
                plan = settle_plan(
                    src,
                    target,
                    sa,
                    send_guess,
                    world,
                    planned_commitments,
                    modes,
                    policy,
                )
                if plan is None:
                    continue
                angle, turns, _, needed, send_cap = plan
                if not time_ok(target, turns, needed, sa) or send_cap < 1:
                    continue
                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue
                score = apply_score_modifiers(
                    value / (send_cap + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                opt = ShotOption(
                    score=score,
                    src_id=src.id,
                    target_id=target.id,
                    angle=angle,
                    turns=turns,
                    needed=needed,
                    send_cap=send_cap,
                    mission="capture",
                )
                if send_cap >= needed:
                    missions.append(
                        Mission(
                            kind="single",
                            score=score,
                            target_id=target.id,
                            turns=turns,
                            options=[opt],
                        )
                    )

            sn = build_snipe_mission(
                src, target, sa, world, planned_commitments, modes, policy
            )
            if sn is not None:
                missions.append(sn)

    # ── Swarm assembly ───────────────────────────────────────
    for target_id, options in source_options_by_target.items():
        if expired() or len(options) < 2:
            continue
        target = world.planet_by_id[target_id]
        top = sorted(options, key=lambda x: -x.score)[:MULTI_SOURCE_TOP_K]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                if a.src_id == b.src_id:
                    continue
                if (
                    a.send_cap < SWARM_MIN_PARTICIPANT_SHIPS
                    or b.send_cap < SWARM_MIN_PARTICIPANT_SHIPS
                ):
                    continue
                tol = swarm_eta_tolerance((a, b), target, world)
                if abs(a.turns - b.turns) > tol:
                    continue
                jt = max(a.turns, b.turns)
                tc = a.send_cap + b.send_cap
                need = world.min_ships_to_own_at(
                    target_id,
                    jt,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=tc,
                )
                if need <= 0 or a.send_cap >= need or b.send_cap >= need or tc < need:
                    continue
                value = target_value(target, jt, "swarm", world, modes, policy)
                if value <= 0:
                    continue
                sc = (
                    apply_score_modifiers(
                        value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                        target,
                        "swarm",
                        world,
                    )
                    * MULTI_SOURCE_PLAN_PENALTY
                )
                missions.append(
                    Mission(
                        kind="swarm",
                        score=sc,
                        target_id=target_id,
                        turns=jt,
                        options=[a, b],
                    )
                )

        if (
            THREE_SOURCE_SWARM_ENABLED
            and allow_heavy()
            and target.owner not in (-1, world.player)
            and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS
            and len(top) >= 3
        ):
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    for k in range(j + 1, len(top)):
                        if expired():
                            return finalize()
                        trio = [top[i], top[j], top[k]]
                        if len({o.src_id for o in trio}) < 3:
                            continue
                        if any(o.send_cap < SWARM_MIN_PARTICIPANT_SHIPS for o in trio):
                            continue
                        ts = [o.turns for o in trio]
                        if max(ts) - min(ts) > THREE_SOURCE_ETA_TOLERANCE:
                            continue
                        jt = max(ts)
                        tc = sum(o.send_cap for o in trio)
                        need = world.min_ships_to_own_at(
                            target_id,
                            jt,
                            world.player,
                            planned_commitments=planned_commitments,
                            upper_bound=tc,
                        )
                        if need <= 0 or tc < need:
                            continue
                        if any(
                            trio[a2].send_cap + trio[b2].send_cap >= need
                            for a2 in range(3)
                            for b2 in range(a2 + 1, 3)
                        ):
                            continue
                        value = target_value(target, jt, "swarm", world, modes, policy)
                        if value <= 0:
                            continue
                        sc = (
                            apply_score_modifiers(
                                value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                target,
                                "swarm",
                                world,
                            )
                            * THREE_SOURCE_PLAN_PENALTY
                        )
                        missions.append(
                            Mission(
                                kind="swarm",
                                score=sc,
                                target_id=target_id,
                                turns=jt,
                                options=trio,
                            )
                        )

    if allow_heavy():
        missions.extend(
            build_crash_exploit_missions(world, policy, planned_commitments, modes)
        )

    missions.sort(key=lambda m: -m.score)

    # ── Commit loop ──────────────────────────────────────────
    for mission in missions:
        if expired():
            return finalize()
        target = world.planet_by_id[mission.target_id]

        if mission.kind in (
            "single",
            "snipe",
            "rescue",
            "recapture",
            "reinforce",
            "crash_exploit",
        ):
            opt = mission.options[0]
            src = world.planet_by_id[opt.src_id]
            left = (
                min(
                    inv_left(opt.src_id), int(src.ships * REINFORCE_MAX_SOURCE_FRACTION)
                )
                if mission.kind == "reinforce"
                else atk_left(opt.src_id)
            )
            if left <= 0:
                continue

            if mission.kind == "reinforce":
                plan = settle_reinforce_plan(
                    src,
                    target,
                    left,
                    min(left, opt.send_cap),
                    world,
                    planned_commitments,
                    opt.anchor_turn,
                    mission.turns,
                )
            elif mission.kind == "rescue":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, opt.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="rescue",
                    eval_turn_fn=lambda t, ft=mission.turns: ft,
                    anchor_turn=opt.anchor_turn,
                )
            elif mission.kind == "snipe":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, opt.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="snipe",
                    eval_turn_fn=lambda t, ee=opt.anchor_turn: max(t, ee),
                    anchor_turn=opt.anchor_turn,
                )
            elif mission.kind == "crash_exploit":
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, opt.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="crash_exploit",
                    eval_turn_fn=lambda t, da=opt.anchor_turn: max(t, da),
                    anchor_turn=opt.anchor_turn,
                    anchor_tolerance=CRASH_EXPLOIT_ETA_WINDOW,
                )
            else:
                plan = settle_plan(
                    src,
                    target,
                    left,
                    min(left, opt.send_cap),
                    world,
                    planned_commitments,
                    modes,
                    policy,
                    mission="capture",
                )

            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need or need > left:
                continue
            sent = append_move(opt.src_id, angle, send)
            if sent < need:
                continue
            planned_commitments[target.id].append((turns, world.player, int(sent)))
            continue

        # Swarm
        limits = [min(atk_left(o.src_id), o.send_cap) for o in mission.options]
        if min(limits) <= 0:
            continue
        missing = world.min_ships_to_own_at(
            target.id,
            mission.turns,
            world.player,
            planned_commitments=planned_commitments,
            upper_bound=sum(limits),
        )
        if missing <= 0 or sum(limits) < missing:
            continue

        ordered = sorted(
            zip(mission.options, limits), key=lambda x: (x[0].turns, -x[1], x[0].src_id)
        )
        remaining = missing
        sends = {}
        for idx, (opt, lim) in enumerate(ordered):
            ro = sum(ol for _, ol in ordered[idx + 1 :])
            sends[opt.src_id] = min(lim, max(0, remaining - ro))
            remaining -= sends[opt.src_id]
        if remaining > 0:
            continue

        reaimed = []
        for opt, _ in ordered:
            send = sends.get(opt.src_id, 0)
            if send <= 0:
                continue
            aim = world.plan_shot(opt.src_id, target.id, send)
            if aim is None:
                reaimed = []
                break
            reaimed.append((opt.src_id, aim[0], aim[1], send))
        if not reaimed:
            continue

        ts = [x[2] for x in reaimed]
        if max(ts) - min(ts) > swarm_eta_tolerance(mission.options, target, world):
            continue

        owner_after, _ = world.projected_state(
            target.id,
            max(ts),
            planned_commitments=planned_commitments,
            extra_arrivals=[(t, world.player, s) for _, _, t, s in reaimed],
        )
        if owner_after != world.player:
            continue

        committed = []
        for src_id, angle, turns, send in reaimed:
            actual = append_move(src_id, angle, send)
            if actual > 0:
                committed.append((turns, world.player, int(actual)))
        if sum(x[2] for x in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # ── Follow-up pass ───────────────────────────────────────
    if not world.is_very_late and allow_optional():
        for src in world.my_planets:
            if expired():
                return finalize()
            sl = atk_left(src.id)
            if sl < FOLLOWUP_MIN_SHIPS:
                continue
            best = None
            for target in world.planets:
                if expired():
                    return finalize()
                if target.id == src.id or target.owner == world.player:
                    continue
                if (
                    target.id in world.comet_ids
                    and target.production <= LOW_VALUE_COMET_PRODUCTION
                ):
                    continue
                if not world.is_total_war and planned_commitments.get(target.id):
                    continue
                seed = world.best_probe_aim(
                    src.id, target.id, sl, hints=(int(target.ships) + 1,)
                )
                if seed is None:
                    continue
                est_turns = seed[1][1]
                if (
                    world.is_late
                    and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER
                ):
                    continue
                need = world.min_ships_to_own_at(
                    target.id,
                    est_turns,
                    world.player,
                    planned_commitments=planned_commitments,
                    upper_bound=sl,
                )
                if need <= 0 or need > sl:
                    continue
                if opening_filter(target, est_turns, need, sl, world, policy):
                    continue
                send = preferred_send(target, need, est_turns, sl, world, modes, policy)
                if send < need:
                    continue
                plan = settle_plan(
                    src, target, sl, send, world, planned_commitments, modes, policy
                )
                if plan is None:
                    continue
                _, turns, _, pn, fs = plan
                if (
                    world.is_late
                    and turns > world.remaining_steps - LATE_CAPTURE_BUFFER
                ):
                    continue
                if fs < pn:
                    continue
                value = target_value(target, turns, "capture", world, modes, policy)
                if value <= 0:
                    continue
                score = apply_score_modifiers(
                    value / (fs + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                    target,
                    "capture",
                    world,
                )
                if best is None or score > best[0]:
                    best = (score, target, plan)
            if best is None:
                continue
            _, target, plan = best
            angle, turns, _, need, send = plan
            sl = atk_left(src.id)
            if need > sl:
                continue
            plan = settle_plan(
                src,
                target,
                sl,
                min(sl, send),
                world,
                planned_commitments,
                modes,
                policy,
            )
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = append_move(src.id, angle, send)
            if actual >= need:
                planned_commitments[target.id].append(
                    (turns, world.player, int(actual))
                )

    # ── Doomed evacuation ────────────────────────────────────
    if not expired():
        live_doomed = compute_live_doomed()
        if live_doomed:
            ft = (
                world.enemy_planets
                or world.static_neutral_planets
                or world.neutral_planets
            )
            fdist = {
                p.id: nearest_distance_to_set(p.x, p.y, ft) if ft else 10**9
                for p in world.my_planets
            }
            for planet in world.my_planets:
                if expired() or planet.id not in live_doomed:
                    continue
                avail = inv_left(planet.id)
                if avail < policy["reserve"].get(planet.id, 0):
                    continue
                best_cap = None
                for target in world.planets:
                    if target.id == planet.id or target.owner == world.player:
                        continue
                    seed = world.best_probe_aim(
                        planet.id,
                        target.id,
                        avail,
                        hints=(avail, int(target.ships) + 1),
                    )
                    if seed is None:
                        continue
                    pt = seed[1][1]
                    if pt > world.remaining_steps - 2:
                        continue
                    need = world.min_ships_to_own_at(
                        target.id,
                        pt,
                        world.player,
                        planned_commitments=planned_commitments,
                        upper_bound=avail,
                    )
                    if need <= 0 or need > avail:
                        continue
                    plan = settle_plan(
                        planet,
                        target,
                        avail,
                        min(avail, max(need, int(target.ships) + 1)),
                        world,
                        planned_commitments,
                        modes,
                        policy,
                    )
                    if plan is None:
                        continue
                    angle, turns, _, pn, send = plan
                    if send < pn:
                        continue
                    score = target_value(
                        target, turns, "capture", world, modes, policy
                    ) / (send + turns + 1.0)
                    if target.owner not in (-1, world.player):
                        score *= 1.05
                    if best_cap is None or score > best_cap[0]:
                        best_cap = (score, target.id, angle, turns, send)
                if best_cap is not None:
                    _, tid, angle, turns, need = best_cap
                    actual = append_move(planet.id, angle, need)
                    if actual >= 1:
                        planned_commitments[tid].append(
                            (turns, world.player, int(actual))
                        )
                    continue
                safe_allies = [
                    a
                    for a in world.my_planets
                    if a.id != planet.id and a.id not in live_doomed
                ]
                if not safe_allies:
                    continue
                retreat = min(
                    safe_allies,
                    key=lambda a: (fdist.get(a.id, 10**9), planet_distance(planet, a)),
                )
                aim = world.plan_shot(planet.id, retreat.id, avail)
                if aim:
                    append_move(planet.id, aim[0], avail)

    # ── Rear funneling ───────────────────────────────────────
    if (
        (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and allow_optional()
    ):
        live_doomed = compute_live_doomed()
        ft = (
            world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        )
        fdist = {p.id: nearest_distance_to_set(p.x, p.y, ft) for p in world.my_planets}
        safe_fronts = [p for p in world.my_planets if p.id not in live_doomed]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda p: fdist[p.id])
            ratio = (
                REAR_SEND_RATIO_FOUR_PLAYER
                if world.is_four_player
                else REAR_SEND_RATIO_TWO_PLAYER
            )
            if modes["is_finishing"]:
                ratio = max(ratio, REAR_SEND_RATIO_FOUR_PLAYER)
            for rear in sorted(world.my_planets, key=lambda p: -fdist[p.id]):
                if expired() or rear.id == front_anchor.id or rear.id in live_doomed:
                    continue
                if atk_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if fdist[rear.id] < fdist[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue
                sc = [
                    p
                    for p in safe_fronts
                    if p.id != rear.id
                    and fdist[p.id] < fdist[rear.id] * REAR_STAGE_PROGRESS
                ]
                if sc:
                    front = min(sc, key=lambda p: planet_distance(rear, p))
                else:
                    obj = min(ft, key=lambda t: planet_distance(rear, t))
                    rf = [p for p in safe_fronts if p.id != rear.id]
                    if not rf:
                        continue
                    front = min(rf, key=lambda p: planet_distance(p, obj))
                if front.id == rear.id:
                    continue
                send = int(atk_left(rear.id) * ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = world.plan_shot(rear.id, front.id, send)
                if aim and aim[1] <= REAR_MAX_TRAVEL_TURNS:
                    append_move(rear.id, aim[0], send)

    # ── Total-war endgame: dump all ships ────────────────────
    if world.is_total_war and world.enemy_planets and allow_optional():

        def enemy_priority(ep):
            biw = ep.owner in world.blood_in_water_owners
            strength = world.owner_strength.get(ep.owner, 99999)
            return (0 if biw else 1, strength)

        targets_sorted = sorted(world.enemy_planets, key=enemy_priority)
        for src in world.my_planets:
            if expired():
                return finalize()
            al = atk_left(src.id)
            if al < 5:
                continue
            sent = False
            for tgt in targets_sorted:
                aim = world.plan_shot(src.id, tgt.id, al)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if turns >= world.remaining_steps:
                    continue
                append_move(src.id, angle, al)
                sent = True
                break
            if not sent:
                for tgt in sorted(
                    world.enemy_planets, key=lambda e: planet_distance(src, e)
                ):
                    aim = world.plan_shot(src.id, tgt.id, al)
                    if aim and aim[1] < world.remaining_steps:
                        append_move(src.id, aim[0], al)
                        break

    return finalize()


# ============================================================
# Agent Entry Point
# ============================================================

_game_id = None
_agent_step = 0


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _detect_game_id(obs):
    raw_init = _read(obs, "initial_planets", []) or []
    if not raw_init:
        return None
    return hash(tuple(tuple(p[:4]) for p in raw_init[:4]))


def build_world(obs, step):
    player = _read(obs, "player", 0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_by_id = {Planet(*p).id: Planet(*p) for p in raw_init}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def agent(obs, config=None):
    global _game_id, _agent_step

    gid = _detect_game_id(obs)
    if gid is not None and gid != _game_id:
        _game_id = gid
        _agent_step = 0

    step = _agent_step
    _agent_step += 1

    start_time = time.perf_counter()
    world = build_world(obs, step)
    if not world.my_planets:
        return []
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "build_world"]
