"""World-building helpers used by WorldModel (arrival ledger, timelines, FFA).

Verbatim port from the LB1224 notebook by Roman Tamrazov (Apache License 2.0).
"""

from __future__ import annotations

import math
from collections import defaultdict

from .config import FFA_LET_FIGHT_MIN_SHIPS, HORIZON
from .physics import dist, fleet_speed


def fleet_target_planet(fleet, planets):
    best_p, best_t = None, 1e9
    dx_f, dy_f = math.cos(fleet.angle), math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dx_f + dy * dy_f
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r2 = planet.radius * planet.radius
        if perp_sq >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp_sq)))
        t = hit_d / speed
        if t <= HORIZON and t < best_t:
            best_t, best_p = t, planet
    if best_p is None:
        return None, None
    return best_p, int(math.ceil(best_t))


def build_arrival_ledger(fleets, planets):
    abt = {p.id: [] for p in planets}
    for f in fleets:
        target, eta = fleet_target_planet(f, planets)
        if target is None:
            continue
        abt[target.id].append((eta, f.owner, int(f.ships)))
    return abt


def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = {}
    for _, o, s in arrivals:
        by_owner[o] = by_owner.get(o, 0) + s
    if not by_owner:
        return owner, max(0.0, garrison)
    sp = sorted(by_owner.items(), key=lambda x: x[1], reverse=True)
    top_o, top_s = sp[0]
    if len(sp) > 1:
        sec_s = sp[1][1]
        if top_s == sec_s:
            top_o, top_s = -1, 0
        else:
            top_s -= sec_s
    if top_s <= 0:
        return owner, max(0.0, garrison)
    if owner == top_o:
        return owner, garrison + top_s
    garrison -= top_s
    if garrison < 0:
        return top_o, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda x: x[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        grp = by_turn.get(turn, [])
        prev_owner = owner
        if grp:
            if prev_owner == player and first_enemy is None:
                if any(x[1] not in (-1, player) for x in grp):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, grp)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn
        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep):
            so, sg = planet.owner, float(keep)
            for t in range(1, horizon + 1):
                if so != -1:
                    sg += planet.production
                grp = by_turn.get(t, [])
                if grp:
                    so, sg = resolve_arrival_event(so, sg, grp)
                    if so != player:
                        return False
            return so == player

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


def state_at_timeline(timeline, arrival_turn):
    t = min(max(0, int(math.ceil(arrival_turn))), timeline["horizon"])
    owner = timeline["owner_at"].get(t, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(t, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    owners = {p.owner for p in planets if p.owner != -1} | {f.owner for f in fleets}
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist(px, py, p.x, p.y) for p in planets)


def indirect_features(planet, planets, player):
    friendly = neutral = enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        f = other.production / (d + 12.0)
        if other.owner == player:
            friendly += f
        elif other.owner == -1:
            neutral += f
        else:
            enemy += f
    return friendly, neutral, enemy


def detect_exposed_enemy_planets(fleets, enemy_planets):
    exposed = set()
    for planet in enemy_planets:
        outbound = sum(
            int(f.ships)
            for f in fleets
            if f.owner == planet.owner
            and f.from_planet_id == planet.id
            and f.ships >= 5
        )
        if outbound >= 12 and outbound >= planet.ships * 0.8:
            exposed.add(planet.id)
    return exposed


def detect_enemy_fights_at_neutrals(arrivals_by_planet, player):
    contested = {}
    for pid, arrivals in arrivals_by_planet.items():
        enemy_owners = set()
        enemy_ships = 0
        for _, owner, ships in arrivals:
            if owner not in (-1, player):
                enemy_owners.add(owner)
                enemy_ships += ships
        if len(enemy_owners) >= 2 and enemy_ships >= FFA_LET_FIGHT_MIN_SHIPS:
            contested[pid] = enemy_ships
    return contested
