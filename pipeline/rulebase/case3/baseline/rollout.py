"""Shallow rollout: reorder top-K missions by single-adoption timeline value.

Modes (ROLLOUT_MODE):
- "bonus": add normalized rollout value to existing score.
- "replace": overwrite mission.score with rollout value (old score as tie-break).
- "true2p": evaluate each top-K mission assuming the strongest enemy takes one
  ply of ``plan_moves`` in reply. Enemy launches are folded into our arrivals
  before simulating the target's timeline, so we score under reaction-aware
  conditions instead of a static arrival set.
"""

from __future__ import annotations

import math

from .core.config import (
    ROLLOUT_DEPTH,
    ROLLOUT_FILTER_DEPTH,
    ROLLOUT_FILTER_MIN_MARGIN,
    ROLLOUT_FILTER_TOP_K,
    ROLLOUT_MODE,
    ROLLOUT_SCORE_BONUS,
    ROLLOUT_TOP_K,
)
from .core.physics import fleet_speed
from .core.types import Mission, Planet
from .core.world_model import WorldModel, simulate_planet_timeline

_ROLLOUT_DEPTH: int = 0
_ROLLOUT_MAX_DEPTH: int = 1


def _baseline_net_ships(world: WorldModel, horizon: int) -> float:
    total = 0.0
    for planet in world.planets:
        timeline = simulate_planet_timeline(
            planet,
            world.predicted_arrivals.get(planet.id, []),
            world.player,
            horizon,
        )
        final_turn = timeline["horizon"]
        owner = timeline["owner_at"].get(final_turn, planet.owner)
        ships = timeline["ships_at"].get(final_turn, float(planet.ships))
        if owner == world.player:
            total += ships
        elif owner != -1:
            total -= ships
    return total


def _mission_value(world: WorldModel, mission: Mission, horizon: int) -> float:
    if not mission.options:
        return 0.0
    option = mission.options[0]
    target = world.planet_by_id.get(mission.target_id)
    if target is None:
        return 0.0

    arrivals = list(world.predicted_arrivals.get(mission.target_id, []))
    arrivals.append((int(option.turns), world.player, int(option.needed)))

    timeline = simulate_planet_timeline(target, arrivals, world.player, horizon)
    final_turn = timeline["horizon"]
    owner = timeline["owner_at"].get(final_turn, target.owner)
    ships = float(timeline["ships_at"].get(final_turn, float(target.ships)))
    if owner == world.player:
        return ships + float(target.production) * max(0, horizon - int(option.turns))
    return 0.0


def _infer_action_target(
    src: Planet, angle: float, planets: list[Planet], ships: int
) -> Planet | None:
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    speed = fleet_speed(ships)
    best: Planet | None = None
    best_turns = 1e9
    for planet in planets:
        if planet.id == src.id:
            continue
        dx = planet.x - src.x
        dy = planet.y - src.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns < best_turns:
            best_turns = turns
            best = planet
    return best


def _strongest_enemy(world: WorldModel) -> int | None:
    best_owner: int | None = None
    best_strength = -1
    for owner, strength in world.owner_strength.items():
        if owner in (-1, world.player):
            continue
        if strength > best_strength:
            best_strength = strength
            best_owner = owner
    return best_owner


def _enemy_reaction_arrivals(
    world: WorldModel,
    enemy_id: int,
    our_send_src: Planet | None,
    our_send_ships: int,
) -> dict[int, list[tuple[int, int, int]]]:
    """Strongest-enemy reaction after we commit our mission fleet.

    We depress ``our_send_src.ships`` by ``our_send_ships`` in the enemy's
    viewpoint so the enemy sees that garrison as already spent; otherwise the
    enemy assumes we still have those ships and over-defends.
    """
    from .strategy import plan_moves  # avoid circular import

    global _ROLLOUT_DEPTH
    if _ROLLOUT_DEPTH >= _ROLLOUT_MAX_DEPTH:
        return {}

    planets = world.planets
    if our_send_src is not None and our_send_ships > 0:
        remaining = max(0, int(our_send_src.ships) - int(our_send_ships))
        planets = [
            Planet(
                id=p.id,
                owner=p.owner,
                x=p.x,
                y=p.y,
                radius=p.radius,
                ships=remaining if p.id == our_send_src.id else p.ships,
                production=p.production,
            )
            for p in world.planets
        ]

    enemy_world = WorldModel(
        player=enemy_id,
        step=world.step,
        planets=planets,
        fleets=world.fleets,
        initial_by_id=world.initial_by_id,
        ang_vel=world.ang_vel,
        comets=world.comets,
        comet_ids=world.comet_ids,
        predicted_arrivals={},
        opponent_threat_score={},
    )

    _ROLLOUT_DEPTH += 1
    try:
        enemy_moves = plan_moves(enemy_world, light=True)
    finally:
        _ROLLOUT_DEPTH -= 1

    predictions: dict[int, list[tuple[int, int, int]]] = {}
    for move in enemy_moves:
        src_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
        if ships <= 0:
            continue
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        target = _infer_action_target(src, angle, world.planets, ships)
        if target is None:
            continue
        dx = target.x - src.x
        dy = target.y - src.y
        speed = fleet_speed(ships)
        eta = max(1, int(math.ceil(math.sqrt(dx * dx + dy * dy) / speed)))
        predictions.setdefault(target.id, []).append((eta, enemy_id, ships))
    return predictions


def _mission_value_2p(
    world: WorldModel,
    mission: Mission,
    horizon: int,
    enemy_arrivals: dict[int, list[tuple[int, int, int]]],
) -> float:
    """Net ship total over all planets after adopting mission + enemy reaction."""
    if not mission.options:
        return 0.0
    option = mission.options[0]
    our_send = (
        int(option.turns),
        world.player,
        int(option.needed),
    )

    total = 0.0
    for planet in world.planets:
        base = list(world.predicted_arrivals.get(planet.id, []))
        if planet.id == mission.target_id:
            base.append(our_send)
        extra = enemy_arrivals.get(planet.id, [])
        if extra:
            base = base + list(extra)
        timeline = simulate_planet_timeline(planet, base, world.player, horizon)
        final_turn = timeline["horizon"]
        owner = timeline["owner_at"].get(final_turn, planet.owner)
        ships = timeline["ships_at"].get(final_turn, float(planet.ships))
        if owner == world.player:
            total += ships
        elif owner != -1:
            total -= ships
    return total


def _target_delta(world: WorldModel, mission: Mission, horizon: int) -> float:
    if not mission.options:
        return 0.0
    option = mission.options[0]
    target = world.planet_by_id.get(mission.target_id)
    if target is None:
        return 0.0

    base_arrivals = list(world.predicted_arrivals.get(mission.target_id, []))
    with_send = base_arrivals + [(int(option.turns), world.player, int(option.needed))]

    def evaluate(arrivals: list[tuple[int, int, int]]) -> float:
        timeline = simulate_planet_timeline(target, arrivals, world.player, horizon)
        final_turn = timeline["horizon"]
        owner = timeline["owner_at"].get(final_turn, target.owner)
        ships = timeline["ships_at"].get(final_turn, float(target.ships))
        if owner == world.player:
            return float(ships)
        if owner == -1:
            return 0.0
        return -float(ships)

    cost = float(option.needed) if mission.kind != "reinforce" else 0.0
    return evaluate(with_send) - evaluate(base_arrivals) - cost * 0.5


def rollout_filter(missions: list[Mission], world: WorldModel) -> list[Mission]:
    if not missions:
        return missions
    top_k = min(ROLLOUT_FILTER_TOP_K, len(missions))
    if top_k < 1:
        return missions

    horizon = max(1, int(ROLLOUT_FILTER_DEPTH))
    min_margin = float(ROLLOUT_FILTER_MIN_MARGIN)

    candidates = missions[:top_k]
    rest = missions[top_k:]

    kept: list[Mission] = []
    for mission in candidates:
        if mission.kind in ("reinforce",):
            kept.append(mission)
            continue
        delta = _target_delta(world, mission, horizon)
        if delta >= min_margin:
            kept.append(mission)
    return kept + rest


def rollout_reorder(missions: list[Mission], world: WorldModel) -> list[Mission]:
    if not missions:
        return missions
    top_k = min(ROLLOUT_TOP_K, len(missions))
    if top_k < 2:
        return missions

    horizon = max(1, int(ROLLOUT_DEPTH))
    candidates = missions[:top_k]
    rest = missions[top_k:]

    mode = ROLLOUT_MODE.lower()

    if mode == "true2p":
        enemy_id = _strongest_enemy(world)
        if enemy_id is None:
            return missions
        scored: list[tuple[float, float, int, Mission]] = []
        for idx, mission in enumerate(candidates):
            option = mission.options[0] if mission.options else None
            if option is None:
                scored.append(
                    (-_baseline_net_ships(world, horizon), -mission.score, idx, mission)
                )
                continue
            src = world.planet_by_id.get(option.src_id)
            enemy_arrivals = _enemy_reaction_arrivals(
                world, enemy_id, src, int(option.needed)
            )
            value = _mission_value_2p(world, mission, horizon, enemy_arrivals)
            scored.append((-value, -mission.score, idx, mission))
        scored.sort()
        reordered = [item[3] for item in scored]
        return reordered + rest

    if mode == "replace":
        placed: list[tuple[float, float, int, Mission]] = []
        for idx, mission in enumerate(candidates):
            value = _mission_value(world, mission, horizon)
            placed.append((-value, -mission.score, idx, mission))
        placed.sort()
        reordered = [item[3] for item in placed]
        return reordered + rest

    baseline = _baseline_net_ships(world, horizon)
    max_abs = max(1.0, abs(baseline))
    adjusted: list[tuple[float, int, Mission]] = []
    for idx, mission in enumerate(candidates):
        value = _mission_value(world, mission, horizon)
        normalized = value / max_abs
        score_adj = mission.score + ROLLOUT_SCORE_BONUS * normalized
        adjusted.append((-score_adj, idx, mission))
    adjusted.sort()
    reordered = [item[2] for item in adjusted]
    return reordered + rest
