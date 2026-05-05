"""1-ply opponent reaction prediction (true2p_light).

Given a candidate ordering's MovesPlan, run `plan_moves_light` from the
strongest enemy's viewpoint with our committed fleet artificially deducted
from the source planet's garrison, and return the resulting predicted enemy
arrivals. This lets the beam evaluator account for the enemy's expected
counter on the same turn.

Adapted from `case3/baseline/rollout.py` (cross-case import is forbidden by
`.claude/rules/bot/pipeline.md`, so the helpers are duplicated here).
"""

from __future__ import annotations

import math

from ..core.physics import fleet_speed
from ..core.types import Planet
from ..core.world_model import WorldModel

_REACTION_DEPTH: int = 0
_REACTION_MAX_DEPTH: int = 1


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


def predict_enemy_reaction(
    world: WorldModel,
    our_send_src: Planet | None,
    our_send_ships: int,
) -> dict[int, list[tuple[int, int, int]]]:
    """Strongest-enemy reaction after we commit our mission fleet.

    Returns predicted_arrivals dict {target_id: [(eta, enemy_id, ships)]}.
    Returns {} when recursion guard triggers or no enemy exists.
    """
    from ..strategy import plan_moves_light  # avoid circular import

    global _REACTION_DEPTH
    if _REACTION_DEPTH >= _REACTION_MAX_DEPTH:
        return {}
    enemy_id = _strongest_enemy(world)
    if enemy_id is None:
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

    _REACTION_DEPTH += 1
    try:
        enemy_moves = plan_moves_light(enemy_world)
    finally:
        _REACTION_DEPTH -= 1

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
