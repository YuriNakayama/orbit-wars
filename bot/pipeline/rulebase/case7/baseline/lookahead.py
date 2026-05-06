"""2-turn lookahead: predict strongest enemy's response via self-play.

`predict_enemy_fleets` builds an enemy-viewpoint ``WorldModel`` from the same
observation and runs one pass of ``plan_moves`` to obtain the enemy's most
likely launches. Those launches are converted to predicted arrivals on our
planets so ``WorldModel.base_timeline`` and ``_compute_defense_buffers`` can
react to them in the next real ``plan_moves`` call.

The lookahead is guarded by ``LOOKAHEAD_ENABLED`` in config. A depth flag
prevents infinite recursion when the enemy's ``plan_moves`` would also try to
look ahead.
"""

from __future__ import annotations

import math

from .core.config import (
    LOOKAHEAD_APPLY_AFTER_STEP,
    LOOKAHEAD_MAX_DEPTH,
    LOOKAHEAD_PREDICTION_WEIGHT,
)
from .core.physics import fleet_speed
from .core.types import Fleet, Planet
from .core.world_model import WorldModel

_LOOKAHEAD_DEPTH: int = 0


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


def predict_enemy_fleets(
    world: WorldModel,
) -> dict[int, list[tuple[int, int, int]]]:
    """Run plan_moves from the strongest enemy's viewpoint, one ply deep.

    Returns predicted arrivals keyed by the target planet id:
    {planet_id: [(eta, enemy_owner, ships), ...]}. Values are weighted by
    ``LOOKAHEAD_PREDICTION_WEIGHT`` to reflect uncertainty.
    """
    global _LOOKAHEAD_DEPTH
    if _LOOKAHEAD_DEPTH >= LOOKAHEAD_MAX_DEPTH:
        return {}
    if world.step < LOOKAHEAD_APPLY_AFTER_STEP:
        return {}

    enemy_id = _strongest_enemy(world)
    if enemy_id is None:
        return {}

    from .strategy import plan_moves  # avoid circular import

    enemy_world = WorldModel(
        player=enemy_id,
        step=world.step,
        planets=world.planets,
        fleets=world.fleets,
        initial_by_id=world.initial_by_id,
        ang_vel=world.ang_vel,
        comets=world.comets,
        comet_ids=world.comet_ids,
        predicted_arrivals={},
        opponent_threat_score={},
    )

    _LOOKAHEAD_DEPTH += 1
    try:
        enemy_moves, _enemy_burst_holds, _enemy_accumulate_holds = plan_moves(
            enemy_world
        )
    finally:
        _LOOKAHEAD_DEPTH -= 1

    predictions: dict[int, list[tuple[int, int, int]]] = {}
    weight = LOOKAHEAD_PREDICTION_WEIGHT
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
        if target.owner != world.player:
            continue
        dx = target.x - src.x
        dy = target.y - src.y
        speed = fleet_speed(ships)
        eta = max(1, int(math.ceil(math.sqrt(dx * dx + dy * dy) / speed)))
        ships_weighted = int(math.ceil(ships * weight))
        if ships_weighted <= 0:
            continue
        predictions.setdefault(target.id, []).append((eta, enemy_id, ships_weighted))
    return predictions


__all__ = ["predict_enemy_fleets"]


# Silence unused-import warnings for symbols reserved for future extensions.
_ = Fleet
