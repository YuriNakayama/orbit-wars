"""Playout-based value function for case11 portfolio search.

`evaluate_assignment` takes a script-per-source assignment, materializes the
resulting moves (one per source, possibly None), and projects the world
forward by `horizon` turns via `simulate_planet_timeline` for each planet.

The score is a weighted aggregate of self net ships and self production at
the horizon, minus enemy threat. Same shape as the case8 evaluator's
`legacy_net_ships` mode but without the case8-specific commitment dict.

The simulation here is a single-side projection (we apply our moves and let
the timeline integrate with already-known fleet arrivals). Enemy reaction
is NOT simulated in iter1 (PGS); iter2 (NGS) adds an enemy mini-PGS layer.
"""

from __future__ import annotations

import math

from ..core.physics import fleet_speed
from ..core.types import Planet
from ..core.world_model import WorldModel, simulate_planet_timeline
from .scripts import SCRIPT_BY_NAME


def _moves_from_assignment(
    world: WorldModel, assignment: dict[int, str]
) -> list[list[float]]:
    """Apply each script to its source, producing a list of moves.

    Caps total per-source ship usage to source.ships (no double-spend if
    a script returns more than the source has). Returns moves as
    `[src_id, angle, ships]` lists; None entries are dropped.
    """
    moves: list[list[float]] = []
    used_per_src: dict[int, int] = {}
    for src in world.my_planets:
        script_name = assignment.get(src.id, "idle")
        script = SCRIPT_BY_NAME.get(script_name)
        if script is None:
            continue
        m = script.fn(src, world)
        if m is None:
            continue
        src_id = int(m[0])
        ships = int(m[2])
        already = used_per_src.get(src_id, 0)
        max_send = max(0, int(src.ships) - already)
        send = min(ships, max_send)
        if send < 1:
            continue
        moves.append([float(src_id), float(m[1]), float(send)])
        used_per_src[src_id] = already + send
    return moves


def _arrivals_from_moves(
    world: WorldModel, moves: list[list[float]]
) -> dict[int, list[tuple[int, int, int]]]:
    """Convert assignment moves to predicted arrivals per target planet.

    Uses analytic angle/speed to decide which planet each move hits and at
    which turn. Targets that don't intersect any planet (a missed shot) are
    dropped — this is fine for evaluation, since such moves are wasted.
    """
    arrivals: dict[int, list[tuple[int, int, int]]] = {}
    for move in moves:
        src_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
        if ships <= 0:
            continue
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        target = _infer_target(src, angle, world.planets, ships)
        if target is None:
            continue
        dx = target.x - src.x
        dy = target.y - src.y
        speed = fleet_speed(ships)
        eta = max(1, int(math.ceil(math.hypot(dx, dy) / max(0.01, speed))))
        arrivals.setdefault(target.id, []).append((eta, world.player, ships))
    return arrivals


def _infer_target(
    src: Planet, angle: float, planets: list[Planet], ships: int
) -> Planet | None:
    """Find the closest planet that the angle ray actually intersects."""
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    best: Planet | None = None
    best_t = 1e9
    for p in planets:
        if p.id == src.id:
            continue
        dx = p.x - src.x
        dy = p.y - src.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r_sq = p.radius * p.radius
        if perp_sq >= r_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
        if hit_d < best_t:
            best_t = hit_d
            best = p
    return best


def evaluate_assignment(
    world: WorldModel,
    assignment: dict[int, str],
    horizon: int,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.3),
) -> float:
    """Score `assignment` by projecting the world forward `horizon` turns.

    Higher is better for `world.player`. The current implementation looks
    only at horizon-end state (ownership and ships); a per-turn integral
    refinement is possible in iter2/iter3.
    """
    player = world.player
    w_ships, w_prod, w_threat = weights

    moves = _moves_from_assignment(world, assignment)
    extra_arrivals = _arrivals_from_moves(world, moves)

    # iter1 v2 fix: don't subtract sent ships from src — they re-appear at
    # the target planet via `extra_arrivals`. Subtracting twice would
    # double-penalize captures. The "smaller is better" v1 bug was actually
    # caused by `_arrivals_from_moves` not propagating eta>horizon arrivals,
    # which v1 fixed by raising horizon to 20.

    net_ships = 0.0
    player_prod = 0.0
    enemy_threat = 0.0
    for planet in world.planets:
        base = list(world.predicted_arrivals.get(planet.id, []))
        extra = extra_arrivals.get(planet.id, [])
        arrivals = base + list(extra)

        timeline = simulate_planet_timeline(planet, arrivals, player, horizon)
        final_turn = timeline["horizon"]
        owner = timeline["owner_at"].get(final_turn, planet.owner)
        ships = float(timeline["ships_at"].get(final_turn, float(planet.ships)))

        if owner == player:
            net_ships += ships
            player_prod += float(planet.production)
        elif owner != -1:
            net_ships -= ships
            enemy_threat += ships + float(planet.production) * horizon

    return w_ships * net_ships + w_prod * player_prod - w_threat * enemy_threat
