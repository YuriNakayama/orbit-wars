"""Plan evaluator: score a candidate ordering's resulting MovesPlan.

iter2: default to `mission_score` mode — sum the `mission.score` of every
mission whose commit emitted at least one move. This makes the beam search
reduce to "find an ordering whose accepted-missions have the highest combined
heuristic score", with the score-desc seed ordering serving as a guaranteed
floor (greedy ordering = same score, beam can only improve).

The legacy `score_commitments_legacy` (horizon-end self net_ships +
production - enemy_threat) is kept for ablation.
"""

from __future__ import annotations

from ..core.types import Mission
from ..core.world_model import WorldModel, simulate_planet_timeline


def score_commitments(
    world: WorldModel,  # noqa: ARG001 — kept for legacy signature parity
    planned_commitments: dict[int, list[tuple[int, int, int]]],  # noqa: ARG001
    accepted_missions: list[Mission],
    horizon: int,  # noqa: ARG001
    weights: tuple[float, float, float],  # noqa: ARG001
) -> float:
    """Return sum of `mission.score` for all missions whose commit emitted
    at least one move. Greedy ordering (score-desc) maximizes this trivially
    when all missions can be accepted; beam can only do better when budget
    contention forces a choice between high-score missions."""
    return sum(m.score for m in accepted_missions)


def score_commitments_legacy(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    horizon: int,
    weights: tuple[float, float, float],
) -> float:
    """iter1 evaluator: horizon-end linear combination of self net_ships,
    self production, and enemy threat. Empirically inferior (vs v4 = 32.3%)
    so retained only for ablation."""
    player = world.player
    w_ships, w_prod, w_threat = weights

    net_ships = 0.0
    player_prod_total = 0.0
    enemy_threat = 0.0

    for planet in world.planets:
        base = list(world.predicted_arrivals.get(planet.id, []))
        committed = planned_commitments.get(planet.id, [])
        arrivals = base + list(committed)

        timeline = simulate_planet_timeline(planet, arrivals, player, horizon)
        final_turn = timeline["horizon"]
        owner = timeline["owner_at"].get(final_turn, planet.owner)
        ships = float(timeline["ships_at"].get(final_turn, float(planet.ships)))

        if owner == player:
            net_ships += ships
            player_prod_total += float(planet.production)
        elif owner != -1:
            net_ships -= ships
            enemy_threat += ships + float(planet.production) * horizon

    return w_ships * net_ships + w_prod * player_prod_total - w_threat * enemy_threat
