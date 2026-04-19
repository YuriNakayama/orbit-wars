"""Target-template action space for imitation/case1.

Replaces planet-id classification (37 classes incl. no-op) with a small set of
strategically meaningful templates. Each template, given a source planet and a
world snapshot, deterministically resolves to a single target planet (or None
if no valid target exists in this state).

Template ids:
  0  NEAREST_NEUTRAL_LOW_GARRISON  — nearest neutral with garrison ≤ src.ships
  1  NEAREST_ENEMY                  — closest enemy planet
  2  HIGHEST_PRODUCTION_NEUTRAL     — neutral with max production reachable
  3  HIGHEST_PRODUCTION_ENEMY       — enemy with max production
  4  REINFORCE_FRONTLINE            — own planet closest to enemy mean
  5  REINFORCE_WEAKEST              — own planet (other than src) with min ships
  6  WEAKEST_ENEMY                  — enemy with fewest ships
  7  NO_OP                          — don't fire from this src

The order matters only for tiebreaking (we always pick the highest-scoring
template via argmax of the model's logits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NUM_TEMPLATES = 8
T_NEAREST_NEUTRAL_LOW = 0
T_NEAREST_ENEMY = 1
T_HIGH_PROD_NEUTRAL = 2
T_HIGH_PROD_ENEMY = 3
T_REINFORCE_FRONTLINE = 4
T_REINFORCE_WEAKEST = 5
T_WEAKEST_ENEMY = 6
T_NO_OP = 7


@dataclass(frozen=True)
class _P:
    id: int
    owner: int
    x: float
    y: float
    ships: int
    production: int


def _to_p(row: list[float]) -> _P:
    return _P(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def _dist(a: _P, b: _P) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def resolve_template(
    template_id: int,
    src_row: list[float],
    planet_rows: list[list[float]],
    player: int,
) -> int | None:
    """Resolve a template id to a target planet id given the current obs.

    Returns None if no valid candidate exists or template_id == NO_OP.
    """
    if template_id == T_NO_OP:
        return None
    src = _to_p(src_row)
    planets = [_to_p(r) for r in planet_rows]
    others = [p for p in planets if p.id != src.id]
    if not others:
        return None

    enemy = [p for p in others if p.owner != player and p.owner != -1]
    neutral = [p for p in others if p.owner == -1]
    mine_others = [p for p in others if p.owner == player]

    def nearest(cands: list[_P]) -> _P | None:
        if not cands:
            return None
        return min(cands, key=lambda p: _dist(src, p))

    def highest_prod(cands: list[_P]) -> _P | None:
        if not cands:
            return None
        return max(cands, key=lambda p: (p.production, -_dist(src, p)))

    if template_id == T_NEAREST_NEUTRAL_LOW:
        cands = [p for p in neutral if p.ships <= max(1, src.ships)]
        chosen = nearest(cands) or nearest(neutral)
    elif template_id == T_NEAREST_ENEMY:
        chosen = nearest(enemy)
    elif template_id == T_HIGH_PROD_NEUTRAL:
        chosen = highest_prod(neutral)
    elif template_id == T_HIGH_PROD_ENEMY:
        chosen = highest_prod(enemy)
    elif template_id == T_REINFORCE_FRONTLINE:
        if not mine_others or not enemy:
            chosen = nearest(mine_others)
        else:
            ex = sum(p.x for p in enemy) / len(enemy)
            ey = sum(p.y for p in enemy) / len(enemy)
            chosen = min(
                mine_others,
                key=lambda p: math.sqrt((p.x - ex) ** 2 + (p.y - ey) ** 2),
            )
    elif template_id == T_REINFORCE_WEAKEST:
        chosen = min(mine_others, key=lambda p: p.ships) if mine_others else None
    elif template_id == T_WEAKEST_ENEMY:
        chosen = min(enemy, key=lambda p: p.ships) if enemy else None
    else:
        return None

    return chosen.id if chosen is not None else None


def classify_actual_target(
    src_row: list[float],
    target_row: list[float],
    planet_rows: list[list[float]],
    player: int,
) -> int:
    """Pick the template whose resolution matches the actual target id.

    For fired sources we never want NO_OP — if no template resolves exactly to
    the actual target, fall back to the template whose target is *closest* to
    the actual target (preferring same-owner-class templates for tie-break).
    """
    actual_id = int(target_row[0])
    target = _to_p(target_row)
    # 1) exact match in priority order
    resolved_per_tid: dict[int, int | None] = {}
    for tid in range(NUM_TEMPLATES - 1):  # exclude NO_OP
        resolved = resolve_template(tid, src_row, planet_rows, player)
        resolved_per_tid[tid] = resolved
        if resolved == actual_id:
            return tid

    # 2) fallback: pick the template whose resolved target is closest to actual.
    planets_by_id = {int(r[0]): _to_p(r) for r in planet_rows}
    best_tid = T_NEAREST_ENEMY  # safe default if even fallback fails
    best_dist = float("inf")
    for tid, rid in resolved_per_tid.items():
        if rid is None:
            continue
        rp = planets_by_id.get(rid)
        if rp is None:
            continue
        d = math.sqrt((rp.x - target.x) ** 2 + (rp.y - target.y) ** 2)
        if d < best_dist:
            best_dist = d
            best_tid = tid
    return best_tid
