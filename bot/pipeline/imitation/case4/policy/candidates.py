"""Per-source candidate set + per-candidate features for imitation/case4.

Mirrors the action / observation design from the Kaggle tutorial
`kashiwaba/orbit-wars-reinforcement-learning-tutorial`:

- For each owned source planet, build a fixed-size candidate slot vector of
  length `CAND_K` (slot 0 reserved for no-op).
- Slots 1..CAND_K-1 are filled in order: enemy / neutral / friendly buckets,
  each sorted by distance ascending; remainder is filled by nearest unused.
- Per-candidate feature dim `CAND_FEAT_DIM == 14` is the same as the notebook.

This module is pure and stateless — fed by the featurizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CAND_K: int = 8
CAND_FEAT_DIM: int = 14
BOARD_SIZE: float = 100.0
MAX_SHIPS: float = 400.0
MAX_PRODUCTION: float = 5.0
MAX_PLANETS_NORM: float = 48.0  # for src.ships normalization in candidate_features
SUN_X: float = 50.0
SUN_Y: float = 50.0
SUN_R: float = 10.0
LAUNCH_CLEARANCE: float = 0.1
ROTATION_LIMIT: float = 50.0


@dataclass(frozen=True)
class _P:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


def _to_p(row: list[Any]) -> _P:
    return _P(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def _dist(a: _P, b: _P) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_rotating(p: _P) -> bool:
    r = math.hypot(p.x - SUN_X, p.y - SUN_Y)
    return r + p.radius < ROTATION_LIMIT


def fixed_ship_count(src: _P, tgt: _P) -> int:
    """Notebook rule: ships = max(target.ships + 1, 20)."""
    return max(tgt.ships + 1, 20)


def _shot_crosses_sun(src: _P, angle: float, tgt: _P) -> bool:
    """Replicates notebook `shot_crosses_sun`: launch point → target centre segment
    distance to sun centre < SUN_R."""
    start_x = src.x + math.cos(angle) * (src.radius + LAUNCH_CLEARANCE)
    start_y = src.y + math.sin(angle) * (src.radius + LAUNCH_CLEARANCE)
    return (
        _point_to_segment_distance(SUN_X, SUN_Y, start_x, start_y, tgt.x, tgt.y) < SUN_R
    )


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def build_candidates(src: _P, planets: list[_P], player: int) -> list[_P]:
    """Return up to CAND_K-1 candidates ordered enemy / neutral / friendly,
    each by distance ascending, with fallback by nearest of remainder."""
    others = [p for p in planets if p.id != src.id]
    target_count = CAND_K - 1
    enemy_quota = target_count // 3
    neutral_quota = target_count // 3
    friendly_quota = target_count - enemy_quota - neutral_quota

    enemies = sorted(
        (p for p in others if p.owner not in {-1, player}),
        key=lambda p: (_dist(src, p), p.id),
    )[:enemy_quota]
    neutrals = sorted(
        (p for p in others if p.owner == -1),
        key=lambda p: (_dist(src, p), p.id),
    )[:neutral_quota]
    friendlies = sorted(
        (p for p in others if p.owner == player),
        key=lambda p: (_dist(src, p), p.id),
    )[:friendly_quota]

    chosen = enemies + neutrals + friendlies
    chosen_ids = {p.id for p in chosen}
    if len(chosen) >= target_count:
        return chosen[:target_count]

    fallback = sorted(
        (p for p in others if p.id not in chosen_ids),
        key=lambda p: (_dist(src, p), p.id),
    )
    chosen.extend(fallback[: target_count - len(chosen)])
    return chosen


def build_candidate_block(
    src_row: list[Any],
    planet_rows: list[list[Any]],
    player: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (candidate_feats (CAND_K, CAND_FEAT_DIM), candidate_mask (CAND_K,),
    candidate_pid (CAND_K,)).

    Slot 0 = no-op: feats are zeros except the leading is_valid=1.0; mask=True;
    pid=-1.
    Slot 1..CAND_K-1 = candidate planets (notebook order). Mask reflects
    `ships_needed > 0 AND not crosses_sun AND src.ships >= ships_needed`.
    """
    src = _to_p(src_row)
    planets = [_to_p(r) for r in planet_rows]

    feats = np.zeros((CAND_K, CAND_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(CAND_K, dtype=np.bool_)
    pids = [-1] * CAND_K

    # slot 0 = no-op: only is_valid set, all others 0; mask True
    feats[0, 0] = 1.0
    mask[0] = True
    pids[0] = -1

    candidates = build_candidates(src, planets, player)
    for idx, tgt in enumerate(candidates, start=1):
        if idx >= CAND_K:
            break
        dx = tgt.x - src.x
        dy = tgt.y - src.y
        angle = math.atan2(dy, dx)
        crosses = _shot_crosses_sun(src, angle, tgt)
        ships_needed = fixed_ship_count(src, tgt)
        is_neutral = 1.0 if tgt.owner == -1 else 0.0
        is_mine = 1.0 if tgt.owner == player else 0.0
        is_enemy = 1.0 if (tgt.owner != -1 and tgt.owner != player) else 0.0

        feats[idx, 0] = 1.0  # is_valid (this slot is populated)
        feats[idx, 1] = is_neutral
        feats[idx, 2] = is_mine
        feats[idx, 3] = is_enemy
        feats[idx, 4] = float(tgt.x) / BOARD_SIZE
        feats[idx, 5] = float(tgt.y) / BOARD_SIZE
        feats[idx, 6] = float(dx) / BOARD_SIZE
        feats[idx, 7] = float(dy) / BOARD_SIZE
        feats[idx, 8] = math.hypot(dx, dy) / BOARD_SIZE
        feats[idx, 9] = min(float(tgt.ships), MAX_SHIPS) / MAX_SHIPS
        feats[idx, 10] = float(tgt.production) / MAX_PRODUCTION
        feats[idx, 11] = 1.0 if _is_rotating(tgt) else 0.0
        feats[idx, 12] = 1.0 if crosses else 0.0
        feats[idx, 13] = min(float(src.ships), MAX_SHIPS) / MAX_SHIPS

        valid = (ships_needed > 0) and (not crosses) and (src.ships >= ships_needed)
        mask[idx] = bool(valid)
        pids[idx] = int(tgt.id)

    return feats, mask, pids


__all__ = [
    "CAND_K",
    "CAND_FEAT_DIM",
    "build_candidates",
    "build_candidate_block",
    "fixed_ship_count",
]
