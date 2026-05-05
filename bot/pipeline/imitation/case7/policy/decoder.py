"""Policy output → Kaggle action list.

The model's target head emits NUM_TEMPLATES logits per source. The decoder:

1. picks the argmax template,
2. resolves it to a concrete target planet via `templates.resolve_template`,
3. computes ships from the bucket head,
4. applies a post-process filter (案4): if multiple sources are about to fire on
   the same target this turn, keep only the one with the smallest required ships
   to capture (avoids the "send everyone at the same neutral" failure mode);
   also drop any fire whose ships exceed the target's current garrison + 50%
   incoming so we don't pile on a target we've already taken.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .featurizer import MAX_PLANETS
from .geometry import Planet, aim_with_prediction
from .templates import T_NO_OP, resolve_template
from .types import PolicyOutput, WorldSnapshot

NO_OP_LABEL = MAX_PLANETS

# bucket index → fraction of source ships to send (4 buckets, midpoints)
SHIPS_BUCKET_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def _build_planet(row: list[Any]) -> Planet:
    return Planet(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def decode(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    from_threshold: float = 0.5,
    min_fire_topk: int = 1,
    max_fire_count: int | None = None,
    target_temperature: float = 1.0,
    ships_temperature: float = 1.0,
) -> list[list[int | float]]:
    """Greedy decode: per-source from-prob gate → template → ships.

    `min_fire_topk` is a fallback: when no source clears `from_threshold`, still
    emit the top-K candidates so the agent does not sit idle when from_prob is
    uniformly low. Default 1 (Phase 1-3 tuning — was 2, which forced two fires
    every turn and contributed to overfire).

    `max_fire_count` caps the number of sources that are allowed to fire in one
    turn; `None` (default) means no cap. Used in Phase 1 to prevent the
    agent from firing from every own planet when the threshold is lowered.

    `target_temperature` / `ships_temperature` divide the raw logits before
    argmax. Temperature scaling is a monotone operation on a single logit
    vector, so `argmax` is actually unchanged for T > 0 — but if a future
    iteration replaces argmax with top-k sampling or masking, these knobs
    become meaningful. Accepting them here keeps the decoder API stable.
    """
    target_T = max(float(target_temperature), 1e-6)
    ships_T = max(float(ships_temperature), 1e-6)
    from_prob = torch.sigmoid(output.from_logits[0])  # (P,)
    target_argmax = (output.target_logits[0] / target_T).argmax(dim=-1)
    ships_argmax = (output.ships_logits[0] / ships_T).argmax(dim=-1)

    raw_planets = list(obs.get("planets", []) or [])
    pid_to_planet = {int(row[0]): _build_planet(row) for row in raw_planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])
    player = snapshot.player

    raw_by_id = {int(row[0]): row for row in raw_planets}

    # Pre-compute incoming friendly ships per target so we can suppress overfire.
    incoming_friendly: dict[int, int] = {}
    for fleet in obs.get("fleets") or []:
        f_owner = int(fleet[1])
        if f_owner != player:
            continue
        # Approximate the target as the planet whose centre lies along the
        # fleet's heading and is closest forward. Skip if no clear hit.
        fx, fy, fa = float(fleet[2]), float(fleet[3]), float(fleet[4])
        f_ships = int(fleet[6])
        dirx, diry = math.cos(fa), math.sin(fa)
        best_pid: int | None = None
        best_proj = float("inf")
        for p in pid_to_planet.values():
            dx, dy = p.x - fx, p.y - fy
            proj = dx * dirx + dy * diry
            if proj < 0:
                continue
            perp = abs(dx * diry - dy * dirx)
            if perp > p.radius:
                continue
            if proj < best_proj:
                best_proj = proj
                best_pid = p.id
        if best_pid is not None:
            incoming_friendly[best_pid] = incoming_friendly.get(best_pid, 0) + f_ships

    # Track per-target commitment so multiple sources don't pile on one target.
    committed: dict[int, int] = dict(incoming_friendly)
    actions: list[list[int | float]] = []

    # Rank all my_planets by from-prob; keep those above threshold AND the top-K
    # regardless of threshold (guarantees the agent always has fire candidates).
    ranked: list[tuple[float, int]] = []
    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        prob = float(from_prob[slot].item())
        ranked.append((prob, src_pid))
    ranked.sort(key=lambda x: -x[0])
    above = sum(1 for p, _ in ranked if p >= from_threshold)
    keep_n = above if above > 0 else min_fire_topk
    if max_fire_count is not None:
        keep_n = min(keep_n, int(max_fire_count))
    src_with_prob = ranked[:keep_n]

    for _, src_pid in src_with_prob:
        slot = snapshot.planet_ids.index(src_pid)
        template_id = int(target_argmax[slot].item())
        if template_id == T_NO_OP:
            continue
        src = pid_to_planet.get(src_pid)
        if src is None:
            continue
        src_row = raw_by_id.get(src_pid)
        if src_row is None:
            continue
        target_pid = resolve_template(template_id, src_row, raw_planets, player)
        if target_pid is None or target_pid == src_pid:
            continue
        target = pid_to_planet.get(target_pid)
        if target is None:
            continue

        bucket = int(ships_argmax[slot].item())
        bucket = max(0, min(bucket, len(SHIPS_BUCKET_FRACTIONS) - 1))
        ships = int(math.floor(SHIPS_BUCKET_FRACTIONS[bucket] * src.ships))
        if ships <= 0 and src.ships > 0:
            ships = 1
        if ships <= 0:
            continue
        if ships > src.ships:
            ships = src.ships

        # 案4: suppress overfire — if we already have enough ships en-route to
        # capture this target, skip. Threshold = target garrison + 1 buffer for
        # enemy / neutral; for own planet just cap committed ≤ ships*4.
        if target.owner != player:
            need = max(0, target.ships + 1 - committed.get(target_pid, 0))
            if need <= 0:
                continue
            if ships > need * 2:
                ships = max(need, 1)
        else:
            # reinforcement: avoid sending more than 2x existing garrison
            if committed.get(target_pid, 0) > target.ships * 2:
                continue

        aim = aim_with_prediction(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
        )
        if aim is None:
            continue
        committed[target_pid] = committed.get(target_pid, 0) + ships
        actions.append([src_pid, float(aim[0]), int(ships)])
    return actions
