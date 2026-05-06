"""Policy output → Kaggle action list for imitation/case8.

Per source, the model emits CAND_K logits + 1 ship-count scalar. Argmax picks
the candidate slot; the ship-count head determines how many ships to send.

  - slot 0 → no-op (do not fire from this source)
  - slot 1..K-1 → corresponding candidate planet (resolved via candidate_pid)

Ships count = clamp(round(ship_pred), [rule_floor, src.ships]) where
rule_floor = max(target.ships + 1, 20). The rule serves as a lower-bound
sanity guard so the learned head cannot under-fire below the notebook minimum.
The post-process overfire suppression (committed dict) is preserved from case4.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .candidates import CAND_K
from .featurizer import MAX_PLANETS
from .geometry import Planet, aim_with_prediction
from .types import PolicyOutput, WorldSnapshot


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


def _fixed_ship_count(target_ships: int) -> int:
    """Notebook rule: ships = max(target.ships + 1, 20)."""
    return max(target_ships + 1, 20)


def decode(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    candidate_pid: torch.Tensor,  # (MAX_PLANETS, CAND_K) int64, batch squeezed
    candidate_mask: torch.Tensor,  # (MAX_PLANETS, CAND_K) bool
    temperature: float = 1.0,
) -> list[list[int | float]]:
    """Greedy decode: per-source argmax over candidate slots.

    `temperature` is reserved for future top-k sampling. With argmax and T > 0
    the answer is unchanged, so temperature is currently a no-op knob.
    """
    T = max(float(temperature), 1e-6)
    cand_logits = output.candidate_logits[0] / T  # (P, K)
    slot_argmax = cand_logits.argmax(dim=-1)  # (P,)
    ship_pred = output.ship_pred[0]  # (P,)

    raw_planets = list(obs.get("planets", []) or [])
    pid_to_planet = {int(row[0]): _build_planet(row) for row in raw_planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])
    player = snapshot.player

    # Pre-compute incoming friendly ships per target so we can suppress overfire.
    incoming_friendly: dict[int, int] = {}
    for fleet in obs.get("fleets") or []:
        f_owner = int(fleet[1])
        if f_owner != player:
            continue
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

    committed: dict[int, int] = dict(incoming_friendly)
    actions: list[list[int | float]] = []

    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        cand_slot = int(slot_argmax[slot].item())
        if cand_slot == 0:
            continue
        if cand_slot < 0 or cand_slot >= CAND_K:
            continue
        if not bool(candidate_mask[slot, cand_slot].item()):
            continue
        target_pid = int(candidate_pid[slot, cand_slot].item())
        if target_pid < 0:
            continue
        src = pid_to_planet.get(src_pid)
        target = pid_to_planet.get(target_pid)
        if src is None or target is None or src.id == target.id:
            continue

        rule_floor = _fixed_ship_count(target.ships)
        pred_ships = int(round(float(ship_pred[slot].item())))
        ships = max(rule_floor, pred_ships)
        if ships > src.ships:
            ships = src.ships  # cap by available
        if ships <= 0:
            continue
        if rule_floor > src.ships:
            continue  # candidate_mask should have caught this; defensive

        # Overfire suppression (案4) — copied from case3 decoder.
        if target.owner != player:
            need = max(0, target.ships + 1 - committed.get(target_pid, 0))
            if need <= 0:
                continue
            if ships > need * 2:
                ships = max(need, 1)
        else:
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


__all__ = ["decode", "MAX_PLANETS"]
