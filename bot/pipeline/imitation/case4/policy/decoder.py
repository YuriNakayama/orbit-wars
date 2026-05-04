"""Policy output → Kaggle action list for imitation/case4.

Per source, the model emits CAND_K logits. Argmax picks slot id:

  - slot 0 → no-op (do not fire from this source)
  - slot 1..K-1 → corresponding candidate planet (resolved via candidate_pid)

Ships count is rule-based: `ships = max(target.ships + 1, 20)` (notebook rule).
The post-process overfire suppression (committed dict) is preserved from case3.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .candidates import CAND_K
from .featurizer import MAX_PLANETS
from .geometry import LAUNCH_CLEARANCE, Planet, aim_with_prediction
from .safety import (
    SafetyPlanet,
    fleet_crosses_other_comet,
    intercept_holds_within_tolerance,
    is_trajectory_sun_safe,
    target_reachable_before_comet_expiry,
)
from .types import PolicyOutput, WorldSnapshot


def _safety_planet(p: Planet) -> SafetyPlanet:
    return SafetyPlanet(
        id=p.id,
        owner=p.owner,
        x=p.x,
        y=p.y,
        radius=p.radius,
        ships=p.ships,
        production=p.production,
    )


def _is_action_safe(
    src: Planet,
    target: Planet,
    ships: int,
    angle: float,
    turns: int,
    intercept_pos: tuple[float, float],
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
    step: int,
) -> bool:
    clearance = src.radius + LAUNCH_CLEARANCE
    launch_x = src.x + math.cos(angle) * clearance
    launch_y = src.y + math.sin(angle) * clearance
    if not is_trajectory_sun_safe(launch_x, launch_y, angle, turns, ships):
        return False
    safety_initial = {pid: _safety_planet(p) for pid, p in initial_by_id.items()}
    if not intercept_holds_within_tolerance(
        target=_safety_planet(target),
        predicted_turns=turns,
        predicted_pos=intercept_pos,
        initial_by_id=safety_initial,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    ):
        return False
    if not target_reachable_before_comet_expiry(target.id, turns, comets):
        return False
    if fleet_crosses_other_comet(
        launch_x=launch_x,
        launch_y=launch_y,
        angle=angle,
        turns=turns,
        ships=ships,
        current_step=step,
        comets=comets,
        exclude_planet_id=target.id,
    ):
        return False
    return True


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

        ships = _fixed_ship_count(target.ships)
        if ships > src.ships:
            continue  # candidate_mask should have caught this; defensive
        if ships <= 0:
            continue

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
        angle, turns, ix, iy = aim
        step = int(obs.get("step") or 0)
        if not _is_action_safe(
            src=src,
            target=target,
            ships=ships,
            angle=angle,
            turns=turns,
            intercept_pos=(ix, iy),
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
            step=step,
        ):
            continue
        committed[target_pid] = committed.get(target_pid, 0) + ships
        actions.append([src_pid, float(angle), int(ships)])
    return actions


__all__ = ["decode", "MAX_PLANETS"]
