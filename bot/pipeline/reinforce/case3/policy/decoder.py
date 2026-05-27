"""SampledAction → Kaggle action list for reinforce/case3.

Trimmed from case9's multi-variant decoder to keep only the per_planet path
plus the case5-derived safety filter. The sampled (target_slot, log1p_ships)
tuple per source is resolved to a (src_pid, angle, ships) action via
`aim_with_prediction`. NO_OP is the (P+1)-th sentinel slot.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from .featurizer import MAX_PLANETS
from .geometry import LAUNCH_CLEARANCE, Planet, aim_with_prediction
from .safety import (
    SafetyPlanet,
    fleet_crosses_other_comet,
    intercept_holds_within_tolerance,
    is_trajectory_sun_safe,
    target_reachable_before_comet_expiry,
)
from .types import WorldSnapshot

NO_OP_INDEX = MAX_PLANETS  # last slot in per_planet_logits


@dataclass(frozen=True)
class SampledAction:
    """Per-source action tuple for one env step (batch_size=1).

    `target_slot[s]` ∈ [0, P] where == P means no-op for source s. `log1p_ships[s]`
    is the Gaussian-sampled log1p(ships) value used to derive the launch size.
    """

    target_slot: torch.Tensor  # (P,) long
    log1p_ships: torch.Tensor  # (P,) float
    log_prob: torch.Tensor  # scalar
    entropy: torch.Tensor  # scalar


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def _fixed_ship_count() -> int:
    return max(1, _env_int("RL_CASE1_MIN_SHIP_FLOOR", 5))


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


def _common_obs_state(
    obs: dict[str, Any], snapshot: WorldSnapshot
) -> tuple[
    dict[int, Planet],
    dict[int, Planet],
    float,
    list[dict[str, Any]],
    set[int],
    int,
    dict[int, int],
]:
    raw_planets = list(obs.get("planets", []) or [])
    pid_to_planet = {int(row[0]): _build_planet(row) for row in raw_planets}
    initial_planets = [_build_planet(row) for row in (obs.get("initial_planets") or [])]
    initial_by_id = {p.id: p for p in initial_planets}
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    comets = list(obs.get("comets") or [])
    comet_ids = set(obs.get("comet_planet_ids") or [])
    step = int(obs.get("step") or 0)
    player = snapshot.player

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
    return (
        pid_to_planet,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        step,
        incoming_friendly,
    )


def _emit_action(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
    step: int,
    committed: dict[int, int],
    player: int,
) -> list[int | float] | None:
    if ships <= 0 or src.id == target.id:
        return None
    if ships > src.ships:
        ships = src.ships
    if ships <= 0:
        return None
    if target.owner != player:
        need = max(0, target.ships + 1 - committed.get(target.id, 0))
        if need <= 0:
            return None
        if ships < need:
            ships = need
        if ships > src.ships:
            ships = src.ships
    elif committed.get(target.id, 0) > target.ships * 2:
        return None
    aim = aim_with_prediction(
        src, target, ships, initial_by_id, ang_vel, comets, comet_ids
    )
    if aim is None:
        return None
    angle, turns, ix, iy = aim
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
        return None
    committed[target.id] = committed.get(target.id, 0) + ships
    return [src.id, float(angle), int(ships)]


def decode(
    action: SampledAction,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
) -> list[list[int | float]]:
    """Resolve sampled (target_slot, log1p_ships) per source to action list."""
    (
        pid_to_planet,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        step,
        incoming_friendly,
    ) = _common_obs_state(obs, snapshot)
    committed = dict(incoming_friendly)
    actions: list[list[int | float]] = []
    player = snapshot.player

    target_slots = action.target_slot.tolist()
    log1p_ships = action.log1p_ships.tolist()

    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        src = pid_to_planet.get(src_pid)
        if src is None:
            continue
        tgt_slot = int(target_slots[slot])
        if tgt_slot == NO_OP_INDEX or tgt_slot < 0:
            continue
        if tgt_slot >= len(snapshot.planet_ids):
            continue
        target_pid = int(snapshot.planet_ids[tgt_slot])
        target = pid_to_planet.get(target_pid)
        if target is None or target.id == src.id:
            continue
        rule_floor = _fixed_ship_count()
        log_ship = max(0.0, float(log1p_ships[slot]))
        pred_ships = int(round(math.expm1(log_ship)))
        ships = max(rule_floor, max(1, pred_ships))
        if ships > src.ships:
            ships = src.ships
        emitted = _emit_action(
            src,
            target,
            ships,
            initial_by_id,
            ang_vel,
            comets,
            comet_ids,
            step,
            committed,
            player,
        )
        if emitted is not None:
            actions.append(emitted)
    return actions


__all__ = ["SampledAction", "decode", "NO_OP_INDEX"]
