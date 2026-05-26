"""Policy output → Kaggle action list for imitation/case11 (per_planet head).

per_planet_logits: (B, P, P+1) — last index is no-op sentinel.
ship_pred:         (B, P)     — log1p(ships) regression output.

Per source planet: argmax over (P+1) targets. If argmax == no-op, no fire.
Otherwise the argmax index identifies the destination planet slot in
`snapshot.planet_ids`. Ships count is `expm1(ship_pred[slot])`, clamped
to a minimum floor and to src.ships.

Mask consistency with training (Lux3 教訓):
  - sources with ships==0 are skipped (matches effective_source_mask in
    losses.py; without this, decode would try to fire from an empty planet
    and the safety filter would drop it silently, creating train/inference
    distribution skew).
  - no-op targets emit no action and the corresponding ship_pred is ignored
    (matches should_learn_ship in losses.py).

Common safety filters (case5 由来) still wrap the final action emission.
"""

from __future__ import annotations

import math
import os
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
from .types import PolicyOutput, WorldSnapshot


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


def _rank_fire_slots(logits: torch.Tensor, noop_idx: int, margin: float) -> list[int]:
    argmax = int(logits.argmax(dim=-1).item())
    fire_logits = logits.clone()
    fire_logits[noop_idx] = -1e9
    fire_idx = int(fire_logits.argmax(dim=-1).item())
    if argmax == noop_idx and float(fire_logits[fire_idx].item()) < (
        float(logits[noop_idx].item()) + margin
    ):
        return []
    return [
        int(slot)
        for slot in fire_logits.argsort(dim=-1, descending=True).tolist()
        if int(slot) != noop_idx
    ]


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
    return max(1, _env_int("IL_CASE9_MIN_SHIP_FLOOR", 5))


def _scale_ship_count(ships: int) -> int:
    scale = _env_float("IL_CASE9_SHIP_SCALE", 1.0)
    if scale >= 1.0:
        return ships
    return max(1, int(math.ceil(ships * max(scale, 0.01))))


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
        attack_mult = max(1.0, _env_float("IL_CASE9_ATTACK_NEED_MULT", 1.0))
        attack_bonus = max(0, _env_int("IL_CASE9_ATTACK_NEED_BONUS", 0))
        desired = max(need, int(math.ceil(need * attack_mult)) + attack_bonus)
        if ships < desired:
            ships = desired
        overcommit_mult = max(1.0, _env_float("IL_CASE9_ATTACK_OVERCOMMIT_MULT", 2.0))
        overcommit_limit = max(desired, int(math.ceil(desired * overcommit_mult)))
        if ships > overcommit_limit:
            ships = max(desired, 1)
        if ships > src.ships:
            ships = src.ships
    else:
        if committed.get(target.id, 0) > target.ships * 2:
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


def _decode_per_planet(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    temperature: float,
) -> list[list[int | float]]:
    """per_planet head decoder.

    per_planet_logits: (1, P, P+1) — last index is no-op sentinel.
    ship_pred:         (1, P)     — log1p(ships) regression output.

    Per source planet: argmax over (P+1) targets. If argmax == no-op, no fire.
    Otherwise the argmax index identifies the destination planet slot in
    `snapshot.planet_ids`. Ships count is `expm1(ship_pred[slot])`, clamped
    to a minimum floor and to src.ships.
    """
    pp_temp = _env_float("IL_CASE9_PP_TEMPERATURE", float(temperature))
    T = max(pp_temp, 1e-6)
    assert output.per_planet_logits is not None
    assert output.ship_pred is not None
    pp_logits = output.per_planet_logits[0] / T  # (P, P+1)
    ship_pred = output.ship_pred[0]  # (P,) log1p space
    no_op_idx = pp_logits.shape[-1] - 1
    noop_bias = _env_float("IL_CASE9_PP_NOOP_BIAS", 0.0)
    if noop_bias != 0.0:
        pp_logits = pp_logits.clone()
        pp_logits[:, no_op_idx] = pp_logits[:, no_op_idx] - noop_bias
    fire_margin = _env_float("IL_CASE9_PP_FIRE_MARGIN", -999.0)

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

    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        src = pid_to_planet.get(src_pid)
        if src is None:
            continue
        # layer2: skip ships==0 sources (matches effective_source_mask in
        # training). Without this guard the safety filter would silently drop
        # the action and the inference-time fire distribution diverges from
        # what the model learned.
        if int(src.ships) <= 0:
            continue
        for tgt_slot in _rank_fire_slots(pp_logits[slot], no_op_idx, fire_margin):
            if tgt_slot < 0 or tgt_slot >= len(snapshot.planet_ids):
                continue
            target_pid = snapshot.planet_ids[tgt_slot]
            target = pid_to_planet.get(int(target_pid))
            if target is None or target.id == src.id:
                continue
            rule_floor = _fixed_ship_count(target.ships)
            if rule_floor > src.ships:
                continue
            log_pred = float(ship_pred[slot].item())
            pred_ships = int(round(math.expm1(max(0.0, log_pred))))
            ships = _scale_ship_count(max(rule_floor, max(1, pred_ships)))
            action = _emit_action(
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
            if action is not None:
                actions.append(action)
                break
    return actions


def decode(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    candidate_pid: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    head_mode: str = "per_planet",
    template_ctx: torch.Tensor | None = None,
) -> list[list[int | float]]:
    """case11 decode (per_planet head only).

    The extra `candidate_pid` / `candidate_mask` / `template_ctx` / `head_mode`
    parameters are accepted for backward-compatible signature only; they are
    not consumed.
    """
    del candidate_pid, candidate_mask, template_ctx
    if head_mode != "per_planet":
        raise ValueError(
            f"case11 supports only head_mode='per_planet', got {head_mode!r}"
        )
    return _decode_per_planet(output, snapshot, obs, temperature)


__all__ = ["decode", "MAX_PLANETS"]
