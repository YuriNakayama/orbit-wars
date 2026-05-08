"""Policy output → Kaggle action list for imitation/case9.

Supports 3 head variants via `head_mode`:
  - "candidate":       cand argmax slot 0 = no-op, slot 1..K-1 → candidate planet.
                       ships = max(rule_floor, round(ship_pred)).
  - "candidate_ships": cand argmax + ships_logits 4-bucket (0..3) → ships.
  - "three_head":      from sigmoid threshold → fire decision.
                       target argmax over NUM_TEMPLATES with template id 7 = no-op.
                       ships_logits 4-bucket → ships.
  - "template_ships":  target argmax over NUM_TEMPLATES directly per source.
                       template id 7 = no-op. ships_logits 4-bucket → ships.
  - "dual":            candidate decoder path. 3-head outputs are auxiliary.

Common safety filters (case5 由来) are applied to every variant.
"""

from __future__ import annotations

import math
import os
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
from .templates import NUM_TEMPLATES, resolve_template
from .types import PolicyOutput, WorldSnapshot

SHIPS_BUCKETS = (5, 15, 30, 60)
NO_OP_TEMPLATE_ID = NUM_TEMPLATES - 1


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


def _decode_candidate(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    candidate_pid: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
    use_learned_ships: bool,
) -> list[list[int | float]]:
    T = max(float(temperature), 1e-6)
    assert output.candidate_logits is not None
    cand_logits = output.candidate_logits[0] / T
    fire_margin = _env_float("IL_CASE9_CAND_FIRE_MARGIN", -0.50)
    if use_learned_ships:
        assert output.ships_logits is not None
        ships_argmax = output.ships_logits[0].argmax(dim=-1)  # (P,) bucket idx
    else:
        assert output.ship_pred is not None
        ship_pred = output.ship_pred[0]

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
        for cand_slot in _rank_fire_slots(cand_logits[slot], 0, fire_margin):
            if cand_slot <= 0 or cand_slot >= CAND_K:
                continue
            if not bool(candidate_mask[slot, cand_slot].item()):
                continue
            target_pid = int(candidate_pid[slot, cand_slot].item())
            target = pid_to_planet.get(target_pid)
            if target is None:
                continue
            rule_floor = _fixed_ship_count(target.ships)
            if rule_floor > src.ships:
                continue
            if use_learned_ships:
                bucket = int(ships_argmax[slot].item())
                bucket = max(0, min(SHIPS_BUCKETS.__len__() - 1, bucket))
                ships = max(rule_floor, SHIPS_BUCKETS[bucket])
            else:
                pred_ships = int(round(float(ship_pred[slot].item())))
                ships = max(rule_floor, pred_ships)
            ships = _scale_ship_count(ships)
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


def _decode_three_head(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    template_ctx: torch.Tensor,
    temperature: float,
) -> list[list[int | float]]:
    del template_ctx
    T = max(float(temperature), 1e-6)
    assert output.from_logits is not None
    assert output.target_logits is not None
    assert output.ships_logits is not None
    from_logits = output.from_logits[0] / T
    target_logits = output.target_logits[0] / T
    ships_argmax = output.ships_logits[0].argmax(dim=-1)
    from_prob = torch.sigmoid(from_logits)
    from_threshold = _env_float("IL_CASE9_THREE_FROM_THRESHOLD", 0.5)

    (
        pid_to_planet,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        step,
        incoming_friendly,
    ) = _common_obs_state(obs, snapshot)
    raw_planets = list(obs.get("planets", []) or [])
    committed = dict(incoming_friendly)
    actions: list[list[int | float]] = []
    player = snapshot.player

    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        if float(from_prob[slot].item()) < from_threshold:
            continue
        src = pid_to_planet.get(src_pid)
        if src is None:
            continue
        for tid in target_logits[slot].argsort(dim=-1, descending=True).tolist():
            tid = int(tid)
            if tid == NO_OP_TEMPLATE_ID:
                continue
            target_pid = resolve_template(
                tid, list(raw_planets[slot]), raw_planets, player
            )
            if target_pid is None:
                continue
            target = pid_to_planet.get(int(target_pid))
            if target is None:
                continue
            bucket = int(ships_argmax[slot].item())
            bucket = max(0, min(SHIPS_BUCKETS.__len__() - 1, bucket))
            rule_floor = _fixed_ship_count(target.ships)
            if rule_floor > src.ships:
                continue
            ships = _scale_ship_count(max(rule_floor, SHIPS_BUCKETS[bucket]))
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


def _decode_template_ships(
    output: PolicyOutput,
    snapshot: WorldSnapshot,
    obs: dict[str, Any],
    temperature: float,
) -> list[list[int | float]]:
    T = max(float(temperature), 1e-6)
    assert output.target_logits is not None
    assert output.ships_logits is not None
    target_logits = output.target_logits[0] / T
    ships_argmax = output.ships_logits[0].argmax(dim=-1)
    fire_margin = _env_float("IL_CASE9_TEMPLATE_FIRE_MARGIN", -999.0)

    (
        pid_to_planet,
        initial_by_id,
        ang_vel,
        comets,
        comet_ids,
        step,
        incoming_friendly,
    ) = _common_obs_state(obs, snapshot)
    raw_planets = list(obs.get("planets", []) or [])
    committed = dict(incoming_friendly)
    actions: list[list[int | float]] = []
    player = snapshot.player

    for src_pid in snapshot.my_planet_ids:
        slot = snapshot.planet_ids.index(src_pid)
        src = pid_to_planet.get(src_pid)
        if src is None:
            continue
        for tid in _rank_fire_slots(
            target_logits[slot], NO_OP_TEMPLATE_ID, fire_margin
        ):
            target_pid = resolve_template(
                tid, list(raw_planets[slot]), raw_planets, player
            )
            if target_pid is None:
                continue
            target = pid_to_planet.get(int(target_pid))
            if target is None:
                continue
            bucket = int(ships_argmax[slot].item())
            bucket = max(0, min(SHIPS_BUCKETS.__len__() - 1, bucket))
            rule_floor = _fixed_ship_count(target.ships)
            if rule_floor > src.ships:
                continue
            ships = _scale_ship_count(max(rule_floor, SHIPS_BUCKETS[bucket]))
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
    candidate_pid: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float = 1.0,
    head_mode: str = "candidate",
    template_ctx: torch.Tensor | None = None,
) -> list[list[int | float]]:
    if head_mode in {"candidate", "dual"}:
        return _decode_candidate(
            output,
            snapshot,
            obs,
            candidate_pid,
            candidate_mask,
            temperature,
            use_learned_ships=False,
        )
    if head_mode == "candidate_ships":
        return _decode_candidate(
            output,
            snapshot,
            obs,
            candidate_pid,
            candidate_mask,
            temperature,
            use_learned_ships=True,
        )
    if head_mode == "three_head":
        if template_ctx is None:
            raise ValueError("three_head decoder requires template_ctx")
        return _decode_three_head(output, snapshot, obs, template_ctx, temperature)
    if head_mode == "template_ships":
        return _decode_template_ships(output, snapshot, obs, temperature)
    raise ValueError(f"unknown head_mode={head_mode!r}")


__all__ = ["decode", "MAX_PLANETS"]
