# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Agent entry point: build WorldModel from observation and delegate to plan_moves.

case9 additionally maintains a module-level ``_DISPATCH_HISTORY`` of
``(src_id, est_dst_id) -> last_step`` populated from the actions returned by
``plan_moves``. The history is read by missions/reinforcement.py and
missions/harass.py for ping-pong cooldown checks. New-game detection clears
the history on a step regression.
"""

from __future__ import annotations

from typing import Any

from . import opponent_model as om
from .core.config import (
    ANTI_PING_PONG_ENABLED,
    LOOKAHEAD_ENABLED,
    OM_V2_ENABLED,
    OPPONENT_MODEL_ENABLED,
    PING_PONG_PAIR_COOLDOWN_TURNS,
)
from .core.types import Fleet, Planet
from .core.world_model import WorldModel
from .lookahead import _infer_action_target, predict_enemy_fleets
from .strategy import plan_moves

_OM_STATE: om.OMState = om.OMState()
_DISPATCH_HISTORY: dict[tuple[int, int], int] = {}
_LAST_STEP: int = -1


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _update_opponent_model(
    step: int, planets: list[Planet], fleets: list[Fleet], player: int
) -> tuple[dict[int, list[tuple[int, int, int]]], dict[int, float]]:
    """Update rolling OM state and return (predicted_arrivals, threat_score)."""
    state = _OM_STATE
    if om.is_new_game(state.last_snapshot, step):
        state.last_snapshot = None
        state.launches.clear()
        state.pref_counts.clear()

    prev = state.last_snapshot
    if prev is not None:
        events = om.detect_launches(prev, step, planets, fleets, player)
        if events:
            state.launches.extend(events)
            om.update_preferences(state.pref_counts, events)
            om.trim_history(state)

    state.last_snapshot = om.make_snapshot(step, planets, fleets)

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    launch_rate = om.compute_launch_rate(state.launches, step)
    if OM_V2_ENABLED:
        predicted = om.predict_future_arrivals_v2(
            enemy_planets,
            my_planets,
            state.launches,
            launch_rate,
            state.pref_counts,
            step,
        )
    else:
        predicted = om.predict_future_arrivals(
            enemy_planets, my_planets, launch_rate, state.pref_counts
        )
    threat = om.compute_threat_score(my_planets, enemy_planets, None, state.pref_counts)
    return predicted, threat


def _maybe_reset_history(step: int) -> None:
    global _LAST_STEP
    if step <= _LAST_STEP:
        _DISPATCH_HISTORY.clear()
    _LAST_STEP = step


def _prune_history(step: int) -> None:
    cutoff = step - PING_PONG_PAIR_COOLDOWN_TURNS - 2
    stale = [k for k, last in _DISPATCH_HISTORY.items() if last < cutoff]
    for key in stale:
        del _DISPATCH_HISTORY[key]


def _record_dispatches(
    step: int, planets: list[Planet], moves: list[list[int | float]]
) -> None:
    by_id = {p.id: p for p in planets}
    for move in moves:
        if not move:
            continue
        src_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
        src = by_id.get(src_id)
        if src is None or ships <= 0:
            continue
        target = _infer_action_target(src, angle, planets, ships)
        if target is None:
            continue
        _DISPATCH_HISTORY[(src_id, int(target.id))] = step


def build_world(obs: Any) -> WorldModel:
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    predicted_arrivals: dict[int, list[tuple[int, int, int]]] = {}
    opponent_threat_score: dict[int, float] = {}
    if OPPONENT_MODEL_ENABLED:
        predicted_arrivals, opponent_threat_score = _update_opponent_model(
            step, planets, fleets, player
        )

    recent_dispatches = dict(_DISPATCH_HISTORY) if ANTI_PING_PONG_ENABLED else {}

    if LOOKAHEAD_ENABLED:
        probe = WorldModel(
            player=player,
            step=step,
            planets=planets,
            fleets=fleets,
            initial_by_id=initial_by_id,
            ang_vel=ang_vel,
            comets=comets,
            comet_ids=comet_ids,
            predicted_arrivals=predicted_arrivals,
            opponent_threat_score=opponent_threat_score,
            recent_dispatches=recent_dispatches,
        )
        lookahead_predictions = predict_enemy_fleets(probe)
        if lookahead_predictions:
            for pid, arrivals in lookahead_predictions.items():
                predicted_arrivals.setdefault(pid, []).extend(arrivals)

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
        predicted_arrivals=predicted_arrivals,
        opponent_threat_score=opponent_threat_score,
        recent_dispatches=recent_dispatches,
    )


def agent(obs: Any) -> list[list[int | float]]:
    step = _read(obs, "step", 0) or 0
    _maybe_reset_history(int(step))
    world = build_world(obs)
    if not world.my_planets:
        return []
    moves = plan_moves(world)
    if ANTI_PING_PONG_ENABLED:
        _record_dispatches(int(step), world.planets, moves)
        _prune_history(int(step))
    return moves
