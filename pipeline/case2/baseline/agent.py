# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Agent entry point: build WorldModel from observation and delegate to plan_moves."""

from __future__ import annotations

from typing import Any

from . import opponent_model as om
from .core.config import LOOKAHEAD_ENABLED, OM_V2_ENABLED, OPPONENT_MODEL_ENABLED
from .core.types import Fleet, Planet
from .core.world_model import WorldModel
from .lookahead import predict_enemy_fleets
from .strategy import plan_moves

_OM_STATE: om.OMState = om.OMState()


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
    )


def agent(obs: Any) -> list[list[int | float]]:
    world = build_world(obs)
    if not world.my_planets:
        return []
    return plan_moves(world)
