# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Agent entry point: build WorldModel from observation and delegate to plan_moves.

case9 = case4 production base + planet thrash filter (case8 iter3 v1 から移植)。
self が直近 THRASH_WINDOW=10 ターンに奪われた planet への capture/snipe/swarm
mission の score を THRASH_SCORE_MULT=0.3 倍に減衰する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import opponent_model as om
from .core.config import (
    LOOKAHEAD_ENABLED,
    OM_V2_ENABLED,
    OPPONENT_MODEL_ENABLED,
    THRASH_WINDOW,
)
from .core.types import Fleet, Planet
from .core.world_model import WorldModel
from .lookahead import predict_enemy_fleets
from .strategy import plan_moves


@dataclass
class StayState:
    """Per-game rolling state for the case9 thrash filter.

    Tracks ownership transitions across turns so the WorldModel can carry
    `recently_lost` and let `apply_score_modifiers` decay capture/snipe/swarm
    scores for planets we just lost.
    """

    last_step: int | None = None
    prev_planet_owners: dict[int, int] = field(default_factory=dict)
    recently_lost: dict[int, int] = field(default_factory=dict)
    mission_commits: dict[int, list[int]] = field(default_factory=dict)


_OM_STATE: om.OMState = om.OMState()
_STAY_STATE: StayState = StayState()


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _reset_stay_state_if_new_episode(step: int) -> None:
    """Clear thrash-tracking state when a new episode starts.

    Detected by step==0, step decreasing, or a large forward jump (mirrors
    ``opponent_model.is_new_game``).
    """
    last = _STAY_STATE.last_step
    if (
        last is None
        or step <= 0
        or step < last
        or step - last > 250  # half of TOTAL_STEPS=500
    ):
        _STAY_STATE.prev_planet_owners = {}
        _STAY_STATE.recently_lost = {}
        _STAY_STATE.mission_commits = {}
    _STAY_STATE.last_step = step


def _update_thrash_state(step: int, planets: list[Planet], player: int) -> None:
    """Detect self→other ownership transitions and prune stale entries."""
    prev = _STAY_STATE.prev_planet_owners
    if prev:
        for planet in planets:
            prior = prev.get(planet.id)
            if prior == player and planet.owner != player:
                _STAY_STATE.recently_lost[planet.id] = step

    _STAY_STATE.prev_planet_owners = {p.id: p.owner for p in planets}

    cutoff = step - THRASH_WINDOW
    _STAY_STATE.recently_lost = {
        pid: t for pid, t in _STAY_STATE.recently_lost.items() if t >= cutoff
    }
    _STAY_STATE.mission_commits = {
        pid: [t for t in turns if t >= cutoff]
        for pid, turns in _STAY_STATE.mission_commits.items()
        if any(t >= cutoff for t in turns)
    }


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
        recently_lost=dict(_STAY_STATE.recently_lost),
        mission_commits={k: list(v) for k, v in _STAY_STATE.mission_commits.items()},
    )


def agent(obs: Any) -> list[list[int | float]]:
    step = _read(obs, "step", 0) or 0
    _reset_stay_state_if_new_episode(step)

    raw_planets = _read(obs, "planets", []) or []
    planets_for_thrash = [Planet(*planet) for planet in raw_planets]
    player = _read(obs, "player", 0)
    _update_thrash_state(step, planets_for_thrash, player)

    world = build_world(obs)
    if not world.my_planets:
        return []
    return plan_moves(world)
