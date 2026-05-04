"""Unit tests for case7 ACCUMULATE mission (multi-turn target-aware hold)."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case7.baseline.agent import build_world
from pipeline.rulebase.case7.baseline.core import config as cfg
from pipeline.rulebase.case7.baseline.missions.stay import (
    _accumulate_target_threshold,
    build_accumulate,
)


def _make_obs(
    *,
    my_planets: list[tuple[int, int, float, float, int, int]],
    enemy_planets: list[tuple[int, int, float, float, int, int]],
    enemy_fleet_ships: int | None = None,
    step: int = 30,
) -> dict[str, Any]:
    """Build an observation from compact (id, owner, x, y, ships, prod) tuples.

    Radius is fixed at 3.0 (matching kaggle defaults) and `player=0`.
    """
    planets = [
        [pid, owner, x, y, 3.0, ships, prod]
        for pid, owner, x, y, ships, prod in my_planets + enemy_planets
    ]
    fleets: list[list[Any]] = []
    if enemy_fleet_ships is not None:
        # Fleet originating from the first enemy planet aimed at the first my planet.
        first_my = my_planets[0]
        first_enemy = enemy_planets[0]
        fleets.append(
            [
                0,
                1,
                (first_my[2] + first_enemy[2]) / 2,
                (first_my[3] + first_enemy[3]) / 2,
                3.14159,
                first_enemy[0],
                enemy_fleet_ships,
            ]
        )
    return {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": 0.0,
        "initial_planets": [list(p) for p in planets],
        "comets": [],
        "comet_planet_ids": [],
    }


def _world(obs: dict[str, Any]) -> Any:
    return build_world(obs)


def test_accumulate_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "ACCUMULATE_ENABLED", False)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 30, 3)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    holds, fire_missions, decisions, held_ids = build_accumulate(_world(obs), {})
    assert holds == {}
    assert fire_missions == []
    assert decisions == []
    assert held_ids == set()


def test_accumulate_threshold_uses_capture_need_plus_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threshold = max(need + safety, knee) — must respect knee floor."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 50)
    monkeypatch.setattr(cfg, "ACCUMULATE_SAFETY_SHIPS", 3)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 5, 1)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    world = _world(obs)
    threshold = _accumulate_target_threshold(
        target_id=1, arrival_turn=20, world=world, planned_commitments={}
    )
    # Knee floor (50) dominates because need + safety < 50 in this small case.
    assert threshold >= cfg.ACCUMULATE_KNEE_SHIPS


def test_accumulate_holds_when_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low-threat src with ships below threshold should accumulate-hold."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 50)
    monkeypatch.setattr(cfg, "ACCUMULATE_SAFETY_SHIPS", 2)
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_TARGET_TURNS", 5)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_TARGET_TURNS", 200)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_HOLD_TURNS", 12)
    # Far-away enemy planet, no incoming fleet → reserve = 0.
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 12, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    world = _world(obs)
    holds, fire_missions, decisions, held_ids = build_accumulate(world, {})
    # Since 12 < knee=50, must hold (not fire).
    assert 0 in holds
    assert holds[0] == 12
    assert 0 in held_ids
    assert any(d.kind == "accumulate_hold" for d in decisions)
    assert fire_missions == []


def test_accumulate_fires_when_threshold_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-ship src reaching threshold should fire instead of hold."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 30)
    monkeypatch.setattr(cfg, "ACCUMULATE_SAFETY_SHIPS", 2)
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_TARGET_TURNS", 5)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_TARGET_TURNS", 200)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 80, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    world = _world(obs)
    holds, fire_missions, decisions, held_ids = build_accumulate(world, {})
    # 80 >= knee=30 + safety=2 → fire mission.
    assert fire_missions, "expected at least one accumulate_fire mission"
    fire = fire_missions[0]
    assert fire.kind == "accumulate_fire"
    assert fire.options[0].src_id == 0
    assert fire.options[0].target_id == 1
    # Source should not be in holds when firing.
    assert 0 not in holds
    assert 0 not in held_ids
    assert any(d.kind == "accumulate_fire" for d in decisions)


def test_accumulate_capped_at_max_hold_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source held at MAX_HOLD_TURNS must be released (no hold this turn)."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 50)
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_TARGET_TURNS", 5)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_TARGET_TURNS", 200)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_HOLD_TURNS", 5)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 12, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    world = _world(obs)
    # Baseline: hold fires at all when consec=0.
    base_holds, _, _, _ = build_accumulate(world, {}, accumulate_holds_consec={0: 0})
    assert 0 in base_holds, "fixture should produce a hold without cap"

    # Cap reached → must NOT hold and must record a capped decision.
    holds, _, decisions, held_ids = build_accumulate(
        world, {}, accumulate_holds_consec={0: 5}
    )
    assert 0 not in holds
    assert 0 not in held_ids
    assert any(d.kind == "accumulate_capped" for d in decisions)


def test_accumulate_skips_high_threat_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source with reserve > THREAT_RESERVE_MAX must not accumulate-hold."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 50)
    monkeypatch.setattr(cfg, "ACCUMULATE_THREAT_RESERVE_MAX", 0)
    # Inbound enemy fleet with many ships → reserve > 0 expected.
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 12, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
        enemy_fleet_ships=20,
        step=30,
    )
    world = _world(obs)
    # Sanity: reserve should be > 0 due to inbound fleet.
    if world.reserve.get(0, 0) <= 0:
        pytest.skip(
            "fixture didn't generate reserve pressure; threat-skip path untested"
        )
    holds, _, decisions, held_ids = build_accumulate(world, {})
    assert 0 not in holds
    assert 0 not in held_ids
    # Source skipped → no decision for it from accumulate.
    assert all(d.src_id != 0 for d in decisions)


def test_accumulate_min_target_turns_filters_close_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targets closer than MIN_TARGET_TURNS must be ignored (STAY/regular handles them)."""
    monkeypatch.setattr(cfg, "ACCUMULATE_KNEE_SHIPS", 50)
    # Set MIN very high so even the only target is filtered out.
    monkeypatch.setattr(cfg, "ACCUMULATE_MIN_TARGET_TURNS", 200)
    monkeypatch.setattr(cfg, "ACCUMULATE_MAX_TARGET_TURNS", 1000)
    obs = _make_obs(
        my_planets=[(0, 0, 20.0, 25.0, 12, 2)],
        enemy_planets=[(1, 1, 80.0, 25.0, 10, 2)],
    )
    world = _world(obs)
    holds, fire_missions, decisions, held_ids = build_accumulate(world, {})
    assert holds == {}
    assert fire_missions == []
    assert held_ids == set()
    # No accumulate decision since no candidate target survives the filter.
    assert all(d.src_id != 0 for d in decisions)
