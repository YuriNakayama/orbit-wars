"""Unit tests for ``build_stay_holds``: defense vs burst gating."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case6.baseline.agent import build_world
from pipeline.rulebase.case6.baseline.core import config as cfg
from pipeline.rulebase.case6.baseline.missions.stay import build_stay_holds


def _make_obs(
    *,
    my_planet_ships: int,
    enemy_planet_ships: int,
    enemy_fleet_ships: int | None,
    extra_planets: list[list[Any]] | None = None,
    step: int = 30,
    player: int = 0,
) -> dict[str, Any]:
    """Construct a minimal 2-planet (+optional extras) observation.

    Planets are placed off-axis from the sun (50, 50) so that
    ``safe_angle_and_distance`` returns a real route rather than a
    sun-crossing rejection. Symmetric layout around y = 25 keeps the
    fixture deterministic across test cases.
    """
    planets = [
        [0, player, 20.0, 25.0, 3.0, my_planet_ships, 3],
        [1, 1 - player, 80.0, 25.0, 3.0, enemy_planet_ships, 2],
    ]
    if extra_planets:
        planets.extend(extra_planets)
    fleets: list[list[Any]] = []
    if enemy_fleet_ships is not None:
        # An enemy fleet already in flight from planet 1 toward planet 0.
        fleets.append([0, 1 - player, 70.0, 25.0, 3.14159, 1, enemy_fleet_ships])
    return {
        "player": player,
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


def test_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "STAY_ENABLED", False)
    obs = _make_obs(my_planet_ships=20, enemy_planet_ships=20, enemy_fleet_ships=None)
    holds, decisions = build_stay_holds(_world(obs))
    assert holds == {}
    assert decisions == []


def test_no_threat_no_burst_no_hold() -> None:
    """Quiet board with low ship count should not trigger any STAY hold."""
    obs = _make_obs(my_planet_ships=4, enemy_planet_ships=20, enemy_fleet_ships=None)
    world = _world(obs)
    # Sanity: no incoming enemy fleet means no defense pressure.
    assert not any(
        any(owner != world.player for _eta, owner, _ships in arrivals)
        for arrivals in world.arrivals_by_planet.values()
    )
    holds, decisions = build_stay_holds(world)
    assert holds == {}
    assert all(d.kind != "defense" for d in decisions)


def test_burst_fires_for_growable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A medium-size source with a reachable target should burst-hold."""
    # Disable defense to isolate the burst signal.
    monkeypatch.setattr(cfg, "STAY_DEFENSE_ENABLED", False)
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_SHIPS", 2)
    monkeypatch.setattr(cfg, "STAY_BURST_MAX_TARGET_TURNS", 200)
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_GAIN", 0)
    # Two own planets so reserve doesn't gobble everything; ship count is
    # low enough that adding production noticeably bumps fleet_speed.
    extra = [[2, 0, 30.0, 25.0, 3.0, 4, 5]]
    obs = _make_obs(
        my_planet_ships=4,
        enemy_planet_ships=10,
        enemy_fleet_ships=None,
        extra_planets=extra,
    )
    world = _world(obs)
    holds, decisions = build_stay_holds(world)
    # At least one source should be flagged for burst when the gain
    # threshold is relaxed; this is the smoke-level guarantee.
    burst_decisions = [d for d in decisions if d.kind == "burst"]
    assert burst_decisions, "expected at least one burst decision"
    assert all(holds.get(d.src_id, 0) >= d.held_ships for d in burst_decisions)


def test_holds_clipped_to_usable_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when both motives fire, hold must not exceed (ships - reserve)."""
    monkeypatch.setattr(cfg, "STAY_DEFENSE_ENABLED", True)
    monkeypatch.setattr(cfg, "STAY_BURST_ENABLED", True)
    obs = _make_obs(
        my_planet_ships=15,
        enemy_planet_ships=8,
        enemy_fleet_ships=12,
        step=30,
    )
    world = _world(obs)
    holds, _ = build_stay_holds(world)
    for src_id, held in holds.items():
        usable = max(
            0, int(world.planet_by_id[src_id].ships) - world.reserve.get(src_id, 0)
        )
        assert 0 <= held <= usable, f"hold {held} exceeds usable {usable}"


def test_decisions_recorded_for_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every non-zero hold should produce at least one StayDecision entry."""
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_SHIPS", 2)
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_GAIN", 0)
    extra = [[2, 0, 30.0, 25.0, 3.0, 6, 4]]
    obs = _make_obs(
        my_planet_ships=6,
        enemy_planet_ships=10,
        enemy_fleet_ships=None,
        extra_planets=extra,
    )
    holds, decisions = build_stay_holds(_world(obs))
    if holds:
        held_srcs = set(holds.keys())
        decided_srcs = {d.src_id for d in decisions}
        assert held_srcs.issubset(decided_srcs)


def _burst_world(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a world that reliably triggers at least one burst hold."""
    monkeypatch.setattr(cfg, "STAY_DEFENSE_ENABLED", False)
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_SHIPS", 2)
    monkeypatch.setattr(cfg, "STAY_BURST_MAX_TARGET_TURNS", 200)
    monkeypatch.setattr(cfg, "STAY_BURST_MIN_GAIN", 0)
    extra = [[2, 0, 30.0, 25.0, 3.0, 4, 5]]
    obs = _make_obs(
        my_planet_ships=4,
        enemy_planet_ships=10,
        enemy_fleet_ships=None,
        extra_planets=extra,
    )
    return _world(obs)


def test_burst_capped_when_consecutive_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sources at MAX_HOLD_TURNS should have their burst hold suppressed."""
    monkeypatch.setattr(cfg, "STAY_BURST_MAX_HOLD_TURNS", 3)
    world = _burst_world(monkeypatch)

    # Baseline (no consecutive_holds gate): identify which src would burst.
    base_holds, base_decisions = build_stay_holds(world)
    burst_srcs = {d.src_id for d in base_decisions if d.kind == "burst"}
    assert burst_srcs, "fixture should produce at least one burst hold"

    # Now feed consecutive_holds at the cap for one of those sources.
    capped_src = next(iter(burst_srcs))
    consecutive = {capped_src: 3}
    holds, decisions = build_stay_holds(world, consecutive_holds=consecutive)

    # The capped source should no longer be in the merged holds (only burst was
    # gating it; defense is disabled).
    assert capped_src not in holds, (
        "burst hold for src at MAX_HOLD_TURNS should be suppressed"
    )
    # And it should be observable as a burst_capped diagnostic decision.
    assert any(d.src_id == capped_src and d.kind == "burst_capped" for d in decisions)
    # Sanity: original burst would have held this src.
    assert capped_src in base_holds


def test_burst_allowed_below_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources below MAX_HOLD_TURNS should still burst-hold normally."""
    monkeypatch.setattr(cfg, "STAY_BURST_MAX_HOLD_TURNS", 3)
    world = _burst_world(monkeypatch)

    base_holds, base_decisions = build_stay_holds(world)
    burst_srcs = {d.src_id for d in base_decisions if d.kind == "burst"}
    assert burst_srcs

    src = next(iter(burst_srcs))
    consecutive = {src: 2}  # below cap
    holds, decisions = build_stay_holds(world, consecutive_holds=consecutive)
    assert holds.get(src, 0) == base_holds.get(src, 0)
    assert any(d.src_id == src and d.kind == "burst" for d in decisions)


def test_consecutive_none_preserves_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing consecutive_holds=None must match the no-arg call exactly."""
    world = _burst_world(monkeypatch)
    legacy_holds, legacy_decisions = build_stay_holds(world)
    explicit_holds, explicit_decisions = build_stay_holds(world, consecutive_holds=None)
    assert legacy_holds == explicit_holds
    assert [(d.src_id, d.kind, d.held_ships) for d in legacy_decisions] == [
        (d.src_id, d.kind, d.held_ships) for d in explicit_decisions
    ]


def test_defense_unaffected_by_consecutive_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense holds should ignore the consecutive_holds gate (burst-only cap)."""
    monkeypatch.setattr(cfg, "STAY_DEFENSE_ENABLED", True)
    monkeypatch.setattr(cfg, "STAY_BURST_ENABLED", False)
    monkeypatch.setattr(cfg, "STAY_BURST_MAX_HOLD_TURNS", 1)
    obs = _make_obs(
        my_planet_ships=15,
        enemy_planet_ships=8,
        enemy_fleet_ships=12,
        step=30,
    )
    world = _world(obs)
    no_hist_holds, _ = build_stay_holds(world)
    # Mark every owned source as if it had been held forever.
    consecutive = {p.id: 999 for p in world.my_planets}
    with_hist_holds, with_hist_decisions = build_stay_holds(
        world, consecutive_holds=consecutive
    )
    assert no_hist_holds == with_hist_holds
    # No burst_capped decisions should appear when burst is disabled.
    assert not any(d.kind == "burst_capped" for d in with_hist_decisions)
