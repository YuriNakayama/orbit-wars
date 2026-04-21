"""Unit tests for case5 core/world_helpers.py — arrival ledger / timelines."""

from __future__ import annotations

import math

from pipeline.rulebase.case5.baseline.core.types import Fleet, Planet
from pipeline.rulebase.case5.baseline.core.world_helpers import (
    count_players,
    detect_enemy_fights_at_neutrals,
    detect_exposed_enemy_planets,
    indirect_features,
    nearest_distance_to_set,
    normalize_arrivals,
    resolve_arrival_event,
    simulate_planet_timeline,
    state_at_timeline,
)


def _planet(pid: int, owner: int, x: float, y: float, ships: int = 10) -> Planet:
    return Planet(id=pid, owner=owner, x=x, y=y, radius=3.0, ships=ships, production=2)


class TestResolveArrivalEvent:
    def test_no_arrivals(self) -> None:
        owner, garrison = resolve_arrival_event(0, 10.0, [])
        assert owner == 0
        assert garrison == 10.0

    def test_friendly_reinforcement(self) -> None:
        owner, garrison = resolve_arrival_event(0, 10.0, [(1, 0, 5)])
        assert owner == 0
        assert garrison == 15.0

    def test_enemy_overpowers(self) -> None:
        owner, garrison = resolve_arrival_event(0, 5.0, [(1, 1, 10)])
        assert owner == 1
        assert garrison == 5.0

    def test_enemy_too_weak_keeps_owner(self) -> None:
        owner, garrison = resolve_arrival_event(0, 10.0, [(1, 1, 5)])
        assert owner == 0
        assert garrison == 5.0

    def test_two_equal_enemies_neutralize(self) -> None:
        owner, garrison = resolve_arrival_event(-1, 0.0, [(1, 1, 5), (1, 2, 5)])
        assert owner == -1


class TestNormalizeArrivals:
    def test_drops_zero_ships(self) -> None:
        events = normalize_arrivals([(1, 0, 0), (2, 1, 5)], 10)
        assert events == [(2, 1, 5)]

    def test_drops_beyond_horizon(self) -> None:
        events = normalize_arrivals([(15, 0, 5)], 10)
        assert events == []

    def test_sorted_by_eta(self) -> None:
        events = normalize_arrivals([(5, 0, 3), (2, 1, 4)], 10)
        assert events[0][0] == 2
        assert events[1][0] == 5

    def test_eta_rounded_up(self) -> None:
        events = normalize_arrivals([(2.3, 0, 5)], 10)
        assert events[0][0] == 3


class TestSimulatePlanetTimeline:
    def test_no_arrivals_planet_grows(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=10)
        tl = simulate_planet_timeline(p, [], player=0, horizon=5)
        assert tl["owner_at"][5] == 0
        assert tl["ships_at"][5] == 10 + 5 * p.production

    def test_neutral_does_not_grow(self) -> None:
        p = _planet(0, -1, 0.0, 0.0, ships=5)
        tl = simulate_planet_timeline(p, [], player=0, horizon=10)
        assert tl["owner_at"][10] == -1
        assert tl["ships_at"][10] == 5

    def test_keep_needed_zero_when_safe(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=10)
        tl = simulate_planet_timeline(p, [], player=0, horizon=10)
        assert tl["keep_needed"] == 0
        assert tl["holds_full"]

    def test_falling_planet_marks_fall_turn(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=2)
        tl = simulate_planet_timeline(p, [(3, 1, 50)], player=0, horizon=10)
        assert tl["fall_turn"] == 3
        assert not tl["holds_full"]


class TestStateAtTimeline:
    def test_returns_initial_state_at_zero(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=10)
        tl = simulate_planet_timeline(p, [], player=0, horizon=5)
        owner, ships = state_at_timeline(tl, 0)
        assert owner == 0
        assert ships == 10

    def test_clamps_beyond_horizon(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=10)
        tl = simulate_planet_timeline(p, [], player=0, horizon=5)
        owner, _ = state_at_timeline(tl, 100)
        assert owner == 0


class TestCountPlayers:
    def test_two_owned_planets(self) -> None:
        planets = [_planet(0, 0, 0, 0), _planet(1, 1, 10, 10)]
        assert count_players(planets, []) == 2

    def test_floor_is_two(self) -> None:
        assert count_players([], []) == 2

    def test_counts_fleet_owners(self) -> None:
        planets = [_planet(0, 0, 0, 0)]
        fleets = [
            Fleet(id=1, owner=2, x=0, y=0, angle=0.0, from_planet_id=0, ships=5),
        ]
        assert count_players(planets, fleets) == 2  # owners {0, 2}, max(2, 2)


class TestNearestDistanceToSet:
    def test_empty_returns_large(self) -> None:
        assert nearest_distance_to_set(0, 0, []) >= 10**9

    def test_finds_nearest(self) -> None:
        planets = [_planet(0, 0, 5, 0), _planet(1, 0, 100, 0)]
        d = nearest_distance_to_set(0, 0, planets)
        assert d == 5


class TestIndirectFeatures:
    def test_empty_neighbors(self) -> None:
        p = _planet(0, 0, 0, 0)
        f, n, e = indirect_features(p, [p], player=0)
        assert (f, n, e) == (0.0, 0.0, 0.0)

    def test_categorizes_by_owner(self) -> None:
        planets = [
            _planet(0, 0, 0, 0),
            _planet(1, 0, 10, 0),
            _planet(2, -1, 20, 0),
            _planet(3, 1, 30, 0),
        ]
        f, n, e = indirect_features(planets[0], planets, player=0)
        assert f > 0
        assert n > 0
        assert e > 0


class TestDetectExposedEnemyPlanets:
    def test_no_outbound_not_exposed(self) -> None:
        enemy = _planet(0, 1, 0, 0, ships=20)
        assert detect_exposed_enemy_planets([], [enemy]) == set()

    def test_high_outbound_marks_exposed(self) -> None:
        enemy = _planet(0, 1, 0, 0, ships=10)
        fleets = [
            Fleet(id=1, owner=1, x=0, y=0, angle=0.0, from_planet_id=0, ships=15),
        ]
        assert detect_exposed_enemy_planets(fleets, [enemy]) == {0}


class TestDetectEnemyFightsAtNeutrals:
    def test_single_enemy_not_fight(self) -> None:
        arrivals = {0: [(5, 1, 20)]}
        assert detect_enemy_fights_at_neutrals(arrivals, player=0) == {}

    def test_two_enemies_above_threshold_is_fight(self) -> None:
        arrivals = {0: [(5, 1, 10), (5, 2, 10)]}
        result = detect_enemy_fights_at_neutrals(arrivals, player=0)
        assert 0 in result
        assert result[0] == 20


class TestNormalizeArrivalsIntegration:
    def test_horizon_rounding_with_simulate(self) -> None:
        p = _planet(0, 0, 0.0, 0.0, ships=10)
        tl = simulate_planet_timeline(p, [(2.7, 1, 100)], player=0, horizon=5)
        assert tl["fall_turn"] == math.ceil(2.7)
