"""Unit tests for case5 core/physics.py — pure geometry helpers."""

from __future__ import annotations

import math

import pytest

from pipeline.rulebase.case5.baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    MAX_SPEED,
    ROTATION_LIMIT,
    SUN_R,
    SUN_SAFETY,
)
from pipeline.rulebase.case5.baseline.core.physics import (
    dist,
    fleet_speed,
    is_static_planet,
    orbital_radius,
    point_to_segment_distance,
    safe_angle_and_distance,
    segment_hits_sun,
)
from pipeline.rulebase.case5.baseline.core.types import Planet


def _planet(pid: int, x: float, y: float, radius: float = 3.0) -> Planet:
    return Planet(id=pid, owner=-1, x=x, y=y, radius=radius, ships=10, production=2)


class TestDist:
    def test_zero_distance(self) -> None:
        assert dist(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_pythagorean(self) -> None:
        assert dist(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


class TestOrbitalRadius:
    def test_at_center(self) -> None:
        assert orbital_radius(_planet(0, CENTER_X, CENTER_Y)) == 0.0

    def test_offset(self) -> None:
        p = _planet(0, CENTER_X + 30, CENTER_Y)
        assert orbital_radius(p) == pytest.approx(30.0)


class TestIsStaticPlanet:
    def test_far_planet_is_static(self) -> None:
        p = _planet(0, CENTER_X + ROTATION_LIMIT, CENTER_Y, radius=3.0)
        assert is_static_planet(p)

    def test_near_planet_is_rotating(self) -> None:
        p = _planet(0, CENTER_X + 20, CENTER_Y, radius=3.0)
        assert not is_static_planet(p)


class TestFleetSpeed:
    def test_one_ship_baseline(self) -> None:
        assert fleet_speed(1) == 1.0

    def test_speed_increases_with_ships(self) -> None:
        assert fleet_speed(10) > fleet_speed(1)
        assert fleet_speed(100) > fleet_speed(10)

    def test_capped_at_max_speed(self) -> None:
        assert fleet_speed(10000) == pytest.approx(MAX_SPEED)


class TestPointToSegmentDistance:
    def test_point_on_segment(self) -> None:
        d = point_to_segment_distance(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        assert d == pytest.approx(0.0)

    def test_point_off_segment(self) -> None:
        d = point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
        assert d == pytest.approx(3.0)

    def test_zero_length_segment(self) -> None:
        d = point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        assert d == pytest.approx(5.0)


class TestSegmentHitsSun:
    def test_segment_through_sun(self) -> None:
        assert segment_hits_sun(CENTER_X - 30, CENTER_Y, CENTER_X + 30, CENTER_Y)

    def test_segment_clear_of_sun(self) -> None:
        clear_y = CENTER_Y + SUN_R + SUN_SAFETY + 5
        assert not segment_hits_sun(0, clear_y, 100, clear_y)


class TestSafeAngleAndDistance:
    def test_blocked_by_sun_returns_none(self) -> None:
        result = safe_angle_and_distance(20.0, 50.0, 3.0, 80.0, 50.0, 3.0)
        assert result is None

    def test_clear_path_returns_angle_and_dist(self) -> None:
        result = safe_angle_and_distance(20.0, 80.0, 3.0, 80.0, 80.0, 3.0)
        assert result is not None
        angle, d = result
        assert angle == pytest.approx(0.0, abs=1e-6)
        assert d > 0.0
        assert d < 60.0

    def test_angle_is_radians(self) -> None:
        result = safe_angle_and_distance(20.0, 80.0, 3.0, 20.0, 90.0, 3.0)
        assert result is not None
        angle, _ = result
        assert angle == pytest.approx(math.pi / 2, abs=1e-6)
