"""Unit tests for pipeline.rulebase.case4.baseline.core.geometry."""

from __future__ import annotations

import math

from pipeline.rulebase.case4.baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    SUN_R,
    SUN_SAFETY,
)
from pipeline.rulebase.case4.baseline.core.geometry import (
    actual_path_geometry,
    dist,
    launch_point,
    orbital_radius,
    point_to_segment_distance,
    safe_angle_and_distance,
    segment_hits_sun,
)
from pipeline.rulebase.case4.baseline.core.types import Planet


def _planet(pid: int, x: float, y: float, *, radius: float = 2.0) -> Planet:
    return Planet(id=pid, owner=0, x=x, y=y, radius=radius, ships=10, production=1)


# ---------- dist ----------


def test_dist_3_4_5_triangle() -> None:
    assert dist(0.0, 0.0, 3.0, 4.0) == 5.0


def test_dist_zero_for_same_point() -> None:
    assert dist(7.0, 7.0, 7.0, 7.0) == 0.0


def test_dist_is_symmetric() -> None:
    assert dist(1.0, 2.0, 5.0, 6.0) == dist(5.0, 6.0, 1.0, 2.0)


# ---------- orbital_radius ----------


def test_orbital_radius_at_center_is_zero() -> None:
    assert orbital_radius(_planet(1, CENTER_X, CENTER_Y)) == 0.0


def test_orbital_radius_offset() -> None:
    assert orbital_radius(_planet(1, CENTER_X + 30.0, CENTER_Y)) == 30.0


# ---------- point_to_segment_distance ----------


def test_point_to_segment_distance_on_segment_is_zero() -> None:
    d = point_to_segment_distance(5.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert math.isclose(d, 0.0, abs_tol=1e-9)


def test_point_to_segment_distance_perpendicular() -> None:
    d = point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert math.isclose(d, 3.0, abs_tol=1e-9)


def test_point_to_segment_distance_endpoint() -> None:
    # Closest point is endpoint x1,y1.
    d = point_to_segment_distance(-3.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    assert math.isclose(d, 3.0, abs_tol=1e-9)


def test_point_to_segment_distance_degenerate_segment() -> None:
    # Both endpoints same → degenerate, returns dist to that point.
    d = point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    assert d == 5.0


# ---------- segment_hits_sun ----------


def test_segment_hits_sun_through_center_returns_true() -> None:
    assert (
        segment_hits_sun(CENTER_X - 30.0, CENTER_Y, CENTER_X + 30.0, CENTER_Y) is True
    )


def test_segment_hits_sun_far_from_sun_returns_false() -> None:
    far = CENTER_Y + SUN_R + SUN_SAFETY + 10.0
    assert segment_hits_sun(CENTER_X - 30.0, far, CENTER_X + 30.0, far) is False


def test_segment_hits_sun_respects_safety_buffer() -> None:
    # Segment just inside the safety buffer should still register a hit.
    boundary = CENTER_Y + SUN_R + SUN_SAFETY - 0.1
    assert segment_hits_sun(CENTER_X - 5.0, boundary, CENTER_X + 5.0, boundary) is True


# ---------- launch_point ----------


def test_launch_point_along_zero_angle() -> None:
    sx, sy, sr = 10.0, 20.0, 2.0
    x, y = launch_point(sx, sy, sr, angle=0.0)
    assert math.isclose(x, sx + sr + LAUNCH_CLEARANCE, abs_tol=1e-9)
    assert math.isclose(y, sy, abs_tol=1e-9)


def test_launch_point_along_quarter_pi() -> None:
    sx, sy, sr = 0.0, 0.0, 1.0
    x, y = launch_point(sx, sy, sr, angle=math.pi / 2.0)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, sr + LAUNCH_CLEARANCE, abs_tol=1e-9)


# ---------- actual_path_geometry ----------


def test_actual_path_geometry_returns_six_floats() -> None:
    res = actual_path_geometry(0.0, 0.0, 1.0, 10.0, 0.0, 1.0)
    assert len(res) == 6
    angle, sx, sy, ex, ey, hd = res
    assert math.isclose(angle, 0.0, abs_tol=1e-9)
    assert hd > 0.0
    # End point is along the angle from start point at distance hd.
    assert math.isclose(ex - sx, hd, abs_tol=1e-9)
    assert math.isclose(ey - sy, 0.0, abs_tol=1e-9)


def test_actual_path_geometry_zero_distance_when_overlap() -> None:
    # src and target overlap; hit distance clamped to 0.
    res = actual_path_geometry(10.0, 10.0, 5.0, 10.0, 10.0, 5.0)
    assert res[5] == 0.0


# ---------- safe_angle_and_distance ----------


def test_safe_angle_clear_path_returns_pair() -> None:
    res = safe_angle_and_distance(0.0, 0.0, 1.0, 20.0, 0.0, 1.0)
    assert res is not None
    angle, distance = res
    assert math.isclose(angle, 0.0, abs_tol=1e-9)
    assert distance > 0.0


def test_safe_angle_blocked_by_sun_returns_none() -> None:
    res = safe_angle_and_distance(
        CENTER_X - 30.0, CENTER_Y, 1.0, CENTER_X + 30.0, CENTER_Y, 1.0
    )
    assert res is None


def test_constants_are_well_defined() -> None:
    assert SUN_R > 0
    assert SUN_SAFETY >= 0
    assert LAUNCH_CLEARANCE > 0
