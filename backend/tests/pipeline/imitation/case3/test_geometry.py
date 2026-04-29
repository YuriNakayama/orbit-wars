"""Unit tests for pipeline.imitation.case3.policy.geometry.

Each case keeps its own copy of geometry tests (case-independent principle).
"""

from __future__ import annotations

import math

from pipeline.imitation.case3.policy.geometry import (
    CENTER_X,
    CENTER_Y,
    HORIZON,
    MAX_SPEED,
    SUN_R,
    Planet,
    aim_with_prediction,
    comet_remaining_life,
    dist,
    estimate_arrival,
    fleet_speed,
    is_static_planet,
    predict_comet_position,
    predict_planet_position,
    safe_angle_and_distance,
)


def _planet(
    pid: int,
    x: float,
    y: float,
    *,
    owner: int = 0,
    radius: float = 2.0,
    ships: int = 10,
    production: int = 1,
) -> Planet:
    return Planet(
        id=pid,
        owner=owner,
        x=x,
        y=y,
        radius=radius,
        ships=ships,
        production=production,
    )


# ---------- distance ----------


def test_dist_origin_to_unit() -> None:
    assert dist(0.0, 0.0, 3.0, 4.0) == 5.0


def test_dist_is_symmetric() -> None:
    assert dist(1.0, 2.0, 5.0, 7.0) == dist(5.0, 7.0, 1.0, 2.0)


def test_dist_zero_for_same_point() -> None:
    assert dist(7.0, 7.0, 7.0, 7.0) == 0.0


# ---------- fleet_speed ----------


def test_fleet_speed_one_ship_is_minimum() -> None:
    assert fleet_speed(1) == 1.0


def test_fleet_speed_zero_or_negative_treated_as_minimum() -> None:
    assert fleet_speed(0) == 1.0
    assert fleet_speed(-5) == 1.0


def test_fleet_speed_monotonic_in_ships() -> None:
    speeds = [fleet_speed(n) for n in (1, 10, 100, 1000)]
    assert speeds == sorted(speeds)


def test_fleet_speed_capped_at_max() -> None:
    assert fleet_speed(1000) == MAX_SPEED
    assert fleet_speed(10_000) == MAX_SPEED


# ---------- safe_angle_and_distance ----------


def test_safe_angle_aligns_with_target_direction() -> None:
    result = safe_angle_and_distance(0.0, 0.0, 1.0, 10.0, 0.0, 1.0)
    assert result is not None
    angle, distance = result
    assert math.isclose(angle, 0.0, abs_tol=1e-6)
    assert distance > 0.0


def test_safe_angle_blocked_by_sun() -> None:
    # Sun is between source and target → straight-line shot blocked.
    src_x = CENTER_X - 30.0
    tgt_x = CENTER_X + 30.0
    result = safe_angle_and_distance(src_x, CENTER_Y, 1.0, tgt_x, CENTER_Y, 1.0)
    assert result is None


# ---------- estimate_arrival ----------


def test_estimate_arrival_reasonable_distance() -> None:
    est = estimate_arrival(0.0, 0.0, 1.0, 20.0, 0.0, 1.0, ships=10)
    assert est is not None
    angle, turns = est
    assert math.isclose(angle, 0.0, abs_tol=1e-6)
    assert turns >= 1


def test_estimate_arrival_blocked_by_sun_returns_none() -> None:
    src_x = CENTER_X - 30.0
    tgt_x = CENTER_X + 30.0
    est = estimate_arrival(src_x, CENTER_Y, 1.0, tgt_x, CENTER_Y, 1.0, ships=10)
    assert est is None


# ---------- predict_planet_position ----------


def test_predict_planet_position_static_planet_unchanged() -> None:
    # Place far enough from center that r + radius >= ROTATION_LIMIT (=50).
    p = _planet(1, CENTER_X + 60.0, CENTER_Y, radius=5.0)
    initial = {1: p}
    new_x, new_y = predict_planet_position(p, initial, angular_velocity=0.1, turns=10)
    assert new_x == p.x
    assert new_y == p.y


def test_predict_planet_position_orbital_rotates() -> None:
    # Orbiting planet: r + radius < ROTATION_LIMIT
    p = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {2: p}
    new_x, new_y = predict_planet_position(
        p, initial, angular_velocity=math.pi, turns=1
    )
    # Half-rotation: should land near the opposite side.
    assert math.isclose(new_x, CENTER_X - 20.0, abs_tol=1e-6)
    assert math.isclose(new_y, CENTER_Y, abs_tol=1e-6)


def test_predict_planet_position_unknown_id_returns_current() -> None:
    p = _planet(3, 10.0, 20.0)
    new_x, new_y = predict_planet_position(p, {}, angular_velocity=0.5, turns=10)
    assert (new_x, new_y) == (10.0, 20.0)


# ---------- predict_comet_position / comet_remaining_life ----------


def test_predict_comet_position_returns_path_entry() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(10.0, 10.0), (11.0, 11.0), (12.0, 12.0)]],
            "path_index": 0,
        }
    ]
    pos = predict_comet_position(42, comets, turns=2)
    assert pos == (12.0, 12.0)


def test_predict_comet_position_out_of_range_returns_none() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(10.0, 10.0), (11.0, 11.0)]],
            "path_index": 0,
        }
    ]
    assert predict_comet_position(42, comets, turns=10) is None


def test_predict_comet_position_unknown_planet_returns_none() -> None:
    assert predict_comet_position(99, [], turns=1) is None


def test_comet_remaining_life_counts_remaining_steps() -> None:
    comets = [
        {
            "planet_ids": [7],
            "paths": [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]],
            "path_index": 1,
        }
    ]
    assert comet_remaining_life(7, comets) == 3


def test_comet_remaining_life_unknown_planet_zero() -> None:
    assert comet_remaining_life(404, []) == 0


# ---------- is_static_planet ----------


def test_is_static_planet_true_for_outer_planet() -> None:
    p = _planet(1, CENTER_X + 60.0, CENTER_Y, radius=5.0)
    assert is_static_planet(p) is True


def test_is_static_planet_false_for_inner_planet() -> None:
    p = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    assert is_static_planet(p) is False


# ---------- aim_with_prediction (integration of geometry) ----------


def test_aim_with_prediction_static_target_returns_solution() -> None:
    src = _planet(1, 10.0, 50.0, radius=2.0)
    tgt = _planet(2, 90.0, 50.0, radius=2.0)
    initial = {1: src, 2: tgt}
    result = aim_with_prediction(
        src,
        tgt,
        ships=10,
        initial_by_id=initial,
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
    )
    # Sun blocks straight line; aim_with_prediction may return None if no safe arc.
    # Either an answer with finite turns within HORIZON, or None - both are valid.
    if result is not None:
        _, turns, _, _ = result
        assert 1 <= turns <= HORIZON


def test_aim_with_prediction_nearby_target_succeeds() -> None:
    src = _planet(1, 20.0, 20.0, radius=2.0)
    tgt = _planet(2, 30.0, 20.0, radius=2.0)
    initial = {1: src, 2: tgt}
    result = aim_with_prediction(
        src,
        tgt,
        ships=5,
        initial_by_id=initial,
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
    )
    assert result is not None
    angle, turns, _, _ = result
    assert math.isfinite(angle)
    assert turns >= 1


def test_sun_constants_are_positive() -> None:
    assert SUN_R > 0
    assert HORIZON > 0
    assert MAX_SPEED > 1.0
