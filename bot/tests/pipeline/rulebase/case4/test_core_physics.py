"""Unit tests for pipeline.rulebase.case4.baseline.core.physics."""

from __future__ import annotations

import math

from pipeline.rulebase.case4.baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    HORIZON,
    MAX_SPEED,
    ROTATION_LIMIT,
)
from pipeline.rulebase.case4.baseline.core.physics import (
    aim_with_prediction,
    comet_remaining_life,
    estimate_arrival,
    fleet_speed,
    is_static_planet,
    predict_comet_position,
    predict_planet_position,
    predict_target_position,
    predict_target_position_fractional,
    travel_time,
)
from pipeline.rulebase.case4.baseline.core.types import Planet


def _planet(
    pid: int,
    x: float,
    y: float,
    *,
    owner: int = 0,
    radius: float = 2.0,
    ships: int = 10,
) -> Planet:
    return Planet(
        id=pid, owner=owner, x=x, y=y, radius=radius, ships=ships, production=1
    )


# ---------- fleet_speed ----------


def test_fleet_speed_floor_for_one_ship() -> None:
    assert fleet_speed(1) == 1.0


def test_fleet_speed_floor_for_zero_or_negative() -> None:
    assert fleet_speed(0) == 1.0
    assert fleet_speed(-3) == 1.0


def test_fleet_speed_increases_with_ships() -> None:
    assert fleet_speed(10) < fleet_speed(100) < fleet_speed(1000)


def test_fleet_speed_caps_at_max() -> None:
    assert fleet_speed(1000) == MAX_SPEED


# ---------- is_static_planet ----------


def test_is_static_planet_outer_orbit() -> None:
    p = _planet(1, CENTER_X + 60.0, CENTER_Y, radius=5.0)
    assert is_static_planet(p) is True


def test_is_static_planet_inner_orbit() -> None:
    p = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    assert is_static_planet(p) is False


def test_is_static_planet_boundary() -> None:
    # r + radius == ROTATION_LIMIT → static.
    radius = 3.0
    p = _planet(3, CENTER_X + (ROTATION_LIMIT - radius), CENTER_Y, radius=radius)
    assert is_static_planet(p) is True


# ---------- predict_planet_position ----------


def test_predict_planet_position_orbital_half_rotation() -> None:
    p = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {2: p}
    nx, ny = predict_planet_position(p, initial, math.pi, turns=1)
    assert math.isclose(nx, CENTER_X - 20.0, abs_tol=1e-6)
    assert math.isclose(ny, CENTER_Y, abs_tol=1e-6)


def test_predict_planet_position_static_unchanged() -> None:
    p = _planet(2, CENTER_X + 60.0, CENTER_Y, radius=5.0)
    initial = {2: p}
    nx, ny = predict_planet_position(p, initial, 1.0, turns=10)
    assert (nx, ny) == (p.x, p.y)


def test_predict_planet_position_unknown_id_returns_current() -> None:
    p = _planet(2, 10.0, 20.0)
    nx, ny = predict_planet_position(p, {}, 1.0, turns=10)
    assert (nx, ny) == (10.0, 20.0)


# ---------- predict_comet_position / comet_remaining_life ----------


def test_predict_comet_position_returns_path_entry() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(10.0, 10.0), (11.0, 11.0), (12.0, 12.0)]],
            "path_index": 0,
        }
    ]
    assert predict_comet_position(42, comets, turns=2) == (12.0, 12.0)


def test_predict_comet_position_out_of_range_returns_none() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(10.0, 10.0)]],
            "path_index": 0,
        }
    ]
    assert predict_comet_position(42, comets, turns=10) is None


def test_predict_comet_position_unknown_id_returns_none() -> None:
    assert predict_comet_position(99, [], turns=1) is None


def test_comet_remaining_life_counts_steps() -> None:
    comets = [
        {
            "planet_ids": [7],
            "paths": [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]],
            "path_index": 1,
        }
    ]
    assert comet_remaining_life(7, comets) == 3


def test_comet_remaining_life_unknown_id_zero() -> None:
    assert comet_remaining_life(404, []) == 0


# ---------- estimate_arrival / travel_time ----------


def test_estimate_arrival_clear_path() -> None:
    est = estimate_arrival(0.0, 0.0, 1.0, 20.0, 0.0, 1.0, ships=10)
    assert est is not None
    angle, turns = est
    assert math.isclose(angle, 0.0, abs_tol=1e-9)
    assert turns >= 1


def test_estimate_arrival_blocked_returns_none() -> None:
    est = estimate_arrival(
        CENTER_X - 30.0, CENTER_Y, 1.0, CENTER_X + 30.0, CENTER_Y, 1.0, ships=10
    )
    assert est is None


def test_travel_time_blocked_returns_sentinel() -> None:
    t = travel_time(
        CENTER_X - 30.0, CENTER_Y, 1.0, CENTER_X + 30.0, CENTER_Y, 1.0, ships=10
    )
    assert t >= 10**8


def test_travel_time_clear_returns_finite() -> None:
    t = travel_time(0.0, 0.0, 1.0, 20.0, 0.0, 1.0, ships=10)
    assert 1 <= t <= HORIZON


# ---------- predict_target_position dispatch ----------


def test_predict_target_position_planet_branch() -> None:
    target = _planet(5, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    pos = predict_target_position(
        target,
        turns=1,
        initial_by_id={5: target},
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
    )
    assert pos is not None
    assert math.isclose(pos[0], target.x, abs_tol=1e-9)


def test_predict_target_position_comet_branch() -> None:
    comets = [
        {
            "planet_ids": [9],
            "paths": [[(0.0, 0.0), (5.0, 5.0)]],
            "path_index": 0,
        }
    ]
    target = _planet(9, 0.0, 0.0)
    pos = predict_target_position(
        target,
        turns=1,
        initial_by_id={9: target},
        ang_vel=0.0,
        comets=comets,
        comet_ids={9},
    )
    assert pos == (5.0, 5.0)


# ---------- aim_with_prediction (integration) ----------


# ---------- predict_target_position_fractional (case4-specific) ----------


def test_predict_target_position_fractional_integer_turns_matches_integer_predictor() -> (
    None
):
    target = _planet(5, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {5: target}
    integer_pos = predict_target_position(
        target,
        turns=2,
        initial_by_id=initial,
        ang_vel=0.5,
        comets=[],
        comet_ids=set(),
    )
    fractional_pos = predict_target_position_fractional(
        target,
        turns=2.0,
        initial_by_id=initial,
        ang_vel=0.5,
        comets=[],
        comet_ids=set(),
    )
    assert integer_pos is not None
    assert fractional_pos is not None
    assert math.isclose(integer_pos[0], fractional_pos[0], abs_tol=1e-6)
    assert math.isclose(integer_pos[1], fractional_pos[1], abs_tol=1e-6)


def test_predict_target_position_fractional_interpolates_between_integers() -> None:
    target = _planet(6, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {6: target}
    pos_lo = predict_target_position(
        target,
        turns=1,
        initial_by_id=initial,
        ang_vel=math.pi / 4.0,
        comets=[],
        comet_ids=set(),
    )
    pos_hi = predict_target_position(
        target,
        turns=2,
        initial_by_id=initial,
        ang_vel=math.pi / 4.0,
        comets=[],
        comet_ids=set(),
    )
    pos_mid = predict_target_position_fractional(
        target,
        turns=1.5,
        initial_by_id=initial,
        ang_vel=math.pi / 4.0,
        comets=[],
        comet_ids=set(),
    )
    assert pos_lo is not None and pos_hi is not None and pos_mid is not None
    # Fractional 1.5 is the midpoint of pos_lo and pos_hi (linear interpolation).
    assert math.isclose(pos_mid[0], (pos_lo[0] + pos_hi[0]) / 2.0, abs_tol=1e-6)
    assert math.isclose(pos_mid[1], (pos_lo[1] + pos_hi[1]) / 2.0, abs_tol=1e-6)


def test_aim_with_prediction_nearby_target_succeeds() -> None:
    src = _planet(1, 20.0, 20.0, radius=2.0)
    tgt = _planet(2, 30.0, 20.0, radius=2.0)
    res = aim_with_prediction(
        src,
        tgt,
        ships=5,
        initial_by_id={1: src, 2: tgt},
        ang_vel=0.0,
        comets=[],
        comet_ids=set(),
    )
    assert res is not None
    angle, turns, _, _ = res
    assert math.isfinite(angle)
    assert turns >= 1


# ---------- aim_with_prediction: engine-aligned segment hit ----------
#
# `aim_with_prediction` returns (angle, turns) such that, if the agent fires
# `ships` ships from `src` at `angle`, the engine's continuous-collision
# detection will catch the fleet on or before the returned `turns`. Verified
# by replaying the engine's per-turn point-to-segment check.


def _engine_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return math.hypot(px - projx, py - projy)


_LAUNCH_OFFSET = 0.1


def _engine_replay_hit(
    src: Planet,
    tgt: Planet,
    angle: float,
    ships: int,
    ang_vel: float,
    initial: dict[int, Planet],
    max_turns: int,
) -> tuple[bool, int | None]:
    """Mirror engine per-turn order (fleet move + planet sweep)."""
    speed = fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    offset = src.radius + _LAUNCH_OFFSET
    start_x = src.x + cos_a * offset
    start_y = src.y + sin_a * offset
    fx_prev, fy_prev = start_x, start_y
    for t in range(1, max_turns + 1):
        fx = start_x + cos_a * speed * t
        fy = start_y + sin_a * speed * t
        prev_tx, prev_ty = predict_planet_position(tgt, initial, ang_vel, t - 1)
        d = _engine_segment_distance(prev_tx, prev_ty, fx_prev, fy_prev, fx, fy)
        if d < tgt.radius:
            return True, t
        tx, ty = predict_planet_position(tgt, initial, ang_vel, t)
        d2 = _engine_segment_distance(fx, fy, prev_tx, prev_ty, tx, ty)
        if d2 < tgt.radius:
            return True, t
        fx_prev, fy_prev = fx, fy
    return False, None


def test_aim_with_prediction_orbital_low_speed_engine_hits() -> None:
    src = _planet(1, CENTER_X + 45.0, CENTER_Y, radius=2.0)
    tgt = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {1: src, 2: tgt}
    ang_vel = 0.05
    ships = 10

    res = aim_with_prediction(src, tgt, ships, initial, ang_vel, [], set())
    assert res is not None
    angle, turns, _, _ = res
    hit, hit_turn = _engine_replay_hit(
        src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 5
    )
    assert hit, (
        f"engine should hit returned aim (angle={angle:.3f}, turns={turns}, "
        f"hit_turn={hit_turn})"
    )


def test_aim_with_prediction_orbital_high_speed_engine_hits() -> None:
    src = _planet(1, CENTER_X + 45.0, CENTER_Y, radius=2.0)
    tgt = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {1: src, 2: tgt}
    ang_vel = 0.10
    ships = 500

    res = aim_with_prediction(src, tgt, ships, initial, ang_vel, [], set())
    assert res is not None
    angle, turns, _, _ = res
    hit, hit_turn = _engine_replay_hit(
        src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 5
    )
    assert hit, f"high-speed: engine should hit (turns={turns}, hit_turn={hit_turn})"


def test_aim_with_prediction_orbital_fast_orbit_engine_hits() -> None:
    src = _planet(1, CENTER_X + 40.0, CENTER_Y - 10.0, radius=2.0)
    tgt = _planet(2, CENTER_X + 25.0, CENTER_Y, radius=2.0)
    initial = {1: src, 2: tgt}
    ang_vel = 0.20
    ships = 50

    res = aim_with_prediction(src, tgt, ships, initial, ang_vel, [], set())
    assert res is not None
    angle, turns, _, _ = res
    hit, hit_turn = _engine_replay_hit(
        src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 5
    )
    assert hit, f"fast-orbit: engine should hit (turns={turns}, hit_turn={hit_turn})"


def test_aim_with_prediction_returned_turns_is_first_engine_hit() -> None:
    """Returned turns should be the engine's first hit turn (no over-shooting)."""
    src = _planet(1, CENTER_X + 45.0, CENTER_Y, radius=2.0)
    tgt = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=2.0)
    initial = {1: src, 2: tgt}
    ang_vel = 0.05
    ships = 10

    res = aim_with_prediction(src, tgt, ships, initial, ang_vel, [], set())
    assert res is not None
    angle, turns, _, _ = res
    _, hit_turn = _engine_replay_hit(
        src, tgt, angle, ships, ang_vel, initial, max_turns=turns + 5
    )
    assert hit_turn == turns, (
        f"returned turns={turns} should equal first engine hit_turn={hit_turn}"
    )
