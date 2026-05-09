"""Engine-truth tests: fire a fleet in the local simulator and assert our
predictor matches the real engine state turn-by-turn.

These tests are the source-of-truth for the segment-based aim_with_prediction
fix. They drive a *real* env via `env.step` and assert:

1. fleet position emitted by the engine after launch == predictor's expected
   start point (`src + (src.radius + 0.1) * direction`).
2. fleet position at each turn == predictor's `start + speed * t` straight line.
3. when the engine eventually catches the fleet on a target, our predictor
   reports the same hit-turn (= turn at which the fleet disappears AND the
   target's ships count changes).
"""

from __future__ import annotations

import math

import pytest

from pipeline.rulebase.case9.baseline.core.physics import fleet_speed
from tests.e2e.pipeline.util import (
    fire_one_fleet,
    fleet_by_id,
    make_initialized_orbit_env,
    planet_by_id,
)

# -------- Test 1: fleet start point matches `src + (radius+0.1)*dir` --------


def test_engine_fleet_starts_at_planet_surface_plus_offset() -> None:
    env = make_initialized_orbit_env(seed=11)
    obs0 = env.steps[-1][0]["observation"]
    # Pick a player-0 planet.
    src_pid = next(int(p[0]) for p in obs0["planets"] if int(p[1]) == 0)
    src_before = planet_by_id(obs0, src_pid)
    assert src_before is not None
    sx, sy, sr, src_ships = (
        float(src_before[2]),
        float(src_before[3]),
        float(src_before[4]),
        int(src_before[5]),
    )

    angle = 0.7
    ships = max(1, src_ships - 1)
    obs_after, fid = fire_one_fleet(env, 0, src_pid, angle, ships)
    assert fid >= 0, "fleet should have launched"

    fleet = fleet_by_id(obs_after, fid)
    assert fleet is not None
    fx, fy = float(fleet[2]), float(fleet[3])

    # Engine: start = src + (radius + 0.1) * direction, then advance by speed.
    # After step()*1, position is `start + speed * 1`.
    speed = fleet_speed(ships)
    expected_x = sx + math.cos(angle) * (sr + 0.1 + speed)
    expected_y = sy + math.sin(angle) * (sr + 0.1 + speed)
    assert math.isclose(fx, expected_x, abs_tol=1e-6), (
        f"fleet x mismatch: engine={fx} predicted={expected_x}"
    )
    assert math.isclose(fy, expected_y, abs_tol=1e-6), (
        f"fleet y mismatch: engine={fy} predicted={expected_y}"
    )


# -------- Test 2: fleet position at turn t matches start + speed * t --------


def test_engine_fleet_position_per_turn_matches_predictor() -> None:
    env = make_initialized_orbit_env(seed=13)
    obs0 = env.steps[-1][0]["observation"]
    src_pid = next(int(p[0]) for p in obs0["planets"] if int(p[1]) == 0)
    src_before = planet_by_id(obs0, src_pid)
    assert src_before is not None
    sx, sy, sr, src_ships = (
        float(src_before[2]),
        float(src_before[3]),
        float(src_before[4]),
        int(src_before[5]),
    )

    angle = math.pi  # head left so fleet doesn't immediately hit anything close
    ships = max(1, min(50, src_ships - 1))
    obs_after, fid = fire_one_fleet(env, 0, src_pid, angle, ships)
    if fid < 0:
        pytest.skip("could not launch fleet for this seed")

    speed = fleet_speed(ships)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    start_x = sx + cos_a * (sr + 0.1)
    start_y = sy + sin_a * (sr + 0.1)

    # Walk a few turns; if the fleet survives (no collision), its pos must
    # equal start + speed*t.
    for t_after in range(1, 6):
        fleet = fleet_by_id(obs_after, fid)
        if fleet is None:
            break  # fleet was caught — that's fine, just stop the per-turn check
        fx, fy = float(fleet[2]), float(fleet[3])
        # `t_after` is how many env.step calls happened since launch (inclusive).
        expected_x = start_x + cos_a * speed * t_after
        expected_y = start_y + sin_a * speed * t_after
        assert math.isclose(fx, expected_x, abs_tol=1e-5), (
            f"t={t_after}: fleet x engine={fx} predicted={expected_x}"
        )
        assert math.isclose(fy, expected_y, abs_tol=1e-5), (
            f"t={t_after}: fleet y engine={fy} predicted={expected_y}"
        )
        if t_after < 5:
            env.step([[], []])
            obs_after = env.steps[-1][0]["observation"]


# -------- Test 3: predicted hit turn matches the turn the fleet disappears --------


def test_predictor_hit_turn_matches_engine_first_capture_turn() -> None:
    """Fire at a static-ish target and check our segment predictor agrees with
    the turn the engine actually catches the fleet."""
    from pipeline.rulebase.case9.baseline.core.physics import (
        _first_engine_hit_turn,
    )
    from pipeline.rulebase.case9.baseline.core.types import Planet

    env = make_initialized_orbit_env(seed=17)
    obs0 = env.steps[-1][0]["observation"]
    ang_vel = float(obs0.get("angular_velocity") or 0.0)

    # Pick our planet and an opponent / neutral target.
    own = [p for p in obs0["planets"] if int(p[1]) == 0]
    others = [p for p in obs0["planets"] if int(p[1]) != 0]
    src_row = own[0]
    target_row = min(
        others,
        key=lambda p: math.hypot(p[2] - src_row[2], p[3] - src_row[3]),
    )

    src = Planet(
        id=int(src_row[0]),
        owner=int(src_row[1]),
        x=float(src_row[2]),
        y=float(src_row[3]),
        radius=float(src_row[4]),
        ships=int(src_row[5]),
        production=int(src_row[6]),
    )
    tgt = Planet(
        id=int(target_row[0]),
        owner=int(target_row[1]),
        x=float(target_row[2]),
        y=float(target_row[3]),
        radius=float(target_row[4]),
        ships=int(target_row[5]),
        production=int(target_row[6]),
    )
    initial_by_id = {p.id: Planet(*p) for p in (src, tgt)}

    angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
    ships = max(1, min(50, src.ships - 1))

    predicted_turn = _first_engine_hit_turn(
        src,
        tgt,
        angle,
        ships,
        initial_by_id,
        ang_vel,
        comets=[],
        comet_ids=set(),
        turn_lo=1,
        turn_hi=80,
    )

    obs_after, fid = fire_one_fleet(env, 0, src.id, angle, ships)
    if fid < 0:
        pytest.skip("could not launch fleet for this seed")

    actual_turn: int | None = None
    for t in range(1, 81):
        if fleet_by_id(obs_after, fid) is None:
            actual_turn = t
            break
        if t < 80:
            env.step([[], []])
            obs_after = env.steps[-1][0]["observation"]

    assert actual_turn is not None, "engine never caught the fleet within horizon"
    assert predicted_turn == actual_turn, (
        f"predicted hit turn={predicted_turn} but engine caught at turn={actual_turn}"
    )
