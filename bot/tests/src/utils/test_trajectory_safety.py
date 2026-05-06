"""Unit tests for src.utils.trajectory_safety.

These checks live alongside `aim_with_prediction` (in each case's geometry/physics)
and are responsible for catching three classes of post-model bugs that the
imitation/rulebase decoder otherwise lets through:

1. Bug 1 — fleet trajectory grazes the sun within INTERCEPT_TOLERANCE.
2. Bug 2 — orbital target leaves its radius across the ±tolerance window.
3. Bug 3 — comet expiry / non-target comet crossing / appearance lookahead.

The module is pure (no kaggle_environments dependency) and shared via bot/src/utils/.
"""

from __future__ import annotations

from src.utils.orbit_constants import (
    CENTER_X,
    CENTER_Y,
    SUN_R,
    SUN_SAFETY,
)
from src.utils.trajectory_safety import (
    Planet,
    comet_appearance_imminent,
    fleet_crosses_other_comet,
    intercept_holds_within_tolerance,
    is_trajectory_sun_safe,
    target_reachable_before_comet_expiry,
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


# ------------------------------------------------------------------ #
# is_trajectory_sun_safe
# ------------------------------------------------------------------ #


def test_sun_safe_path_clear_returns_true() -> None:
    # Launch from (10, 10) heading away from the sun (toward south-west) -> safe.
    assert is_trajectory_sun_safe(
        launch_x=10.0,
        launch_y=10.0,
        angle=-2.356,  # south-west, away from center
        turns=10,
        ships=5,
    )


def test_sun_safe_direct_hit_returns_false() -> None:
    # Launch from the left edge heading straight at the sun (centre = 50, 50).
    assert not is_trajectory_sun_safe(
        launch_x=20.0,
        launch_y=CENTER_Y,
        angle=0.0,  # heading east, straight through (50, 50)
        turns=20,
        ships=10,
    )


def test_sun_safe_zero_turns_returns_true_outside_safety() -> None:
    # turns=0 => only check the launch point. Far from sun => safe.
    assert is_trajectory_sun_safe(
        launch_x=10.0, launch_y=10.0, angle=0.0, turns=0, ships=1
    )


def test_sun_safe_path_grazes_within_safety_returns_false() -> None:
    # Trajectory passes within SUN_SAFETY of the sun's edge.
    # Source (CENTER_X-30, CENTER_Y - SUN_R - 1.0) heading east => path is 1.0
    # above the sun edge, inside (SUN_R + SUN_SAFETY=11.5) - SUN_R = 1.5 buffer.
    assert not is_trajectory_sun_safe(
        launch_x=CENTER_X - 30.0,
        launch_y=CENTER_Y - SUN_R - 1.0,
        angle=0.0,
        turns=20,
        ships=10,
    )


# ------------------------------------------------------------------ #
# intercept_holds_within_tolerance
# ------------------------------------------------------------------ #


def test_intercept_static_target_holds() -> None:
    # Static planet (far from center, beyond ROTATION_LIMIT) does not move.
    target = _planet(1, CENTER_X + 60.0, CENTER_Y, radius=3.0)
    assert intercept_holds_within_tolerance(
        target=target,
        predicted_turns=5,
        predicted_pos=(CENTER_X + 60.0, CENTER_Y),
        initial_by_id={1: target},
        ang_vel=0.5,  # static planet ignores ang_vel
        comets=[],
        comet_ids=set(),
    )


def test_intercept_high_angvel_orbital_target_misses() -> None:
    # Orbital target with very high ang_vel: predicted_pos at turn k differs
    # significantly from positions at k-1 / k+1 -> miss.
    initial = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=1.0)
    # Predicted intercept at turn 5 -> use the actual prediction as predicted_pos.
    # With ang_vel=0.5 rad/turn, target moves ~10 units over 5 turns.
    # Test: pin predicted_pos at the target's CURRENT position; with high ang_vel
    # the actual position at turn 5 will be far away => fail.
    assert not intercept_holds_within_tolerance(
        target=initial,
        predicted_turns=5,
        predicted_pos=(initial.x, initial.y),  # WRONG: stale current pos
        initial_by_id={2: initial},
        ang_vel=0.5,
        comets=[],
        comet_ids=set(),
        miss_radius_factor=0.7,
    )


def test_intercept_low_angvel_orbital_target_holds() -> None:
    # Same orbital target but ang_vel small enough that movement < radius * 0.7
    # across the ±1 turn window centered on predicted_turns.
    initial = _planet(2, CENTER_X + 20.0, CENTER_Y, radius=3.0)
    # ang_vel=0.01 -> motion across ±1 turn = 20 * 0.02 = 0.4 < 3 * 0.7 = 2.1
    # predicted_pos = position at turn 5 with this ang_vel.
    import math

    cur_ang = 0.0
    new_ang = cur_ang + 0.01 * 5
    px = CENTER_X + 20.0 * math.cos(new_ang)
    py = CENTER_Y + 20.0 * math.sin(new_ang)
    assert intercept_holds_within_tolerance(
        target=initial,
        predicted_turns=5,
        predicted_pos=(px, py),
        initial_by_id={2: initial},
        ang_vel=0.01,
        comets=[],
        comet_ids=set(),
        miss_radius_factor=0.7,
    )


# ------------------------------------------------------------------ #
# fleet_crosses_other_comet
# ------------------------------------------------------------------ #


def test_fleet_crosses_comet_on_collision_path_returns_true() -> None:
    # Comet path passes through the fleet's trajectory at turn ~3.
    # Fleet launches at (10, 50) east, ships=1 => speed 1.0 => turn 3 at (13, 50).
    comets = [
        {
            "planet_ids": [99],
            "paths": [[(20.0, 50.0), (17.0, 50.0), (14.0, 50.0), (13.0, 50.0)]],
            "path_index": 0,
            "radius": 2.0,
        }
    ]
    assert fleet_crosses_other_comet(
        launch_x=10.0,
        launch_y=50.0,
        angle=0.0,  # east
        turns=5,
        ships=1,
        current_step=0,
        comets=comets,
        exclude_planet_id=-1,
    )


def test_fleet_crosses_other_comet_off_path_returns_false() -> None:
    # Comet runs in a far-away quadrant; fleet trajectory doesn't approach.
    comets = [
        {
            "planet_ids": [99],
            "paths": [[(90.0, 90.0), (88.0, 88.0), (85.0, 85.0), (80.0, 80.0)]],
            "path_index": 0,
            "radius": 2.0,
        }
    ]
    assert not fleet_crosses_other_comet(
        launch_x=10.0,
        launch_y=10.0,
        angle=-2.356,
        turns=10,
        ships=5,
        current_step=0,
        comets=comets,
        exclude_planet_id=-1,
    )


def test_fleet_crosses_other_comet_excludes_target_planet() -> None:
    # The collision-path comet IS the target -> exclude_planet_id silences check.
    comets = [
        {
            "planet_ids": [99],
            "paths": [[(20.0, 50.0), (17.0, 50.0), (14.0, 50.0), (13.0, 50.0)]],
            "path_index": 0,
            "radius": 2.0,
        }
    ]
    assert not fleet_crosses_other_comet(
        launch_x=10.0,
        launch_y=50.0,
        angle=0.0,
        turns=5,
        ships=1,
        current_step=0,
        comets=comets,
        exclude_planet_id=99,
    )


# ------------------------------------------------------------------ #
# target_reachable_before_comet_expiry
# ------------------------------------------------------------------ #


def test_comet_expiry_sufficient_life_returns_true() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(0.0, 0.0)] * 10],
            "path_index": 0,
        }
    ]
    assert target_reachable_before_comet_expiry(
        target_id=42, predicted_turns=5, comets=comets, safety_margin=1
    )


def test_comet_expiry_insufficient_life_returns_false() -> None:
    comets = [
        {
            "planet_ids": [42],
            "paths": [[(0.0, 0.0)] * 4],
            "path_index": 0,
        }
    ]
    assert not target_reachable_before_comet_expiry(
        target_id=42, predicted_turns=5, comets=comets, safety_margin=1
    )


def test_comet_expiry_non_comet_target_returns_true() -> None:
    # Empty comets -> always reachable.
    assert target_reachable_before_comet_expiry(
        target_id=10, predicted_turns=5, comets=[], safety_margin=1
    )


# ------------------------------------------------------------------ #
# comet_appearance_imminent
# ------------------------------------------------------------------ #


def test_comet_appearance_imminent_in_window() -> None:
    # step=48, predicted_turns=10, buffer=5 -> window [48, 63] catches 50.
    assert comet_appearance_imminent(current_step=48, predicted_turns=10) == 50


def test_comet_appearance_not_imminent() -> None:
    # step=60 already past 50, far from 150 -> None.
    assert comet_appearance_imminent(current_step=60, predicted_turns=10) is None


def test_comet_appearance_caught_by_buffer() -> None:
    # step=44, predicted=2 -> normal window [44, 46] doesn't reach 50.
    # With look_ahead_buffer=5 -> [44, 51] catches 50.
    assert comet_appearance_imminent(current_step=44, predicted_turns=2) == 50


def test_sun_safety_constants_sane() -> None:
    assert SUN_R > 0
    assert SUN_SAFETY > 0
    assert CENTER_X == 50.0
    assert CENTER_Y == 50.0
