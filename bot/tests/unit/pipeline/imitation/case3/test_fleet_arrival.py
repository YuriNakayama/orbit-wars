"""Reproduction tests for "ships が planet に到達しない" cases.

Env-free self-consistency failure detection:

* Case A — env-free self-consistency: ``aim_with_prediction`` returns
  ``(angle, turns, hit_x, hit_y)``. We re-walk the fleet from the env's
  launch point at ``fleet_speed(ships) * turns`` along ``angle`` and check
  the resulting position against ``predict_planet_position(target, turns)``.
  If the aim solver is internally consistent the gap must not exceed
  ``target.radius`` (a hit by the env's continuous-collision rule).
"""

from __future__ import annotations

import math

from pipeline.imitation.case3.policy.geometry import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    ROTATION_LIMIT,
    Planet,
    aim_with_prediction,
    fleet_speed,
    predict_planet_position,
)


def _planet(
    pid: int,
    x: float,
    y: float,
    *,
    owner: int = 0,
    radius: float = 2.0,
    ships: int = 50,
    production: int = 2,
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


# ---------- Case A: env-free self-consistency ----------


def test_aim_self_consistency_orbiting_target() -> None:
    """The fleet, walked at fleet_speed(ships)*turns from the env launch
    point along the aim angle, must end up within ``target.radius`` of the
    target's predicted position at ``turn=turns``.

    This isolates the policy-side aim/predict pipeline from the env: any
    gap is a self-inconsistency in geometry.aim_with_prediction itself.
    """

    src = _planet(1, x=20.0, y=80.0, ships=200)

    orbital_r = 25.0
    init_ang = math.radians(45.0)
    target_init = _planet(
        2,
        x=CENTER_X + orbital_r * math.cos(init_ang),
        y=CENTER_Y + orbital_r * math.sin(init_ang),
        owner=-1,
        ships=10,
    )
    assert orbital_r + target_init.radius < ROTATION_LIMIT, (
        "target must be in the rotating band for this test to be meaningful"
    )

    ang_vel = 0.04
    elapsed_steps = 7
    cur_ang = init_ang + ang_vel * elapsed_steps
    target_now = target_init._replace(
        x=CENTER_X + orbital_r * math.cos(cur_ang),
        y=CENTER_Y + orbital_r * math.sin(cur_ang),
    )

    initial_by_id = {target_init.id: target_init}
    ships = 100

    aim = aim_with_prediction(
        src,
        target_now,
        ships,
        initial_by_id,
        ang_vel,
        comets=[],
        comet_ids=set(),
    )
    assert aim is not None, "aim solver must find a solution for this geometry"

    angle, turns, hit_x, hit_y = aim

    # Same launch point formula as the local simulator orbit_wars env
    # (orbit_wars.py L496-497): start = planet pos + (radius + 0.1) * unit(angle).
    start_x = src.x + math.cos(angle) * (src.radius + LAUNCH_CLEARANCE)
    start_y = src.y + math.sin(angle) * (src.radius + LAUNCH_CLEARANCE)

    speed = fleet_speed(ships)
    fleet_end_x = start_x + math.cos(angle) * speed * turns
    fleet_end_y = start_y + math.sin(angle) * speed * turns

    target_pred_x, target_pred_y = predict_planet_position(
        target_now, initial_by_id, ang_vel, turns
    )

    miss = math.hypot(fleet_end_x - target_pred_x, fleet_end_y - target_pred_y)
    assert miss <= target_now.radius, (
        f"aim self-inconsistent: fleet at turn={turns} is {miss:.3f} units from "
        f"predicted target position (radius={target_now.radius}); "
        f"hit_xy=({hit_x:.3f},{hit_y:.3f}) vs predicted=({target_pred_x:.3f},{target_pred_y:.3f})"
    )
