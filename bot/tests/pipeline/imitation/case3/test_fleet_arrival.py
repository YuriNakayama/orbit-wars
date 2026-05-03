"""Reproduction tests for "ships が planet に到達しない" cases.

Two layers of failure detection:

* Case A — env-free self-consistency: ``aim_with_prediction`` returns
  ``(angle, turns, hit_x, hit_y)``. We re-walk the fleet from the env's
  launch point at ``fleet_speed(ships) * turns`` along ``angle`` and check
  the resulting position against ``predict_planet_position(target, turns)``.
  If the aim solver is internally consistent the gap must not exceed
  ``target.radius`` (a hit by the env's continuous-collision rule).

* Case B — env-in-the-loop: drive a real ``kaggle_environments`` orbit_wars
  match for a few turns, fire one fleet via the same decode path, and
  assert the fleet is consumed at the target planet (combat resolution
  changed the planet owner / ship count).

Both cases focus on orbiting (rotating) targets, where the half-step
phase mismatch between policy prediction and env collision detection is
expected to surface.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest
from kaggle_environments import make

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

    # Same launch point formula as kaggle_environments orbit_wars env
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


# ---------- Case B: env-in-the-loop ----------


def _is_orbiting(p: list[Any]) -> bool:
    r = math.hypot(float(p[2]) - CENTER_X, float(p[3]) - CENTER_Y)
    return bool(r + float(p[4]) < ROTATION_LIMIT)


def _init_env_with_seed(seed: int) -> tuple[Any, dict[str, Any]] | None:
    random.seed(seed)
    env = make("orbit_wars", debug=True)
    env.reset(num_agents=2)
    # Orbit Wars env populates planets on the first step(); a no-op action
    # does not get rejected, so we use it to trigger init.
    env.step([[], []])
    obs = env.state[0]["observation"]
    if not obs.get("planets"):
        return None
    return env, obs


def _simulate_single_shot(
    seed: int, src_pid: int, target_pid: int
) -> dict[str, Any] | None:
    """Replay one shot in a fresh env and return diagnostics, or None if
    the seed/pair is unusable.
    """
    from pipeline.imitation.case3.policy.decoder import _build_planet

    bundle = _init_env_with_seed(seed)
    if bundle is None:
        return None
    env, obs0 = bundle
    src_row = next((p for p in obs0["planets"] if p[0] == src_pid), None)
    tgt_row = next((p for p in obs0["planets"] if p[0] == target_pid), None)
    if src_row is None or tgt_row is None or int(src_row[1]) != 0:
        return None
    src = _build_planet(src_row)
    target = _build_planet(tgt_row)
    if src.ships < 5:
        return None

    ships = max(1, src.ships // 2)
    initial_by_id = {
        int(row[0]): _build_planet(row) for row in obs0.get("initial_planets") or []
    }
    ang_vel = float(obs0.get("angular_velocity", 0.0) or 0.0)
    aim = aim_with_prediction(src, target, ships, initial_by_id, ang_vel, [], set())
    if aim is None:
        return None
    angle, aim_turns, _, _ = aim

    owner_before = int(tgt_row[1])

    env.step([[[src_pid, float(angle), int(ships)]], []])
    fleet_seen = False
    consumed_turn = -1
    last_pos: tuple[float, float] | None = None

    for t in range(aim_turns + 5):
        post = env.state[0]["observation"]
        fleets = post.get("fleets") or []
        my_fleet = next(
            (f for f in fleets if int(f[1]) == 0 and int(f[5]) == src_pid),
            None,
        )
        if my_fleet is not None:
            fleet_seen = True
            last_pos = (float(my_fleet[2]), float(my_fleet[3]))
        elif fleet_seen and consumed_turn < 0:
            consumed_turn = t
            break
        env.step([[], []])

    final = env.state[0]["observation"]
    tgt_after = next((p for p in final["planets"] if p[0] == target_pid), None)
    owner_changed = tgt_after is not None and int(tgt_after[1]) != owner_before

    return {
        "seed": seed,
        "src_pid": src_pid,
        "target_pid": target_pid,
        "ships": ships,
        "angle": angle,
        "aim_turns": aim_turns,
        "fleet_seen": fleet_seen,
        "consumed_turn": consumed_turn,
        "owner_changed": owner_changed,
        "last_pos": last_pos,
        "target_xy_before": (target.x, target.y),
    }


def test_fleet_actually_hits_orbiting_target() -> None:
    """Sweep many (seed, src, target) pairs and fail if any single shot
    clearly misses its orbiting target.

    A clear miss is defined as:
    * the fleet never disappeared within ``aim_turns + 5`` env steps, AND
    * the target planet's owner did not change.

    This formulation matches the user's framing — "ships が planet に到達して
    いない (外れている) ケースがあります" — i.e. one or more failure cases
    is enough to flag the bug.
    """
    misses: list[dict[str, Any]] = []
    attempts = 0

    for seed in range(15):
        bundle = _init_env_with_seed(seed)
        if bundle is None:
            continue
        _, obs0 = bundle
        my = [p for p in obs0["planets"] if p[1] == 0]
        for src_row in my[:3]:
            src_pid = int(src_row[0])
            targets = [
                p
                for p in obs0["planets"]
                if int(p[0]) != src_pid and int(p[1]) != 0 and _is_orbiting(p)
            ]
            targets.sort(key=lambda p: math.hypot(p[2] - src_row[2], p[3] - src_row[3]))
            for tgt_row in targets[:3]:
                target_pid = int(tgt_row[0])
                diag = _simulate_single_shot(seed, src_pid, target_pid)
                if diag is None:
                    continue
                attempts += 1
                miss = (
                    diag["fleet_seen"]
                    and diag["consumed_turn"] < 0
                    and not diag["owner_changed"]
                )
                if miss:
                    misses.append(diag)

    assert attempts > 0, "no shots were attempted — env init likely broken"
    if misses:
        first = misses[0]
        pytest.fail(
            f"Found {len(misses)}/{attempts} shots where the fleet failed to "
            f"reach its orbiting target. First miss: "
            f"seed={first['seed']} src={first['src_pid']} target={first['target_pid']} "
            f"ships={first['ships']} angle={first['angle']:.4f} "
            f"aim_turns={first['aim_turns']} last_pos={first['last_pos']} "
            f"target_xy_before={first['target_xy_before']}"
        )
