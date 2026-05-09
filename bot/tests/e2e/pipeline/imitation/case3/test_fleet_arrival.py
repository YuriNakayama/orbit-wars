"""Env-in-the-loop fleet arrival regression tests."""

from __future__ import annotations

import math
from typing import Any

import pytest

from pipeline.imitation.case3.policy.geometry import (
    CENTER_X,
    CENTER_Y,
    ROTATION_LIMIT,
    aim_with_prediction,
)
from tests.e2e.pipeline.util import make_initialized_orbit_env

# ---------- Case B: env-in-the-loop ----------


def _is_orbiting(p: list[Any]) -> bool:
    r = math.hypot(float(p[2]) - CENTER_X, float(p[3]) - CENTER_Y)
    return bool(r + float(p[4]) < ROTATION_LIMIT)


def _init_env_with_seed(seed: int) -> tuple[Any, dict[str, Any]] | None:
    env = make_initialized_orbit_env(seed=seed)
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
