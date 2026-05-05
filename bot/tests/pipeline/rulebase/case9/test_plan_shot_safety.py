"""Bug-reproduction tests for WorldModel.plan_shot post-aim safety filter.

Mirrors `tests/pipeline/imitation/case5/test_decoder_safety.py` for the rulebase
side, hooking into `plan_shot` (the single chokepoint used by every mission).
"""

from __future__ import annotations

from pipeline.rulebase.case9.baseline.agent import build_world


def _planet(
    pid: int,
    owner: int,
    x: float,
    y: float,
    *,
    radius: float = 2.0,
    ships: int = 10,
    production: int = 1,
) -> list[float]:
    return [pid, owner, x, y, radius, ships, production]


# ------------------------------------------------------------------ #
# Bug 2: high-angvel orbital target — plan_shot returns None.
# ------------------------------------------------------------------ #


def test_plan_shot_drops_high_angvel_orbital_target() -> None:
    obs = {
        "player": 0,
        "step": 5,
        "planets": [
            _planet(1, 0, 20.0, 10.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 70.0, 50.0, radius=1.0, ships=30, production=4),
        ],
        "fleets": [],
        "angular_velocity": 0.5,  # very high
        "initial_planets": [
            _planet(1, 0, 20.0, 10.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 70.0, 50.0, radius=1.0, ships=30, production=4),
        ],
        "comets": [],
        "comet_planet_ids": [],
    }
    world = build_world(obs)
    aim = world.plan_shot(src_id=1, target_id=2, ships=20)
    assert aim is None, f"high-angvel orbital plan_shot should return None, got {aim}"


# ------------------------------------------------------------------ #
# Regression: a clean static-target shot still succeeds.
# ------------------------------------------------------------------ #


def test_plan_shot_still_returns_static_target_aim() -> None:
    obs = {
        "player": 0,
        "step": 5,
        "planets": [
            _planet(1, 0, 20.0, 15.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 40.0, 15.0, radius=2.0, ships=10, production=4),
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [
            _planet(1, 0, 20.0, 15.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 40.0, 15.0, radius=2.0, ships=10, production=4),
        ],
        "comets": [],
        "comet_planet_ids": [],
    }
    world = build_world(obs)
    aim = world.plan_shot(src_id=1, target_id=2, ships=15)
    assert aim is not None
    angle, turns, ix, iy = aim
    assert turns >= 1


# ------------------------------------------------------------------ #
# Bug 3a: comet target expiring before fleet arrival.
# ------------------------------------------------------------------ #


def test_plan_shot_drops_comet_target_expiring_before_arrival() -> None:
    comet_path = [(60.0, 50.0), (61.0, 50.0), (62.0, 50.0), (63.0, 50.0)]
    obs = {
        "player": 0,
        "step": 5,
        "planets": [
            _planet(1, 0, 10.0, 5.0, radius=2.0, ships=50),  # far south
            _planet(99, 1, 60.0, 50.0, radius=1.0, ships=5),
        ],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [
            _planet(1, 0, 10.0, 5.0, radius=2.0, ships=50),
            _planet(99, 1, 60.0, 50.0, radius=1.0, ships=5),
        ],
        "comets": [
            {
                "planet_ids": [99],
                "paths": [comet_path],
                "path_index": 0,
                "radius": 1.0,
            }
        ],
        "comet_planet_ids": [99],
    }
    world = build_world(obs)
    aim = world.plan_shot(src_id=1, target_id=99, ships=10)
    assert aim is None, f"comet expiring plan_shot should return None, got {aim}"
