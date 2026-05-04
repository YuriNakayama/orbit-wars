"""Bug-reproduction tests for rulebase/case5 WorldModel.plan_shot safety.

Mirrors `tests/pipeline/rulebase/case1/test_plan_shot_safety.py` but case5's
build_world takes `(obs, step)` and lives in `agent_full.py`.
"""

from __future__ import annotations

from pipeline.rulebase.case5.baseline.agent_full import build_world


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


def test_plan_shot_drops_high_angvel_orbital_target() -> None:
    obs = {
        "player": 0,
        "step": 5,
        "planets": [
            _planet(1, 0, 20.0, 10.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 70.0, 50.0, radius=1.0, ships=30, production=4),
        ],
        "fleets": [],
        "angular_velocity": 0.5,
        "initial_planets": [
            _planet(1, 0, 20.0, 10.0, radius=2.0, ships=50, production=5),
            _planet(2, 1, 70.0, 50.0, radius=1.0, ships=30, production=4),
        ],
        "comets": [],
        "comet_planet_ids": [],
    }
    world = build_world(obs, step=5)
    aim = world.plan_shot(src_id=1, target_id=2, ships=20)
    assert aim is None, f"high-angvel orbital plan_shot should return None, got {aim}"


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
    world = build_world(obs, step=5)
    aim = world.plan_shot(src_id=1, target_id=2, ships=15)
    assert aim is not None
    angle, turns, ix, iy = aim
    assert turns >= 1
