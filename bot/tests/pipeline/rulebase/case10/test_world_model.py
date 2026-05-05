"""Unit tests for case10 `WorldModel` construction from a synthetic observation."""

from __future__ import annotations

from pipeline.rulebase.case10.baseline.agent import build_world
from pipeline.rulebase.case10.baseline.core.types import Planet


def _planet(
    pid: int, owner: int, x: float, y: float, ships: int = 10, prod: int = 1
) -> list[int | float]:
    return [pid, owner, x, y, 3.0, ships, prod]


def test_build_world_basic_two_player() -> None:
    obs = {
        "player": 0,
        "step": 10,
        "planets": [
            _planet(0, 0, 20.0, 50.0, ships=15, prod=2),
            _planet(1, 1, 80.0, 50.0, ships=15, prod=2),
            _planet(2, -1, 50.0, 30.0, ships=5, prod=1),
        ],
        "fleets": [
            [100, 0, 30.0, 50.0, 0.0, 0, 3],
        ],
        "angular_velocity": 0.0,
        "initial_planets": [
            _planet(0, 0, 20.0, 50.0, ships=10, prod=2),
            _planet(1, 1, 80.0, 50.0, ships=10, prod=2),
            _planet(2, -1, 50.0, 30.0, ships=5, prod=1),
        ],
        "comets": [],
        "comet_planet_ids": [],
    }

    world = build_world(obs)

    assert world.player == 0
    assert world.step == 10
    assert {p.id for p in world.my_planets} == {0}
    assert {p.id for p in world.enemy_planets} == {1}
    assert {p.id for p in world.neutral_planets} == {2}
    assert isinstance(world.my_planets[0], Planet)
    assert world.my_total > 0
    assert world.enemy_total > 0


def test_build_world_empty_observation() -> None:
    obs = {
        "player": 0,
        "step": 0,
        "planets": [],
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": [],
        "comets": [],
        "comet_planet_ids": [],
    }
    world = build_world(obs)
    assert world.my_planets == []
    assert world.enemy_planets == []
    assert world.neutral_planets == []
