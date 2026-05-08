"""Pipeline physics must mirror the updated simulator swept-pair hit test."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest


@pytest.mark.parametrize("case", ["case4", "case6", "case7", "case8", "case9"])
def test_first_engine_hit_turn_uses_swept_pair_collision(case: str) -> None:
    physics = import_module(f"pipeline.rulebase.{case}.baseline.core.physics")
    types = import_module(f"pipeline.rulebase.{case}.baseline.core.types")
    planet_cls: Any = types.Planet

    src = planet_cls(99, 0, 5.0, -4.1, 1.0, 100, 0)
    target = planet_cls(7, -1, 0.0, 0.0, 1.0, 10, 0)
    comets = [
        {"planet_ids": [7], "paths": [[[0.0, 0.0], [10.0, 0.0]]], "path_index": 0}
    ]

    hit_turn = physics._first_engine_hit_turn(
        src=src,
        target=target,
        angle=1.5707963267948966,
        ships=1000,
        initial_by_id={src.id: src, target.id: target},
        ang_vel=0.0,
        comets=comets,
        comet_ids={target.id},
        turn_lo=1,
        turn_hi=1,
    )

    assert hit_turn == 1
