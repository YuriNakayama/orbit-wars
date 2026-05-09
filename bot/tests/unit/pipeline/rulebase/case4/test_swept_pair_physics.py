"""Case 4 physics mirrors the simulator swept-pair hit test."""

from __future__ import annotations

from pipeline.rulebase.case4.baseline.core import physics
from pipeline.rulebase.case4.baseline.core.types import Planet


def test_first_engine_hit_turn_uses_swept_pair_collision() -> None:
    src = Planet(99, 0, 5.0, -4.1, 1.0, 100, 0)
    target = Planet(7, -1, 0.0, 0.0, 1.0, 10, 0)
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
