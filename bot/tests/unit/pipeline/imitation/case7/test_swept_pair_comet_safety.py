"""Case 7 comet safety mirrors swept-pair simulator collision checks."""

from __future__ import annotations

from pipeline.imitation.case7.policy import safety


def test_fleet_crosses_other_comet_uses_swept_pair_collision() -> None:
    comets = [
        {"planet_ids": [7], "paths": [[[0.0, 0.0], [10.0, 0.0]]], "path_index": 0}
    ]

    crosses = safety.fleet_crosses_other_comet(
        launch_x=5.0,
        launch_y=-4.1,
        angle=1.5707963267948966,
        turns=1,
        ships=1000,
        current_step=0,
        comets=comets,
        exclude_planet_id=-1,
        safety=0.0,
    )

    assert crosses is True
