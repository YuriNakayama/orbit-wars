"""Imitation comet safety mirrors swept-pair simulator collision checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

IMITATION_CASES = [
    "case1",
    "case2",
    "case3",
    "case4",
    "case5",
    "case6",
    "case7",
    "case8",
]


BOT_ROOT = Path(__file__).resolve().parents[3]  # bot/


@pytest.mark.parametrize("case", IMITATION_CASES)
def test_fleet_crosses_other_comet_uses_swept_pair_collision(case: str) -> None:
    policy_dir = BOT_ROOT / "pipeline" / "imitation" / case / "policy"
    package_name = f"_test_imitation_{case}_policy"
    package = ModuleType(package_name)
    package.__path__ = [str(policy_dir)]
    import sys

    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.safety", policy_dir / "safety.py"
    )
    assert spec is not None and spec.loader is not None
    safety = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = safety
    spec.loader.exec_module(safety)
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
