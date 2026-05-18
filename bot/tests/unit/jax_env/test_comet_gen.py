"""Parity: comet_gen.generate_comet_paths vs vendor."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "python"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    generate_comet_paths as vendor_generate_comet_paths,
)
from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    generate_planets as vendor_generate_planets,
)

from jax_env.comet_gen import generate_comet_paths  # noqa: E402


@pytest.mark.parametrize("seed", list(range(40)))
@pytest.mark.parametrize("spawn_step", [50, 150, 250, 350, 450])
def test_comet_paths_match_vendor(seed: int, spawn_step: int) -> None:
    # Build a vendor-compatible initial state for the call.
    init_rng = random.Random(seed)
    angular_velocity = init_rng.uniform(0.025, 0.05)
    initial_planets = vendor_generate_planets(init_rng)

    comet_rng_a = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
    comet_rng_b = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")

    ours = generate_comet_paths(
        initial_planets, angular_velocity, spawn_step, None, 4.0, rng=comet_rng_a
    )
    theirs = vendor_generate_comet_paths(
        initial_planets, angular_velocity, spawn_step, None, 4.0, rng=comet_rng_b
    )
    assert ours == theirs, f"seed={seed} step={spawn_step}"
