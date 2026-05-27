"""Parity tests: planet_gen.generate_planets vs vendor generate_planets."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "python"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_jax.planet_gen import generate_planets  # noqa: E402
from orbit_wars_vendor.orbit_wars import (  # noqa: E402
    generate_planets as vendor_generate_planets,
)


@pytest.mark.parametrize("seed", list(range(100)))
def test_planet_gen_matches_vendor(seed: int) -> None:
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)
    ours = generate_planets(rng_a)
    theirs = vendor_generate_planets(rng_b)
    assert ours == theirs, (
        f"seed={seed} mismatch: len ours={len(ours)} theirs={len(theirs)}"
    )
