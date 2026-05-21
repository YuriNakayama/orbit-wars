"""L0 parity: jax_env.reset vs vendor interpreter initialization."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "simulator" / "python"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from orbit_wars_vendor.orbit_wars import interpreter as vendor_interpreter  # noqa: E402

from jax_env.reset import reset  # noqa: E402


def _vendor_init(seed: int, num_agents: int) -> SimpleNamespace:
    """Run only the vendor init block and return the shared observation."""
    state = [
        SimpleNamespace(observation=SimpleNamespace(), action=None)
        for _ in range(num_agents)
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(episodeSteps=500, shipSpeed=6.0, cometSpeed=4.0),
        info={"seed": seed},
        done=False,
    )
    vendor_interpreter(state, env)
    return state[0].observation


@pytest.mark.parametrize("seed", list(range(100)))
@pytest.mark.parametrize("num_agents", [2, 4])
def test_reset_matches_vendor_init(seed: int, num_agents: int) -> None:
    js = reset(seed, num_agents=num_agents)
    vendor_obs = _vendor_init(seed, num_agents)

    # angular_velocity is stored as float32 (vendor uses Python float = float64);
    # tolerate the rounding gap.
    assert float(js.angular_velocity) == pytest.approx(
        vendor_obs.angular_velocity, rel=1e-6, abs=1e-7
    )

    # planet count matches
    n_valid = int(np.asarray(js.planet_valid).sum())
    assert n_valid == len(vendor_obs.planets)

    # Compare each planet (slot order matches insertion order).
    planet_id = np.asarray(js.planet_id)
    planet_owner = np.asarray(js.planet_owner)
    planet_xy = np.asarray(js.planet_xy)
    planet_radius = np.asarray(js.planet_radius)
    planet_ships = np.asarray(js.planet_ships)
    planet_prod = np.asarray(js.planet_prod)
    for i, vp in enumerate(vendor_obs.planets):
        assert int(planet_id[i]) == vp[0]
        assert int(planet_owner[i]) == vp[1]
        # planet_xy stored as (y, x) — vendor row planet[2]=y, planet[3]=x.
        # float32 vs Python float64 — allow ~1e-4 abs gap (board span is 100.0).
        assert float(planet_xy[i, 0]) == pytest.approx(vp[2], abs=1e-4)
        assert float(planet_xy[i, 1]) == pytest.approx(vp[3], abs=1e-4)
        assert float(planet_radius[i]) == pytest.approx(vp[4], abs=1e-4)
        assert int(planet_ships[i]) == vp[5]
        assert int(planet_prod[i]) == vp[6]
