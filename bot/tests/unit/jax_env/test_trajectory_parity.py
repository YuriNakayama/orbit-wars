"""Trajectory-level parity between JAX env and vendor sim.

Two fixes preserved here:

1. **Orbital step offset** (step.py:_compute_planet_new_xy). Pre-fix the JAX
   env used `state.step + 1` for the rotation argument; vendor uses
   `obs0.step` which equals `state.step` at that point. The off-by-one
   produced ~50-unit drift within 5 steps.

2. **Per-quadrant comet xy swap** (step.py, same function). Vendor builds
   the four rotationally symmetric comet paths with asymmetric column
   ordering: q=0 stores `[y, x]`, q=1/q=2 already in `(x, y)`-like layout,
   q=3 stores `[BS-y, BS-x]`. Rust mirrors this and the BC weights were
   trained on Rust obs, so JAX must apply the per-quadrant swap (q=0 / q=3
   only) when writing comet positions into `planet_xy`.

This file pins both properties via a no-op-action trajectory across multiple
seeds, including past the first comet spawn at step 50.

NOTE: For some seeds Rust and Python vendor diverge in *which* rejection
sampling attempt succeeds during `generate_comet_paths`, producing different
(but each individually valid) comet paths. JAX matches Python vendor's RNG
path, so for those seeds the comet planet positions will differ from Rust
sim's obs even with all swaps applied correctly. We exclude such seeds from
the deep-horizon assertion.
"""

from __future__ import annotations

from typing import Any

import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step
from orbit_wars_sim import make_orbit_wars_env

# Float32 representation error budget for planet xy after N steps.
# Empirically <1e-4 across 5 seeds × 49 steps (no comets yet).
PRE_COMET_TOL = 5e-4


def _run_both(seed: int, turns: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run JAX env + vendor sim with no-op actions and return final obs."""
    js = reset(seed=seed, num_agents=2)
    env = make_orbit_wars_env(agents=2, seed=seed)
    ea = empty_actions()
    for _ in range(turns):
        js, _, _ = step(js, ea)
        env.step([[], []])
    jax_obs = state_to_obs(js, player=0)
    vendor_obs = dict(env.steps[-1][0]["observation"])
    return jax_obs, vendor_obs


@pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
@pytest.mark.parametrize("turns", [1, 10, 25, 49])
def test_planet_xy_pre_comet(seed: int, turns: int) -> None:
    """Planet xy stays within float32 precision of vendor before comet spawn."""
    jax_obs, vendor_obs = _run_both(seed, turns)
    j_by_id = {row[0]: row for row in jax_obs["planets"]}
    v_by_id = {row[0]: row for row in vendor_obs["planets"]}
    assert j_by_id.keys() == v_by_id.keys(), (
        f"seed={seed} turns={turns}: planet id sets differ"
    )
    for pid in v_by_id:
        for col in (2, 3):
            diff = abs(j_by_id[pid][col] - v_by_id[pid][col])
            assert diff < PRE_COMET_TOL, (
                f"seed={seed} turns={turns} pid={pid} col={col}: "
                f"diff={diff:.3e} (tol={PRE_COMET_TOL})"
            )


# Seeds where Rust sim picks the same comet rejection attempt as Python
# vendor (so JAX matches Rust exactly through and past the first comet
# spawn). Other seeds diverge from Rust at comet generation due to Rust's
# RNG differing from vendor; JAX matches vendor by design.
_COMET_PARITY_SEEDS = [0, 7]
_COMET_PARITY_TOL = 1e-4


@pytest.mark.parametrize("seed", _COMET_PARITY_SEEDS)
@pytest.mark.parametrize("turns", [50, 60, 80, 100])
def test_planet_xy_first_comet_phase(seed: int, turns: int) -> None:
    """Trajectory parity through and past the first comet spawn (step 50).

    Pins both the orbital-step fix and the per-quadrant comet swap. Limited
    to seeds where Rust matches Python vendor in comet rejection sampling.
    """
    jax_obs, vendor_obs = _run_both(seed, turns)
    j_by_id = {row[0]: row for row in jax_obs["planets"]}
    v_by_id = {row[0]: row for row in vendor_obs["planets"]}
    assert j_by_id.keys() == v_by_id.keys(), (
        f"seed={seed} turns={turns}: planet id sets differ"
    )
    for pid in v_by_id:
        for col in (2, 3):
            diff = abs(j_by_id[pid][col] - v_by_id[pid][col])
            assert diff < _COMET_PARITY_TOL, (
                f"seed={seed} turns={turns} pid={pid} col={col}: "
                f"diff={diff:.3e} (tol={_COMET_PARITY_TOL})"
            )
