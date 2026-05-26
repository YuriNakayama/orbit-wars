"""Verify the extended `state_to_obs` emits all fields the reinforce featurizer reads.

C-1 spike check: ensure the JAX env produces obs dicts that are structurally
compatible with the vendor sim's obs at reset (turn 0) and after a few steps.
This is a structural parity test — value-level parity with the vendor sim is
covered by `test_parity.py`.
"""

from __future__ import annotations

from typing import Any

from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from env.orbit_wars import make_orbit_wars_env


def _vendor_obs(seed: int, turns: int = 0) -> dict[str, Any]:
    env = make_orbit_wars_env(agents=2, seed=seed)
    for _ in range(turns):
        env.step([[], []])
    return dict(env.steps[-1][0]["observation"])


def test_state_to_obs_has_featurizer_keys() -> None:
    state = reset(seed=0, num_agents=2)
    obs = state_to_obs(state, player=0)

    required_keys = {
        "planets",
        "fleets",
        "initial_planets",
        "comet_planet_ids",
        "comets",
        "player",
        "step",
        "angular_velocity",
    }
    missing = required_keys - obs.keys()
    assert not missing, f"obs missing required keys: {missing}"


def test_initial_obs_planets_match_vendor_count() -> None:
    seed = 42
    state = reset(seed=seed, num_agents=2)
    jax_obs = state_to_obs(state, player=0)
    vendor_obs = _vendor_obs(seed, turns=0)

    assert len(jax_obs["planets"]) == len(vendor_obs["planets"]), (
        f"planet count diff: jax={len(jax_obs['planets'])} "
        f"vendor={len(vendor_obs['planets'])}"
    )
    assert len(jax_obs["initial_planets"]) == len(vendor_obs["initial_planets"])
    assert len(jax_obs["fleets"]) == len(vendor_obs["fleets"])
    # No comets at turn 0
    assert jax_obs["comet_planet_ids"] == []
    assert jax_obs["comets"] == []


def test_initial_obs_planet_rows_match_vendor() -> None:
    # JAX env stores xy/radius as float32 while vendor uses Python float
    # (float64). Tolerance covers single-precision representation error.
    FLOAT_TOL = 1e-4

    seed = 7
    state = reset(seed=seed, num_agents=2)
    jax_obs = state_to_obs(state, player=0)
    vendor_obs = _vendor_obs(seed, turns=0)

    j_by_id = {row[0]: row for row in jax_obs["planets"]}
    v_by_id = {row[0]: row for row in vendor_obs["planets"]}
    assert j_by_id.keys() == v_by_id.keys()
    # Columns: [id, owner, x, y, radius, ships, prod]
    # Integer columns must match exactly; float columns (2,3,4) within tol.
    int_cols = (0, 1, 5, 6)
    float_cols = (2, 3, 4)
    for pid in v_by_id:
        for col in int_cols:
            assert j_by_id[pid][col] == v_by_id[pid][col], (
                f"planet pid={pid} col={col} (int): "
                f"jax={j_by_id[pid][col]} vendor={v_by_id[pid][col]}"
            )
        for col in float_cols:
            assert abs(j_by_id[pid][col] - v_by_id[pid][col]) < FLOAT_TOL, (
                f"planet pid={pid} col={col} (float): "
                f"jax={j_by_id[pid][col]} vendor={v_by_id[pid][col]}"
            )
