"""Smoke tests for step() — compiles, no-op runs 500 steps, no NaNs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step


def test_step_compiles_and_runs_no_op() -> None:
    state = reset(seed=0, num_agents=2)
    actions = empty_actions()
    new_state, rewards, terminated = step(state, actions)
    assert int(new_state.step) == 1
    assert bool(terminated) is False
    # All planet_ships >= 0
    valid_ships = np.asarray(new_state.planet_ships)[np.asarray(new_state.planet_valid)]
    assert (valid_ships >= 0).all()


def test_step_run_500_steps_no_op() -> None:
    state = reset(seed=0, num_agents=2)
    actions = empty_actions()
    for _ in range(500):
        state, _, term = step(state, actions)
        if bool(term):
            break
    # Either we terminated by alive_count (likely, since one home planet each
    # and no actions doesn't change much) or hit step cap. Either way no NaN.
    assert not jnp.any(jnp.isnan(state.planet_xy))
    assert not jnp.any(jnp.isnan(state.fleet_xy))


def test_step_with_launch_action() -> None:
    state = reset(seed=0, num_agents=2)
    # Find player 0's home planet.
    owners = np.asarray(state.planet_owner)
    home_idx = int(np.where(owners == 0)[0][0])
    home_id = int(state.planet_id[home_idx])
    home_ships = int(state.planet_ships[home_idx])

    # Build action: agent 0 launches 5 ships from home, angle 0.
    from orbit_wars_jax.constants import NUM_AGENTS_MAX
    from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT

    actions = np.full(
        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=np.float32
    )
    actions[0, 0, 0] = home_id
    actions[0, 0, 1] = 0.0
    actions[0, 0, 2] = 5
    actions_j = jnp.asarray(actions)

    new_state, _, _ = step(state, actions_j)
    # Home planet should have lost 5 ships (then gained `prod`).
    new_home_ships = int(new_state.planet_ships[home_idx])
    prod = int(state.planet_prod[home_idx])
    assert new_home_ships == home_ships - 5 + prod
    # One fleet appeared.
    assert int(jnp.sum(new_state.fleet_valid)) == 1
    assert int(new_state.fleet_owner[0]) == 0
    assert int(new_state.fleet_ships[0]) == 5
