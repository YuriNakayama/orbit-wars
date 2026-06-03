"""Integration (e2e): case8 JAX port vs real case8 Python agent.

Mirrors case1's test_agent_jax_identity (docs/experiment/rulebase/
20260603_case8_jax_port/plan.md). The #1 risk is the JAX rewrite degrading to a
~0 win-rate; this gate targets it directly with a small (10-game) tripwire rather
than a large eval.

case8 = case4 + t14 predict-cache (a perf optimization, behaviorally equivalent).
The JAX port reuses case1's parity-verified core_jax with case8's config deltas
(PARTIAL_SOURCE_MIN_SHIPS 16, REINFORCE margins, opening mults). OM / lookahead
are default-OFF in case8 config, so the runtime agent reduces to plan_moves with
the forked missions — close to case1's core_jax.

Marked `slow` (full 500-turn games). Run locally on CPU:
    uv run pytest tests/e2e/pipeline/rulebase/case8/test_agent_jax_identity.py -q
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

from pipeline.rulebase.case8.baseline.agent import agent as v8_py
from pipeline.rulebase.case8.baseline_jax.core_jax.agent_full_jax import (
    compute_actions_jax_jit as compute_actions_jax,
)

ANGLE_TOL = 1e-4


def _py_row(moves: list[Any]) -> jnp.ndarray:
    row = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        row = row.at[i].set(jnp.asarray([m[0], m[1], m[2]], dtype=jnp.float32))
    return row


def _play_jax_vs_python(seed: int, jax_seat: int) -> int:
    """Play case8 JAX port (jax_seat) vs real case8 Python. Return winner seat or -1."""
    state = reset(seed=seed, num_agents=2)
    py_seat = 1 - jax_seat
    rewards = None
    for _turn in range(500):
        a_jax = compute_actions_jax(state, seat=jax_seat)
        a_py = _py_row(v8_py(state_to_obs(state, player=py_seat)))
        actions = empty_actions().at[jax_seat].set(a_jax).at[py_seat].set(a_py)
        state, rewards, term = step(state, actions)
        if bool(term):
            break
    if rewards is None:
        return -1
    rj, rp = float(rewards[jax_seat]), float(rewards[py_seat])
    return jax_seat if rj > rp else (py_seat if rp > rj else -1)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_jax_vs_python_runs_clean(seed: int) -> None:
    """Smoke: case8 JAX port vs real Python runs a full game with no NaN / bad shapes."""
    state = reset(seed=seed, num_agents=2)
    for _turn in range(500):
        a0 = compute_actions_jax(state, seat=0)
        assert a0.shape == (MAX_LAUNCHES_PER_AGENT, 3)
        assert not bool(jnp.any(jnp.isnan(a0)))
        a1 = _py_row(v8_py(state_to_obs(state, player=1)))
        actions = empty_actions().at[0].set(a0).at[1].set(a1)
        state, _r, term = step(state, actions)
        assert not bool(jnp.any(jnp.isnan(state.planet_xy)))
        if bool(term):
            break


@pytest.mark.slow
def test_jax_port_not_catastrophically_worse_than_python() -> None:
    """Anti-regression: case8 JAX port must not collapse to a near-0 win-rate.

    Directly targets the failure mode (JAX rewrite degrades to ~0 wins). 10 games
    (5 seeds x 2 seat assignments), threshold >=3/10 — a catastrophically-degraded
    (~0-win) port cannot pass while any competitive-or-faithful port clears easily.
    """
    jax_wins = 0
    games = 0
    for seed in range(5):
        for jax_seat in (0, 1):
            winner = _play_jax_vs_python(seed, jax_seat)
            games += 1
            if winner == jax_seat:
                jax_wins += 1
    assert jax_wins >= 3, (
        f"case8 JAX port won {jax_wins}/{games} vs real Python — catastrophic "
        f"degradation (the ~0-win failure mode we must avoid)"
    )
