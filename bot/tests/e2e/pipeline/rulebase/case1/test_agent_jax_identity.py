"""Integration (e2e): JAX port action-equivalence with real Python baseline_v1.

TDD RED-first contract for the rulebase→JAX port (docs/plans/rulebase-to-jax/
10-integration-test-design.md).

Equivalence definition: on the SAME observation, the JAX agent
(`compute_actions_jax(state, seat)`) must emit the SAME launch list as the real
Python agent (`baseline.agent.agent(obs)`) — exact on source planet id and ship
count, float32-tolerant only on angle. The observation sequence is drawn from a
REAL self-play game (both seats played by the Python agent) so the comparison
covers realistic, contested mid/late-game boards, not just noop-rolled states.

Currently RED: `baseline_jax` is the `lite` approximation (NOT a 1:1 port), so
the match rate is ~0%. This test goes GREEN only when the faithful full port
(Step 1-4) reproduces the Python agent's decisions exactly.

Marked `slow` (full 500-turn games). Run locally on CPU:
    uv run pytest tests/e2e/pipeline/rulebase/case1/test_agent_jax_identity.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py

# The faithful full port (capture + reserve + plan_shot guards). NOTE: the
# legacy lite `baseline_jax.compute_actions_jax` is ~0-win; the integration gate
# must test the full port under development.
from pipeline.rulebase.case1.baseline_jax.core_jax.agent_full_jax import (
    compute_actions_jax,
)

ANGLE_TOL = 1e-4


def _py_row(moves: list) -> jnp.ndarray:
    row = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        row = row.at[i].set(jnp.asarray([m[0], m[1], m[2]], dtype=jnp.float32))
    return row


def _jax_to_moves(row: jnp.ndarray) -> list[tuple[int, float, int]]:
    return sorted(
        (int(r[0]), float(r[1]), int(r[2])) for r in row if int(r[0]) >= 0
    )


def _py_to_moves(moves: list) -> list[tuple[int, float, int]]:
    return sorted((int(m[0]), float(m[1]), int(m[2])) for m in moves)


def _actions_equal(jax_row: jnp.ndarray, py_moves: list) -> bool:
    j = _jax_to_moves(jax_row)
    p = _py_to_moves(py_moves)
    if len(j) != len(p):
        return False
    return all(
        a[0] == b[0] and abs(a[1] - b[1]) < ANGLE_TOL and a[2] == b[2]
        for a, b in zip(j, p, strict=True)
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_jax_port_action_equivalence_over_selfplay(seed: int) -> None:
    """JAX port must match Python agent's actions on every reachable board."""
    state = reset(seed=seed, num_agents=2)
    compared = 0
    matches = 0

    for _turn in range(500):
        obs0 = state_to_obs(state, player=0)
        py_moves = v1_py(obs0)
        jax_row = compute_actions_jax(state, seat=0)
        compared += 1
        if _actions_equal(jax_row, py_moves):
            matches += 1

        # advance the board with BOTH seats played by the real Python agent so
        # the observation sequence matches what baseline_v1 actually produces.
        obs1 = state_to_obs(state, player=1)
        actions = (
            empty_actions().at[0].set(_py_row(py_moves)).at[1].set(_py_row(v1_py(obs1)))
        )
        state, _r, term = step(state, actions)
        if bool(term):
            break

    rate = matches / compared if compared else 0.0
    assert rate == 1.0, (
        f"seed={seed}: JAX port matched Python on {matches}/{compared} boards "
        f"({rate:.1%}); require 100% for equivalence"
    )


def _play_jax_vs_python(seed: int, jax_seat: int) -> int:
    """Play JAX port (jax_seat) vs real Python (other seat). Return winner seat or -1."""
    state = reset(seed=seed, num_agents=2)
    py_seat = 1 - jax_seat
    rewards = None
    for _turn in range(500):
        a_jax = compute_actions_jax(state, seat=jax_seat)
        a_py = _py_row(v1_py(state_to_obs(state, player=py_seat)))
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
    """Smoke: JAX port vs real Python runs a full game with no NaN / bad shapes."""
    state = reset(seed=seed, num_agents=2)
    for _turn in range(500):
        a0 = compute_actions_jax(state, seat=0)
        assert a0.shape == (MAX_LAUNCHES_PER_AGENT, 3)
        assert not bool(jnp.any(jnp.isnan(a0)))
        a1 = _py_row(v1_py(state_to_obs(state, player=1)))
        actions = empty_actions().at[0].set(a0).at[1].set(a1)
        state, _r, term = step(state, actions)
        assert not bool(jnp.any(jnp.isnan(state.planet_xy)))
        if bool(term):
            break


@pytest.mark.slow
def test_jax_port_not_catastrophically_worse_than_python() -> None:
    """Anti-regression: the JAX port must not collapse to a near-0 win-rate.

    Directly targets the failure mode (JAX rewrite degrades to ~0 wins). Uses 10
    games (5 seeds × 2 seat assignments) — the smallest sample that is reliable:
    a 4-game version is too noisy in this near-mirror match (a faithful port sits
    near 50%, so 4 games swing 1/4↔3/4 by luck). Threshold = >=3/10, which a
    catastrophically-degraded (~0-win) port cannot pass while any
    competitive-or-faithful port clears easily. Still NOT a large-scale eval.
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
        f"JAX port won {jax_wins}/{games} vs real Python — catastrophic "
        f"degradation (the ~0-win failure mode we must avoid)"
    )
