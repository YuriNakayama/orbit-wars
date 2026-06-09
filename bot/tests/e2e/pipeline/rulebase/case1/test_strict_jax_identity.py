"""Action-parity gate for the case1 STRICT JAX port (grid + allocator).

The strict port (`baseline_jax/strict/`) is the case8 grid+allocator pipeline
copied into case1 with config redirected to case1 values via `_config_compat`.
Unlike the legacy `core_jax` port (ship-miscount / under-launch, ~33% exact),
the strict port fires from the correct source on every board and matches ship
counts; the residual is target-selection / aim tie-breaks (the same benign class
case8 lives with at ~90%).

Gates, mirroring case6's registered-agent convention:
  * smoke: shape + no-NaN over a full game,
  * source-match floor: every launched source must match Python (no under-launch,
    no false positives) — this is the property core_jax violated,
  * full-exact floor: a regression tripwire (>= 70%), not 100% (tie-breaks).

Marked `slow` (full self-play). Run locally on CPU:
    uv run pytest tests/e2e/pipeline/rulebase/case1/test_strict_jax_identity.py -q
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline_jax.strict.agent_jax import (
    _modes_from_features,
    compute_actions,
)
from pipeline.rulebase.case1.baseline_jax.strict.world_features import (
    build_world_features_from_state,
)


def _jax_strict(state: Any, seat: int) -> jnp.ndarray:
    feats = build_world_features_from_state(state, seat)
    return compute_actions(feats, _modes_from_features(feats))


def _jax_set(arr: jnp.ndarray) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for row in np.asarray(arr):
        pid = int(round(float(row[0])))
        if pid >= 0:
            out.add((pid, int(round(float(row[1]))), int(round(float(row[2])))))
    return out


def _py_set(moves: list[Any]) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for mv in moves or []:
        pid = int(round(float(mv[0])))
        if pid >= 0:
            out.add((pid, int(round(float(mv[1]))), int(round(float(mv[2])))))
    return out


def _py_row(moves: list[Any]) -> jnp.ndarray:
    row = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        row = row.at[i].set(jnp.asarray([m[0], m[1], m[2]], dtype=jnp.float32))
    return row


@pytest.mark.slow
def test_strict_source_match_and_exact_floor() -> None:
    """Source planets must match Python on every state; full-exact >= floor.

    Sampled like case8_parity (30 noop-rolled states, seat=1). The strict port's
    defining property over core_jax: source-match is exact (no under-launch).
    """
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    seat = 1
    n = 30
    exact = 0
    src_match = 0
    for s in range(n):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 3):
            state, _, term = step(state, noop)
            if bool(term):
                break
        a = _jax_set(_jax_strict(state, seat))
        b = _py_set(v1_py(state_to_obs(state, player=seat)))
        if a == b:
            exact += 1
        if {x[0] for x in a} == {x[0] for x in b}:
            src_match += 1
    assert src_match == n, (
        f"source-match {src_match}/{n}: strict port fired from a source Python "
        f"did not (or missed one) — the under-launch failure mode core_jax had"
    )
    assert exact / n >= 0.70, (
        f"full-exact {exact}/{n} ({exact / n:.0%}) below 70% regression floor"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_strict_runs_clean(seed: int) -> None:
    """Smoke: strict port vs real Python plays a full game with no NaN / bad shapes."""
    state = reset(seed=seed, num_agents=2)
    for _turn in range(500):
        a0 = _jax_strict(state, seat=0)
        assert a0.shape == (MAX_LAUNCHES_PER_AGENT, 3)
        assert not bool(jnp.any(jnp.isnan(a0)))
        a1 = _py_row(v1_py(state_to_obs(state, player=1)))
        actions = empty_actions().at[0].set(a0).at[1].set(a1)
        state, _r, term = step(state, actions)
        assert not bool(jnp.any(jnp.isnan(state.planet_xy)))
        if bool(term):
            break
