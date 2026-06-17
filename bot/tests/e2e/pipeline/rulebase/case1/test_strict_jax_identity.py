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


def _actions_with_k(state: Any, seat: int, k: int) -> jnp.ndarray:
    """compute_actions with MAX_LIVE_SRC temporarily set to ``k`` (restored after)."""
    import pipeline.rulebase.case1.baseline_jax.strict.missions_capture_jax as mc

    orig = mc.MAX_LIVE_SRC
    mc.MAX_LIVE_SRC = k
    try:
        feats = build_world_features_from_state(state, seat)
        return compute_actions(feats, _modes_from_features(feats))
    finally:
        mc.MAX_LIVE_SRC = orig


@pytest.mark.slow
def test_gather_live_sources_preserves_actions() -> None:
    """The gather-live-sources grid speedup must not change emitted actions.

    `_gather_scatter_grid` vmaps the per-source HORIZON scan over only the live
    (owned, ship-bearing) source planets instead of all 48, scattering rows back.
    Dead sources score -inf and are never selected, so the action set must be
    byte-identical to the full 48-wide build. We compare the hybrid path (K=16),
    a forced-fallback path (K=2, exercising the full-grid lax.cond branch), and a
    forced-gather path (K=999 == full reference) across sampled states — source
    id and ship count EXACT, angle within 1e-3 (benign float reordering only).
    """
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    for s in range(20):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 4):
            state, _, term = step(state, noop)
            if bool(term):
                break
        for seat in (0, 1):
            ref = np.asarray(_actions_with_k(state, seat, 999))  # full reference
            for k in (16, 2):
                got = np.asarray(_actions_with_k(state, seat, k))
                assert np.array_equal(got[:, 0], ref[:, 0]), (
                    f"source id changed at K={k}, seed={s}, seat={seat}"
                )
                assert np.array_equal(got[:, 2], ref[:, 2]), (
                    f"ship count changed at K={k}, seed={s}, seat={seat}"
                )
                assert np.allclose(got[:, 1], ref[:, 1], atol=1e-3), (
                    f"angle drifted >1e-3 at K={k}, seed={s}, seat={seat}"
                )


def _actions_with_alloc_k(state: Any, seat: int, k: int) -> jnp.ndarray:
    """compute_actions with MAX_ALLOC_CANDIDATES temporarily set to ``k``."""
    import pipeline.rulebase.case1.baseline_jax.strict.allocator_jax as al

    orig = al.MAX_ALLOC_CANDIDATES
    al.MAX_ALLOC_CANDIDATES = k
    try:
        feats = build_world_features_from_state(state, seat)
        return compute_actions(feats, _modes_from_features(feats))
    finally:
        al.MAX_ALLOC_CANDIDATES = orig


@pytest.mark.slow
@pytest.mark.parametrize("k", [256, 64])
def test_allocator_truncation_preserves_actions(k: int) -> None:
    """Truncating the mission scan to the top-K candidates must not change actions.

    The allocator walks `argsort(-score)` over the N=4608 candidate table, but at
    most ~10 candidates are ever valid and they sort first; the tail is -inf
    no-ops. Top-K scanning must therefore equal the full scan (K=4608): source id
    and ship count EXACT, angle within 1e-3. K=64 keeps a 6x margin over the
    observed valid peak (10); K=256 a 25x margin.
    """
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    for s in range(12):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 6) * 7):
            state, _, term = step(state, noop)
            if bool(term):
                break
        for seat in (0, 1):
            ref = np.asarray(_actions_with_alloc_k(state, seat, 4608))  # full scan
            got = np.asarray(_actions_with_alloc_k(state, seat, k))
            assert np.array_equal(got[:, 0], ref[:, 0]), (
                f"source id changed at K={k}, seed={s}, seat={seat}"
            )
            assert np.array_equal(got[:, 2], ref[:, 2]), (
                f"ship count changed at K={k}, seed={s}, seat={seat}"
            )
            assert np.allclose(got[:, 1], ref[:, 1], atol=1e-3), (
                f"angle drifted >1e-3 at K={k}, seed={s}, seat={seat}"
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
