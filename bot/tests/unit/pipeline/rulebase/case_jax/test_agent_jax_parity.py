"""End-to-end target-match: case_jax compute_actions vs full Python plan_moves.

This measures how close the JAX agent's emitted launches are to the complete
Python `plan_moves` (all mission families + movements) — the real Phase-3 gate.
The JAX driver currently wires capture "single" + snipe; reinforce / swarm /
harass / movements are layered in later. So this test REPORTS the target-match
(set of source planets that fire) rather than asserting 90% yet — it documents
the gap as families are added. A hard assert guards the capture-dominant floor.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case_jax.baseline.agent import agent as py_agent
from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.agent_jax import compute_actions
from pipeline.rulebase.case_jax.baseline_jax.scoring_jax import ModesArrays
from pipeline.rulebase.case_jax.baseline_jax.world_features import build_world_features

pytestmark = pytest.mark.slow


def _modes_arr(modes: dict[str, Any]) -> ModesArrays:
    return ModesArrays(
        domination=jnp.float32(modes["domination"]),
        is_behind=jnp.bool_(modes["is_behind"]),
        is_ahead=jnp.bool_(modes["is_ahead"]),
        is_dominating=jnp.bool_(modes["is_dominating"]),
        is_finishing=jnp.bool_(modes["is_finishing"]),
        attack_margin_mult=jnp.float32(modes["attack_margin_mult"]),
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_compute_actions_targets_subset_of_plan_moves(seed: int) -> None:
    """JAX launches must be a near-subset of Python's (capture+snipe ⊂ full)."""
    state = reset(seed=seed, num_agents=2)
    py_total = 0
    jax_total = 0
    jax_src_in_py = 0  # JAX sources that Python also fires (precision)
    boards = 0

    for _ in range(8):
        obs = state_to_obs(state, player=0)

        reset_predict_cache()
        py_moves = py_agent(obs)
        py_src = {int(m[0]) for m in py_moves}

        feats = build_world_features(obs)
        modes = _modes_arr(build_modes(build_world(obs)))
        act = compute_actions(feats, modes)
        jax_src = {int(act[i, 0]) for i in range(act.shape[0]) if int(act[i, 0]) >= 0}

        boards += 1
        py_total += len(py_src)
        jax_total += len(jax_src)
        jax_src_in_py += len(jax_src & py_src)

        state, _r, term = step(state, empty_actions())
        if bool(term):
            break

    assert boards > 0
    # Precision: of the sources JAX fires, the fraction Python also fires.
    # capture+snipe should be a near-subset of full plan_moves; require high.
    precision = jax_src_in_py / max(1, jax_total)
    assert precision >= 0.80, (
        f"seed={seed}: JAX-fires-that-Python-also-fires {precision:.2%} "
        f"(jax={jax_total}, overlap={jax_src_in_py}, py={py_total})"
    )
