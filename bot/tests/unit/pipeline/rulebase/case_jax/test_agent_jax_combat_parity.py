"""Combat-board target-match: case_jax compute_actions vs full Python plan_moves.

The noop-stepped boards only ever produce capture missions, so they can't reveal
how the agent matches Python under combat (where harass / swarm / reinforce
fire). This test drives a real game by stepping the JAX env with the Python
agent on BOTH seats, snapshots mid-game obs, and compares the JAX agent's emitted
source set against full Python `plan_moves` on the same obs.

It REPORTS precision (JAX sources Python also fires) and recall (Python sources
JAX covers). As mission families are added, recall rises toward the ≥90% gate.
A floor assert guards regressions. Source-set is the target-match proxy.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case_jax.baseline.agent import agent as py_agent
from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.agent_jax import compute_actions_jit
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


def _to_action_tensor(m0: list[Any], m1: list[Any]) -> jnp.ndarray:
    a = np.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=np.float32)
    a[:, :, 1:] = 0.0
    for j, mv in enumerate(m0[:MAX_LAUNCHES_PER_AGENT]):
        a[0, j] = [mv[0], mv[1], mv[2]]
    for j, mv in enumerate(m1[:MAX_LAUNCHES_PER_AGENT]):
        a[1, j] = [mv[0], mv[1], mv[2]]
    return jnp.asarray(a)


def test_combat_target_match() -> None:
    py_total = 0
    jax_total = 0
    overlap = 0
    samples = 0

    for seed in (0, 1, 7):
        state = reset(seed=seed, num_agents=2)
        for stp in range(120):
            obs0 = state_to_obs(state, player=0)
            obs1 = state_to_obs(state, player=1)
            reset_predict_cache()
            m0 = py_agent(obs0)
            reset_predict_cache()
            m1 = py_agent(obs1)

            if stp >= 30 and stp % 15 == 0:
                reset_predict_cache()
                py_src = {int(m[0]) for m in py_agent(obs0)}
                feats = build_world_features(obs0)
                modes = _modes_arr(build_modes(build_world(obs0)))
                act = compute_actions_jit(feats, modes)
                jax_src = {
                    int(act[i, 0]) for i in range(act.shape[0]) if int(act[i, 0]) >= 0
                }
                py_total += len(py_src)
                jax_total += len(jax_src)
                overlap += len(jax_src & py_src)
                samples += 1

            state, _r, term = step(state, _to_action_tensor(m0, m1))
            if bool(term):
                break

    assert samples > 0
    precision = overlap / max(1, jax_total)
    recall = overlap / max(1, py_total)
    print(
        f"\n[combat parity] samples={samples} py_src={py_total} jax_src={jax_total} "
        f"overlap={overlap} precision={precision:.2%} recall={recall:.2%}"
    )
    # Precision: JAX should rarely fire a source Python doesn't (capture+snipe+
    # harass ⊂ full plan_moves). Require high precision; recall rises as the
    # remaining families (swarm/reinforce/movements) are added.
    assert precision >= 0.85, (
        f"combat precision {precision:.2%} (jax={jax_total}, overlap={overlap}); "
        f"recall={recall:.2%} (py={py_total}, samples={samples})"
    )
