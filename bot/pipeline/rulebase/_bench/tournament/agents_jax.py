"""Adapter: map each registered JAX rule agent (jax_vN) to a vmappable ActionFn.

Every JAX case exposes the same three in-JAX entry points operating on the shared
``EnvState`` and producing the shared ``(MAX_LAUNCHES_PER_AGENT, 3)`` action row:

    build_world_features_from_state(state, seat) -> WorldFeatures   (case-specific type)
    _modes_from_features(features)               -> ModesArrays
    compute_actions(features, modes)             -> (L, 3)

So an ``ActionFn = (EnvState) -> (L, 3)`` for agent X at a static seat is just the
composition of X's three functions. This module resolves each ``jax_vN`` name to
its module and returns the bound ActionFn — letting ``selfplay_jax.run_selfplay_batch``
play any pair of cases head-to-head in a single vmapped device call.

Only the in-JAX (vmappable, GPU-fast) agents are listed here; Python host-callback
agents are intentionally excluded (the tournament is JAX-only).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import jax
import jax.numpy as jnp
from orbit_wars_jax.step import EnvState

ActionFn = Callable[[EnvState], jax.Array]

# jax_vN -> module path holding (build_world_features_from_state,
# _modes_from_features, compute_actions). Kept in sync with AGENT_REGISTRY.
TOURNAMENT_AGENTS: dict[str, str] = {
    "jax_v1": "pipeline.rulebase.case1.baseline_jax.strict",
    "jax_v2": "pipeline.rulebase.case2.baseline_jax.strict",
    "jax_v3": "pipeline.rulebase.case3.baseline_jax.strict",
    "jax_v4": "pipeline.rulebase.case4.baseline_jax",
    "jax_v6": "pipeline.rulebase.case6.baseline_jax",
    "jax_v8": "pipeline.rulebase.case8.baseline_jax",
    "jax_v9": "pipeline.rulebase.case9.baseline_jax",
}


def _resolve_fns(
    name: str,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Import (build_world_features_from_state, _modes_from_features, compute_actions)."""  # noqa: E501
    pkg = TOURNAMENT_AGENTS[name]
    wf = importlib.import_module(f"{pkg}.world_features")
    aj = importlib.import_module(f"{pkg}.agent_jax")
    return (
        wf.build_world_features_from_state,
        aj._modes_from_features,
        aj.compute_actions,
    )


def action_fn(name: str, seat: int) -> ActionFn:
    """Return a vmappable ``ActionFn`` for agent ``name`` bound to a static seat."""
    build_features, modes_from, compute = _resolve_fns(name)
    seat_i = int(seat)

    def _fn(state: EnvState) -> jax.Array:
        features = build_features(state, seat_i)
        return jnp.asarray(compute(features, modes_from(features)))

    return _fn


def list_agents() -> list[str]:
    """Tournament participant names (sorted, stable order)."""
    return sorted(TOURNAMENT_AGENTS)
