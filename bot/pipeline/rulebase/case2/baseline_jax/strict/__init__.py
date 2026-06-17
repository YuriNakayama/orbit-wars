"""Strict action-parity JAX port of case2 (case8 strategy + case1 geometric aim).

case2 is case1-lineage on aim (geometric estimate_arrival, half-step intercept)
and case8-lineage on strategy (plan_moves grid + allocator, harass/swarm missions).
Config redirected to case2 values via `_config_compat`.
"""

from .agent_jax import agent, compute_actions, compute_actions_jit

__all__ = ["agent", "compute_actions", "compute_actions_jit"]
