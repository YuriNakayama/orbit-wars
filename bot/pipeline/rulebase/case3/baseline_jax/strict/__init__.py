"""Strict action-parity JAX port of case3 (case2 base; rollout not yet applied).

case3 = case2 (case8 strategy + case1 geometric aim) + shallow true2p rollout
re-ordering. This base ports everything EXCEPT rollout_reorder; see agent_jax.
"""

from .agent_jax import agent, compute_actions, compute_actions_jit

__all__ = ["agent", "compute_actions", "compute_actions_jit"]
