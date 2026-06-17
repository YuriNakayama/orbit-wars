"""Strict action-parity JAX port of case7 (case8 base; STAY/ACCUMULATE pending).

case7 = case8 engine-replay base + STAY_BURST (case6) + ACCUMULATE (multi-turn).
This base ports the case8 mission family; the cross-turn STAY/ACCUMULATE state
machine is layered on separately (host-side hold counters).
"""

from .agent_jax import agent, compute_actions, compute_actions_jit

__all__ = ["agent", "compute_actions", "compute_actions_jit"]
