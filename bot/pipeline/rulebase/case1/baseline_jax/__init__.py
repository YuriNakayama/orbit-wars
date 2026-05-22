"""JAX-native reimplementation of rulebase/case1 baseline_v1.

This is **not** a 1:1 port of the Python `baseline/` agent. The Python
version relies on dynamic mission lists, binary-searched defense buffers,
and per-source greedy budget allocation — all incompatible with JAX's
trace-once + fixed-shape contract.

`baseline_jax_lite` keeps the same strategic intent (high production,
nearby, capturable enemy/neutral target) but expresses it as a single
jit-friendly priority score + per-source argmax. Strength target is
≈50% win-rate vs the Python `baseline_v1` so it works as a PPO opponent.
"""

from .agent_jax import compute_actions_jax

__all__ = ["compute_actions_jax"]
