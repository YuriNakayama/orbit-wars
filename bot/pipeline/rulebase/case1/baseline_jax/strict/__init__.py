"""Strict action-parity JAX port of case1 (grid + allocator architecture).

Ported from case8/baseline_jax with config redirected to case1 values via
``_config_compat``. Targets full source+angle+ships parity with case1 Python.
"""

from .agent_jax import agent, compute_actions, compute_actions_jit

__all__ = ["agent", "compute_actions", "compute_actions_jit"]
