"""baseline_jax_full — JAX rule-based opponent with baseline_v1 feature parity.

Stronger than `baseline_jax_lite` (single-pass argmax) while keeping the
function `jax.jit` / `jax.vmap` friendly so PPO rollouts (`rollout_jax.py`)
can use it as a `curriculum.late` opponent without paying host-callback
penalties.

Public API:
    from baseline_jax_full import compute_actions_jax

Strategy (incremental, Phases 1-6):
    Phase 1: orbit prediction (predict_planet_xy) + lite reproduction
    Phase 2: snipe (fleet intercept) + crash_exploit (sun-near targets)
    Phase 3: pair-source swarm (top-K, fixed-size)
    Phase 4: nearest-enemy-ETA defense reserve + inventory_cap
    Phase 5: registration into rollout_jax + opponent_mode dispatch
    Phase 6: 100-game eval vs baseline_v1, weight tuning
"""

from .agent_jax_full import compute_actions_jax

__all__ = ["compute_actions_jax"]
