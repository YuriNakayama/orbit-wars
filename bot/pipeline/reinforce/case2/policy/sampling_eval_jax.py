"""`evaluate_actions_jax`: recompute (log_prob, entropy) under the current
policy for a stored (target_slot, log1p_ships) action minibatch.

Mirrors `sampling.evaluate_actions` in PyTorch but stays in JAX so the
PPO update can be `jit`'d end-to-end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .model_jax import PolicyOutputJax


def evaluate_actions_jax(
    output: PolicyOutputJax,
    my_planet_mask: jax.Array,  # (B, P) bool
    target_slot: jax.Array,  # (B, P) int32
    log1p_ships: jax.Array,  # (B, P) float32
) -> tuple[jax.Array, jax.Array]:
    """Return (log_prob, entropy) per minibatch row (B,)."""
    pp_logits = output.per_planet_logits  # (B, P, P+1)
    n_classes = pp_logits.shape[-1]
    safe_logits = jnp.where(
        my_planet_mask[..., None], pp_logits, jnp.zeros_like(pp_logits)
    )
    log_softmax = jax.nn.log_softmax(safe_logits, axis=-1)
    clamped_idx = jnp.clip(target_slot, 0, n_classes - 1)
    t_lp = jnp.take_along_axis(log_softmax, clamped_idx[..., None], axis=-1).squeeze(
        -1
    )  # (B, P)
    probs = jnp.exp(log_softmax)
    t_ent = -jnp.sum(probs * log_softmax, axis=-1)  # (B, P)

    # Normal log-prob and entropy.
    mean = output.ship_mean  # (B, P)
    log_std = output.ship_log_std  # (1,)
    std = jnp.maximum(1e-3, jnp.exp(log_std))
    std_b = jnp.broadcast_to(std, mean.shape)
    var = std_b**2
    s_lp = -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * ((log1p_ships - mean) ** 2) / var
    s_ent = 0.5 * (jnp.log(2 * jnp.pi * var) + 1.0)

    mask_f = my_planet_mask.astype(jnp.float32)
    log_prob = jnp.sum(t_lp * mask_f, axis=-1) + jnp.sum(s_lp * mask_f, axis=-1)
    entropy = jnp.sum(t_ent * mask_f, axis=-1) + jnp.sum(s_ent * mask_f, axis=-1)
    return log_prob, entropy


__all__ = ["evaluate_actions_jax"]
