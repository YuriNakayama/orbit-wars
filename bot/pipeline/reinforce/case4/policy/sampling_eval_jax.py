"""`evaluate_actions_jax`: recompute (log_prob, entropy) under the current
policy for a stored (target_slot, log1p_ships) action minibatch.

Mirrors `sampling.evaluate_actions` in PyTorch but stays in JAX so the
PPO update can be `jit`'d end-to-end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .featurizer_jax import MAX_PLANETS
from .model_jax import PolicyOutputJax

NO_OP_INDEX = MAX_PLANETS  # stored target_slot == NO_OP_INDEX means Z = 0


def evaluate_actions_jax(
    output: PolicyOutputJax,
    my_planet_mask: jax.Array,  # (B, P) bool
    target_slot: jax.Array,  # (B, P) int32 — in [0, P], P = NO_OP
    log1p_ships: jax.Array,  # (B, P) float32
) -> tuple[jax.Array, jax.Array]:
    """Recompute (log_prob, entropy) per minibatch row (B,) for the 3-head
    factorization, given a stored action in the case3 representation.

      Z = (target_slot != NO_OP)
      log_prob = log P(Z) + Z · (log P(Y=target_slot | Z=1) + log P(X | Z=1))
      entropy  = H(Z) + P(Z=1) · (H(Y|Z=1) + H(X))

    Mirrors `sample_action_jax` exactly so the PPO ratio is unbiased.
    """
    # Launch (binary Z) log-prob / entropy.
    z = (target_slot != NO_OP_INDEX).astype(jnp.float32)  # (B, P)
    log_p1 = jax.nn.log_sigmoid(output.launch_logit)  # log P(Z=1)
    log_p0 = jax.nn.log_sigmoid(-output.launch_logit)  # log P(Z=0)
    z_lp = z * log_p1 + (1.0 - z) * log_p0
    p1 = jnp.exp(log_p1)
    z_ent = -(p1 * log_p1 + (1.0 - p1) * log_p0)

    # Target categorical P(Y | Z=1) over P planets (no NO_OP slot).
    tgt_logits = jnp.where(
        my_planet_mask[..., None],
        output.target_logits,
        jnp.zeros_like(output.target_logits),
    )
    log_softmax = jax.nn.log_softmax(tgt_logits, axis=-1)
    clamped_idx = jnp.clip(target_slot, 0, MAX_PLANETS - 1)
    t_lp = jnp.take_along_axis(log_softmax, clamped_idx[..., None], axis=-1).squeeze(-1)
    probs = jnp.exp(log_softmax)
    t_ent = -jnp.sum(probs * log_softmax, axis=-1)  # (B, P)

    # Ships Normal P(X | Z=1).
    mean = output.ship_mean  # (B, P)
    std = jnp.maximum(1e-3, jnp.exp(output.ship_log_std))
    std_b = jnp.broadcast_to(std, mean.shape)
    var = std_b**2
    s_lp = -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * ((log1p_ships - mean) ** 2) / var
    s_ent = 0.5 * (jnp.log(2 * jnp.pi * var) + 1.0)

    per_lp = z_lp + z * (t_lp + s_lp)
    per_ent = z_ent + p1 * (t_ent + s_ent)

    mask_f = my_planet_mask.astype(jnp.float32)
    log_prob = jnp.sum(per_lp * mask_f, axis=-1)
    entropy = jnp.sum(per_ent * mask_f, axis=-1)
    return log_prob, entropy


__all__ = ["evaluate_actions_jax"]
