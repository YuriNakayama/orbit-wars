"""JAX/Optax PPO update for end-to-end GPU training.

Mirrors `training/ppo.py` (PyTorch) so the optimization step can sit on
the same device as the rollout. The contract:

  ppo_update_jax(model, opt_state, rollout_batch, bc_reference, cfg, key)
  → (new_model, new_opt_state, stats)

Loss = clipped surrogate + value_coef * value MSE - entropy_coef *
entropy + kl_beta * KL(π || π_BC). target_kl early stops between
epochs (after the current epoch's minibatches are processed) matching
the PyTorch CleanRL/SB3 pattern.

Adam (`optax.adamw`) is created externally and passed via `opt_state`;
the caller threads it through iters. Returns the same stats schema as
PyTorch `PPOStats` for parity with logging.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..policy.model_jax import ActorCriticJax, PolicyOutputJax
from ..policy.sampling_eval_jax import evaluate_actions_jax


class PPOConfigJax(NamedTuple):
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    kl_beta: float = 0.1
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: float | None = None
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5


class PPOStatsJax(NamedTuple):
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    bc_kl: jax.Array
    clip_fraction: jax.Array
    epochs_run: jax.Array


class FlatRollout(NamedTuple):
    """Per-step transitions flattened to (N, ...) for shuffled minibatch
    sampling. Same fields as `JaxRolloutBatch` minus the (B, T) leading
    pair — the caller flattens the rollout once before calling
    `ppo_update_jax`."""

    planet_feats: jax.Array  # (N, P, F)
    planet_mask: jax.Array  # (N, P)
    my_planet_mask: jax.Array  # (N, P)
    target_mask: jax.Array  # (N, P)
    global_feats: jax.Array  # (N, G)
    template_ctx: jax.Array  # (N, P, T)
    candidate_feats: jax.Array  # (N, P, K, C)
    candidate_mask: jax.Array  # (N, P, K)
    candidate_pid: jax.Array  # (N, P, K)
    target_slot: jax.Array  # (N, P)
    log1p_ships: jax.Array  # (N, P)
    log_probs: jax.Array  # (N,) — old log_probs
    values: jax.Array  # (N,) — old values (used to compute GAE; we
    #                                       pass returns separately)
    advantages: jax.Array  # (N,)
    returns: jax.Array  # (N,)
    done_mask: jax.Array  # (N,) bool — True iff valid step (pre-terminal)


def make_optimizer(cfg: PPOConfigJax) -> optax.GradientTransformation:
    """Adam(W) + global gradient clipping, mirroring PyTorch
    `optim.AdamW + clip_grad_norm_`."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )


def _bc_kl_jax(
    new_out: PolicyOutputJax,
    bc_out: PolicyOutputJax,
    my_mask: jax.Array,
) -> jax.Array:
    """Approximate KL(π || π_BC) summed over my-planet rows, averaged
    across the minibatch."""
    log_p_new = jax.nn.log_softmax(new_out.per_planet_logits, axis=-1)
    log_p_bc = jax.nn.log_softmax(bc_out.per_planet_logits, axis=-1)
    p_new = jnp.exp(log_p_new)
    cat_kl = jnp.sum(p_new * (log_p_new - log_p_bc), axis=-1)  # (B, P)
    cat_kl = jnp.sum(jnp.where(my_mask, cat_kl, 0.0), axis=-1)  # (B,)
    std_new = jnp.maximum(1e-3, jnp.exp(new_out.ship_log_std))
    std_bc = jnp.maximum(1e-3, jnp.exp(bc_out.ship_log_std))
    var_new = std_new**2
    var_bc = std_bc**2
    mean_new = new_out.ship_mean  # (B, P)
    mean_bc = bc_out.ship_mean
    gauss_kl_per = (
        jnp.log(std_bc)
        - jnp.log(std_new)
        + (var_new + (mean_new - mean_bc) ** 2) / (2 * var_bc)
        - 0.5
    )  # (B, P)
    gauss_kl = jnp.sum(jnp.where(my_mask, gauss_kl_per, 0.0), axis=-1)
    return jnp.mean(cat_kl + gauss_kl)


def _ppo_loss(
    model: ActorCriticJax,
    bc_reference: ActorCriticJax | None,
    batch: FlatRollout,
    idx: jax.Array,  # (M,) int32 — minibatch indices
    cfg: PPOConfigJax,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute PPO loss for one minibatch. Returns (loss, aux stats)."""
    from ..policy.featurizer_jax import BatchFeaturesJax

    mb = BatchFeaturesJax(
        planet_feats=batch.planet_feats[idx],
        planet_mask=batch.planet_mask[idx],
        my_planet_mask=batch.my_planet_mask[idx],
        target_mask=batch.target_mask[idx],
        global_feats=batch.global_feats[idx],
        template_ctx=batch.template_ctx[idx],
        candidate_feats=batch.candidate_feats[idx],
        candidate_mask=batch.candidate_mask[idx],
        candidate_pid=batch.candidate_pid[idx],
    )
    target_slot = batch.target_slot[idx]
    log1p_ships = batch.log1p_ships[idx]
    old_log_probs = batch.log_probs[idx]
    advantages = batch.advantages[idx]
    returns = batch.returns[idx]

    output = model(mb)
    new_lp, entropy = evaluate_actions_jax(
        output, mb.my_planet_mask, target_slot, log1p_ships
    )
    ratio = jnp.exp(new_lp - old_log_probs)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((output.value - returns) ** 2)
    entropy_mean = jnp.mean(entropy)

    if bc_reference is not None and cfg.kl_beta > 0:
        bc_out = bc_reference(mb)
        bc_kl = _bc_kl_jax(output, bc_out, mb.my_planet_mask)
    else:
        bc_kl = jnp.float32(0.0)

    loss = (
        policy_loss
        + cfg.value_coef * value_loss
        - cfg.entropy_coef * entropy_mean
        + cfg.kl_beta * bc_kl
    )
    approx_kl = jnp.mean(old_log_probs - new_lp)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32))
    aux = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
        "bc_kl": bc_kl,
        "clip_fraction": clip_frac,
    }
    return loss, aux


def _minibatch_step(
    carry: tuple[ActorCriticJax, optax.OptState, jax.Array],
    idx: jax.Array,
    bc_reference: ActorCriticJax | None,
    rollout: FlatRollout,
    cfg: PPOConfigJax,
    optimizer: optax.GradientTransformation,
    epoch_alive: jax.Array,
) -> tuple[
    tuple[ActorCriticJax, optax.OptState, jax.Array],
    dict[str, jax.Array],
]:
    """One PPO minibatch step. `epoch_alive` is a bool scalar — when
    False (target_kl breached on a prior epoch) we keep the model and
    opt_state untouched but still compute aux stats (returning the
    real numbers; the caller's mean weighting compensates).
    """
    model, opt_state, kl_acc = carry
    grad_fn = eqx.filter_value_and_grad(_ppo_loss, has_aux=True)
    (_loss, aux), grads = grad_fn(model, bc_reference, rollout, idx, cfg)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    proposed = eqx.apply_updates(model, updates)
    # If epoch_alive is False, snap back to the carry model & opt_state.
    new_model = jax.tree.map(
        lambda new, old: jnp.where(epoch_alive, new, old), proposed, model
    )
    new_opt_state_eff = jax.tree.map(
        lambda new, old: jnp.where(epoch_alive, new, old),
        new_opt_state,
        opt_state,
    )
    new_kl_acc = kl_acc + aux["approx_kl"]
    return (new_model, new_opt_state_eff, new_kl_acc), aux


def ppo_update_jax(
    model: ActorCriticJax,
    bc_reference: ActorCriticJax | None,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: FlatRollout,
    cfg: PPOConfigJax,
    key: jax.Array,
) -> tuple[ActorCriticJax, optax.OptState, PPOStatsJax]:
    """One PPO update over the full rollout (cfg.epochs × minibatches).

    All loops are inside JAX (lax.scan over epochs × scan over
    minibatches), eliminating host↔device synchronization between
    minibatches. target_kl early-stop is enforced via an epoch_alive
    bool flag tracked in the scan carry; once an epoch's mean approx_kl
    exceeds cfg.target_kl, subsequent epochs become no-ops at the
    apply_updates step but still produce stats (the caller's
    epochs_run reports how many epochs actually advanced).
    """
    n = rollout.planet_feats.shape[0]
    M = cfg.minibatch_size
    num_mb = n // M
    if num_mb == 0:
        zero = jnp.float32(0.0)
        return (
            model,
            opt_state,
            PPOStatsJax(zero, zero, zero, zero, zero, zero, zero),
        )

    advantages = rollout.advantages
    if cfg.normalize_advantage and n > 1:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    rollout = rollout._replace(advantages=advantages)

    target_kl_thresh = (
        jnp.float32(cfg.target_kl)
        if cfg.target_kl is not None
        else jnp.float32(jnp.inf)
    )

    def epoch_step(
        carry: tuple[ActorCriticJax, optax.OptState, jax.Array, jax.Array],
        k_perm: jax.Array,
    ) -> tuple[
        tuple[ActorCriticJax, optax.OptState, jax.Array, jax.Array],
        dict[str, jax.Array],
    ]:
        model_c, opt_c, alive_c, epochs_done_c = carry
        # Shuffle indices and reshape to (num_mb, M).
        perm = jax.random.permutation(k_perm, n)
        idx_2d = perm[: num_mb * M].reshape((num_mb, M))

        def _body(
            mb_carry: tuple[ActorCriticJax, optax.OptState, jax.Array],
            mb_idx: jax.Array,
        ) -> tuple[
            tuple[ActorCriticJax, optax.OptState, jax.Array],
            dict[str, jax.Array],
        ]:
            return _minibatch_step(
                mb_carry,
                mb_idx,
                bc_reference,
                rollout,
                cfg,
                optimizer,
                alive_c,
            )

        init_mb_carry = (model_c, opt_c, jnp.float32(0.0))
        (model_new, opt_new, kl_sum), mb_aux = jax.lax.scan(
            _body, init_mb_carry, idx_2d
        )
        epoch_mean_kl = kl_sum / jnp.float32(num_mb)
        # If we breached target_kl during this epoch, mark dead so
        # FUTURE epochs no-op. The current epoch's updates already went
        # through (matching PyTorch's "complete the current epoch then
        # decide" semantics).
        new_alive = alive_c & (epoch_mean_kl <= target_kl_thresh)
        new_epochs_done = epochs_done_c + alive_c.astype(jnp.float32)
        # Average the per-minibatch aux for this epoch.
        epoch_aux = jax.tree.map(lambda x: jnp.mean(x, axis=0), mb_aux)
        return (model_new, opt_new, new_alive, new_epochs_done), epoch_aux

    # Per-epoch PRNG keys (distinct from `key` which is consumed).
    epoch_keys = jax.random.split(key, cfg.epochs)
    init_carry = (
        model,
        opt_state,
        jnp.bool_(True),
        jnp.float32(0.0),
    )
    (model_final, opt_final, _alive_final, epochs_run), epoch_aux_stack = jax.lax.scan(
        epoch_step, init_carry, epoch_keys
    )

    # Means over epochs (each is already an epoch-averaged scalar).
    means = jax.tree.map(lambda x: jnp.mean(x, axis=0), epoch_aux_stack)

    return (
        model_final,
        opt_final,
        PPOStatsJax(
            policy_loss=means["policy_loss"],
            value_loss=means["value_loss"],
            entropy=means["entropy"],
            approx_kl=means["approx_kl"],
            bc_kl=means["bc_kl"],
            clip_fraction=means["clip_fraction"],
            epochs_run=epochs_run,
        ),
    )


__all__ = [
    "PPOConfigJax",
    "PPOStatsJax",
    "FlatRollout",
    "make_optimizer",
    "ppo_update_jax",
]
