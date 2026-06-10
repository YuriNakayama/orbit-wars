"""JAX/Optax V-MPO update for end-to-end GPU training.

V-MPO (Song et al., DeepMind, ICLR 2020 — arXiv:1909.12238). On-policy MPO:
instead of the PPO clipped surrogate, V-MPO builds a non-parametric target
distribution ψ from the TOP-HALF advantages weighted by a learned temperature η,
then fits the policy to ψ by weighted maximum likelihood, under a decoupled KL
trust region enforced by a Lagrange multiplier α. No importance weighting, no
entropy regularization, no clipping.

Loss = L_η (temperature dual) + L_π (weighted MLE) + L_α (trust region)
       + value_coef * value MSE.

Mirrors `ppo_jax.py`'s structure (same FlatRollout, optimizer, epoch×minibatch
lax.scan scaffold, PPOStatsJax-compatible aux) so V-MPO and PPO share the entire
PFSP / held-out / rollout harness and differ ONLY in the loss — the A/B contract
of the case8 experiment.

η and α are learnable positive scalars (stored in log-space in `VMPOParams`,
exp() on use) optimized jointly with the policy by the same Optax optimizer.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..policy.model_jax import ActorCriticJax, PolicyOutputJax
from ..policy.sampling_eval_jax import evaluate_actions_jax
from .ppo_jax import FlatRollout, PPOStatsJax

_EPS = 1e-8


class VMPOConfigJax(NamedTuple):
    """V-MPO hyperparameters + shared optimizer config.

    Paper defaults: eps_eta ~ 0.1 (temperature KL bound), eps_alpha ~ 0.01
    (trust-region KL bound), top-k fraction = 0.5 (top-half advantages).
    """

    eps_eta: float = 0.1
    eps_alpha: float = 0.01
    topk_frac: float = 0.5
    value_coef: float = 0.5
    init_eta: float = 1.0
    init_alpha: float = 5.0
    epochs: int = 2
    minibatch_size: int = 128
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    lr: float = 3.0e-5
    weight_decay: float = 1.0e-5
    lr_end: float = 0.0
    lr_schedule_steps: int = 0


class VMPOParams(eqx.Module):
    """Learnable V-MPO Lagrange scalars (log-space for positivity)."""

    log_eta: jax.Array
    log_alpha: jax.Array

    @property
    def eta(self) -> jax.Array:
        return jnp.exp(self.log_eta)

    @property
    def alpha(self) -> jax.Array:
        return jnp.exp(self.log_alpha)


def init_vmpo_params(cfg: VMPOConfigJax) -> VMPOParams:
    return VMPOParams(
        log_eta=jnp.log(jnp.float32(cfg.init_eta)),
        log_alpha=jnp.log(jnp.float32(cfg.init_alpha)),
    )


def make_optimizer_vmpo(cfg: VMPOConfigJax) -> optax.GradientTransformation:
    """AdamW + global grad clip, mirroring ppo_jax.make_optimizer."""
    if cfg.lr_end > 0 and cfg.lr_schedule_steps > 0:
        schedule = optax.linear_schedule(
            init_value=cfg.lr,
            end_value=cfg.lr_end,
            transition_steps=cfg.lr_schedule_steps,
        )
        adam: optax.GradientTransformation = optax.adamw(
            learning_rate=schedule, weight_decay=cfg.weight_decay
        )
    else:
        adam = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), adam)


def _policy_kl(
    old_out: PolicyOutputJax,
    new_out: PolicyOutputJax,
    my_mask: jax.Array,
) -> jax.Array:
    """Per-sample KL(old || new) over my-planet rows (categorical + gaussian).

    Decoupled trust region: KL is measured FROM the rollout-time policy (old) TO
    the current policy (new), per sample, returned as (B,). Mirrors the structure
    of ppo_jax._bc_kl_jax but keeps the per-sample axis (no mean) so the caller
    can stop-grad and average for the Lagrangian.
    """
    log_p_old = jax.nn.log_softmax(old_out.per_planet_logits, axis=-1)
    log_p_new = jax.nn.log_softmax(new_out.per_planet_logits, axis=-1)
    p_old = jnp.exp(log_p_old)
    cat_kl = jnp.sum(p_old * (log_p_old - log_p_new), axis=-1)  # (B, P)
    cat_kl = jnp.sum(jnp.where(my_mask, cat_kl, 0.0), axis=-1)  # (B,)

    std_old = jnp.maximum(1e-3, jnp.exp(old_out.ship_log_std))
    std_new = jnp.maximum(1e-3, jnp.exp(new_out.ship_log_std))
    var_old = std_old**2
    var_new = std_new**2
    gauss_kl_per = (
        jnp.log(std_new)
        - jnp.log(std_old)
        + (var_old + (old_out.ship_mean - new_out.ship_mean) ** 2) / (2 * var_new)
        - 0.5
    )  # (B, P)
    gauss_kl = jnp.sum(jnp.where(my_mask, gauss_kl_per, 0.0), axis=-1)  # (B,)
    return cat_kl + gauss_kl


def _vmpo_loss(
    model: ActorCriticJax,
    vp: VMPOParams,
    old_model: ActorCriticJax,
    batch: FlatRollout,
    idx: jax.Array,
    cfg: VMPOConfigJax,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """V-MPO loss for one minibatch. Returns (loss, aux)."""
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
    m = advantages.shape[0]

    output = model(mb)
    new_lp, entropy = evaluate_actions_jax(
        output, mb.my_planet_mask, target_slot, log1p_ships
    )

    # ---- top-half advantage mask (V-MPO uses the best 50% of samples) --------
    # threshold = median advantage; keep samples with adv >= median.
    thresh = jnp.median(advantages)
    top_mask = advantages >= thresh  # (M,) bool, ~half True
    top_f = top_mask.astype(jnp.float32)
    n_top = jnp.maximum(jnp.sum(top_f), 1.0)

    eta = vp.eta

    # ---- L_η: temperature dual (Eq. in V-MPO) --------------------------------
    # L_η = η·ε_η + η·log( (1/n_top) Σ_top exp(A_i/η) )
    # numerically-stable logsumexp over top-half advantages.
    scaled = advantages / eta
    neg_inf = jnp.float32(-1e30)
    scaled_masked = jnp.where(top_mask, scaled, neg_inf)
    lse = jax.nn.logsumexp(scaled_masked) - jnp.log(n_top)
    l_eta = eta * cfg.eps_eta + eta * lse

    # ---- ψ target weights (non-parametric, stop-grad through η) --------------
    # ψ_i = softmax_top(A_i / η)  (only top-half; others weight 0)
    psi = jax.nn.softmax(jax.lax.stop_gradient(scaled_masked))  # (M,), sums to 1

    # ---- L_π: weighted maximum likelihood ------------------------------------
    l_pi = -jnp.sum(psi * new_lp)

    # ---- L_α: decoupled KL trust region --------------------------------------
    old_out = old_model(mb)
    kl = _policy_kl(jax.lax.stop_gradient(old_out), output, mb.my_planet_mask)  # (M,)
    kl_mean = jnp.mean(kl)
    alpha = vp.alpha
    # L_α = α·(ε_α - sg(KL)) + sg(α)·KL   (Lagrangian; α pulled toward feasibility)
    l_alpha = alpha * (cfg.eps_alpha - jax.lax.stop_gradient(kl_mean)) + (
        jax.lax.stop_gradient(alpha) * kl_mean
    )

    # ---- value loss (same MSE as PPO) ----------------------------------------
    value_loss = jnp.mean((output.value - returns) ** 2)

    loss = l_pi + l_eta + l_alpha + cfg.value_coef * value_loss

    approx_kl = jnp.mean(old_log_probs - new_lp)
    aux = {
        "policy_loss": l_pi,
        "value_loss": value_loss,
        "entropy": jnp.mean(entropy),
        "approx_kl": approx_kl,
        "bc_kl": jnp.float32(0.0),
        "clip_fraction": jnp.float32(0.0),
        "l_eta": l_eta,
        "l_alpha": l_alpha,
        "eta": eta,
        "alpha": alpha,
        "vmpo_kl": kl_mean,
        "top_frac": jnp.sum(top_f) / jnp.float32(m),
    }
    return loss, aux


class _MBCarry(NamedTuple):
    model: ActorCriticJax
    vp: VMPOParams
    opt_state: optax.OptState


def vmpo_update_jax(
    model: ActorCriticJax,
    vp: VMPOParams,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rollout: FlatRollout,
    cfg: VMPOConfigJax,
    key: jax.Array,
) -> tuple[ActorCriticJax, VMPOParams, optax.OptState, PPOStatsJax]:
    """One V-MPO update over the full rollout (cfg.epochs × minibatches).

    `old_model` (rollout-time policy) is snapshotted once at the start for the
    decoupled KL. All loops are inside JAX (lax.scan over epochs × minibatches).
    No target_kl early-stop (V-MPO controls divergence via L_α instead).
    """
    n = rollout.planet_feats.shape[0]
    mbs = cfg.minibatch_size
    num_mb = n // mbs
    zero = jnp.float32(0.0)
    if num_mb == 0:
        return (
            model,
            vp,
            opt_state,
            PPOStatsJax(zero, zero, zero, zero, zero, zero, zero),
        )

    advantages = rollout.advantages
    if cfg.normalize_advantage and n > 1:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + _EPS)
    rollout = rollout._replace(advantages=advantages)

    old_model = jax.lax.stop_gradient(model)  # rollout-time policy snapshot

    def mb_step(
        carry: _MBCarry, mb_idx: jax.Array
    ) -> tuple[_MBCarry, dict[str, jax.Array]]:
        # grad w.r.t. (model, vp) jointly — packed as one pytree arg since
        # eqx.filter_value_and_grad differentiates only its first argument.
        def loss_fn(
            mv: tuple[ActorCriticJax, VMPOParams],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            md, vpp = mv
            return _vmpo_loss(md, vpp, old_model, rollout, mb_idx, cfg)

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (_loss, aux), grads = grad_fn((carry.model, carry.vp))
        params = eqx.filter((carry.model, carry.vp), eqx.is_inexact_array)
        updates, new_opt = optimizer.update(grads, carry.opt_state, params)
        new_model, new_vp = eqx.apply_updates((carry.model, carry.vp), updates)
        return _MBCarry(new_model, new_vp, new_opt), aux

    def epoch_step(
        carry: _MBCarry, k_perm: jax.Array
    ) -> tuple[_MBCarry, dict[str, jax.Array]]:
        perm = jax.random.permutation(k_perm, n)
        idx_2d = perm[: num_mb * mbs].reshape((num_mb, mbs))
        new_carry, mb_aux = jax.lax.scan(mb_step, carry, idx_2d)
        return new_carry, jax.tree.map(lambda x: jnp.mean(x, axis=0), mb_aux)

    epoch_keys = jax.random.split(key, cfg.epochs)
    init = _MBCarry(model, vp, opt_state)
    final, epoch_aux_stack = jax.lax.scan(epoch_step, init, epoch_keys)
    means = jax.tree.map(lambda x: jnp.mean(x, axis=0), epoch_aux_stack)

    stats = PPOStatsJax(
        policy_loss=means["policy_loss"],
        value_loss=means["value_loss"],
        entropy=means["entropy"],
        approx_kl=means["approx_kl"],
        # bc_kl slot carries the V-MPO trust-region KL (rollout→current policy)
        # so train_jax can surface it for the A/B without changing the schema.
        bc_kl=means["vmpo_kl"],
        clip_fraction=means["clip_fraction"],
        epochs_run=jnp.float32(cfg.epochs),
    )
    return final.model, final.vp, final.opt_state, stats
