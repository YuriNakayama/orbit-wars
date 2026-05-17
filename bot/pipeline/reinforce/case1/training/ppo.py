"""PPO update step for the per_planet policy with optional KL anchor to BC.

Loss = clipped surrogate + value MSE − entropy bonus + β · KL(π_old_BC || π).

The BC anchor is a frozen reference policy (loaded with the same warm-start
weights). It keeps the updated policy from drifting too far from the BC prior
during the first few thousand updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.distributions import Categorical, Normal

from ..policy.model import ActorCritic, PolicyOutput
from ..policy.sampling import evaluate_actions
from ..policy.types import BatchFeatures
from .rollout import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    kl_beta: float = 0.1
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


@dataclass(frozen=True)
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    bc_kl: float
    clip_fraction: float


def _slice_batch(
    rollout: RolloutBatch, idx: torch.Tensor
) -> tuple[BatchFeatures, dict[str, torch.Tensor]]:
    batch = BatchFeatures(
        planet_feats=rollout.planet_feats[idx],
        planet_mask=rollout.planet_mask[idx],
        my_planet_mask=rollout.my_planet_mask[idx],
        target_mask=rollout.target_mask[idx],
        global_feats=rollout.global_feats[idx],
        template_ctx=rollout.template_ctx[idx],
        candidate_feats=rollout.candidate_feats[idx],
        candidate_mask=rollout.candidate_mask[idx],
        candidate_pid=rollout.candidate_pid[idx],
    )
    aux = {
        "target_slot": rollout.target_slot[idx],
        "log1p_ships": rollout.log1p_ships[idx],
        "old_log_probs": rollout.log_probs[idx],
        "advantages": rollout.advantages[idx],
        "returns": rollout.returns[idx],
    }
    return batch, aux


def _bc_kl(
    new_out: PolicyOutput, bc_out: PolicyOutput, my_mask: torch.Tensor
) -> torch.Tensor:
    """Approximate KL(π || π_BC) summed over my-planet rows.

    Categorical KL is exact; Gaussian KL uses the closed form for diagonal
    Normal. We sum over sources, average over the minibatch.
    """
    # Categorical KL: Σ p_new (log p_new - log p_bc), masked to my_mask rows.
    log_p_new = nn.functional.log_softmax(new_out.per_planet_logits, dim=-1)
    log_p_bc = nn.functional.log_softmax(bc_out.per_planet_logits, dim=-1)
    p_new = log_p_new.exp()
    cat_kl = (p_new * (log_p_new - log_p_bc)).sum(dim=-1)  # (B, P)
    cat_kl = torch.where(my_mask, cat_kl, torch.zeros_like(cat_kl)).sum(dim=-1)

    # Gaussian KL (state-independent σ): identical σ → reduces to (μ_new − μ_bc)^2 / (2σ^2)
    std_new = new_out.ship_log_std.exp().clamp_min(1e-3)
    std_bc = bc_out.ship_log_std.exp().clamp_min(1e-3)
    var_new = std_new**2
    var_bc = std_bc**2
    diff = new_out.ship_mean - bc_out.ship_mean
    gauss_kl = (
        torch.log(std_bc / std_new) + (var_new + diff**2) / (2 * var_bc) - 0.5
    )  # (B, P) broadcast
    gauss_kl = torch.where(my_mask, gauss_kl, torch.zeros_like(gauss_kl)).sum(dim=-1)

    return (cat_kl + gauss_kl).mean()


def ppo_update(
    model: ActorCritic,
    bc_reference: ActorCritic | None,
    optimizer: optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PPOConfig,
) -> PPOStats:
    n = rollout.planet_feats.shape[0]
    if n == 0:
        return PPOStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    advantages = rollout.advantages
    if cfg.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Substitute normalized advantages into a shallow copy.
    rollout = RolloutBatch(
        planet_feats=rollout.planet_feats,
        planet_mask=rollout.planet_mask,
        my_planet_mask=rollout.my_planet_mask,
        target_mask=rollout.target_mask,
        global_feats=rollout.global_feats,
        template_ctx=rollout.template_ctx,
        candidate_feats=rollout.candidate_feats,
        candidate_mask=rollout.candidate_mask,
        candidate_pid=rollout.candidate_pid,
        target_slot=rollout.target_slot,
        log1p_ships=rollout.log1p_ships,
        log_probs=rollout.log_probs,
        values=rollout.values,
        rewards=rollout.rewards,
        advantages=advantages,
        returns=rollout.returns,
        episode_ends=rollout.episode_ends,
        episode_outcomes=rollout.episode_outcomes,
    )

    model.train()
    if bc_reference is not None:
        bc_reference.eval()

    p_losses: list[float] = []
    v_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    bc_kls: list[float] = []
    clips: list[float] = []

    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            batch, aux = _slice_batch(rollout, idx)
            output = model(batch)
            new_lp, entropy = evaluate_actions(
                output, batch, aux["target_slot"], aux["log1p_ships"]
            )
            ratio = torch.exp(new_lp - aux["old_log_probs"])
            adv = aux["advantages"]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(output.value, aux["returns"])
            entropy_mean = entropy.mean()

            if bc_reference is not None and cfg.kl_beta > 0:
                with torch.no_grad():
                    bc_out = bc_reference(batch)
                bc_kl = _bc_kl(output, bc_out, batch.my_planet_mask)
            else:
                bc_kl = torch.tensor(0.0, device=output.value.device)

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_mean
                + cfg.kl_beta * bc_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (aux["old_log_probs"] - new_lp).mean()
                clip_frac = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()
            p_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))
            kls.append(float(approx_kl.item()))
            bc_kls.append(float(bc_kl.item()))
            clips.append(float(clip_frac.item()))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # Quiet unused imports — Normal/Categorical may be re-exported later.
    _ = (Normal, Categorical)

    return PPOStats(
        policy_loss=_mean(p_losses),
        value_loss=_mean(v_losses),
        entropy=_mean(entropies),
        approx_kl=_mean(kls),
        bc_kl=_mean(bc_kls),
        clip_fraction=_mean(clips),
    )


__all__ = ["PPOConfig", "PPOStats", "ppo_update"]
