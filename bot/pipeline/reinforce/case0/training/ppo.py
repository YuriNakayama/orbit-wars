"""PPO update step.

Standard clipped surrogate + value MSE + entropy bonus.

For one collected RolloutBatch we run `epochs` passes over shuffled
minibatches, recompute log-probs / values with the current policy, and apply
gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, optim

from ..policy.model import ActorCritic
from ..policy.sampling import evaluate_actions
from ..policy.types import BatchFeatures
from .rollout import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
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
    )
    aux = {
        "from_choice": rollout.from_choice[idx],
        "target_choice": rollout.target_choice[idx],
        "ships_choice": rollout.ships_choice[idx],
        "log_probs": rollout.log_probs[idx],
        "advantages": rollout.advantages[idx],
        "returns": rollout.returns[idx],
    }
    return batch, aux


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PPOConfig,
) -> PPOStats:
    n = rollout.planet_feats.shape[0]
    if n == 0:
        return PPOStats(0.0, 0.0, 0.0, 0.0, 0.0)

    advantages = rollout.advantages
    if cfg.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rollout = RolloutBatch(
        planet_feats=rollout.planet_feats,
        planet_mask=rollout.planet_mask,
        my_planet_mask=rollout.my_planet_mask,
        target_mask=rollout.target_mask,
        global_feats=rollout.global_feats,
        from_choice=rollout.from_choice,
        target_choice=rollout.target_choice,
        ships_choice=rollout.ships_choice,
        log_probs=rollout.log_probs,
        values=rollout.values,
        rewards=rollout.rewards,
        advantages=advantages,
        returns=rollout.returns,
        episode_ends=rollout.episode_ends,
        episode_outcomes=rollout.episode_outcomes,
    )

    model.train()
    p_losses: list[float] = []
    v_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clips: list[float] = []

    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            batch, aux = _slice_batch(rollout, idx)
            output = model(batch)
            new_lp, entropy = evaluate_actions(
                output,
                batch,
                aux["from_choice"],
                aux["target_choice"],
                aux["ships_choice"],
            )
            ratio = torch.exp(new_lp - aux["log_probs"])
            adv = aux["advantages"]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_loss = nn.functional.mse_loss(output.value, aux["returns"])
            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (aux["log_probs"] - new_lp).mean()
                clip_frac = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()
            p_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))
            kls.append(float(approx_kl.item()))
            clips.append(float(clip_frac.item()))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return PPOStats(
        policy_loss=_mean(p_losses),
        value_loss=_mean(v_losses),
        entropy=_mean(entropies),
        approx_kl=_mean(kls),
        clip_fraction=_mean(clips),
    )


__all__ = ["PPOConfig", "PPOStats", "ppo_update"]
