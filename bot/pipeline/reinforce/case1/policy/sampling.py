"""Action sampling + log-prob computation for the PPO policy heads.

Action factorization per source planet `s`:
  - target_slot[s] ~ Categorical(per_planet_logits[s])          # over P+1 slots
  - log1p_ships[s] ~ Normal(ship_mean[s], exp(ship_log_std))    # state-indep σ

log_prob and entropy contributions are summed only over `my_planet_mask`
sources. Fire decision is implicit: `target_slot == NO_OP_INDEX` means no-op,
in which case the Gaussian ships sample is still kept in the buffer (so PPO
recomputes the same log_prob) but the decoder ignores it.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical, Normal

from .decoder import NO_OP_INDEX, SampledAction
from .model import PolicyOutput
from .types import BatchFeatures


def _safe_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, x, torch.zeros_like(x))
    return masked.sum()


def _safe_sum_2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, x, torch.zeros_like(x))
    return masked.sum(dim=-1)


def _ship_dist(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
    std = log_std.exp().clamp_min(1e-3)
    return Normal(loc=mean, scale=std.expand_as(mean))


def sample_action(output: PolicyOutput, batch: BatchFeatures) -> SampledAction:
    """Sample one action from the policy (batch_size=1)."""
    my_mask = batch.my_planet_mask[0]  # (P,)
    pp_logits = output.per_planet_logits[0]  # (P, P+1)

    # Replace fully-invalid rows (no source) with uniform 0s so Categorical
    # stays finite; we mask sampled output afterwards.
    safe_logits = torch.where(
        my_mask.unsqueeze(-1).expand_as(pp_logits),
        pp_logits,
        torch.zeros_like(pp_logits),
    )
    target_dist = Categorical(logits=safe_logits)
    target_sample = target_dist.sample()  # (P,)
    t_lp = target_dist.log_prob(target_sample)
    t_ent = target_dist.entropy()

    ship_dist = _ship_dist(output.ship_mean[0], output.ship_log_std)
    ship_sample = ship_dist.sample()  # (P,)
    s_lp = ship_dist.log_prob(ship_sample)
    s_ent = ship_dist.entropy()

    # Mask out non-source rows; non-fire rows still contribute Gaussian lp/ent
    # so the buffered sample is consistent with the policy's joint.
    log_prob = _safe_sum(t_lp, my_mask) + _safe_sum(s_lp, my_mask)
    entropy = _safe_sum(t_ent, my_mask) + _safe_sum(s_ent, my_mask)

    # Force non-source rows to NO_OP / zero ship for downstream decoder cleanliness.
    target_slot = torch.where(
        my_mask,
        target_sample,
        torch.full_like(target_sample, NO_OP_INDEX),
    )
    log1p_ships = torch.where(my_mask, ship_sample, torch.zeros_like(ship_sample))

    return SampledAction(
        target_slot=target_slot,
        log1p_ships=log1p_ships,
        log_prob=log_prob,
        entropy=entropy,
    )


def greedy_action(
    output: PolicyOutput,
    batch: BatchFeatures,
    fire_margin: float = 999.0,
) -> SampledAction:
    """Argmax decode for inference.

    The BC head is heavily biased toward no-op (per case9's inference notes),
    so we follow case9's `IL_CASE9_PP_FIRE_MARGIN` convention: a source is
    allowed to fire its best non-no-op slot whenever
    ``noop_logit − best_fire_logit < fire_margin``. Default `fire_margin=999`
    reproduces case9's "always fire" inference behavior; downstream safety
    filters in `decoder.py` still drop unsafe launches.
    """
    my_mask = batch.my_planet_mask[0]
    pp_logits = output.per_planet_logits[0]  # (P, P+1)
    noop_logits = pp_logits[..., NO_OP_INDEX]  # (P,)
    fire_logits = pp_logits.clone()
    fire_logits[..., NO_OP_INDEX] = float("-inf")
    best_fire = fire_logits.argmax(dim=-1)  # (P,)
    best_fire_val = fire_logits.gather(-1, best_fire.unsqueeze(-1)).squeeze(-1)
    force_fire = noop_logits - best_fire_val < fire_margin
    use_fire = my_mask & force_fire
    fallback = pp_logits.argmax(dim=-1)
    target_slot = torch.where(
        use_fire,
        best_fire,
        torch.where(my_mask, fallback, torch.full_like(fallback, NO_OP_INDEX)),
    )
    log1p_ships = output.ship_mean[0]
    zero = torch.tensor(0.0, device=pp_logits.device)
    return SampledAction(
        target_slot=target_slot,
        log1p_ships=log1p_ships,
        log_prob=zero,
        entropy=zero,
    )


def evaluate_actions(
    output: PolicyOutput,
    batch: BatchFeatures,
    target_slot: torch.Tensor,  # (B, P) long
    log1p_ships: torch.Tensor,  # (B, P) float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (log_prob, entropy) under the current policy for a minibatch."""
    my_mask = batch.my_planet_mask
    pp_logits = output.per_planet_logits
    safe_logits = torch.where(
        my_mask.unsqueeze(-1).expand_as(pp_logits),
        pp_logits,
        torch.zeros_like(pp_logits),
    )
    target_dist = Categorical(logits=safe_logits)
    t_lp = target_dist.log_prob(
        target_slot.clamp_min(0).clamp_max(pp_logits.shape[-1] - 1)
    )
    t_ent = target_dist.entropy()

    ship_dist = _ship_dist(output.ship_mean, output.ship_log_std)
    s_lp = ship_dist.log_prob(log1p_ships)
    s_ent = ship_dist.entropy()

    log_prob = _safe_sum_2d(t_lp, my_mask) + _safe_sum_2d(s_lp, my_mask)
    entropy = _safe_sum_2d(t_ent, my_mask) + _safe_sum_2d(s_ent, my_mask)
    return log_prob, entropy


__all__ = ["sample_action", "greedy_action", "evaluate_actions"]
