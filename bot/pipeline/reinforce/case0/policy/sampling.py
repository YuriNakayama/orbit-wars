"""Action sampling + log-prob computation for the PPO policy heads.

Shared between rollout (stochastic sampling, requires log_probs) and
inference (greedy argmax). Centralized here so the same masking logic is
applied in training and at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Bernoulli, Categorical

from .types import BatchFeatures, PolicyOutput


@dataclass(frozen=True)
class SampledAction:
    """Per-source action choices for a single env step (batch_size=1).

    `from_choice`: (P,) bool — which my-planets fire this turn.
    `target_choice`: (P,) long — target planet slot per source (only meaningful
       where from_choice is True; otherwise 0).
    `ships_choice`: (P,) long — ships bucket per source.
    """

    from_choice: torch.Tensor  # (P,) bool
    target_choice: torch.Tensor  # (P,) long
    ships_choice: torch.Tensor  # (P,) long
    log_prob: torch.Tensor  # scalar
    entropy: torch.Tensor  # scalar


def _valid_target_rows(batch: BatchFeatures) -> torch.Tensor:
    """(B, P) bool — True for rows that have at least one valid target."""
    return batch.target_mask.any(dim=-1, keepdim=True).expand_as(batch.target_mask)


def _safe_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum x over rows selected by mask, ignoring NaN/Inf from masked rows.

    Bernoulli/Categorical with all -inf logits produces NaN entropy on those
    rows; the mask already excludes them but a 0 * NaN still poisons the sum.
    """
    masked = torch.where(mask, x, torch.zeros_like(x))
    return masked.sum()


def sample_action(
    output: PolicyOutput,
    batch: BatchFeatures,
) -> SampledAction:
    """Sample one action from the policy (batch_size=1)."""
    from_logits = output.from_logits[0]  # (P,)
    target_logits = output.target_logits[0]  # (P_src, P_tgt)
    ships_logits = output.ships_logits[0]  # (P, K)

    my_mask = batch.my_planet_mask[0]  # (P,)
    has_target = batch.target_mask[0].any().item()

    # Replace -inf rows with a safe constant so Bernoulli math stays finite;
    # we ignore those rows via my_mask anyway.
    safe_from = torch.where(my_mask, from_logits, torch.zeros_like(from_logits))
    from_dist = Bernoulli(logits=safe_from)
    from_sample = from_dist.sample()  # (P,) float in {0, 1}
    from_choice = (from_sample > 0.5) & my_mask
    log_prob = _safe_sum(from_dist.log_prob(from_sample), my_mask)
    entropy = _safe_sum(from_dist.entropy(), my_mask)

    p = from_logits.shape[0]
    target_choice = torch.zeros(p, dtype=torch.long, device=from_logits.device)
    ships_choice = torch.zeros(p, dtype=torch.long, device=from_logits.device)

    if has_target:
        # Categorical needs at least one finite logit per row; rows with no
        # valid target get a uniform row but their samples are masked out.
        target_row_valid = batch.target_mask[0].any().expand_as(target_logits[..., 0])
        safe_target = torch.where(
            torch.isfinite(target_logits),
            target_logits,
            torch.full_like(target_logits, -1e9),
        )
        # Ensure every row has at least one finite entry (broadcast valid mask).
        safe_target = safe_target.masked_fill(
            ~target_row_valid.unsqueeze(-1).expand_as(safe_target),
            0.0,
        )
        target_dist = Categorical(logits=safe_target)
        ships_dist = Categorical(logits=ships_logits)
        t_sample = target_dist.sample()  # (P,)
        s_sample = ships_dist.sample()  # (P,)
        t_lp = target_dist.log_prob(t_sample)  # (P,)
        s_lp = ships_dist.log_prob(s_sample)
        fire_mask = from_choice
        target_choice = torch.where(from_choice, t_sample, target_choice)
        ships_choice = torch.where(from_choice, s_sample, ships_choice)
        log_prob = log_prob + _safe_sum(t_lp, fire_mask) + _safe_sum(s_lp, fire_mask)
        entropy = (
            entropy
            + _safe_sum(target_dist.entropy(), my_mask)
            + _safe_sum(ships_dist.entropy(), my_mask)
        )

    return SampledAction(
        from_choice=from_choice,
        target_choice=target_choice,
        ships_choice=ships_choice,
        log_prob=log_prob,
        entropy=entropy,
    )


def greedy_action(
    output: PolicyOutput,
    batch: BatchFeatures,
    from_threshold: float = 0.5,
) -> SampledAction:
    """Argmax decoding for inference."""
    from_logits = output.from_logits[0]
    target_logits = output.target_logits[0]
    ships_logits = output.ships_logits[0]
    my_mask = batch.my_planet_mask[0]

    from_prob = torch.sigmoid(from_logits)
    from_choice = (from_prob >= from_threshold) & my_mask
    target_choice = target_logits.argmax(dim=-1)
    ships_choice = ships_logits.argmax(dim=-1)

    zero = torch.tensor(0.0, device=from_logits.device)
    return SampledAction(
        from_choice=from_choice,
        target_choice=target_choice,
        ships_choice=ships_choice,
        log_prob=zero,
        entropy=zero,
    )


def _masked_sum_2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = torch.where(mask, x, torch.zeros_like(x))
    return safe.sum(dim=-1)


def evaluate_actions(
    output: PolicyOutput,
    batch: BatchFeatures,
    from_choice: torch.Tensor,  # (B, P) bool
    target_choice: torch.Tensor,  # (B, P) long
    ships_choice: torch.Tensor,  # (B, P) long
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute (log_prob, entropy) under the current policy for a batch.

    Mirrors `sample_action` but operates on full batches and on actions that
    were drawn under an older policy (for the PPO ratio).
    """
    my_mask = batch.my_planet_mask
    safe_from = torch.where(
        my_mask, output.from_logits, torch.zeros_like(output.from_logits)
    )
    from_dist = Bernoulli(logits=safe_from)
    from_lp = from_dist.log_prob(from_choice.float())
    log_prob = _masked_sum_2d(from_lp, my_mask)
    entropy = _masked_sum_2d(from_dist.entropy(), my_mask)

    has_target = batch.target_mask.any(dim=-1)  # (B,)

    safe_target = torch.where(
        torch.isfinite(output.target_logits),
        output.target_logits,
        torch.full_like(output.target_logits, -1e9),
    )
    # Ensure each (B, P_src, *) row has at least one finite entry.
    safe_target = safe_target.masked_fill(
        ~has_target.unsqueeze(-1).unsqueeze(-1).expand_as(safe_target),
        0.0,
    )
    target_dist = Categorical(logits=safe_target)
    ships_dist = Categorical(logits=output.ships_logits)
    target_lp = target_dist.log_prob(target_choice.clamp_min(0))
    ships_lp = ships_dist.log_prob(ships_choice.clamp_min(0))
    log_prob = (
        log_prob
        + _masked_sum_2d(target_lp, from_choice)
        + _masked_sum_2d(ships_lp, from_choice)
    )
    entropy = (
        entropy
        + (
            _masked_sum_2d(target_dist.entropy(), my_mask)
            + _masked_sum_2d(ships_dist.entropy(), my_mask)
        )
        * has_target.float()
    )
    return log_prob, entropy


__all__ = ["SampledAction", "sample_action", "greedy_action", "evaluate_actions"]
