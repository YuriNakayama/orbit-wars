"""3-head BC loss for imitation/case1 IL baseline.

Per-frame supervision:
- from_head: per-planet BCE over my_planet_mask. Positives = sources actually
  fired in this frame (multi-hot). Negatives = my_planet_mask & ~from_multihot.
- target_head: cross-entropy on every fired source's pairwise logits.
- ships_head: cross-entropy on every fired source's ships-bucket logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case1.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case1.policy.types import PolicyOutput

NO_OP_LABEL = MAX_PLANETS


@dataclass(frozen=True)
class LossWeights:
    from_w: float = 1.0
    target_w: float = 1.0
    ships_w: float = 0.5
    from_pos_weight: float = 8.5  # neg/pos ratio in training data
    from_focal_gamma: float = 2.0  # focal loss focusing on hard examples
    from_focal_alpha: float = 0.75  # weight on positive class
    target_label_smoothing: float = 0.1
    target_entropy_bonus: float = 0.05  # weight on -H(softmax(target_logits))


@dataclass(frozen=True)
class LossReport:
    total: torch.Tensor
    from_loss: torch.Tensor
    target_loss: torch.Tensor
    ships_loss: torch.Tensor
    from_acc: float
    target_acc: float
    ships_acc: float


def compute_loss(
    output: PolicyOutput,
    from_multihot: torch.Tensor,  # (B, P) bool
    target_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    ships_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    my_planet_mask: torch.Tensor,  # (B, P) bool
    weights: LossWeights,
) -> LossReport:
    device = from_multihot.device

    # ---- from head: multi-hot focal loss over my_planet_mask ----
    # Focal loss (Lin et al. 2017): alpha-balanced, gamma-focused BCE.
    # Solves the problem that median(sigmoid(from_logit)) was 0.01 in iter1-3
    # because easy negatives dominate the gradient under plain BCE+pos_weight.
    from_target = from_multihot.float()
    valid = my_planet_mask
    safe_logits = torch.where(valid, output.from_logits, torch.zeros_like(from_target))
    bce_per_elem = nn.functional.binary_cross_entropy_with_logits(
        safe_logits, from_target, reduction="none"
    )
    p = torch.sigmoid(safe_logits)
    p_t = from_target * p + (1.0 - from_target) * (1.0 - p)
    alpha_t = weights.from_focal_alpha * from_target + (1.0 - weights.from_focal_alpha) * (1.0 - from_target)
    focal_factor = alpha_t * (1.0 - p_t).clamp_min(1e-6).pow(weights.from_focal_gamma)
    focal = focal_factor * bce_per_elem
    focal = focal * valid.float()
    from_loss = focal.sum() / valid.float().sum().clamp_min(1.0)

    # ---- gather "fired" rows across the batch ----
    fired_mask = from_multihot & valid  # (B, P)
    if fired_mask.any():
        b_idx, src_idx = fired_mask.nonzero(as_tuple=True)
        sel_target_logits = output.target_logits[b_idx, src_idx]  # (N, NUM_TEMPLATES)
        sel_ships_logits = output.ships_logits[b_idx, src_idx]  # (N, K)
        target_labels = target_per_src[b_idx, src_idx]  # (N,)
        ships_labels = ships_per_src[b_idx, src_idx]  # (N,)

        target_loss = nn.functional.cross_entropy(
            sel_target_logits,
            target_labels,
            label_smoothing=weights.target_label_smoothing,
        )
        if weights.target_entropy_bonus > 0.0:
            log_p = nn.functional.log_softmax(sel_target_logits, dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1).mean()
            # subtract entropy → encourages higher entropy (anti-collapse)
            target_loss = target_loss - weights.target_entropy_bonus * entropy
        ships_loss = nn.functional.cross_entropy(sel_ships_logits, ships_labels)

        target_pred = sel_target_logits.argmax(dim=-1)
        ships_pred = sel_ships_logits.argmax(dim=-1)
        target_acc = float((target_pred == target_labels).float().mean().item())
        ships_acc = float((ships_pred == ships_labels).float().mean().item())
    else:
        target_loss = torch.zeros((), device=device)
        ships_loss = torch.zeros((), device=device)
        target_acc = 0.0
        ships_acc = 0.0

    # from accuracy: per-(B,P) match between sigmoid>0.5 and gt, on my_planets only.
    with torch.no_grad():
        from_pred = (torch.sigmoid(output.from_logits) > 0.5) & valid
        denom = valid.float().sum().clamp_min(1.0)
        match = (from_pred == (from_multihot & valid)) & valid
        from_acc = float((match.float().sum() / denom).item())

    total = (
        weights.from_w * from_loss
        + weights.target_w * target_loss
        + weights.ships_w * ships_loss
    )
    return LossReport(
        total=total,
        from_loss=from_loss.detach(),
        target_loss=target_loss.detach(),
        ships_loss=ships_loss.detach(),
        from_acc=from_acc,
        target_acc=target_acc,
        ships_acc=ships_acc,
    )
