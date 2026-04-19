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

    # ---- from head: multi-hot BCE over my_planet_mask ----
    from_target = from_multihot.float()
    # `output.from_logits` already has -inf at non-my slots (masked in model.py).
    # Replace with 0 only for the BCE compute path; mask out non-my contributions
    # afterwards so the gradient stays on valid sources.
    valid = my_planet_mask
    safe_logits = torch.where(valid, output.from_logits, torch.zeros_like(from_target))
    pos_weight = torch.tensor(weights.from_pos_weight, device=device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        safe_logits, from_target, reduction="none", pos_weight=pos_weight
    )
    bce = bce * valid.float()
    from_loss = bce.sum() / valid.float().sum().clamp_min(1.0)

    # ---- gather "fired" rows across the batch ----
    fired_mask = from_multihot & valid  # (B, P)
    if fired_mask.any():
        b_idx, src_idx = fired_mask.nonzero(as_tuple=True)
        sel_target_logits = output.target_logits[b_idx, src_idx]  # (N, NUM_TEMPLATES)
        sel_ships_logits = output.ships_logits[b_idx, src_idx]  # (N, K)
        target_labels = target_per_src[b_idx, src_idx]  # (N,)
        ships_labels = ships_per_src[b_idx, src_idx]  # (N,)

        target_loss = nn.functional.cross_entropy(sel_target_logits, target_labels)
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
