"""1-head BC loss for imitation/case4.

Per active source (my_planet_mask AND cand_slot_per_src != -1):

  CE(candidate_logits[fired_src], cand_slot_label, weight=class_weights)

Slot 0 = no-op. Slots 1..K-1 = candidate planet (notebook order). Class weights
optional via `class_weights` tensor (shape (CAND_K,)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case4.policy.types import PolicyOutput


@dataclass(frozen=True)
class LossWeights:
    cand_w: float = 1.0
    cand_class_weights: torch.Tensor | None = None
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class LossReport:
    total: torch.Tensor
    cand_loss: torch.Tensor
    cand_acc: float
    cand_noop_acc: float
    cand_fire_acc: float


def compute_loss(
    output: PolicyOutput,
    cand_slot_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    my_planet_mask: torch.Tensor,  # (B, P) bool
    weights: LossWeights,
) -> LossReport:
    device = cand_slot_per_src.device

    valid = my_planet_mask & (cand_slot_per_src != -1)  # (B, P)
    if not valid.any():
        zero = torch.zeros((), device=device)
        return LossReport(
            total=zero,
            cand_loss=zero,
            cand_acc=0.0,
            cand_noop_acc=0.0,
            cand_fire_acc=0.0,
        )

    b_idx, src_idx = valid.nonzero(as_tuple=True)
    sel_logits = output.candidate_logits[b_idx, src_idx]  # (N, CAND_K)
    sel_labels = cand_slot_per_src[b_idx, src_idx]  # (N,)

    cand_cw = (
        weights.cand_class_weights.to(sel_logits.device)
        if weights.cand_class_weights is not None
        else None
    )
    cand_loss = nn.functional.cross_entropy(
        sel_logits,
        sel_labels,
        weight=cand_cw,
        label_smoothing=weights.label_smoothing,
    )

    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)
        match = (pred == sel_labels).float()
        cand_acc = float(match.mean().item())
        is_noop = sel_labels == 0
        is_fire = ~is_noop
        n_noop = int(is_noop.sum().item())
        n_fire = int(is_fire.sum().item())
        cand_noop_acc = float(match[is_noop].mean().item()) if n_noop > 0 else 0.0
        cand_fire_acc = float(match[is_fire].mean().item()) if n_fire > 0 else 0.0

    total = weights.cand_w * cand_loss
    return LossReport(
        total=total,
        cand_loss=cand_loss.detach(),
        cand_acc=cand_acc,
        cand_noop_acc=cand_noop_acc,
        cand_fire_acc=cand_fire_acc,
    )
