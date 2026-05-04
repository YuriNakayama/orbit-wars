"""2-head BC loss for imitation/case8 (candidate categorical + ship-count regression).

Per active source (my_planet_mask AND cand_slot_per_src != -1):

  cand_loss = CE(candidate_logits, cand_slot_label, weight=class_weights)

For sources whose label slot != 0 (= fire) AND ship_label_per_src != -1:

  ship_loss = SmoothL1(ship_pred, ship_label)  # Huber delta=1, robust to outliers

Total: total = cand_w * cand_loss + ship_w * ship_loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case8.policy.types import PolicyOutput


@dataclass(frozen=True)
class LossWeights:
    cand_w: float = 1.0
    cand_class_weights: torch.Tensor | None = None
    label_smoothing: float = 0.0
    ship_w: float = 1.0


@dataclass(frozen=True)
class LossReport:
    total: torch.Tensor
    cand_loss: torch.Tensor
    cand_acc: float
    cand_noop_acc: float
    cand_fire_acc: float
    ship_loss: torch.Tensor
    ship_mae: float
    ship_count: int


def compute_loss(
    output: PolicyOutput,
    cand_slot_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    my_planet_mask: torch.Tensor,  # (B, P) bool
    weights: LossWeights,
    ship_label_per_src: torch.Tensor | None = None,  # (B, P) int64; -1 = unused
) -> LossReport:
    device = cand_slot_per_src.device
    zero = torch.zeros((), device=device)

    valid = my_planet_mask & (cand_slot_per_src != -1)  # (B, P)
    if not valid.any():
        return LossReport(
            total=zero,
            cand_loss=zero,
            cand_acc=0.0,
            cand_noop_acc=0.0,
            cand_fire_acc=0.0,
            ship_loss=zero,
            ship_mae=0.0,
            ship_count=0,
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

    ship_loss = zero
    ship_mae = 0.0
    ship_count = 0
    if ship_label_per_src is not None:
        ship_valid = my_planet_mask & (ship_label_per_src != -1)  # (B, P)
        if ship_valid.any():
            sb_idx, ss_idx = ship_valid.nonzero(as_tuple=True)
            ship_pred_sel = output.ship_pred[sb_idx, ss_idx]
            ship_target_sel = ship_label_per_src[sb_idx, ss_idx].to(ship_pred_sel.dtype)
            ship_loss = nn.functional.smooth_l1_loss(
                ship_pred_sel, ship_target_sel, reduction="mean"
            )
            with torch.no_grad():
                ship_mae = float((ship_pred_sel - ship_target_sel).abs().mean().item())
                ship_count = int(ship_valid.sum().item())

    total = weights.cand_w * cand_loss + weights.ship_w * ship_loss
    return LossReport(
        total=total,
        cand_loss=cand_loss.detach(),
        cand_acc=cand_acc,
        cand_noop_acc=cand_noop_acc,
        cand_fire_acc=cand_fire_acc,
        ship_loss=ship_loss.detach() if isinstance(ship_loss, torch.Tensor) else zero,
        ship_mae=ship_mae,
        ship_count=ship_count,
    )
