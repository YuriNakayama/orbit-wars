"""per_planet BC loss for imitation/case11 (3-layer mask design).

Two heads, two masks:

  target_head : (B, P, P+1) cross-entropy on `target_pid_per_src`.
                Only rows where `effective_source_mask` is True contribute.
                (layer2: my_planet & ships>0)

  ship_head   : (B, P) SmoothL1 on `ship_pred_label` (log1p(ships)).
                Only rows where `should_learn_ship` is True contribute.
                (layer3: effective_source_mask & (target != no_op))

Total = target_w * target_loss + ship_w * ship_loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case11.policy.types import PolicyOutput


@dataclass(frozen=True)
class LossWeights:
    target_w: float = 1.0
    ship_w: float = 1.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class PerPlanetLossReport:
    total: torch.Tensor
    target_loss: torch.Tensor
    ship_loss: torch.Tensor
    target_acc: float
    target_noop_acc: float
    target_fire_acc: float
    ship_mae: float
    fire_count: int


def compute_per_planet_loss(
    output: PolicyOutput,
    target_pid_per_src: torch.Tensor,  # (B, P) int64; 0..P-1 = fire target slot,
    # no_op_idx (== P) = teacher chose no-op
    ship_pred_label: torch.Tensor,  # (B, P) float; log1p(ships) for fired, -1
    effective_source_mask: torch.Tensor,  # (B, P) bool — layer2 source mask
    should_learn_ship: torch.Tensor,  # (B, P) bool — layer3 ship-head mask
    weights: LossWeights,
    no_op_idx: int,
) -> PerPlanetLossReport:
    """(P+1)-class CE on target_head + SmoothL1 on ship_head.

    The two masks are computed in preprocess and serialized in parquet so
    train-time and inference-time semantics stay aligned (Lux3 教訓).
    """
    device = effective_source_mask.device
    zero = torch.zeros((), device=device)

    if not effective_source_mask.any():
        return PerPlanetLossReport(
            total=zero,
            target_loss=zero,
            ship_loss=zero,
            target_acc=0.0,
            target_noop_acc=0.0,
            target_fire_acc=0.0,
            ship_mae=0.0,
            fire_count=0,
        )

    b_idx, s_idx = effective_source_mask.nonzero(as_tuple=True)
    sel_logits = output.per_planet_logits[b_idx, s_idx]  # (N, P+1)
    sel_labels = target_pid_per_src[b_idx, s_idx]  # (N,)

    target_loss = nn.functional.cross_entropy(
        sel_logits,
        sel_labels,
        label_smoothing=weights.label_smoothing,
    )
    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)
        match = (pred == sel_labels).float()
        target_acc = float(match.mean().item())
        is_noop = sel_labels == no_op_idx
        is_fire = ~is_noop
        n_noop = int(is_noop.sum().item())
        n_fire = int(is_fire.sum().item())
        target_noop_acc = float(match[is_noop].mean().item()) if n_noop > 0 else 0.0
        target_fire_acc = float(match[is_fire].mean().item()) if n_fire > 0 else 0.0

    ship_loss = zero
    ship_mae = 0.0
    fire_count = 0
    if should_learn_ship.any():
        sb_idx, ss_idx = should_learn_ship.nonzero(as_tuple=True)
        ship_pred_sel = output.ship_pred[sb_idx, ss_idx]
        ship_target_sel = ship_pred_label[sb_idx, ss_idx].to(ship_pred_sel.dtype)
        ship_loss = nn.functional.smooth_l1_loss(
            ship_pred_sel, ship_target_sel, reduction="mean"
        )
        with torch.no_grad():
            ship_mae = float((ship_pred_sel - ship_target_sel).abs().mean().item())
            fire_count = int(should_learn_ship.sum().item())

    total = weights.target_w * target_loss + weights.ship_w * ship_loss
    return PerPlanetLossReport(
        total=total,
        target_loss=target_loss.detach(),
        ship_loss=ship_loss.detach() if isinstance(ship_loss, torch.Tensor) else zero,
        target_acc=target_acc,
        target_noop_acc=target_noop_acc,
        target_fire_acc=target_fire_acc,
        ship_mae=ship_mae,
        fire_count=fire_count,
    )


__all__ = [
    "LossWeights",
    "PerPlanetLossReport",
    "compute_per_planet_loss",
]
