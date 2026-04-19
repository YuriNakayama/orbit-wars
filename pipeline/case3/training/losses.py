"""3-head BC loss for case3 IL baseline.

We treat the action label as (from_planet, target_planet, ships_bucket).
For no-op frames we still supervise the from-head (predict no fire from any
planet) but skip target/ships heads — they are conditioned on a chosen src.

Implementation:
- from_head: BCEWithLogits per planet, with positive class = "fire from this".
- target_head: cross-entropy over MAX_PLANETS+1 candidates conditioned on the
  ground-truth from slot. Skipped when from_label == NO_OP_LABEL.
- ships_head: cross-entropy over ships_buckets conditioned on the ground-truth
  from slot. Skipped when from_label == NO_OP_LABEL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.case3.policy.featurizer import MAX_PLANETS
from pipeline.case3.policy.types import PolicyOutput

NO_OP_LABEL = MAX_PLANETS


@dataclass(frozen=True)
class LossWeights:
    from_w: float = 1.0
    target_w: float = 1.0
    ships_w: float = 0.5


@dataclass(frozen=True)
class LossReport:
    total: torch.Tensor
    from_loss: torch.Tensor
    target_loss: torch.Tensor
    ships_loss: torch.Tensor
    from_acc: float
    target_acc: float
    ships_acc: float


def _select_per_src_logits(
    logits: torch.Tensor, from_label: torch.Tensor
) -> torch.Tensor:
    """Pick (B, K) slice from (B, P, K) using from_label index per row.

    Rows with from_label == NO_OP_LABEL get a zero placeholder (caller masks).
    """
    safe_idx = from_label.clamp(max=MAX_PLANETS - 1).unsqueeze(-1).unsqueeze(-1)
    safe_idx = safe_idx.expand(-1, 1, logits.size(-1))
    return logits.gather(1, safe_idx).squeeze(1)


def compute_loss(
    output: PolicyOutput,
    from_label: torch.Tensor,
    target_label: torch.Tensor,
    ships_label: torch.Tensor,
    my_planet_mask: torch.Tensor,
    weights: LossWeights,
) -> LossReport:
    b = from_label.size(0)
    device = from_label.device

    # ---- from head: per-planet BCE ----
    from_target = torch.zeros_like(output.from_logits)  # (B, P)
    is_action = from_label != NO_OP_LABEL
    if is_action.any():
        rows = torch.arange(b, device=device)[is_action]
        from_target[rows, from_label[is_action]] = 1.0
    bce = nn.functional.binary_cross_entropy_with_logits(
        output.from_logits.masked_fill(~my_planet_mask, 0.0),
        from_target,
        reduction="none",
    )
    bce = bce * my_planet_mask.float()
    from_loss = bce.sum() / my_planet_mask.float().sum().clamp_min(1.0)

    # ---- target head: CE conditioned on gt from ----
    if is_action.any():
        sel_target_logits = _select_per_src_logits(
            output.target_logits[is_action], from_label[is_action]
        )
        target_loss = nn.functional.cross_entropy(
            sel_target_logits, target_label[is_action]
        )
        target_pred = sel_target_logits.argmax(dim=-1)
        target_acc = float(
            (target_pred == target_label[is_action]).float().mean().item()
        )
    else:
        target_loss = torch.zeros((), device=device)
        target_acc = 0.0

    # ---- ships head: CE conditioned on gt from ----
    if is_action.any():
        sel_ships_logits = _select_per_src_logits(
            output.ships_logits[is_action], from_label[is_action]
        )
        ships_loss = nn.functional.cross_entropy(
            sel_ships_logits, ships_label[is_action]
        )
        ships_pred = sel_ships_logits.argmax(dim=-1)
        ships_acc = float((ships_pred == ships_label[is_action]).float().mean().item())
    else:
        ships_loss = torch.zeros((), device=device)
        ships_acc = 0.0

    # from accuracy: argmax of from_logits == from_label (or no-op when all -inf)
    from_pred = output.from_logits.argmax(dim=-1)
    from_acc = float(
        ((from_pred == from_label) | (~is_action & (from_label == NO_OP_LABEL)))
        .float()
        .mean()
        .item()
    )

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
