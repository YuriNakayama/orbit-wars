"""2-head BC loss for imitation/case8 (candidate categorical + ship-count regression).

Per active source (my_planet_mask AND cand_slot_per_src != -1):

  cand_loss = CE(candidate_logits, cand_slot_label, weight=class_weights)
              or focal_loss(candidate_logits, label, alpha, gamma, weight)

For sources whose label slot != 0 (= fire) AND ship_label_per_src != -1:

  ship_loss = SmoothL1(ship_pred, ship_label)  # Huber delta=1, robust to outliers

Total: total = cand_w * cand_loss + ship_w * ship_loss

iter3 (2026-05-05): focal loss option added to mitigate cand head oscillation
caused by extreme noop/fire label imbalance. CE + label_smoothing kept as
default for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case9.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case9.policy.types import PolicyOutput


@dataclass(frozen=True)
class LossWeights:
    cand_w: float = 1.0
    cand_class_weights: torch.Tensor | None = None
    label_smoothing: float = 0.0
    ship_w: float = 1.0
    # iter3: cand_loss_type="focal" で focal loss、"ce" (default) で従来 CE。
    # focal は label imbalance に頑健 (easy examples を down-weight)、
    # FL(p_t) = -α_t (1 - p_t)^γ log(p_t) (Lin et al., 2017).
    cand_loss_type: str = "ce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dual_alpha: float = 0.5


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


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al., 2017).

    `weight` (per-class) is applied multiplicatively after the focal term, so
    the iter1/iter2 inverse-frequency class_weight remains compatible.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal_factor = (1.0 - pt).clamp_min(1e-12).pow(gamma)
    loss = -alpha * focal_factor * log_pt
    if weight is not None:
        weight_t = weight.to(logits.device).gather(-1, targets)
        loss = loss * weight_t
    mean_loss: torch.Tensor = loss.mean()
    return mean_loss


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
    assert output.candidate_logits is not None
    sel_logits = output.candidate_logits[b_idx, src_idx]  # (N, CAND_K)
    sel_labels = cand_slot_per_src[b_idx, src_idx]  # (N,)

    cand_cw = (
        weights.cand_class_weights.to(sel_logits.device)
        if weights.cand_class_weights is not None
        else None
    )
    loss_type = weights.cand_loss_type.lower()
    if loss_type == "focal":
        cand_loss = _focal_cross_entropy(
            sel_logits,
            sel_labels,
            alpha=weights.focal_alpha,
            gamma=weights.focal_gamma,
            weight=cand_cw,
        )
    elif loss_type == "ce":
        cand_loss = nn.functional.cross_entropy(
            sel_logits,
            sel_labels,
            weight=cand_cw,
            label_smoothing=weights.label_smoothing,
        )
    else:
        raise ValueError(
            f"unknown cand_loss_type {weights.cand_loss_type!r}"
            " (expected 'ce' or 'focal')"
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
            assert output.ship_pred is not None
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


@dataclass(frozen=True)
class ThreeHeadLossReport:
    total: torch.Tensor
    from_loss: torch.Tensor
    target_loss: torch.Tensor
    ships_loss: torch.Tensor
    from_acc: float
    target_acc: float
    ships_acc: float
    fire_count: int


def compute_three_head_loss(
    output: PolicyOutput,
    from_multihot: torch.Tensor,  # (B, P) bool
    target_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    ships_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    my_planet_mask: torch.Tensor,  # (B, P) bool
    from_w: float = 1.0,
    target_w: float = 1.0,
    ships_w: float = 0.5,
    pos_weight: float = 5.0,
    label_smoothing: float = 0.1,
) -> ThreeHeadLossReport:
    """3-head BC loss (case7 流): from BCE + target CE + ships CE."""
    device = my_planet_mask.device
    zero = torch.zeros((), device=device)

    assert output.from_logits is not None
    assert output.target_logits is not None
    assert output.ships_logits is not None

    # from head: per-source binary fire/no-fire (focal-style with pos_weight).
    from_logits_sel = output.from_logits[my_planet_mask]
    from_target_sel = from_multihot[my_planet_mask].to(from_logits_sel.dtype)
    pw = torch.tensor([pos_weight], device=device, dtype=from_logits_sel.dtype)
    from_loss = nn.functional.binary_cross_entropy_with_logits(
        from_logits_sel, from_target_sel, pos_weight=pw
    )
    with torch.no_grad():
        from_pred = (from_logits_sel > 0.0).to(from_target_sel.dtype)
        from_acc = float((from_pred == from_target_sel).float().mean().item())

    # target head: per-fired-source CE over NUM_TEMPLATES.
    fire_valid = my_planet_mask & (target_per_src != -1)
    if fire_valid.any():
        b_idx, s_idx = fire_valid.nonzero(as_tuple=True)
        tgt_logits = output.target_logits[b_idx, s_idx]  # (N, NUM_TEMPLATES)
        tgt_labels = target_per_src[b_idx, s_idx]
        target_loss = nn.functional.cross_entropy(
            tgt_logits, tgt_labels, label_smoothing=label_smoothing
        )
        with torch.no_grad():
            target_acc = float(
                (tgt_logits.argmax(dim=-1) == tgt_labels).float().mean().item()
            )
        fire_count = int(fire_valid.sum().item())
    else:
        target_loss = zero
        target_acc = 0.0
        fire_count = 0

    ships_valid = my_planet_mask & (ships_per_src != -1)
    if ships_valid.any():
        b_idx, s_idx = ships_valid.nonzero(as_tuple=True)
        ships_logits_sel = output.ships_logits[b_idx, s_idx]  # (N, ships_buckets)
        ships_labels = ships_per_src[b_idx, s_idx]
        ships_loss = nn.functional.cross_entropy(
            ships_logits_sel, ships_labels, label_smoothing=label_smoothing
        )
        with torch.no_grad():
            ships_acc = float(
                (ships_logits_sel.argmax(dim=-1) == ships_labels).float().mean().item()
            )
    else:
        ships_loss = zero
        ships_acc = 0.0

    total = from_w * from_loss + target_w * target_loss + ships_w * ships_loss
    return ThreeHeadLossReport(
        total=total,
        from_loss=from_loss.detach(),
        target_loss=target_loss.detach()
        if isinstance(target_loss, torch.Tensor)
        else zero,
        ships_loss=ships_loss.detach()
        if isinstance(ships_loss, torch.Tensor)
        else zero,
        from_acc=from_acc,
        target_acc=target_acc,
        ships_acc=ships_acc,
        fire_count=fire_count,
    )


@dataclass(frozen=True)
class CandidateShipsLossReport:
    total: torch.Tensor
    cand_loss: torch.Tensor
    ships_loss: torch.Tensor
    cand_acc: float
    cand_noop_acc: float
    cand_fire_acc: float
    ships_acc: float
    fire_count: int


@dataclass(frozen=True)
class TemplateShipsLossReport:
    total: torch.Tensor
    template_loss: torch.Tensor
    ships_loss: torch.Tensor
    template_acc: float
    template_noop_acc: float
    template_fire_acc: float
    ships_acc: float
    fire_count: int


def compute_candidate_ships_loss(
    output: PolicyOutput,
    cand_slot_per_src: torch.Tensor,  # (B, P) int64; -1 = unused
    ships_bucket_per_src: torch.Tensor,  # (B, P) int64; -1 = unused (fired only)
    my_planet_mask: torch.Tensor,  # (B, P) bool
    weights: LossWeights,
    ships_w: float = 0.5,
    label_smoothing: float = 0.1,
) -> CandidateShipsLossReport:
    """candidate categorical (cand_logits) + ships 4-bucket categorical."""
    device = my_planet_mask.device
    zero = torch.zeros((), device=device)

    valid = my_planet_mask & (cand_slot_per_src != -1)
    if not valid.any():
        return CandidateShipsLossReport(
            total=zero,
            cand_loss=zero,
            ships_loss=zero,
            cand_acc=0.0,
            cand_noop_acc=0.0,
            cand_fire_acc=0.0,
            ships_acc=0.0,
            fire_count=0,
        )

    b_idx, src_idx = valid.nonzero(as_tuple=True)
    assert output.candidate_logits is not None
    sel_logits = output.candidate_logits[b_idx, src_idx]
    sel_labels = cand_slot_per_src[b_idx, src_idx]

    cand_cw = (
        weights.cand_class_weights.to(sel_logits.device)
        if weights.cand_class_weights is not None
        else None
    )
    if weights.cand_loss_type.lower() == "focal":
        cand_loss = _focal_cross_entropy(
            sel_logits,
            sel_labels,
            alpha=weights.focal_alpha,
            gamma=weights.focal_gamma,
            weight=cand_cw,
        )
    else:
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

    ships_valid = my_planet_mask & (ships_bucket_per_src != -1)
    fire_count = 0
    if ships_valid.any():
        assert output.ships_logits is not None
        sb_idx, ss_idx = ships_valid.nonzero(as_tuple=True)
        ships_logits_sel = output.ships_logits[sb_idx, ss_idx]
        ships_labels = ships_bucket_per_src[sb_idx, ss_idx]
        ships_loss = nn.functional.cross_entropy(
            ships_logits_sel, ships_labels, label_smoothing=label_smoothing
        )
        with torch.no_grad():
            ships_acc = float(
                (ships_logits_sel.argmax(dim=-1) == ships_labels).float().mean().item()
            )
            fire_count = int(ships_valid.sum().item())
    else:
        ships_loss = zero
        ships_acc = 0.0

    total = weights.cand_w * cand_loss + ships_w * ships_loss
    ships_loss_detached = ships_loss.detach()
    return CandidateShipsLossReport(
        total=total,
        cand_loss=cand_loss.detach(),
        ships_loss=ships_loss_detached,
        cand_acc=cand_acc,
        cand_noop_acc=cand_noop_acc,
        cand_fire_acc=cand_fire_acc,
        ships_acc=ships_acc,
        fire_count=fire_count,
    )


def compute_template_ships_loss(
    output: PolicyOutput,
    target_per_src: torch.Tensor,  # (B, P) int64; -1 = no-op / unused
    ships_per_src: torch.Tensor,  # (B, P) int64; -1 = unused (fired only)
    my_planet_mask: torch.Tensor,  # (B, P) bool
    template_w: float = 1.0,
    ships_w: float = 0.5,
    label_smoothing: float = 0.1,
) -> TemplateShipsLossReport:
    """Template categorical incl no-op + ships 4-bucket categorical.

    Unlike `three_head`, no-op/fire is represented directly as the final
    template class.  This makes it structurally comparable with
    `candidate_ships`, where slot 0 is no-op.
    """
    device = my_planet_mask.device
    zero = torch.zeros((), device=device)

    assert output.target_logits is not None
    assert output.ships_logits is not None

    if not my_planet_mask.any():
        return TemplateShipsLossReport(
            total=zero,
            template_loss=zero,
            ships_loss=zero,
            template_acc=0.0,
            template_noop_acc=0.0,
            template_fire_acc=0.0,
            ships_acc=0.0,
            fire_count=0,
        )

    noop_id = NUM_TEMPLATES - 1
    template_labels = torch.where(
        target_per_src >= 0,
        target_per_src,
        torch.full_like(target_per_src, noop_id),
    )
    b_idx, s_idx = my_planet_mask.nonzero(as_tuple=True)
    tmpl_logits = output.target_logits[b_idx, s_idx]
    tmpl_labels = template_labels[b_idx, s_idx]
    template_loss = nn.functional.cross_entropy(
        tmpl_logits, tmpl_labels, label_smoothing=label_smoothing
    )
    with torch.no_grad():
        pred = tmpl_logits.argmax(dim=-1)
        match = (pred == tmpl_labels).float()
        template_acc = float(match.mean().item())
        is_noop = tmpl_labels == noop_id
        is_fire = ~is_noop
        n_noop = int(is_noop.sum().item())
        n_fire = int(is_fire.sum().item())
        template_noop_acc = float(match[is_noop].mean().item()) if n_noop else 0.0
        template_fire_acc = float(match[is_fire].mean().item()) if n_fire else 0.0

    ships_valid = my_planet_mask & (ships_per_src != -1)
    fire_count = 0
    if ships_valid.any():
        sb_idx, ss_idx = ships_valid.nonzero(as_tuple=True)
        ships_logits_sel = output.ships_logits[sb_idx, ss_idx]
        ships_labels = ships_per_src[sb_idx, ss_idx]
        ships_loss = nn.functional.cross_entropy(
            ships_logits_sel, ships_labels, label_smoothing=label_smoothing
        )
        with torch.no_grad():
            ships_acc = float(
                (ships_logits_sel.argmax(dim=-1) == ships_labels)
                .float()
                .mean()
                .item()
            )
            fire_count = int(ships_valid.sum().item())
    else:
        ships_loss = zero
        ships_acc = 0.0

    total = template_w * template_loss + ships_w * ships_loss
    return TemplateShipsLossReport(
        total=total,
        template_loss=template_loss.detach(),
        ships_loss=ships_loss.detach(),
        template_acc=template_acc,
        template_noop_acc=template_noop_acc,
        template_fire_acc=template_fire_acc,
        ships_acc=ships_acc,
        fire_count=fire_count,
    )


@dataclass(frozen=True)
class DualHeadLossReport:
    total: torch.Tensor
    three: ThreeHeadLossReport
    candidate: LossReport
    alpha: float


def compute_dual_head_loss(
    output: PolicyOutput,
    from_multihot: torch.Tensor,
    target_per_src: torch.Tensor,
    ships_per_src: torch.Tensor,
    cand_slot_per_src: torch.Tensor,
    ship_label_per_src: torch.Tensor,
    my_planet_mask: torch.Tensor,
    weights: LossWeights,
) -> DualHeadLossReport:
    """Blend 3-head and candidate losses on the shared backbone."""
    alpha = max(0.0, min(1.0, float(weights.dual_alpha)))
    three = compute_three_head_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
    )
    candidate = compute_loss(
        output,
        cand_slot_per_src=cand_slot_per_src,
        my_planet_mask=my_planet_mask,
        weights=weights,
        ship_label_per_src=ship_label_per_src,
    )
    total = alpha * three.total + (1.0 - alpha) * candidate.total
    return DualHeadLossReport(
        total=total, three=three, candidate=candidate, alpha=alpha
    )
