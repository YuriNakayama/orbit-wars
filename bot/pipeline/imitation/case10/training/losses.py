"""Losses for imitation/case10 candidate/template heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pipeline.imitation.case10.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case10.policy.types import PolicyOutput


@dataclass(frozen=True)
class LossWeights:
    cand_w: float = 1.0
    cand_class_weights: torch.Tensor | None = None
    template_class_weights: torch.Tensor | None = None
    ships_class_weights: torch.Tensor | None = None
    label_smoothing: float = 0.0
    ship_w: float = 1.0
    template_w: float = 1.0
    cand_loss_type: str = "ce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


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


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    loss = -alpha * (1.0 - pt).clamp_min(1e-12).pow(gamma) * log_pt
    if weight is not None:
        loss = loss * weight.to(logits.device).gather(-1, targets)
    return loss.mean()


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: LossWeights,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights.cand_loss_type.lower() == "focal":
        return _focal_cross_entropy(
            logits,
            labels,
            alpha=weights.focal_alpha,
            gamma=weights.focal_gamma,
            weight=class_weights,
        )
    return nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=weights.label_smoothing,
    )


def compute_candidate_ships_loss(
    output: PolicyOutput,
    cand_slot_per_src: torch.Tensor,
    ships_bucket_per_src: torch.Tensor,
    my_planet_mask: torch.Tensor,
    weights: LossWeights,
    ships_w: float = 0.5,
    label_smoothing: float = 0.1,
) -> CandidateShipsLossReport:
    """candidate categorical + ships 4-bucket categorical."""
    device = my_planet_mask.device
    zero = torch.zeros((), device=device)
    valid = my_planet_mask & (cand_slot_per_src != -1)
    if not valid.any():
        return CandidateShipsLossReport(zero, zero, zero, 0.0, 0.0, 0.0, 0.0, 0)

    b_idx, src_idx = valid.nonzero(as_tuple=True)
    assert output.candidate_logits is not None
    sel_logits = output.candidate_logits[b_idx, src_idx]
    sel_labels = cand_slot_per_src[b_idx, src_idx]
    cand_cw = (
        weights.cand_class_weights.to(sel_logits.device)
        if weights.cand_class_weights is not None
        else None
    )
    cand_loss = _classification_loss(sel_logits, sel_labels, weights, cand_cw)

    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)
        match = (pred == sel_labels).float()
        is_noop = sel_labels == 0
        is_fire = ~is_noop
        cand_acc = float(match.mean().item())
        cand_noop_acc = float(match[is_noop].mean().item()) if is_noop.any() else 0.0
        cand_fire_acc = float(match[is_fire].mean().item()) if is_fire.any() else 0.0

    ships_valid = my_planet_mask & (ships_bucket_per_src != -1)
    if ships_valid.any():
        assert output.ships_logits is not None
        sb_idx, ss_idx = ships_valid.nonzero(as_tuple=True)
        ships_logits_sel = output.ships_logits[sb_idx, ss_idx]
        ships_labels = ships_bucket_per_src[sb_idx, ss_idx]
        ship_cw = (
            weights.ships_class_weights.to(ships_logits_sel.device)
            if weights.ships_class_weights is not None
            else None
        )
        ships_loss = nn.functional.cross_entropy(
            ships_logits_sel,
            ships_labels,
            weight=ship_cw,
            label_smoothing=label_smoothing,
        )
        with torch.no_grad():
            ships_acc = float(
                (ships_logits_sel.argmax(dim=-1) == ships_labels).float().mean().item()
            )
            fire_count = int(ships_valid.sum().item())
    else:
        ships_loss = zero
        ships_acc = 0.0
        fire_count = 0

    total = weights.cand_w * cand_loss + ships_w * ships_loss
    return CandidateShipsLossReport(
        total=total,
        cand_loss=cand_loss.detach(),
        ships_loss=ships_loss.detach(),
        cand_acc=cand_acc,
        cand_noop_acc=cand_noop_acc,
        cand_fire_acc=cand_fire_acc,
        ships_acc=ships_acc,
        fire_count=fire_count,
    )


def compute_template_ships_loss(
    output: PolicyOutput,
    target_per_src: torch.Tensor,
    ships_per_src: torch.Tensor,
    my_planet_mask: torch.Tensor,
    weights: LossWeights,
) -> TemplateShipsLossReport:
    """Template categorical incl no-op + ships 4-bucket categorical."""
    device = my_planet_mask.device
    zero = torch.zeros((), device=device)
    if not my_planet_mask.any():
        return TemplateShipsLossReport(zero, zero, zero, 0.0, 0.0, 0.0, 0.0, 0)

    assert output.target_logits is not None
    assert output.ships_logits is not None
    noop_id = NUM_TEMPLATES - 1
    template_labels = torch.where(
        target_per_src >= 0,
        target_per_src,
        torch.full_like(target_per_src, noop_id),
    )
    b_idx, s_idx = my_planet_mask.nonzero(as_tuple=True)
    tmpl_logits = output.target_logits[b_idx, s_idx]
    tmpl_labels = template_labels[b_idx, s_idx]
    tmpl_cw = (
        weights.template_class_weights.to(tmpl_logits.device)
        if weights.template_class_weights is not None
        else None
    )
    template_loss = _classification_loss(tmpl_logits, tmpl_labels, weights, tmpl_cw)

    with torch.no_grad():
        pred = tmpl_logits.argmax(dim=-1)
        match = (pred == tmpl_labels).float()
        is_noop = tmpl_labels == noop_id
        is_fire = ~is_noop
        template_acc = float(match.mean().item())
        template_noop_acc = (
            float(match[is_noop].mean().item()) if is_noop.any() else 0.0
        )
        template_fire_acc = (
            float(match[is_fire].mean().item()) if is_fire.any() else 0.0
        )

    ships_valid = my_planet_mask & (ships_per_src != -1)
    if ships_valid.any():
        sb_idx, ss_idx = ships_valid.nonzero(as_tuple=True)
        ships_logits_sel = output.ships_logits[sb_idx, ss_idx]
        ships_labels = ships_per_src[sb_idx, ss_idx]
        ship_cw = (
            weights.ships_class_weights.to(ships_logits_sel.device)
            if weights.ships_class_weights is not None
            else None
        )
        ships_loss = nn.functional.cross_entropy(
            ships_logits_sel,
            ships_labels,
            weight=ship_cw,
            label_smoothing=weights.label_smoothing,
        )
        with torch.no_grad():
            ships_acc = float(
                (ships_logits_sel.argmax(dim=-1) == ships_labels).float().mean().item()
            )
            fire_count = int(ships_valid.sum().item())
    else:
        ships_loss = zero
        ships_acc = 0.0
        fire_count = 0

    total = weights.template_w * template_loss + weights.ship_w * ships_loss
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
