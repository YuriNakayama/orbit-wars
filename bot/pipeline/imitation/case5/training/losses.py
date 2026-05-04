"""3-head BC loss for imitation/case5 IL baseline.

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

from pipeline.imitation.case5.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case5.policy.types import PolicyOutput

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
    # Phase 2: per-class weight tensors (length NUM_TEMPLATES / SHIPS_BUCKETS),
    # passed into F.cross_entropy(weight=...). None disables weighting.
    target_class_weights: torch.Tensor | None = None
    ships_class_weights: torch.Tensor | None = None
    # Phase 2: ordinal-aware label smoothing for ships head. When > 0, the true
    # bucket receives (1 - α) mass and each *adjacent* bucket gets α/2 (edge
    # buckets keep the full spill on the single neighbour). 0 disables it.
    ships_ordinal_smoothing: float = 0.0
    # Phase 3-B2: multiclass focal on ships head. Replaces plain CE with
    # (1-p_t)^γ * CE so well-classified majority bucket 3 examples contribute
    # less to gradients. γ=0 disables it (= plain CE). α is optional per-class
    # scaling — use to lift minority buckets further or leave as None.
    ships_focal_gamma: float = 0.0
    ships_focal_alpha: torch.Tensor | None = None


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    beta: float = 0.9999,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Effective-number-of-samples class weights (Cui et al. 2019).

    For each class c: w_c = (1 - β) / (1 - β^{n_c}), then normalize so the mean
    weight is 1.0 — this keeps the overall loss scale comparable to unweighted
    CE and avoids having to retune per-head loss coefficients.

    β=0.9999 gives a gentler re-balance than pure 1/freq: majority classes are
    only modestly down-weighted, minority classes are lifted without exploding.
    """
    labels = labels.flatten()
    labels = labels[labels != ignore_index]
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float64)
    # "effective number": (1 - β^n) / (1 - β)
    # Empty classes get count=0 → eff_num=0 → weight would be inf; clamp to 1.
    eff_num = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)) / (
        1.0 - beta
    )
    eff_num = eff_num.clamp_min(1.0)
    weights = (1.0 - beta) / (
        (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)).clamp_min(
            1e-12
        )
    )
    # Replace inf (empty class) with 0 so it does not affect training.
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    # Normalize so weighted mean over present classes == 1.0.
    present = (counts > 0).to(torch.float64)
    n_present = present.sum().clamp_min(1.0)
    weights = weights * (n_present / weights.sum().clamp_min(1e-12))
    return weights.to(torch.float32)


def _smooth_ordinal_labels(
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> torch.Tensor:
    """Build soft targets that leak `alpha` mass to the adjacent bucket(s).

    Example (num_classes=4, alpha=0.1): bucket 2 → [0, 0, 0.90, 0.10];
    bucket 1 → [0, 0.90, 0.05, 0.05] if both neighbours exist, otherwise the
    α mass spills entirely to the single neighbour.
    """
    n = labels.numel()
    soft = torch.zeros((n, num_classes), dtype=torch.float32, device=labels.device)
    for i, c in enumerate(labels.tolist()):
        below = c - 1 >= 0
        above = c + 1 < num_classes
        spill_slots = int(below) + int(above)
        if spill_slots == 0:
            soft[i, c] = 1.0
            continue
        per_neighbour = alpha / spill_slots
        soft[i, c] = 1.0 - alpha
        if below:
            soft[i, c - 1] = per_neighbour
        if above:
            soft[i, c + 1] = per_neighbour
    return soft


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
    alpha_t = weights.from_focal_alpha * from_target + (
        1.0 - weights.from_focal_alpha
    ) * (1.0 - from_target)
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

        target_cw = None
        if weights.target_class_weights is not None:
            target_cw = weights.target_class_weights.to(sel_target_logits.device)
        target_loss = nn.functional.cross_entropy(
            sel_target_logits,
            target_labels,
            weight=target_cw,
            label_smoothing=weights.target_label_smoothing,
        )
        if weights.target_entropy_bonus > 0.0:
            log_p = nn.functional.log_softmax(sel_target_logits, dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1).mean()
            # subtract entropy → encourages higher entropy (anti-collapse)
            target_loss = target_loss - weights.target_entropy_bonus * entropy

        ships_cw = None
        if weights.ships_class_weights is not None:
            ships_cw = weights.ships_class_weights.to(sel_ships_logits.device)
        if weights.ships_focal_gamma > 0.0:
            # Multiclass focal: -α_t * (1 - p_t)^γ * log p_t.
            # p_t is the softmax prob on the true class for each sample.
            log_p = nn.functional.log_softmax(sel_ships_logits, dim=-1)
            p = log_p.exp()
            p_t = p.gather(1, ships_labels.view(-1, 1)).squeeze(1).clamp_min(1e-8)
            log_p_t = log_p.gather(1, ships_labels.view(-1, 1)).squeeze(1)
            focal_factor = (1.0 - p_t).pow(weights.ships_focal_gamma)
            per_sample = -focal_factor * log_p_t
            if weights.ships_focal_alpha is not None:
                alpha = weights.ships_focal_alpha.to(sel_ships_logits.device)
                per_sample = per_sample * alpha[ships_labels]
            if ships_cw is not None:
                sample_w = ships_cw[ships_labels]
                ships_loss = (per_sample * sample_w).sum() / sample_w.sum().clamp_min(
                    1e-6
                )
            else:
                ships_loss = per_sample.mean()
        elif weights.ships_ordinal_smoothing > 0.0:
            # Manual CE over soft ordinal targets; class weights applied per-sample
            # via the true-label column (matches CE's per-sample weighting).
            num_ships_classes = sel_ships_logits.size(-1)
            soft = _smooth_ordinal_labels(
                ships_labels, num_ships_classes, weights.ships_ordinal_smoothing
            )
            log_p = nn.functional.log_softmax(sel_ships_logits, dim=-1)
            per_sample = -(soft * log_p).sum(dim=-1)
            if ships_cw is not None:
                sample_w = ships_cw[ships_labels]
                ships_loss = (per_sample * sample_w).sum() / sample_w.sum().clamp_min(
                    1e-6
                )
            else:
                ships_loss = per_sample.mean()
        else:
            ships_loss = nn.functional.cross_entropy(
                sel_ships_logits, ships_labels, weight=ships_cw
            )

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
