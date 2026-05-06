"""Unit tests for iter3 focal loss in losses.py."""

from __future__ import annotations

import math

import torch
from torch import nn

from pipeline.imitation.case8.policy.candidates import CAND_K
from pipeline.imitation.case8.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case8.policy.types import PolicyOutput
from pipeline.imitation.case8.training.losses import (
    LossWeights,
    _focal_cross_entropy,
    compute_loss,
)


def test_focal_with_gamma_zero_alpha_one_matches_ce_up_to_scale() -> None:
    """gamma=0, alpha=1 → focal = -log_pt = CE (no class_weight)."""
    torch.manual_seed(0)
    logits = torch.randn(8, CAND_K, dtype=torch.float32)
    targets = torch.randint(0, CAND_K, (8,), dtype=torch.long)

    focal = _focal_cross_entropy(logits, targets, alpha=1.0, gamma=0.0, weight=None)
    ce = nn.functional.cross_entropy(logits, targets, reduction="mean")

    assert math.isclose(float(focal.item()), float(ce.item()), rel_tol=1e-5)


def test_focal_down_weights_easy_examples() -> None:
    """For correctly-classified easy examples, focal loss < CE loss."""
    # 2 samples: both correctly predicted with high confidence (logit=10)
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    focal = _focal_cross_entropy(logits, targets, alpha=1.0, gamma=2.0, weight=None)
    ce = nn.functional.cross_entropy(logits, targets, reduction="mean")

    # focal は (1-pt)^gamma が極めて小さくなるので CE より大幅に小
    assert float(focal.item()) < float(ce.item())
    assert float(focal.item()) < 1.0e-6


def test_focal_keeps_hard_examples() -> None:
    """For hard examples (uniform logits), focal ~= alpha * gamma_factor * CE."""
    # Uniform → pt = 1/3, log_pt = -log(3), (1-pt)^gamma = (2/3)^2 = 4/9
    logits = torch.zeros(4, 3, dtype=torch.float32)
    targets = torch.zeros(4, dtype=torch.long)

    focal = _focal_cross_entropy(logits, targets, alpha=0.25, gamma=2.0, weight=None)
    expected = 0.25 * (2.0 / 3.0) ** 2 * math.log(3.0)
    assert math.isclose(float(focal.item()), expected, rel_tol=1e-5)


def test_focal_gradient_finite() -> None:
    """Backward through focal loss produces finite gradients."""
    torch.manual_seed(0)
    logits = torch.randn(16, CAND_K, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, CAND_K, (16,), dtype=torch.long)

    focal = _focal_cross_entropy(logits, targets, alpha=0.25, gamma=2.0, weight=None)
    focal.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_with_class_weight_applied_multiplicatively() -> None:
    """Class weight multiplies after focal term."""
    logits = torch.zeros(2, 3, dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    weight = torch.tensor([1.0, 5.0, 1.0], dtype=torch.float32)

    focal_no_w = _focal_cross_entropy(
        logits, targets, alpha=1.0, gamma=2.0, weight=None
    )
    focal_w = _focal_cross_entropy(logits, targets, alpha=1.0, gamma=2.0, weight=weight)
    # weight=[1, 5] applied to two equal-loss samples → mean = (1*L + 5*L)/2 = 3L
    assert math.isclose(
        float(focal_w.item()), 3.0 * float(focal_no_w.item()), rel_tol=1e-5
    )


def test_compute_loss_focal_path_runs() -> None:
    """End-to-end: compute_loss with cand_loss_type='focal' returns finite loss."""
    cand_logits = torch.randn(1, MAX_PLANETS, CAND_K, dtype=torch.float32)
    ship_pred = torch.full((1, MAX_PLANETS), 25.0, dtype=torch.float32)
    output = PolicyOutput(candidate_logits=cand_logits, ship_pred=ship_pred)

    cand_slot = torch.full((1, MAX_PLANETS), -1, dtype=torch.long)
    cand_slot[0, 0] = 1  # one fire src
    cand_slot[0, 1] = 0  # one no-op src
    my_mask = torch.zeros((1, MAX_PLANETS), dtype=torch.bool)
    my_mask[0, 0] = True
    my_mask[0, 1] = True

    weights = LossWeights(
        cand_w=1.0,
        ship_w=1.0,
        cand_loss_type="focal",
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    report = compute_loss(output, cand_slot, my_mask, weights)
    assert torch.isfinite(report.cand_loss)
    assert torch.isfinite(report.total)


def test_compute_loss_unknown_cand_loss_type_raises() -> None:
    cand_logits = torch.randn(1, MAX_PLANETS, CAND_K, dtype=torch.float32)
    ship_pred = torch.zeros((1, MAX_PLANETS), dtype=torch.float32)
    output = PolicyOutput(candidate_logits=cand_logits, ship_pred=ship_pred)
    cand_slot = torch.zeros((1, MAX_PLANETS), dtype=torch.long)
    my_mask = torch.ones((1, MAX_PLANETS), dtype=torch.bool)

    weights = LossWeights(cand_w=1.0, cand_loss_type="bogus")
    try:
        compute_loss(output, cand_slot, my_mask, weights)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("ValueError was not raised")
