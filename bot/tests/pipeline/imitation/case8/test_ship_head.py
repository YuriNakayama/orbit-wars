"""Tests for case8 ship-count regression head and 2-head loss."""

from __future__ import annotations

import torch

from pipeline.imitation.case8.policy.candidates import CAND_K
from pipeline.imitation.case8.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case8.policy.types import PolicyOutput
from pipeline.imitation.case8.training.losses import LossWeights, compute_loss


def _make_output(b: int, p: int, k: int, ship_value: float = 30.0) -> PolicyOutput:
    cand_logits = torch.zeros((b, p, k), dtype=torch.float32)
    cand_logits[..., 1] = 5.0  # prefer slot 1
    ship_pred = torch.full((b, p), ship_value, dtype=torch.float32)
    return PolicyOutput(candidate_logits=cand_logits, ship_pred=ship_pred)


def test_ship_loss_zero_when_no_fired_labels() -> None:
    out = _make_output(1, MAX_PLANETS, CAND_K)
    cand_slot = torch.zeros((1, MAX_PLANETS), dtype=torch.long)  # all no-op label
    my_mask = torch.ones((1, MAX_PLANETS), dtype=torch.bool)
    ship_label = torch.full((1, MAX_PLANETS), -1, dtype=torch.long)

    weights = LossWeights(cand_w=1.0, ship_w=1.0)
    report = compute_loss(
        out, cand_slot, my_mask, weights, ship_label_per_src=ship_label
    )

    assert float(report.ship_loss.item()) == 0.0
    assert report.ship_count == 0


def test_ship_loss_positive_when_pred_off_target() -> None:
    out = _make_output(1, MAX_PLANETS, CAND_K, ship_value=30.0)
    cand_slot = torch.zeros((1, MAX_PLANETS), dtype=torch.long)
    cand_slot[0, 0] = 1  # source 0 fired (slot 1)
    my_mask = torch.zeros((1, MAX_PLANETS), dtype=torch.bool)
    my_mask[0, 0] = True
    ship_label = torch.full((1, MAX_PLANETS), -1, dtype=torch.long)
    ship_label[0, 0] = 100  # actual fired 100 ships

    weights = LossWeights(cand_w=1.0, ship_w=1.0)
    report = compute_loss(
        out, cand_slot, my_mask, weights, ship_label_per_src=ship_label
    )

    assert report.ship_count == 1
    # SmoothL1(30, 100) = |30-100| - 0.5 = 69.5
    assert float(report.ship_loss.item()) > 50.0
    assert report.ship_mae > 60.0


def test_ship_loss_uses_smooth_l1() -> None:
    """SmoothL1 is quadratic for small errors; verify at |diff|=0.5."""
    out = _make_output(1, 1, CAND_K, ship_value=20.5)
    cand_slot = torch.tensor([[1]], dtype=torch.long)
    my_mask = torch.tensor([[True]], dtype=torch.bool)
    ship_label = torch.tensor([[20]], dtype=torch.long)

    weights = LossWeights(cand_w=0.0, ship_w=1.0)  # isolate ship loss
    report = compute_loss(
        out, cand_slot, my_mask, weights, ship_label_per_src=ship_label
    )

    # SmoothL1(0.5) = 0.5 * 0.5^2 = 0.125
    assert abs(float(report.ship_loss.item()) - 0.125) < 1e-4


def test_ship_label_omitted_falls_back_to_zero() -> None:
    out = _make_output(1, 4, CAND_K)
    cand_slot = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    my_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    weights = LossWeights(cand_w=1.0, ship_w=1.0)
    report = compute_loss(out, cand_slot, my_mask, weights)  # no ship_label arg

    assert float(report.ship_loss.item()) == 0.0
    assert report.ship_count == 0
    assert torch.isfinite(report.total)
