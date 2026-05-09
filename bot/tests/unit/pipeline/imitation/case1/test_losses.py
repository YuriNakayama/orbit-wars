"""Tests for Phase 2 class-weighted loss additions in imitation/case1.

Covers:
- compute_class_weights: correct normalization, empty-class handling, shape.
- _smooth_ordinal_labels: adjacent-bucket leak, edge-bucket spill.
- compute_loss plumbing: target/ships class weights and ordinal smoothing are
  actually routed through F.cross_entropy / the manual soft-CE branch.
"""

from __future__ import annotations

import torch

from pipeline.imitation.case1.policy.featurizer import MAX_PLANETS
from pipeline.imitation.case1.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case1.policy.types import PolicyOutput
from pipeline.imitation.case1.training.losses import (
    LossWeights,
    _smooth_ordinal_labels,
    compute_class_weights,
    compute_loss,
)


def _mk_output(batch: int, planets: int, ships_buckets: int = 4) -> PolicyOutput:
    torch.manual_seed(0)
    return PolicyOutput(
        from_logits=torch.randn(batch, planets),
        target_logits=torch.randn(batch, planets, NUM_TEMPLATES),
        ships_logits=torch.randn(batch, planets, ships_buckets),
    )


def test_compute_class_weights_normalized_to_present_mean_one() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, -1, -1])  # class 3 absent
    w = compute_class_weights(labels, num_classes=4, beta=0.9999)

    assert w.shape == (4,)
    assert w[3].item() == 0.0  # empty class gets 0

    present = w[:3]
    assert abs(present.mean().item() - 1.0) < 1e-5
    # minority classes should be up-weighted vs majority
    assert w[1].item() > w[0].item()
    assert w[2].item() > w[1].item()


def test_compute_class_weights_ignore_index_filters() -> None:
    labels = torch.tensor([-1, -1, -1, 0, 1])
    w = compute_class_weights(labels, num_classes=2, beta=0.9999)
    # both classes have 1 sample → equal weights normalised to 1.0
    assert abs(w[0].item() - 1.0) < 1e-5
    assert abs(w[1].item() - 1.0) < 1e-5


def test_smooth_ordinal_labels_middle_bucket_leaks_both_sides() -> None:
    soft = _smooth_ordinal_labels(torch.tensor([2]), num_classes=4, alpha=0.1)
    assert soft.shape == (1, 4)
    assert abs(soft[0, 2].item() - 0.9) < 1e-6
    assert abs(soft[0, 1].item() - 0.05) < 1e-6
    assert abs(soft[0, 3].item() - 0.05) < 1e-6
    assert soft[0, 0].item() == 0.0


def test_smooth_ordinal_labels_edge_bucket_spills_one_side() -> None:
    soft = _smooth_ordinal_labels(torch.tensor([0, 3]), num_classes=4, alpha=0.1)
    # lowest bucket → all leak goes to bucket 1
    assert abs(soft[0, 0].item() - 0.9) < 1e-6
    assert abs(soft[0, 1].item() - 0.1) < 1e-6
    # highest bucket → all leak goes to bucket 2
    assert abs(soft[1, 3].item() - 0.9) < 1e-6
    assert abs(soft[1, 2].item() - 0.1) < 1e-6


def test_compute_loss_target_class_weights_change_loss() -> None:
    """Passing non-uniform target class weights must change target_loss."""
    batch, planets = 2, MAX_PLANETS
    output = _mk_output(batch, planets)
    from_multihot = torch.zeros(batch, planets, dtype=torch.bool)
    from_multihot[0, 0] = True
    from_multihot[1, 1] = True
    target_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    target_per_src[0, 0] = 0  # first fired sample → class 0
    target_per_src[1, 1] = 1  # second fired sample → class 1
    ships_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    ships_per_src[0, 0] = 0
    ships_per_src[1, 1] = 3
    my_planet_mask = torch.zeros(batch, planets, dtype=torch.bool)
    my_planet_mask[0, 0] = True
    my_planet_mask[1, 1] = True

    base = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(target_label_smoothing=0.0, target_entropy_bonus=0.0),
    )
    skewed = torch.ones(NUM_TEMPLATES)
    skewed[1] = 5.0  # heavily up-weight class 1
    weighted = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(
            target_label_smoothing=0.0,
            target_entropy_bonus=0.0,
            target_class_weights=skewed,
        ),
    )
    assert abs(weighted.target_loss.item() - base.target_loss.item()) > 1e-4


def test_compute_loss_ships_ordinal_smoothing_changes_loss() -> None:
    batch, planets = 1, MAX_PLANETS
    output = _mk_output(batch, planets)
    from_multihot = torch.zeros(batch, planets, dtype=torch.bool)
    from_multihot[0, 0] = True
    target_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    target_per_src[0, 0] = 0
    ships_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    ships_per_src[0, 0] = 2
    my_planet_mask = torch.zeros(batch, planets, dtype=torch.bool)
    my_planet_mask[0, 0] = True

    base = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(),
    )
    smoothed = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(ships_ordinal_smoothing=0.2),
    )
    assert abs(smoothed.ships_loss.item() - base.ships_loss.item()) > 1e-4


def test_compute_loss_ships_focal_reduces_well_classified_contribution() -> None:
    batch, planets, K = 2, MAX_PLANETS, 4
    output = _mk_output(batch, planets, ships_buckets=K)
    # Make class-2 the "confident correct" sample and class-0 the "hard" one
    output.ships_logits.zero_()
    output.ships_logits[0, 0, 2] = 5.0  # very high logit on true class 2
    output.ships_logits[1, 0, 0] = 0.2  # weakly correct on class 0
    from_multihot = torch.zeros(batch, planets, dtype=torch.bool)
    from_multihot[:, 0] = True
    target_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    target_per_src[:, 0] = 0
    ships_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    ships_per_src[0, 0] = 2
    ships_per_src[1, 0] = 0
    my_planet_mask = torch.zeros(batch, planets, dtype=torch.bool)
    my_planet_mask[:, 0] = True

    base = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(),
    )
    focal = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(ships_focal_gamma=2.0),
    )
    # Focal loss on a confident+weak mix must differ from plain CE.
    assert abs(focal.ships_loss.item() - base.ships_loss.item()) > 1e-4
    # With γ=2 the easy (confident) example's loss is heavily discounted, so
    # the average should land below plain CE on this mix.
    assert focal.ships_loss.item() < base.ships_loss.item()


def test_compute_loss_ships_focal_alpha_scales_per_class() -> None:
    batch, planets, K = 1, MAX_PLANETS, 4
    output = _mk_output(batch, planets, ships_buckets=K)
    from_multihot = torch.zeros(batch, planets, dtype=torch.bool)
    from_multihot[0, 0] = True
    target_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    target_per_src[0, 0] = 0
    ships_per_src = torch.full((batch, planets), -1, dtype=torch.int64)
    ships_per_src[0, 0] = 1
    my_planet_mask = torch.zeros(batch, planets, dtype=torch.bool)
    my_planet_mask[0, 0] = True

    unweighted = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(ships_focal_gamma=2.0),
    )
    alpha = torch.tensor([1.0, 5.0, 1.0, 1.0])  # upweight class 1
    scaled = compute_loss(
        output,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        my_planet_mask=my_planet_mask,
        weights=LossWeights(ships_focal_gamma=2.0, ships_focal_alpha=alpha),
    )
    assert scaled.ships_loss.item() > unweighted.ships_loss.item() * 3.0
