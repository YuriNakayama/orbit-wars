"""Unit tests for pipeline.case3.policy.model."""

from __future__ import annotations

import io

import torch

from pipeline.case3.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.case3.policy.model import (
    DeepSetsPolicy,
    ModelConfig,
    build_model,
    count_parameters,
)
from pipeline.case3.policy.types import BatchFeatures


def _make_batch(b: int = 2, num_valid: int = 5) -> BatchFeatures:
    torch.manual_seed(0)
    planet_feats = torch.randn(b, MAX_PLANETS, PLANET_FEAT_DIM)
    planet_mask = torch.zeros(b, MAX_PLANETS, dtype=torch.bool)
    planet_mask[:, :num_valid] = True
    my_planet_mask = torch.zeros_like(planet_mask)
    my_planet_mask[:, 0] = True  # only slot 0 is mine
    target_mask = planet_mask & ~my_planet_mask
    global_feats = torch.randn(b, GLOBAL_FEAT_DIM)
    return BatchFeatures(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        global_feats=global_feats,
    )


def test_forward_shapes() -> None:
    model = build_model()
    out = model(_make_batch())
    assert out.from_logits.shape == (2, MAX_PLANETS)
    assert out.target_logits.shape == (2, MAX_PLANETS, MAX_PLANETS + 1)
    assert out.ships_logits.shape == (2, MAX_PLANETS, ModelConfig().ships_buckets)


def test_from_mask_applied() -> None:
    model = build_model()
    out = model(_make_batch())
    # non-my planets must be -inf in from_logits
    assert torch.all(torch.isinf(out.from_logits[:, 1:]))
    assert torch.all(out.from_logits[:, 1:] < 0)
    # my planet (slot 0) must be finite
    assert torch.all(torch.isfinite(out.from_logits[:, 0]))


def test_target_mask_applied() -> None:
    model = build_model()
    batch = _make_batch()
    out = model(batch)
    # padding slots (>=5) must be -inf (reinforcing owned planets is legal)
    assert torch.all(torch.isinf(out.target_logits[:, :, 5:MAX_PLANETS]))
    # no-op slot (last) is finite
    assert torch.all(torch.isfinite(out.target_logits[:, :, -1]))
    # valid planet slots (0..4) are finite
    assert torch.all(torch.isfinite(out.target_logits[:, :, :5]))


def test_param_count_under_100k() -> None:
    n = count_parameters(DeepSetsPolicy())
    assert n < 100_000, f"too many params: {n}"


def test_state_dict_under_1mb() -> None:
    model = DeepSetsPolicy()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    assert buf.tell() < 1_000_000
