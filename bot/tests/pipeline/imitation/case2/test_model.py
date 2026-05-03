"""Unit tests for pipeline.imitation.case2.policy.model."""

from __future__ import annotations

import io

import torch

from pipeline.imitation.case2.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case2.policy.model import (
    DeepSetsPolicy,
    ModelConfig,
    build_model,
    count_parameters,
)
from pipeline.imitation.case2.policy.templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from pipeline.imitation.case2.policy.types import BatchFeatures


def _make_batch(b: int = 2, num_valid: int = 5) -> BatchFeatures:
    torch.manual_seed(0)
    planet_feats = torch.randn(b, MAX_PLANETS, PLANET_FEAT_DIM)
    planet_mask = torch.zeros(b, MAX_PLANETS, dtype=torch.bool)
    planet_mask[:, :num_valid] = True
    my_planet_mask = torch.zeros_like(planet_mask)
    my_planet_mask[:, 0] = True
    target_mask = planet_mask & ~my_planet_mask
    global_feats = torch.randn(b, GLOBAL_FEAT_DIM)
    template_ctx = torch.zeros(b, MAX_PLANETS, TEMPLATE_CTX_DIM)
    return BatchFeatures(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        global_feats=global_feats,
        template_ctx=template_ctx,
    )


def test_forward_shapes() -> None:
    model = build_model()
    out = model(_make_batch())
    assert out.from_logits.shape == (2, MAX_PLANETS)
    assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
    assert out.ships_logits.shape == (2, MAX_PLANETS, ModelConfig().ships_buckets)


def test_from_mask_applied() -> None:
    model = build_model()
    out = model(_make_batch())
    assert torch.all(torch.isinf(out.from_logits[:, 1:]))
    assert torch.all(out.from_logits[:, 1:] < 0)
    assert torch.all(torch.isfinite(out.from_logits[:, 0]))


def test_target_logits_finite() -> None:
    model = build_model()
    out = model(_make_batch())
    assert torch.all(torch.isfinite(out.target_logits))


def test_model_config_defaults_match_featurizer() -> None:
    cfg = ModelConfig()
    assert cfg.planet_in_dim == PLANET_FEAT_DIM
    assert cfg.global_in_dim == GLOBAL_FEAT_DIM


def test_param_count_under_kaggle_limit() -> None:
    n = count_parameters(DeepSetsPolicy())
    assert n < 250_000, f"too many params: {n}"


def test_state_dict_under_1mb() -> None:
    model = DeepSetsPolicy()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    assert buf.tell() < 1_000_000
