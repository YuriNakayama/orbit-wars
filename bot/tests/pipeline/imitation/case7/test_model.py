"""Sanity tests for case7 Set Transformer + cross-attention head model."""

from __future__ import annotations

import torch

from pipeline.imitation.case7.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case7.policy.model import (
    PMA,
    HierarchicalPointerPolicy,
    InducedSetAttentionBlock,
    ModelConfig,
    SetTransformerPolicy,
    build_model,
    count_parameters,
)
from pipeline.imitation.case7.policy.templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from pipeline.imitation.case7.policy.types import BatchFeatures


def _fake_batch(batch_size: int = 2, num_planets: int = 6) -> BatchFeatures:
    p = MAX_PLANETS
    planet_feats = torch.zeros(batch_size, p, PLANET_FEAT_DIM)
    planet_feats[:, :num_planets] = torch.randn(
        batch_size, num_planets, PLANET_FEAT_DIM
    )
    planet_mask = torch.zeros(batch_size, p, dtype=torch.bool)
    planet_mask[:, :num_planets] = True
    my_planet_mask = torch.zeros(batch_size, p, dtype=torch.bool)
    my_planet_mask[:, :2] = True
    target_mask = planet_mask.clone()
    global_feats = torch.randn(batch_size, GLOBAL_FEAT_DIM)
    template_ctx = torch.randn(batch_size, p, TEMPLATE_CTX_DIM)
    return BatchFeatures(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        global_feats=global_feats,
        template_ctx=template_ctx,
    )


def test_forward_shapes() -> None:
    model = SetTransformerPolicy(ModelConfig())
    batch = _fake_batch(batch_size=2, num_planets=8)
    out = model(batch)
    assert out.from_logits.shape == (2, MAX_PLANETS)
    assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
    assert out.ships_logits.shape == (2, MAX_PLANETS, 4)


def test_padding_planets_get_minus_inf_in_from_head() -> None:
    model = SetTransformerPolicy(ModelConfig())
    batch = _fake_batch(batch_size=1, num_planets=4)
    out = model(batch)
    invalid_rows = ~batch.my_planet_mask
    assert torch.all(out.from_logits[invalid_rows] == float("-inf"))


def test_no_nan_with_few_valid_planets() -> None:
    """ISAB key_padding_mask での NaN 回避 sanity."""
    model = SetTransformerPolicy(ModelConfig())
    batch = _fake_batch(batch_size=2, num_planets=3)
    out = model(batch)
    finite_cells = batch.my_planet_mask
    assert torch.all(torch.isfinite(out.from_logits[finite_cells]))
    assert torch.all(torch.isfinite(out.target_logits))
    assert torch.all(torch.isfinite(out.ships_logits))


def test_isab_block_shape() -> None:
    h_dim = 32
    layer = InducedSetAttentionBlock(dim=h_dim, num_heads=4, num_inducing=8)
    b, p = 2, 5
    h = torch.randn(b, p, h_dim)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    out = layer(h, mask)
    assert out.shape == (b, p, h_dim)
    assert torch.all(out[~mask] == 0)


def test_pma_shape() -> None:
    h_dim = 32
    pma = PMA(dim=h_dim, num_heads=4, num_seeds=1)
    b, p = 2, 5
    h = torch.randn(b, p, h_dim)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    out = pma(h, mask)
    assert out.shape == (b, 1, h_dim)


def test_param_count_reasonable() -> None:
    model = SetTransformerPolicy(ModelConfig())
    n = count_parameters(model)
    assert n > 200_000, f"too few params: {n}"
    assert n < 1_500_000, f"unexpectedly many params: {n}"


def test_target_logits_per_template_distinct() -> None:
    """cross-attention で template ごとの logit が異なる値を返すこと。"""
    model = SetTransformerPolicy(ModelConfig())
    batch = _fake_batch(batch_size=1, num_planets=8)
    out = model(batch)
    src_idx = 0
    template_logits = out.target_logits[0, src_idx]
    assert template_logits.std().item() > 1e-4, (
        f"all templates same logit: {template_logits}"
    )


def test_factory_set_transformer() -> None:
    """build_model(model_type='set_transformer') が SetTransformerPolicy を返す。"""
    cfg = ModelConfig(model_type="set_transformer")
    model = build_model(cfg)
    assert isinstance(model, SetTransformerPolicy)


def test_factory_pointer() -> None:
    """build_model(model_type='pointer') が HierarchicalPointerPolicy を返す。"""
    cfg = ModelConfig(model_type="pointer")
    model = build_model(cfg)
    assert isinstance(model, HierarchicalPointerPolicy)


def test_factory_unknown_raises() -> None:
    """unknown model_type で ValueError。"""
    import pytest

    cfg = ModelConfig(model_type="diffusion")
    with pytest.raises(ValueError, match="unknown model_type"):
        build_model(cfg)


def test_pointer_forward_shape() -> None:
    """HierarchicalPointerPolicy forward shape sanity (旧 case8 と同型)."""
    cfg = ModelConfig(model_type="pointer")
    model = build_model(cfg)
    batch = _fake_batch(batch_size=2, num_planets=8)
    out = model(batch)
    assert out.from_logits.shape == (2, MAX_PLANETS)
    assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
    assert out.ships_logits.shape == (2, MAX_PLANETS, 4)
