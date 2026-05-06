"""Sanity tests for case6 GraphAttention U-Net model."""

from __future__ import annotations

import torch

from pipeline.imitation.case6.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case6.policy.model import (
    GraphAttention,
    GraphAttentionUNetPolicy,
    ModelConfig,
    _knn_adjacency,
    _pairwise_geometry,
    count_parameters,
)
from pipeline.imitation.case6.policy.templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from pipeline.imitation.case6.policy.types import BatchFeatures


def _fake_batch(batch_size: int = 2, num_planets: int = 6) -> BatchFeatures:
    p = MAX_PLANETS
    planet_feats = torch.zeros(batch_size, p, PLANET_FEAT_DIM)
    # populate first num_planets rows with non-trivial values
    planet_feats[:, :num_planets] = torch.randn(
        batch_size, num_planets, PLANET_FEAT_DIM
    )
    planet_mask = torch.zeros(batch_size, p, dtype=torch.bool)
    planet_mask[:, :num_planets] = True
    my_planet_mask = torch.zeros(batch_size, p, dtype=torch.bool)
    my_planet_mask[:, :2] = True  # 2 own planets per batch
    global_feats = torch.randn(batch_size, GLOBAL_FEAT_DIM)
    template_ctx = torch.randn(batch_size, p, TEMPLATE_CTX_DIM)
    target_mask = planet_mask.clone()
    return BatchFeatures(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        global_feats=global_feats,
        template_ctx=template_ctx,
    )


def test_forward_shapes() -> None:
    model = GraphAttentionUNetPolicy(ModelConfig())
    batch = _fake_batch(batch_size=2, num_planets=8)
    out = model(batch)
    assert out.from_logits.shape == (2, MAX_PLANETS)
    assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
    assert out.ships_logits.shape == (2, MAX_PLANETS, 4)


def test_padding_planets_get_minus_inf_in_from_head() -> None:
    model = GraphAttentionUNetPolicy(ModelConfig())
    batch = _fake_batch(batch_size=1, num_planets=4)
    out = model(batch)
    # rows where my_planet_mask is False must get -inf so softmax yields 0 prob
    invalid_rows = ~batch.my_planet_mask
    assert torch.all(out.from_logits[invalid_rows] == float("-inf"))


def test_no_nan_with_isolated_padding_rows() -> None:
    """Rows with no valid neighbours (masked padding) must not produce NaN."""
    model = GraphAttentionUNetPolicy(ModelConfig())
    batch = _fake_batch(batch_size=2, num_planets=3)  # very few valid planets
    out = model(batch)
    # Valid (my-planet) cells must be finite
    finite_cells = batch.my_planet_mask
    assert torch.all(torch.isfinite(out.from_logits[finite_cells]))
    assert torch.all(torch.isfinite(out.target_logits))
    assert torch.all(torch.isfinite(out.ships_logits))


def test_attention_layer_shape_and_mask() -> None:
    h_dim = 32
    layer = GraphAttention(in_dim=h_dim, out_dim=h_dim, num_heads=4)
    b, p = 2, 5
    h = torch.randn(b, p, h_dim)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    coords = torch.randn(b, p, 2)
    adj = _knn_adjacency(coords, mask, k=2)
    edge = torch.randn(b, p, p, 6)
    out = layer(h, adj, mask, edge)
    assert out.shape == (b, p, h_dim)
    # masked rows should be zero
    assert torch.all(out[~mask] == 0)


def test_param_count_grows_vs_graphconv_baseline() -> None:
    """Attention has more params than the case5 GraphConv baseline (sanity check)."""
    model = GraphAttentionUNetPolicy(ModelConfig())
    n = count_parameters(model)
    # case5 baseline is ~165k. Attention with 4 heads should be larger.
    assert n > 200_000, f"too few params: {n}"
    assert n < 1_000_000, f"unexpectedly many params: {n}"


def test_pairwise_geometry_shape() -> None:
    feats = torch.randn(2, MAX_PLANETS, PLANET_FEAT_DIM)
    pair = _pairwise_geometry(feats)
    assert pair.shape == (2, MAX_PLANETS, MAX_PLANETS, 6)
    # Diagonal dist should be ~0
    diag_dist = pair[..., 2].diagonal(dim1=1, dim2=2)
    assert torch.all(diag_dist < 1e-3)
