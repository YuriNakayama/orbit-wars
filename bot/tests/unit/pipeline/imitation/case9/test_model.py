"""Forward-pass smoke test for imitation/case9 head variants."""

from __future__ import annotations

import pytest
import torch

from pipeline.imitation.case9.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case9.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case9.policy.model import (
    SUPPORTED_HEAD_MODES,
    Case9Policy,
    ModelConfig,
    count_parameters,
)
from pipeline.imitation.case9.policy.templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from pipeline.imitation.case9.policy.types import BatchFeatures


def _make_batch(batch_size: int = 2) -> BatchFeatures:
    b, p, k = batch_size, MAX_PLANETS, CAND_K
    my_mask = torch.zeros(b, p, dtype=torch.bool)
    my_mask[:, : p // 2] = True
    return BatchFeatures(
        planet_feats=torch.randn(b, p, PLANET_FEAT_DIM),
        planet_mask=torch.ones(b, p, dtype=torch.bool),
        my_planet_mask=my_mask,
        target_mask=torch.ones(b, p, dtype=torch.bool),
        global_feats=torch.randn(b, GLOBAL_FEAT_DIM),
        template_ctx=torch.randn(b, p, TEMPLATE_CTX_DIM),
        candidate_feats=torch.randn(b, p, k, CAND_FEAT_DIM),
        candidate_mask=torch.ones(b, p, k, dtype=torch.bool),
        candidate_pid=torch.zeros(b, p, k, dtype=torch.int64),
    )


@pytest.mark.parametrize("head_mode", SUPPORTED_HEAD_MODES)
def test_forward_shape(head_mode: str) -> None:
    cfg = ModelConfig(head_mode=head_mode)
    model = Case9Policy(cfg)
    batch = _make_batch(batch_size=2)
    out = model(batch)
    if head_mode == "three_head":
        assert out.from_logits is not None
        assert out.target_logits is not None
        assert out.ships_logits is not None
        assert out.from_logits.shape == (2, MAX_PLANETS)
        assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
        assert out.ships_logits.shape == (2, MAX_PLANETS, cfg.ships_buckets)
        assert out.candidate_logits is None
    elif head_mode == "candidate":
        assert out.candidate_logits is not None
        assert out.ship_pred is not None
        assert out.candidate_logits.shape == (2, MAX_PLANETS, CAND_K)
        assert out.ship_pred.shape == (2, MAX_PLANETS)
        assert out.from_logits is None
    elif head_mode == "candidate_ships":
        assert out.candidate_logits is not None
        assert out.ships_logits is not None
        assert out.candidate_logits.shape == (2, MAX_PLANETS, CAND_K)
        assert out.ships_logits.shape == (2, MAX_PLANETS, cfg.ships_buckets)
        assert out.from_logits is None
        assert out.ship_pred is None
    elif head_mode == "template_ships":
        assert out.target_logits is not None
        assert out.ships_logits is not None
        assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
        assert out.ships_logits.shape == (2, MAX_PLANETS, cfg.ships_buckets)
        assert out.from_logits is None
        assert out.candidate_logits is None
        assert out.ship_pred is None
    elif head_mode == "per_planet":
        assert out.per_planet_logits is not None
        assert out.ship_pred is not None
        assert out.per_planet_logits.shape == (2, MAX_PLANETS, MAX_PLANETS + 1)
        assert out.ship_pred.shape == (2, MAX_PLANETS)
        assert out.from_logits is None
        assert out.target_logits is None
        assert out.ships_logits is None
        assert out.candidate_logits is None
    else:  # dual
        assert out.from_logits is not None
        assert out.target_logits is not None
        assert out.ships_logits is not None
        assert out.candidate_logits is not None
        assert out.ship_pred is not None
        assert out.from_logits.shape == (2, MAX_PLANETS)
        assert out.target_logits.shape == (2, MAX_PLANETS, NUM_TEMPLATES)
        assert out.ships_logits.shape == (2, MAX_PLANETS, cfg.ships_buckets)
        assert out.candidate_logits.shape == (2, MAX_PLANETS, CAND_K)
        assert out.ship_pred.shape == (2, MAX_PLANETS)


@pytest.mark.parametrize("head_mode", SUPPORTED_HEAD_MODES)
def test_forward_no_nan(head_mode: str) -> None:
    cfg = ModelConfig(head_mode=head_mode)
    model = Case9Policy(cfg)
    batch = _make_batch(batch_size=2)
    out = model(batch)
    for name in (
        "from_logits",
        "target_logits",
        "ships_logits",
        "candidate_logits",
        "ship_pred",
        "per_planet_logits",
    ):
        v = getattr(out, name)
        if v is None:
            continue
        finite = torch.isfinite(v) | (v == float("-inf"))
        assert finite.all(), f"NaN/Inf detected in {head_mode}.{name}"


@pytest.mark.parametrize("head_mode", SUPPORTED_HEAD_MODES)
def test_param_count_within_range(head_mode: str) -> None:
    """Sanity check each variant stays within its expected parameter budget."""
    cfg = ModelConfig(head_mode=head_mode)
    model = Case9Policy(cfg)
    n = count_parameters(model)
    if head_mode == "dual":
        upper = 1_900_000
    elif head_mode == "per_planet":
        # bilinear pair_proj + larger src/tgt projections widen the head budget.
        upper = 4_000_000
    else:
        upper = 1_500_000
    assert 800_000 < n < upper, f"{head_mode} has {n} params"


def test_unknown_head_mode_raises() -> None:
    cfg = ModelConfig(head_mode="bogus")
    with pytest.raises(ValueError, match="unknown head_mode"):
        Case9Policy(cfg)
