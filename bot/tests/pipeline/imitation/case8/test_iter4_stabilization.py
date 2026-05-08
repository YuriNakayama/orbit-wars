"""Unit tests for iter4 head + training stabilization (grad_clip / EMA / dropout)."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn, optim
from torch.optim import swa_utils

from pipeline.imitation.case8.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case8.policy.model import CandidatePolicy, ModelConfig


def test_head_dropout_zero_default() -> None:
    """ModelConfig().head_dropout defaults to 0.0 (iter1-3 backward compat)."""
    cfg = ModelConfig()
    assert cfg.head_dropout == 0.0
    model = CandidatePolicy(cfg)
    # cand_score is Sequential(Linear, ReLU, Dropout(0.0), Linear)
    dropout_layer = model.cand_score[2]
    assert isinstance(dropout_layer, nn.Dropout)
    assert dropout_layer.p == 0.0


def test_head_dropout_active() -> None:
    """head_dropout=0.2 sets Dropout.p on both cand_score and ship_head."""
    cfg = ModelConfig(head_dropout=0.2)
    model = CandidatePolicy(cfg)
    cand_drop = model.cand_score[2]
    ship_drop = model.ship_head[2]
    assert isinstance(cand_drop, nn.Dropout)
    assert isinstance(ship_drop, nn.Dropout)
    assert cand_drop.p == 0.2
    assert ship_drop.p == 0.2


def test_dropout_train_eval_switch() -> None:
    """Dropout fires only in train mode."""
    torch.manual_seed(0)
    cfg = ModelConfig(head_dropout=0.5, hidden=16)
    model = CandidatePolicy(cfg)
    h = 16
    x = torch.randn(4, h * 3)

    model.train()
    train_outputs = [model.cand_score(x) for _ in range(50)]
    train_var = torch.stack(train_outputs).var().item()

    model.eval()
    eval_outputs = [model.cand_score(x) for _ in range(50)]
    eval_var = torch.stack(eval_outputs).var().item()

    # train mode: dropout は random で variance 大、eval は deterministic (var≈0)。
    # 比は理論上 ∞ だが seed 依存の noise を許容する保守的な assertion。
    assert train_var > eval_var + 1e-4


def test_grad_clip_compresses_large_norm() -> None:
    """clip_grad_norm_ rescales gradients so total L2 norm ≤ max_norm."""
    layer = nn.Linear(10, 1)
    layer.weight.grad = torch.full_like(layer.weight, 100.0)
    layer.bias.grad = torch.full_like(layer.bias, 100.0)

    pre_clip_norm = float(
        torch.cat([layer.weight.grad.flatten(), layer.bias.grad.flatten()]).norm()
    )
    assert pre_clip_norm > 100.0

    returned_norm = float(nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0))
    # clip_grad_norm_ returns the pre-clip total L2 norm
    assert abs(returned_norm - pre_clip_norm) < 1e-3

    post_clip_norm = float(
        torch.cat([layer.weight.grad.flatten(), layer.bias.grad.flatten()]).norm()
    )
    assert post_clip_norm <= 1.0 + 1e-5


def test_grad_clip_no_op_when_norm_below_threshold() -> None:
    """If gradients are already small, clip is a no-op."""
    layer = nn.Linear(10, 1)
    layer.weight.grad = torch.full_like(layer.weight, 0.01)
    layer.bias.grad = torch.full_like(layer.bias, 0.01)
    before = layer.weight.grad.clone()
    nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
    assert torch.allclose(layer.weight.grad, before)


def test_averaged_model_ema_update() -> None:
    """AveragedModel(ema_fn) maintains an EMA of the live model.

    PyTorch 2.x semantics: 1st `update_parameters` snapshots live weights
    (n_averaged=0→1), 2nd onwards apply EMA: ema = decay·ema + (1-decay)·live.
    """
    _ = optim  # used to silence unused-import lint when SGD not exercised
    layer = nn.Linear(4, 4)
    ema = swa_utils.AveragedModel(
        layer,
        multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.9),  # type: ignore[no-untyped-call]
    )

    def ema_weight() -> torch.Tensor:
        inner = cast(nn.Linear, ema.module)
        return inner.weight.detach().clone()

    # 1st update: snapshot
    ema.update_parameters(layer)
    snap_w = ema_weight()
    assert torch.allclose(snap_w, layer.weight)

    # Now perturb live and 2nd update — EMA should partially follow
    with torch.no_grad():
        layer.weight.add_(10.0)
    ema.update_parameters(layer)
    after_w = ema_weight()

    moved = float((after_w - snap_w).abs().mean())
    diff_to_live = float((after_w - layer.weight).abs().mean())
    assert moved > 0.0, "EMA should have moved toward perturbed live"
    assert diff_to_live > 0.0, "EMA should not equal live after one EMA step"


def test_candidate_policy_ema_state_dict_keys_match() -> None:
    """Saving EMA weights via ema.module.state_dict() yields keys identical to
    the live model — required for downstream policy/agent.py to load them."""
    cfg = ModelConfig(head_dropout=0.2, hidden=16)
    model = CandidatePolicy(cfg)
    ema = swa_utils.AveragedModel(
        model,
        multi_avg_fn=swa_utils.get_ema_multi_avg_fn(0.999),  # type: ignore[no-untyped-call]
    )
    live_keys = set(model.state_dict().keys())
    ema_keys = set(ema.module.state_dict().keys())
    # Both should have identical keys; AveragedModel.module is the wrapped
    # CandidatePolicy, not a transformed version.
    assert live_keys == ema_keys


def test_compute_loss_compatible_with_dropout_model() -> None:
    """End-to-end: forward pass through dropout-enabled model in eval mode."""
    cfg = ModelConfig(head_dropout=0.2)
    model = CandidatePolicy(cfg).eval()

    from pipeline.imitation.case8.policy.featurizer import HistoryState, featurize

    planets = [
        [0, 0, 20.0, 50.0, 1.5, 10, 1],
        [1, 1, 80.0, 50.0, 1.5, 10, 1],
        [2, -1, 50.0, 20.0, 1.5, 10, 1],
    ]
    obs = {
        "step": 1,
        "player": 0,
        "planets": planets,
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
    }
    batch, _ = featurize(obs, HistoryState())
    with torch.no_grad():
        out = model(batch)
    assert out.candidate_logits.shape == (1, MAX_PLANETS, CAND_K)
    assert out.ship_pred.shape == (1, MAX_PLANETS)
    assert torch.isfinite(out.ship_pred).all()
    # avoid unused import warnings
    _ = (CAND_FEAT_DIM, GLOBAL_FEAT_DIM, PLANET_FEAT_DIM)
