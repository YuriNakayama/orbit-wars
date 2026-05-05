"""Unit tests for iter2 lr scheduler + early stop helpers in train.py."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import optim

from pipeline.imitation.case8.training.train import (
    _build_scheduler,
    _select_metric_value,
)


def _toy_optimizer(lr: float = 1.0e-3) -> optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return optim.AdamW([param], lr=lr)


def test_build_scheduler_returns_none_when_cfg_empty() -> None:
    opt = _toy_optimizer()
    assert _build_scheduler(opt, None, epochs=10) is None
    assert _build_scheduler(opt, {}, epochs=10) is None


def test_cosine_warmup_lr_curve() -> None:
    """warmup phase ramps lr 0.1*base → base, then cosine decays to eta_min."""
    base_lr = 1.0e-3
    eta_min = 1.0e-5
    opt = _toy_optimizer(base_lr)
    cfg = {
        "type": "cosine_warmup",
        "t_max": 10,
        "eta_min": eta_min,
        "warmup_epochs": 2,
        "warmup_start_factor": 0.1,
    }
    sched = _build_scheduler(opt, cfg, epochs=10)
    assert sched is not None

    lrs: list[float] = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    # epoch 0: warmup start (~0.1 × base_lr)
    assert lrs[0] == 1.0e-4 or math.isclose(lrs[0], 1.0e-4, rel_tol=1e-6)
    # epoch 2: warmup ends, lr at base_lr
    assert math.isclose(lrs[2], base_lr, rel_tol=1e-3)
    # epoch 9 (last): near eta_min
    assert lrs[-1] < base_lr / 5.0
    assert lrs[-1] >= eta_min - 1e-9


def test_cosine_warmup_no_warmup_returns_pure_cosine() -> None:
    base_lr = 1.0e-3
    opt = _toy_optimizer(base_lr)
    cfg = {
        "type": "cosine_warmup",
        "t_max": 5,
        "eta_min": 0.0,
        "warmup_epochs": 0,
    }
    sched = _build_scheduler(opt, cfg, epochs=5)
    assert sched is not None
    # epoch 0 lr equals base_lr (no warmup applied)
    assert math.isclose(opt.param_groups[0]["lr"], base_lr)


def test_unsupported_scheduler_raises() -> None:
    opt = _toy_optimizer()
    try:
        _build_scheduler(opt, {"type": "step"}, epochs=10)
    except ValueError as e:
        assert "unsupported scheduler" in str(e)
    else:
        raise AssertionError("ValueError was not raised")


def test_select_metric_value_known_keys() -> None:
    val_metrics: dict[str, float] = {
        "total": 1.5,
        "cand": 1.0,
        "cand_acc": 0.3,
        "cand_noop_acc": 0.2,
        "cand_fire_acc": 0.1,
        "ship": 30.0,
        "ship_mae": 30.5,
    }
    assert _select_metric_value(val_metrics, "val_total") == 1.5
    assert _select_metric_value(val_metrics, "val_cand_fire_acc") == 0.1
    assert _select_metric_value(val_metrics, "val_ship_mae") == 30.5


def test_select_metric_value_unknown_raises() -> None:
    val_metrics: dict[str, Any] = {"total": 0.0}
    try:
        _select_metric_value(val_metrics, "val_bogus")
    except KeyError as e:
        assert "val_bogus" in str(e)
    else:
        raise AssertionError("KeyError was not raised")
