"""Featurizer smoke tests for imitation/case9 (P=41, G=20)."""

from __future__ import annotations

import math

import torch

from pipeline.imitation.case9.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
    update_history,
)
from pipeline.imitation.case9.policy.templates import TEMPLATE_CTX_DIM


def _make_obs(step: int = 0) -> dict[str, object]:
    """Minimal Orbit Wars-shaped observation with 4 planets."""
    return {
        "step": step,
        "player": 0,
        "angular_velocity": 0.05,
        "planets": [
            [0, 0, 30.0, 50.0, 3.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
            [3, -1, 50.0, 70.0, 3.0, 5, 1],
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [0, 0, 30.0, 50.0, 3.0, 10, 1],
            [1, 1, 70.0, 50.0, 3.0, 10, 1],
            [2, -1, 50.0, 30.0, 3.0, 5, 1],
            [3, -1, 50.0, 70.0, 3.0, 5, 1],
        ],
    }


def test_dims_and_shapes() -> None:
    assert PLANET_FEAT_DIM == 41
    assert GLOBAL_FEAT_DIM == 20

    obs = _make_obs()
    history = HistoryState()
    batch, snap = featurize(obs, history)

    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)
    assert batch.template_ctx.shape == (1, MAX_PLANETS, TEMPLATE_CTX_DIM)
    assert batch.candidate_feats.shape[0] == 1
    assert batch.planet_mask.dtype == torch.bool
    assert snap.player == 0
    assert snap.step == 0


def test_no_nan_or_inf() -> None:
    obs = _make_obs()
    history = HistoryState()
    batch, _ = featurize(obs, history)
    assert torch.isfinite(batch.planet_feats).all()
    assert torch.isfinite(batch.global_feats).all()
    assert torch.isfinite(batch.template_ctx).all()
    assert torch.isfinite(batch.candidate_feats).all()


def test_history_update_grows_buffer() -> None:
    history = HistoryState()
    assert len(history.prev_planet_snapshots) == 0
    obs0 = _make_obs(step=0)
    featurize(obs0, history)
    update_history(history, obs0, actions=[])
    obs1 = _make_obs(step=1)
    featurize(obs1, history)
    update_history(history, obs1, actions=[])
    assert 1 <= len(history.prev_planet_snapshots) <= 3


def test_timeline_columns_finite_under_arrivals() -> None:
    """timeline 6 列 (idx 35..40) は arrivals があっても finite であること。"""
    obs = _make_obs(step=10)
    obs["fleets"] = [
        # [id, owner, x, y, angle, from_planet_id, ships]
        [0, 1, 60.0, 50.0, math.pi, 1, 8],  # enemy fleet heading to player's planet 0
    ]
    history = HistoryState()
    batch, _ = featurize(obs, history)
    timeline_cols = batch.planet_feats[0, :, 35:41]
    assert torch.isfinite(timeline_cols).all()
