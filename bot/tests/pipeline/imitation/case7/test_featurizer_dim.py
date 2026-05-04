"""case7 featurizer の dim sanity test。case5 (17/6) → case7 (24/10) を確認。"""

from __future__ import annotations

from pipeline.imitation.case7.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
)


def test_planet_feat_dim_is_24() -> None:
    """case5 17 + 予測距離 2 + history 3 + 敵 ship 発射 2 = 24。"""
    assert PLANET_FEAT_DIM == 24


def test_global_feat_dim_is_10() -> None:
    """case5 6 + enemy/ally launch history 4 = 10。"""
    assert GLOBAL_FEAT_DIM == 10


def test_featurize_minimal_obs_without_history_returns_correct_shape() -> None:
    """history=None でも shape が合い、新規列はゼロ埋め (sane default)。"""
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 60.0, 60.0, 1.0, 10, 2],
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    batch, snap = featurize(obs)

    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)

    # history 列 (idx 19/20/21) は 0、敵 ship event (22/23) も 0
    for col in (19, 20, 21, 22, 23):
        assert batch.planet_feats[0, 0, col].item() == 0.0

    # 新規 global 6-9 もゼロ
    for col in (6, 7, 8, 9):
        assert batch.global_feats[0, col].item() == 0.0


def test_featurize_with_history_state_accepts_call() -> None:
    """HistoryState を渡しても shape が一致する。"""
    obs = {
        "player": 0,
        "step": 1,
        "planets": [[0, 0, 50.0, 50.0, 1.0, 10, 2]],
        "fleets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    history = HistoryState()
    batch, _ = featurize(obs, history=history)
    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
