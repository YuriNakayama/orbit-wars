"""case7 featurizer の dim sanity test。

iter1: case5 17 + 7 (predicted dist 2 + history 3 + enemy ship event 2) = 24
iter2: iter1 24 + 10 (fleet trajectory 4 + multi-horizon 4 + production/centroid 2) = 34
       global iter1 10 + 4 (comet 2 + home/centroid 2) = 14
iter3: iter2 34 + 29 (Pairwise Top-K 20 + Defense surplus 4 + Sparse mask 5) = 63
       global iter2 14 (変更なし)
"""

from __future__ import annotations

from pipeline.imitation.case7.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
)


def test_planet_feat_dim_is_61() -> None:
    """iter4: PLANET_FEAT_DIM = 61 (iter3 63 - G6 2 - K1 縮小 8 + K2 4 + K3 4)。"""
    assert PLANET_FEAT_DIM == 61


def test_global_feat_dim_is_12() -> None:
    """iter4: GLOBAL_FEAT_DIM = 12 (iter3 14 - H1/H4/H5 6 + K4 4)。"""
    assert GLOBAL_FEAT_DIM == 12


def test_featurize_minimal_obs_without_history_returns_correct_shape() -> None:
    """history=None でも shape が合い、history 系列はゼロ埋め (sane default)。"""
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

    # iter1 history 列 (idx 19/20/21) は 0
    for col in (19, 20, 21):
        assert batch.planet_feats[0, 0, col].item() == 0.0

    # iter4 launch history (global 4-7) ゼロ
    for col in (4, 5, 6, 7):
        assert batch.global_feats[0, col].item() == 0.0

    # iter4 inbound fleet (22,23) はゼロ、dist (24) は -1 (no-inbound sentinel)、ships_log (25) はゼロ
    assert batch.planet_feats[0, 0, 22].item() == 0.0
    assert batch.planet_feats[0, 0, 23].item() == 0.0
    assert batch.planet_feats[0, 0, 24].item() == -1.0
    assert batch.planet_feats[0, 0, 25].item() == 0.0


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
