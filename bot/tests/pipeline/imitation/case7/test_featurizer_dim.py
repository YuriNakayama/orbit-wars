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


def test_planet_feat_dim_is_63() -> None:
    """iter3: PLANET_FEAT_DIM = 63。"""
    assert PLANET_FEAT_DIM == 63


def test_global_feat_dim_is_14() -> None:
    """iter2 以降: GLOBAL_FEAT_DIM = 14。"""
    assert GLOBAL_FEAT_DIM == 14


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

    # iter1 history 列 (idx 19/20/21) は 0、enemy ship event (22/23) も 0
    for col in (19, 20, 21, 22, 23):
        assert batch.planet_feats[0, 0, col].item() == 0.0

    # iter1 global launch (6-9) もゼロ
    for col in (6, 7, 8, 9):
        assert batch.global_feats[0, col].item() == 0.0

    # iter2 inbound fleet (24,25) はゼロ、dist (26) は -1 (no-inbound sentinel)、ships_log (27) はゼロ
    assert batch.planet_feats[0, 0, 24].item() == 0.0
    assert batch.planet_feats[0, 0, 25].item() == 0.0
    assert batch.planet_feats[0, 0, 26].item() == -1.0
    assert batch.planet_feats[0, 0, 27].item() == 0.0


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
