"""case5 featurizer の dim sanity test。case1 (11 列) → case5 (17 列) を確認。"""

from __future__ import annotations

from pipeline.imitation.case7.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize,
)


def test_planet_feat_dim_is_17() -> None:
    """case1 base 11 + ship-prediction 6 = 17。"""
    assert PLANET_FEAT_DIM == 17


def test_featurize_minimal_obs_returns_correct_shape() -> None:
    """最小限の obs (planets/fleets 空) で BatchFeatures shape が一致する。"""
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            # [pid, owner, x, y, radius, ships, production]
            [0, 0, 50.0, 50.0, 1.0, 10, 2],  # 自分
            [1, 1, 60.0, 60.0, 1.0, 10, 2],  # 敵
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    batch, snap = featurize(obs)

    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)
    assert batch.planet_mask[0, 0].item() is True
    assert batch.planet_mask[0, 1].item() is True
    assert batch.planet_mask[0, 2].item() is False  # 未使用 slot

    # 自分 planet (slot 0) の ship-prediction 列は陥落なしで埋まる
    # col 11=loss_3turn, 12=ttf_norm, 13=min_owned, 14=surplus, 15=fall_predicted, 16=keep_needed
    fall_predicted = batch.planet_feats[0, 0, 15].item()
    assert fall_predicted == 0.0  # 敵 fleet なしなので陥落しない

    ttf_norm = batch.planet_feats[0, 0, 12].item()
    assert ttf_norm == 1.0


def test_featurize_with_incoming_enemy_predicts_fall() -> None:
    """敵 fleet が向かっている自 planet で fall_predicted=1 になる。"""
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 50.0, 50.0, 1.0, 5, 1],  # 自分 (弱小)
        ],
        "fleets": [
            # [fid, fowner, fx, fy, fangle, from_pid, fships] — 向かって来る大群
            [
                10,
                1,
                51.0,
                50.0,
                3.14159,
                99,
                100,
            ],  # angle=π で +x → -x、planet (50,50) に近い
        ],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    batch, _ = featurize(obs)

    # 自 planet の planet_mask が立つこと
    assert batch.planet_mask[0, 0].item() is True
