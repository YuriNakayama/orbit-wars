"""case8 予測距離 (orbit-aware future position) の動作確認。"""

from __future__ import annotations

import math

from pipeline.imitation.case8.policy.featurizer import (
    featurize,
)

COL_FUTURE_DIST_MY = 17
COL_FUTURE_DIST_ENEMY = 18


def test_future_dist_zero_for_self_against_my_centroid_with_single_my_planet() -> None:
    """自軍 planet が 1 つだけなら、その planet の future position == my-centroid → 距離 0。"""
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 60.0, 50.0, 1.0, 10, 2],  # 自分のみ
            [1, 1, 30.0, 50.0, 1.0, 10, 2],  # 敵
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_FUTURE_DIST_MY].item() == 0.0
    # 敵への距離は (60-30)/100 = 0.30
    assert abs(batch.planet_feats[0, 0, COL_FUTURE_DIST_ENEMY].item() - 0.30) < 1e-4


def test_future_dist_static_matches_euclidean_distance() -> None:
    """ang_vel=0 では future position == 現在位置 → 距離は raw Euclidean / BOARD_SIZE。"""
    obs = {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 60.0, 50.0, 1.0, 10, 2],  # 自分
            [1, 1, 50.0, 70.0, 1.0, 10, 2],  # 敵
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }
    batch, _ = featurize(obs)
    # 自 planet (60,50) → 敵 centroid (50,70): √(10² + 20²) / 100 = √500 / 100
    expected = math.sqrt(500.0) / 100.0
    assert abs(batch.planet_feats[0, 0, COL_FUTURE_DIST_ENEMY].item() - expected) < 1e-4
