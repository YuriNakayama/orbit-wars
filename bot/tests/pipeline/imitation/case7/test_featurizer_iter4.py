"""case7 iter4 特徴量のテスト (iter6 で K2/K4 削除済み)。

iter4 で導入したのは K2/K3/K4 の 3 つだったが、iter6 で causal leak の K2 と
dead group の K4 を削除。残るのは K3 frontline_distance のみ。

idx layout (iter6 反映後):
  planet 53-56: K3 Frontline distance (d1_log, r1_log, d2_log, r2_log)
    元 iter4 layout では 57-60 にあったが、K2 削除で -4 shift して 53-56 に。
"""

from __future__ import annotations

import math

from pipeline.imitation.case7.policy.featurizer import (
    featurize,
)

# K3 Frontline distance (iter6: K2 削除で 57-60 → 53-56 に shift)
COL_FL_D1 = 53
COL_FL_R1 = 54
COL_FL_D2 = 55
COL_FL_R2 = 56


def _obs(
    step: int = 0,
    planets: list[list[float | int]] | None = None,
    fleets: list[list[float | int]] | None = None,
    initial_planets: list[list[float | int]] | None = None,
    ang_vel: float = 0.0,
    player: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "step": step,
        "planets": planets if planets is not None else [],
        "fleets": fleets if fleets is not None else [],
        "comet_planet_ids": [],
        "initial_planets": initial_planets if initial_planets is not None else [],
        "angular_velocity": ang_vel,
    }


# ----- K3 Frontline distance -----


def test_k3_frontline_no_enemy_returns_sentinel() -> None:
    """敵 planet 無し → frontline_d1/d2 = -1。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, -1, 60.0, 50.0, 1.0, 5, 1],  # 中立だけ
        ]
    )
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_FL_D1].item() == -1.0
    assert batch.planet_feats[0, 0, COL_FL_D2].item() == -1.0


def test_k3_frontline_d1_correct_distance() -> None:
    """敵が 1 個だけなら d1 = log1p(dist) / log1p(BOARD_SIZE)、d2 = -1 (sentinel)。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 70.0, 50.0, 1.0, 5, 3],  # 敵、距離 20
        ]
    )
    batch, _ = featurize(obs)
    expected_d1 = math.log1p(20.0) / math.log1p(100.0)
    assert abs(batch.planet_feats[0, 0, COL_FL_D1].item() - expected_d1) < 1e-5
    # d2 は敵が 1 個しかないので sentinel
    assert batch.planet_feats[0, 0, COL_FL_D2].item() == -1.0


def test_k3_frontline_two_enemies_sorted_by_distance() -> None:
    """敵 2 個。d1 が最近、d2 が次に近い。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 90.0, 50.0, 1.0, 5, 1],  # 敵、距離 40 (遠い)
            [2, 1, 60.0, 50.0, 1.0, 8, 2],  # 敵、距離 10 (近い)
        ]
    )
    batch, _ = featurize(obs)
    # d1 = log1p(10) (最近の P2)、d2 = log1p(40) (P1)
    expected_d1 = math.log1p(10.0) / math.log1p(100.0)
    expected_d2 = math.log1p(40.0) / math.log1p(100.0)
    assert abs(batch.planet_feats[0, 0, COL_FL_D1].item() - expected_d1) < 1e-5
    assert abs(batch.planet_feats[0, 0, COL_FL_D2].item() - expected_d2) < 1e-5
