"""case7 iter4 特徴量 (K2 Outgoing fleet trajectory + K3 Frontline distance + K4 Aux global) のテスト。

idx layout (iter4):
  planet 53-56: K2 Outgoing fleet trajectory (dx, dy, dist, ships_log)
  planet 57-60: K3 Frontline distance (d1_log, r1_log, d2_log, r2_log)
  global 8-11:  K4 Aux multi-horizon (my_ships_h5, my_ships_h15, my_prod_log, ships_ratio)
"""

from __future__ import annotations

import math

from pipeline.imitation.case7.policy.featurizer import (
    featurize,
)

# K2 Outgoing fleet
COL_OUT_DX = 53
COL_OUT_DY = 54
COL_OUT_DIST = 55
COL_OUT_SHIPS_LOG = 56

# K3 Frontline distance
COL_FL_D1 = 57
COL_FL_R1 = 58
COL_FL_D2 = 59
COL_FL_R2 = 60

# K4 Aux global
GCOL_AUX_H5 = 8
GCOL_AUX_H15 = 9
GCOL_AUX_PROD_LOG = 10
GCOL_AUX_RATIO = 11


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


# ----- K2 Outgoing fleet trajectory -----


def test_k2_outgoing_no_own_fleet_returns_sentinel_dist() -> None:
    """自軍 fleet 無し → outgoing_dist=-1, 他 0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_OUT_DX].item() == 0.0
    assert batch.planet_feats[0, 0, COL_OUT_DY].item() == 0.0
    assert batch.planet_feats[0, 0, COL_OUT_DIST].item() == -1.0
    assert batch.planet_feats[0, 0, COL_OUT_SHIPS_LOG].item() == 0.0


def test_k2_outgoing_own_fleet_writes_nonzero() -> None:
    """自軍 fleet が source planet 0 から出ている → outgoing trajectory が立つ。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.5, 10, 2],
        ],
        fleets=[
            # fid, fowner=0 (own), fx, fy, fangle, from_pid=0, fships=5
            [100, 0, 60.0, 50.0, 0.0, 0, 5],
        ],
    )
    batch, _ = featurize(obs)
    # dist != -1 (outgoing 認識された)
    assert batch.planet_feats[0, 0, COL_OUT_DIST].item() != -1.0
    # ships_log = log1p(5) > 0
    assert batch.planet_feats[0, 0, COL_OUT_SHIPS_LOG].item() > 0.0


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


# ----- K4 Aux multi-horizon global -----


def test_k4_aux_global_h5_h15_grows_with_my_production() -> None:
    """自軍 production を増やすと aux_h5/h15 (log1p ベース) が増える。"""
    obs_low = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    obs_high = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 10]])
    b_low, _ = featurize(obs_low)
    b_high, _ = featurize(obs_high)
    # h=5: log1p(10 + 5 * 2) = log1p(20) vs log1p(10 + 5 * 10) = log1p(60)
    assert (
        b_high.global_feats[0, GCOL_AUX_H5].item()
        > b_low.global_feats[0, GCOL_AUX_H5].item()
    )
    assert (
        b_high.global_feats[0, GCOL_AUX_H15].item()
        > b_low.global_feats[0, GCOL_AUX_H15].item()
    )


def test_k4_aux_ratio_one_when_only_my_planet() -> None:
    """自軍だけなら ratio = 1.0、enemy がいるなら < 1.0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.global_feats[0, GCOL_AUX_RATIO].item() == 1.0


def test_k4_aux_ratio_half_when_equal_ships() -> None:
    """自軍と敵で同 ships → ratio = 0.5。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 70.0, 50.0, 1.0, 10, 2],
        ]
    )
    batch, _ = featurize(obs)
    assert abs(batch.global_feats[0, GCOL_AUX_RATIO].item() - 0.5) < 1e-5


def test_k4_aux_prod_log_log1p_of_my_total_prod() -> None:
    """my_prod_log = log1p(自軍 production sum)。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 3],
            [1, 0, 60.0, 50.0, 1.0, 5, 4],  # 自軍 2 個目
        ]
    )
    batch, _ = featurize(obs)
    expected = math.log1p(3 + 4)
    assert abs(batch.global_feats[0, GCOL_AUX_PROD_LOG].item() - expected) < 1e-5
