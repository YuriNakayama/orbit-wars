"""case7 iter2 特徴量 (fleet trajectory / multi-horizon / production-centroid / comet) のテスト。

idx layout (iter2):
  planet 24-27: fleet trajectory (dx, dy, dist, ships_log)
  planet 28-31: multi-horizon (loss_5turn, loss_15turn, min_owned_5turn, min_owned_15turn)
  planet 32-33: home_dist_log, prod_centroid_dist_log
  global 10-11: next_comet_eta_norm, comet_active_flag
  global 12-13: home_planet_owner_flag, prod_centroid_dist
"""

from __future__ import annotations

import math

from pipeline.imitation.case7.policy.featurizer import (
    HistoryState,
    featurize,
    fleet_snapshot_from_obs,
    planet_snapshot_from_obs,
)

# iter4 column indices (G6 削除 + H1/H4/H5 削除を反映)
COL_INBOUND_DX = 22
COL_INBOUND_DY = 23
COL_INBOUND_DIST = 24
COL_INBOUND_SHIPS_LOG = 25
COL_LOSS_5 = 26
COL_LOSS_15 = 27
COL_MIN_OWNED_5 = 28
COL_MIN_OWNED_15 = 29
COL_HOME_DIST = 30
COL_PROD_CENTROID_DIST = 31

# iter4: H1/H4/H5 削除済 — comet/home_owner/prod_centroid global は廃止


def _obs(
    step: int = 0,
    planets: list[list[float | int]] | None = None,
    fleets: list[list[float | int]] | None = None,
    comet_ids: list[int] | None = None,
    initial_planets: list[list[float | int]] | None = None,
    ang_vel: float = 0.0,
    player: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "step": step,
        "planets": planets if planets is not None else [],
        "fleets": fleets if fleets is not None else [],
        "comet_planet_ids": comet_ids if comet_ids is not None else [],
        "initial_planets": initial_planets if initial_planets is not None else [],
        "angular_velocity": ang_vel,
    }


# ----- Fleet trajectory -----


def test_no_inbound_fleet_returns_sentinel_minus_one_for_dist() -> None:
    """敵 fleet が無いと inbound_dist = -1 (sentinel)、他は 0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_INBOUND_DX].item() == 0.0
    assert batch.planet_feats[0, 0, COL_INBOUND_DY].item() == 0.0
    assert batch.planet_feats[0, 0, COL_INBOUND_DIST].item() == -1.0
    assert batch.planet_feats[0, 0, COL_INBOUND_SHIPS_LOG].item() == 0.0


def test_inbound_enemy_fleet_writes_nonzero_trajectory() -> None:
    """敵 fleet が向かってくる自軍 planet に対して inbound trajectory が立つ。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.5, 10, 2],  # 自軍
        ],
        fleets=[
            # fid, fowner=1 (enemy), fx, fy, fangle = π (=> moving toward x=50,y=50), from_pid, ships
            [100, 1, 60.0, 50.0, math.pi, 1, 5],
        ],
    )
    batch, _ = featurize(obs)
    # dist != -1 (有効、inbound 認識された)
    assert batch.planet_feats[0, 0, COL_INBOUND_DIST].item() != -1.0
    # ships_log = log1p(5) > 0
    assert batch.planet_feats[0, 0, COL_INBOUND_SHIPS_LOG].item() > 0.0


# ----- Multi-horizon ship-prediction -----


def test_multi_horizon_default_zero_for_neutral_planet() -> None:
    """neutral planet は自所有でないので min_owned_{5,15} = 0、loss も 0 (静的)。"""
    obs = _obs(planets=[[0, -1, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_MIN_OWNED_5].item() == 0.0
    assert batch.planet_feats[0, 0, COL_MIN_OWNED_15].item() == 0.0


def test_multi_horizon_writes_log1p_min_owned_for_my_planet() -> None:
    """自軍 planet は production で増えていく → min_owned >= ships_now、log1p で >0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    # log1p(10) ≒ 2.398 (float32 精度内で一致)
    assert abs(batch.planet_feats[0, 0, COL_MIN_OWNED_5].item() - math.log1p(10)) < 1e-5
    assert (
        abs(batch.planet_feats[0, 0, COL_MIN_OWNED_15].item() - math.log1p(10)) < 1e-5
    )


# ----- Home distance / production-weighted centroid -----


def test_home_distance_zero_when_planet_is_home() -> None:
    """home planet そのものなら home_dist = log1p(0) / log1p(100) = 0。"""
    obs = _obs(
        planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]],
        initial_planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]],
    )
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_HOME_DIST].item() == 0.0


def test_prod_centroid_dist_zero_for_single_my_planet() -> None:
    """自軍 planet が 1 つだけなら、その planet の production centroid との距離 = 0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 5]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_PROD_CENTROID_DIST].item() == 0.0


# ----- Global features (iter4: H4/H5 削除済、comet/home_owner/prod_centroid テストは廃止) -----


# ----- Causal safety -----


def test_iter2_features_deterministic_without_history() -> None:
    """iter2 の追加列 (fleet trajectory / multi-horizon / centroid / comet) は
    history を参照しない: 同 obs を 2 回 featurize して全列が一致する。"""
    obs = _obs(
        step=10,
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 70.0, 50.0, 1.0, 8, 1],
        ],
        fleets=[
            [100, 1, 60.0, 50.0, math.pi, 1, 5],
        ],
    )
    batch_a, _ = featurize(obs)
    history = HistoryState()
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    batch_b, _ = featurize(obs, history=history)
    # iter4 列 (22-31): fleet trajectory (22-25) + multi-horizon (26-29) + home/centroid (30-31)
    for col in range(22, 32):
        a = batch_a.planet_feats[0, 0, col].item()
        b = batch_b.planet_feats[0, 0, col].item()
        assert a == b, f"col {col}: history-independent expected, got {a} vs {b}"
