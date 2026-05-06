"""case8 history 列の動作確認: delta_ships_t1 / delta_ships_t2 / owner_changed_t1。

case3 phase2 で発生した causal leak (obs_{N-1} 参照) を回避するため、
case8 では obs_{N-2} / obs_{N-3} 参照のみを使う。
"""

from __future__ import annotations

from pipeline.imitation.case8.policy.featurizer import (
    HistoryState,
    featurize,
    fleet_snapshot_from_obs,
    planet_snapshot_from_obs,
)

# planet 列インデックス (case8 layout)
COL_DELTA_T1 = 19
COL_DELTA_T2 = 20
COL_OWNER_CHANGED_T1 = 21


def _make_obs(
    step: int,
    planets: list[list[float | int]],
    fleets: list[list[float | int]] | None = None,
) -> dict[str, object]:
    return {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }


def test_delta_ships_t1_reflects_obs_n_minus_2_difference() -> None:
    """obs_{N} と obs_{N−2} の ships 差分が delta_ships_t1 に反映される。"""
    history = HistoryState()
    # turn 0: ships=10
    obs0 = _make_obs(0, [[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    featurize(obs0, history=history)
    history.push(planet_snapshot_from_obs(obs0), fleet_snapshot_from_obs(obs0))

    # turn 1: ships=12
    obs1 = _make_obs(1, [[0, 0, 50.0, 50.0, 1.0, 12, 2]])
    featurize(obs1, history=history)
    history.push(planet_snapshot_from_obs(obs1), fleet_snapshot_from_obs(obs1))

    # turn 2: ships=14 → obs_{N-2} = obs_0 (ships=10) との差分が見える
    obs2 = _make_obs(2, [[0, 0, 50.0, 50.0, 1.0, 14, 2]])
    batch, _ = featurize(obs2, history=history)

    # delta_t1 = (14 − 10) / 14 ≒ 0.286
    delta_t1 = batch.planet_feats[0, 0, COL_DELTA_T1].item()
    assert abs(delta_t1 - 4.0 / 14.0) < 1e-5


def test_owner_changed_flag_set_when_owner_differs_from_n_minus_2() -> None:
    """owner_changed_t1: obs_{N−2} と owner が違う場合 1。"""
    history = HistoryState()
    obs0 = _make_obs(0, [[0, 1, 50.0, 50.0, 1.0, 5, 1]])  # 敵所有
    featurize(obs0, history=history)
    history.push(planet_snapshot_from_obs(obs0), fleet_snapshot_from_obs(obs0))

    obs1 = _make_obs(1, [[0, 1, 50.0, 50.0, 1.0, 5, 1]])
    featurize(obs1, history=history)
    history.push(planet_snapshot_from_obs(obs1), fleet_snapshot_from_obs(obs1))

    # turn 2: 自軍に変わった → owner が obs_0 (=1) と異なる → flag=1
    obs2 = _make_obs(2, [[0, 0, 50.0, 50.0, 1.0, 3, 1]])
    batch, _ = featurize(obs2, history=history)
    assert batch.planet_feats[0, 0, COL_OWNER_CHANGED_T1].item() == 1.0


def test_history_columns_zero_when_no_prior_snapshots() -> None:
    """history が短い (turn 0) と全 0、リーク余地なし。"""
    history = HistoryState()
    obs0 = _make_obs(0, [[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs0, history=history)
    assert batch.planet_feats[0, 0, COL_DELTA_T1].item() == 0.0
    assert batch.planet_feats[0, 0, COL_DELTA_T2].item() == 0.0
    assert batch.planet_feats[0, 0, COL_OWNER_CHANGED_T1].item() == 0.0
