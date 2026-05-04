"""case7 ship launch event (per-planet enemy targeting + global counts) の動作確認。"""

from __future__ import annotations

from pipeline.imitation.case7.policy.featurizer import (
    HistoryState,
    featurize,
    fleet_snapshot_from_obs,
    planet_snapshot_from_obs,
)

COL_ENEMY_TARGETED_COUNT = 22
COL_ENEMY_TARGETED_SHIPS = 23
GCOL_ENEMY_LAUNCH_COUNT = 6
GCOL_ENEMY_LAUNCH_SHIPS = 7
GCOL_ALLY_LAUNCH_COUNT = 8
GCOL_ALLY_LAUNCH_SHIPS = 9


def _obs(
    step: int, planets: list[list[float | int]], fleets: list[list[float | int]]
) -> dict[str, object]:
    return {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
    }


def test_enemy_launch_increments_global_counts() -> None:
    """敵 fleet が新規に出現したら global enemy_launch_count_last4 が増える。"""
    history = HistoryState()
    planets = [
        [0, 0, 50.0, 50.0, 1.0, 10, 2],
        [1, 1, 70.0, 50.0, 1.0, 10, 2],
    ]
    obs0 = _obs(0, planets, fleets=[])
    featurize(obs0, history=history)
    history.push(planet_snapshot_from_obs(obs0), fleet_snapshot_from_obs(obs0))

    # turn 1: 敵 fleet 追加 (= turn 0→1 で発射された)
    obs1 = _obs(1, planets, fleets=[[100, 1, 65.0, 50.0, 3.14, 1, 8]])
    featurize(obs1, history=history)
    history.push(planet_snapshot_from_obs(obs1), fleet_snapshot_from_obs(obs1))

    # turn 2: featurize 時点で history は (snap_0, snap_1) → 差分 = 敵 fleet 1 件
    obs2 = _obs(2, planets, fleets=[[100, 1, 65.0, 50.0, 3.14, 1, 8]])
    batch, _ = featurize(obs2, history=history)
    assert batch.global_feats[0, GCOL_ENEMY_LAUNCH_COUNT].item() > 0.0
    assert batch.global_feats[0, GCOL_ENEMY_LAUNCH_SHIPS].item() > 0.0
    # 自軍 launch は 0 のまま
    assert batch.global_feats[0, GCOL_ALLY_LAUNCH_COUNT].item() == 0.0


def test_no_history_means_zero_launch_counts() -> None:
    """history なしでは launch 集計 0 (cold start)。"""
    obs = _obs(0, [[0, 0, 50.0, 50.0, 1.0, 10, 2]], fleets=[])
    batch, _ = featurize(obs)
    assert batch.global_feats[0, GCOL_ENEMY_LAUNCH_COUNT].item() == 0.0
    assert batch.global_feats[0, GCOL_ALLY_LAUNCH_COUNT].item() == 0.0
