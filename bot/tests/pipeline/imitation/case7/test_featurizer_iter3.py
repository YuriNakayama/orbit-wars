"""case7 iter3 特徴量 (Pairwise Top-K / Defense surplus / Sparse mask) のテスト。

idx layout (iter3):
  planet 34-53: Pairwise Top-K (5 neighbors × 4 feat)
    34..37: k=0 (dist_log, ships_ratio_log, owner_signed, prod_log)
    38..41: k=1
    42..45: k=2
    46..49: k=3
    50..53: k=4
  planet 54-57: Defense surplus (h=5, h=15, h=30, now)
  planet 58-62: Sparse mask flags (inbound, history, enemy_targeted, is_home, neighbor_enemy)
"""

from __future__ import annotations

import math

from pipeline.imitation.case7.policy.featurizer import (
    HistoryState,
    featurize,
    fleet_snapshot_from_obs,
    planet_snapshot_from_obs,
)

# Pairwise Top-K
COL_TOPK_K0_DIST = 34
COL_TOPK_K0_SHIPS_RATIO = 35
COL_TOPK_K0_OWNER_SIGNED = 36
COL_TOPK_K0_PROD_LOG = 37
COL_TOPK_K4_DIST = 50
# Defense surplus
COL_DEF_5 = 54
COL_DEF_15 = 55
COL_DEF_30 = 56
COL_DEF_NOW = 57
# Sparse mask flags
COL_HAS_INBOUND = 58
COL_HAS_HISTORY = 59
COL_HAS_ENEMY_TARGETED = 60
COL_IS_HOME = 61
COL_IS_NEIGHBOR_ENEMY = 62


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


# ----- Pairwise Top-K -----


def test_topk_sentinel_when_only_one_planet() -> None:
    """planet が 1 個だけなら Top-K は全 sentinel (dist_log = -1)。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    for k in range(5):
        col = 34 + k * 4
        assert batch.planet_feats[0, 0, col].item() == -1.0


def test_topk_k0_is_nearest_with_correct_owner() -> None:
    """K=0 (最も近い隣) が 期待 planet と一致するか。

    自軍 P0 (50, 50)、敵 P1 (60, 50, dist=10)、中立 P2 (90, 50, dist=40)。
    P0 から見て K=0 は P1 (敵)、K=1 は P2 (中立)。
    """
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],  # 自軍
            [1, 1, 60.0, 50.0, 1.0, 5, 3],  # 敵
            [2, -1, 90.0, 50.0, 1.0, 3, 1],  # 中立
        ]
    )
    batch, _ = featurize(obs)
    # K=0: P1 (敵) → owner_signed = -1
    assert batch.planet_feats[0, 0, COL_TOPK_K0_OWNER_SIGNED].item() == -1.0
    # K=0 dist_log = log1p(10) / log1p(100)
    expected_dist = math.log1p(10.0) / math.log1p(100.0)
    assert abs(batch.planet_feats[0, 0, COL_TOPK_K0_DIST].item() - expected_dist) < 1e-5
    # K=0 prod_log = log1p(3) (P1's production)
    assert (
        abs(batch.planet_feats[0, 0, COL_TOPK_K0_PROD_LOG].item() - math.log1p(3))
        < 1e-5
    )


def test_topk_remaining_slots_sentinel_when_few_neighbors() -> None:
    """planet が 3 個 (K=2 までしか埋まらない) → K=3, K=4 は sentinel。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 60.0, 50.0, 1.0, 5, 3],
            [2, -1, 70.0, 50.0, 1.0, 3, 1],
        ]
    )
    batch, _ = featurize(obs)
    # K=2 は埋まる (P2)
    # K=3, K=4 は sentinel: dist_log = -1 (この時点で 36 や 37 は不変、テストは dist_log のみ)
    col_k3_dist = 34 + 3 * 4  # = 46
    col_k4_dist = 34 + 4 * 4  # = 50
    assert batch.planet_feats[0, 0, col_k3_dist].item() == -1.0
    assert batch.planet_feats[0, 0, col_k4_dist].item() == -1.0


# ----- Defense surplus -----


def test_defense_surplus_zero_when_neutral_planet() -> None:
    """neutral planet は own_ships=0 / production=0 扱いで margin = 0 / 100 → 0。"""
    obs = _obs(planets=[[0, -1, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_DEF_NOW].item() == 0.0
    assert batch.planet_feats[0, 0, COL_DEF_5].item() == 0.0
    assert batch.planet_feats[0, 0, COL_DEF_15].item() == 0.0
    assert batch.planet_feats[0, 0, COL_DEF_30].item() == 0.0


def test_defense_surplus_positive_for_safe_my_planet() -> None:
    """敵 fleet 無しの自軍 planet → margin_now = ships / 100、_5/15/30 は production で更に増。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    # margin_now = (10 - 0) / 100 = 0.10
    assert abs(batch.planet_feats[0, 0, COL_DEF_NOW].item() - 0.10) < 1e-4
    # margin_5 = (10 + 5*2 - 0) / 100 = 0.20
    assert abs(batch.planet_feats[0, 0, COL_DEF_5].item() - 0.20) < 1e-4
    # margin_30 clamped at 1.0 because (10 + 30*2 = 70) / 100 = 0.70
    assert abs(batch.planet_feats[0, 0, COL_DEF_30].item() - 0.70) < 1e-4


def test_defense_surplus_clamped_to_one() -> None:
    """Production が大きいと clamp 1.0。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 50, 10]])
    batch, _ = featurize(obs)
    # margin_30 = (50 + 30*10) / 100 = 3.5 → clamped 1.0
    assert batch.planet_feats[0, 0, COL_DEF_30].item() == 1.0


# ----- Sparse mask flags -----


def test_has_history_flag_zero_without_history() -> None:
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_HAS_HISTORY].item() == 0.0


def test_has_history_flag_one_with_2_snapshots() -> None:
    """obs を 2 回 push した HistoryState で has_history_flag = 1。"""
    obs = _obs(planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]])
    history = HistoryState()
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    batch, _ = featurize(obs, history=history)
    assert batch.planet_feats[0, 0, COL_HAS_HISTORY].item() == 1.0


def test_is_home_planet_flag_set_when_initial_planet_present() -> None:
    obs = _obs(
        planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]],
        initial_planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]],
    )
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_IS_HOME].item() == 1.0


def test_is_home_planet_flag_zero_when_not_home() -> None:
    """この planet は home ではない (initial_planets[0].id != current pid)。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [99, 0, 30.0, 30.0, 1.0, 5, 1],  # 自軍だが home ではない
        ],
        initial_planets=[[0, 0, 50.0, 50.0, 1.0, 10, 2]],  # home id = 0
    )
    batch, _ = featurize(obs)
    # planet slot 1 (id=99) は home ではない
    assert batch.planet_feats[0, 1, COL_IS_HOME].item() == 0.0


def test_is_neighbor_to_enemy_flag_set_when_enemy_within_quarter_board() -> None:
    """半径 BOARD_SIZE/4 = 25 以内に敵 planet → flag = 1。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 70.0, 50.0, 1.0, 8, 1],  # 敵、距離 20 < 25
        ]
    )
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_IS_NEIGHBOR_ENEMY].item() == 1.0


def test_is_neighbor_to_enemy_flag_zero_when_enemy_far() -> None:
    """敵が半径 25 より遠い → flag = 0。"""
    obs = _obs(
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 90.0, 50.0, 1.0, 8, 1],  # 敵、距離 40 > 25
        ]
    )
    batch, _ = featurize(obs)
    assert batch.planet_feats[0, 0, COL_IS_NEIGHBOR_ENEMY].item() == 0.0


# ----- Causal safety -----


def test_iter3_features_deterministic_without_history() -> None:
    """iter3 列 (34-62) は HistoryState 有無で値が変わらない (history 非依存) ものが大半。
    has_history_flag (59) のみ history で変わるが他は同値。"""
    obs = _obs(
        step=10,
        planets=[
            [0, 0, 50.0, 50.0, 1.0, 10, 2],
            [1, 1, 70.0, 50.0, 1.0, 8, 1],
        ],
    )
    batch_a, _ = featurize(obs)
    history = HistoryState()
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    history.push(planet_snapshot_from_obs(obs), fleet_snapshot_from_obs(obs))
    batch_b, _ = featurize(obs, history=history)
    # Pairwise Top-K (34-53) と Defense surplus (54-57) と mask 60-62 は history 非依存
    for col in list(range(34, 58)) + [60, 61, 62]:
        a = batch_a.planet_feats[0, 0, col].item()
        b = batch_b.planet_feats[0, 0, col].item()
        assert a == b, f"col {col}: history-independent expected, got {a} vs {b}"
    # has_history_flag (59) だけは異なる
    assert batch_a.planet_feats[0, 0, COL_HAS_HISTORY].item() == 0.0
    assert batch_b.planet_feats[0, 0, COL_HAS_HISTORY].item() == 1.0
