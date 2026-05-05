"""obs (dict) → BatchFeatures (torch.Tensor) for imitation/case7 IL baseline.

case5 base (11 + 6 = 17 列 / planet, 6 列 / global) に、以下の 11 列を追加:
  planet 17: future_dist_to_my_centroid   (orbit 予測 5-turn 先の自軍重心距離)
  planet 18: future_dist_to_enemy_centroid (同 敵重心距離)
  planet 19: delta_ships_t1                (history: ships_now − ships_{N−2})
  planet 20: delta_ships_t2                (history: ships_now − ships_{N−3})
  planet 21: owner_changed_t1              (history: owner が N−2 から変わったか)
  planet 22: enemy_targeted_count_last4    (敵 fleet 直近 4 ターン発射回数)
  planet 23: enemy_targeted_ships_last4    (敵 fleet 直近 4 ターン発射 ships 合計)
  global  6: enemy_launch_count_last4      (case3 phase2 流用)
  global  7: enemy_launch_ships_last4
  global  8: ally_launch_count_last4
  global  9: ally_launch_ships_last4

history 参照は obs_{N-2} / obs_{N-3} のみ (case3 phase2 で確認した causal leak 対策)。
launch event は prev_fleet_snapshot 差分 (`prev_fleets_{N-2..N-5}` の N−1 と N−2 比較) で算出し、
action_N との直接相関を避ける。HistoryState は agent.py / preprocess.py が per-match ring
buffer として保持する。

Pure function — no torch.nn, no autograd, no random state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .templates import TEMPLATE_CTX_DIM, template_context_features
from .timeline import (
    DEFAULT_HORIZON as TIMELINE_HORIZON,
)
from .timeline import (
    TimelinePlanet,
    simulate_planet_timeline,
    summarize_timeline,
    summarize_timeline_multi_horizon,
)
from .types import BatchFeatures, WorldSnapshot

# iter6: K2/K4 削除で iter4 の causal leak を除去。
# 削除:
#   K2 outgoing_fleet_trajectory (planet 53-56, 4列)
#     causal leak: obs.fleets は action_N 適用後の値。自軍 fleet の from_pid と ships が
#     ships_per_src ラベルと完全一致 → policy が逆算可能。iter5 permutation で sum |Δ|=1.70 検出。
#   K4 aux_multi_horizon_global (global 8-11, 4列)
#     iter5 permutation で sum |Δ|=0.001、H2 と相関で意義なし。dead group。
# 結果: planet 61→57、global 12→8、K3 frontline_distance は planet 57→53 に slot shift。
PLANET_FEAT_DIM = 57  # iter4 61 - K2 4 = 57
GLOBAL_FEAT_DIM = 8  # iter4 12 - K4 4 = 8
MAX_PLANETS = 36
TOPK_NEIGHBORS = 3  # iter4 K1: pairwise top-K count (5 から削減)
TOPK_FEATS_PER_NEIGHBOR = 4  # dist_log, ships_ratio_log, owner_signed, prod_log
BOARD_SIZE = 100.0
HORIZON_TURNS = 30  # for incoming-fleet eta normalization
PREDICT_TURNS = 5  # future-position prediction horizon (orbit rotation)
INBOUND_FLEET_FUTURE_TURNS = (
    5  # how far ahead to project the earliest inbound enemy fleet
)
COMET_TURNS = (50, 150, 250, 350, 450)  # comet spawn turns (from competition spec)
SUN_X = 50.0
SUN_Y = 50.0
LAUNCH_HISTORY_WINDOW = 4  # last-N turns for ally/enemy launch event aggregation
PLANET_SNAPSHOT_MAXLEN = 4  # need obs_{N-2} / obs_{N-3} → keep 4 history slots
FLEET_SNAPSHOT_MAXLEN = 5  # last 4 turns of launch differences → 5 fleet snapshots


@dataclass
class HistoryState:
    """Per-match ring buffer fed by agent.py / preprocess.py before each featurize().

    All snapshots are *raw obs payloads*, not derived features, so featurizer can
    re-compute derived columns deterministically from history alone.
    """

    prev_planet_snapshots: deque[dict[int, tuple[int, int]]] = field(
        default_factory=lambda: deque(maxlen=PLANET_SNAPSHOT_MAXLEN)
    )
    prev_fleet_snapshots: deque[
        list[tuple[int, int, float, float, float, int, int]]
    ] = field(default_factory=lambda: deque(maxlen=FLEET_SNAPSHOT_MAXLEN))

    def push(
        self,
        planets_by_id: dict[int, tuple[int, int]],
        fleets: list[tuple[int, int, float, float, float, int, int]],
    ) -> None:
        self.prev_planet_snapshots.append(planets_by_id)
        self.prev_fleet_snapshots.append(fleets)


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
    # Mirrors pipeline/rulebase/case1/baseline/core/physics.py::fleet_speed.
    # Fewer ships fly faster; we only need a monotone proxy here.
    return max(0.5, 2.0 - 0.05 * math.sqrt(max(1, ships)))


def _fleet_target_eta(
    fleet_x: float,
    fleet_y: float,
    angle: float,
    ships: int,
    planet_x: float,
    planet_y: float,
    planet_radius: float,
) -> float | None:
    dx = planet_x - fleet_x
    dy = planet_y - fleet_y
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    proj = dx * dir_x + dy * dir_y
    if proj < 0:
        return None
    perp_sq = dx * dx + dy * dy - proj * proj
    radius_sq = planet_radius * planet_radius
    if perp_sq >= radius_sq:
        return None
    hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
    speed = _fleet_speed(ships)
    if speed <= 0:
        return None
    return hit_d / speed


def _batch_fleet_target_eta(
    fleet_x: np.ndarray,  # (F,)
    fleet_y: np.ndarray,
    angle: np.ndarray,
    ships: np.ndarray,
    planet_x: np.ndarray,  # (P,)
    planet_y: np.ndarray,
    planet_radius: np.ndarray,
) -> np.ndarray:
    """Vectorized version of _fleet_target_eta. Returns (F, P) ETA matrix.

    Cells where the fleet does not hit the planet within HORIZON_TURNS are set to
    +inf so the caller can apply a single threshold mask.
    """
    if fleet_x.size == 0 or planet_x.size == 0:
        return np.empty((fleet_x.size, planet_x.size), dtype=np.float64)
    dx = planet_x[None, :] - fleet_x[:, None]  # (F, P)
    dy = planet_y[None, :] - fleet_y[:, None]
    dir_x = np.cos(angle)[:, None]  # (F, 1)
    dir_y = np.sin(angle)[:, None]
    proj = dx * dir_x + dy * dir_y
    perp_sq = dx * dx + dy * dy - proj * proj
    radius_sq = (planet_radius * planet_radius)[None, :]  # (1, P)

    speed = np.maximum(0.5, 2.0 - 0.05 * np.sqrt(np.maximum(1, ships)))[
        :, None
    ]  # (F, 1)
    hit_d = np.maximum(0.0, proj - np.sqrt(np.maximum(0.0, radius_sq - perp_sq)))
    eta = hit_d / np.where(speed > 0, speed, np.inf)
    invalid = (proj < 0) | (perp_sq >= radius_sq) | (eta > HORIZON_TURNS)
    eta = np.where(invalid, np.inf, eta)
    return eta


def _future_position(
    x: float, y: float, ang_vel: float, turns: int
) -> tuple[float, float]:
    """Rotate (x, y) around (SUN_X, SUN_Y) by ang_vel * turns. Static planets pass through."""
    if abs(ang_vel) < 1e-9:
        return x, y
    rx = x - SUN_X
    ry = y - SUN_Y
    theta = ang_vel * float(turns)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    new_x = rx * cos_t - ry * sin_t + SUN_X
    new_y = rx * sin_t + ry * cos_t + SUN_Y
    return new_x, new_y


def _fleet_action_target(
    fleet_x: float,
    fleet_y: float,
    angle: float,
    ships: int,
    planet_rows: list[Any],
) -> int | None:
    """Reverse-resolve which planet a fleet is heading toward (best ETA <= horizon)."""
    best_pid: int | None = None
    best_eta = HORIZON_TURNS + 1.0
    for row in planet_rows:
        pid_, _, px, py, pradius, _, _ = row
        eta = _fleet_target_eta(
            fleet_x, fleet_y, angle, ships, float(px), float(py), float(pradius)
        )
        if eta is None or eta > HORIZON_TURNS:
            continue
        if eta < best_eta:
            best_eta = eta
            best_pid = int(pid_)
    return best_pid


def _diff_launches(
    snap_a: list[tuple[int, int, float, float, float, int, int]] | None,
    snap_b: list[tuple[int, int, float, float, float, int, int]] | None,
) -> list[tuple[int, int, float, float, float, int, int]]:
    """Fleets present in snap_b but not in snap_a — i.e. launched between A→B."""
    if snap_a is None or snap_b is None:
        return []
    ids_a = {int(f[0]) for f in snap_a}
    return [f for f in snap_b if int(f[0]) not in ids_a]


def featurize(
    obs: Any,
    history: HistoryState | None = None,
) -> tuple[BatchFeatures, WorldSnapshot]:
    """Convert a single observation dict to a BatchFeatures of batch_size=1.

    Args:
        obs: obs dict (Kaggle env step output for one player).
        history: optional per-match ring buffer for delta_ships / launch history.
            When None, history-derived columns are zero (cold-start, smoke tests).
    """
    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = list(_read(obs, "planets", []) or [])
    raw_fleets = list(_read(obs, "fleets", []) or [])
    raw_comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    raw_initial_planets = list(_read(obs, "initial_planets", []) or [])
    ang_vel = float(_read(obs, "angular_velocity", 0.0) or 0.0)

    n = min(len(raw_planets), MAX_PLANETS)
    planet_feats = torch.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=torch.float32)
    planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    my_planet_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    target_mask = torch.zeros(MAX_PLANETS, dtype=torch.bool)
    template_ctx = torch.zeros((MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=torch.float32)

    planet_ids: list[int] = []
    my_planet_ids: list[int] = []

    incoming = [[0.0, 0.0, 0.0] for _ in range(MAX_PLANETS)]
    nearest_eta = [HORIZON_TURNS + 1.0] * MAX_PLANETS
    arrivals_by_slot: list[list[tuple[float, int, int]]] = [
        [] for _ in range(MAX_PLANETS)
    ]

    planet_index_by_id: dict[int, int] = {}
    planet_x_arr = np.zeros(n, dtype=np.float64)
    planet_y_arr = np.zeros(n, dtype=np.float64)
    planet_r_arr = np.zeros(n, dtype=np.float64)
    for slot in range(n):
        pid = int(raw_planets[slot][0])
        planet_index_by_id[pid] = slot
        planet_x_arr[slot] = float(raw_planets[slot][2])
        planet_y_arr[slot] = float(raw_planets[slot][3])
        planet_r_arr[slot] = float(raw_planets[slot][4])

    if raw_fleets:
        n_fleets = len(raw_fleets)
        fx_arr = np.zeros(n_fleets, dtype=np.float64)
        fy_arr = np.zeros(n_fleets, dtype=np.float64)
        fa_arr = np.zeros(n_fleets, dtype=np.float64)
        fs_arr = np.zeros(n_fleets, dtype=np.int64)
        fo_arr = np.zeros(n_fleets, dtype=np.int64)
        for i, fleet_row in enumerate(raw_fleets):
            fid, fowner, fx, fy, fangle, _from_pid, fships = fleet_row
            fo_arr[i] = int(fowner)
            fs_arr[i] = int(fships)
            fx_arr[i] = float(fx)
            fy_arr[i] = float(fy)
            fa_arr[i] = float(fangle)

        eta_mat = _batch_fleet_target_eta(
            fx_arr,
            fy_arr,
            fa_arr,
            fs_arr,
            planet_x_arr,
            planet_y_arr,
            planet_r_arr,
        )  # (F, P), inf for misses

        # Aggregate per (fleet, slot) hits
        valid = np.isfinite(eta_mat)
        # iter2: track earliest enemy fleet per slot for trajectory features
        earliest_enemy_fleet_idx_per_slot: list[int | None] = [None] * MAX_PLANETS
        # For each slot, find nearest_eta and accumulate per-owner ships
        for slot in range(n):
            slot_valid = valid[:, slot]
            if not slot_valid.any():
                continue
            slot_etas = eta_mat[slot_valid, slot]
            slot_owners = fo_arr[slot_valid]
            slot_ships = fs_arr[slot_valid]
            slot_indices = np.where(slot_valid)[0]
            # min eta
            min_eta = float(slot_etas.min())
            if min_eta < nearest_eta[slot]:
                nearest_eta[slot] = min_eta
            # owner-bucketed ship sums
            mine_mask = slot_owners == player
            neut_mask = slot_owners == -1
            enemy_mask = ~mine_mask & ~neut_mask
            incoming[slot][0] += float(slot_ships[mine_mask].sum())
            incoming[slot][2] += float(slot_ships[neut_mask].sum())
            incoming[slot][1] += float(slot_ships[enemy_mask].sum())
            # iter2: pick earliest enemy inbound fleet for this slot
            if enemy_mask.any():
                enemy_etas_in_slot = slot_etas[enemy_mask]
                enemy_indices_in_slot = slot_indices[enemy_mask]
                k_min = int(enemy_etas_in_slot.argmin())
                earliest_enemy_fleet_idx_per_slot[slot] = int(
                    enemy_indices_in_slot[k_min]
                )
            # arrivals (timeline 用)
            for eta_v, ow_v, sh_v in zip(
                slot_etas, slot_owners, slot_ships, strict=True
            ):
                arrivals_by_slot[slot].append((float(eta_v), int(ow_v), int(sh_v)))

        # iter2: compute per-slot inbound fleet trajectory features (5-turn future pos)
        inbound_fleet_dx = [0.0] * MAX_PLANETS
        inbound_fleet_dy = [0.0] * MAX_PLANETS
        inbound_fleet_dist = [-1.0] * MAX_PLANETS  # sentinel: no inbound
        inbound_fleet_ships_log = [0.0] * MAX_PLANETS
        for slot in range(n):
            f_idx = earliest_enemy_fleet_idx_per_slot[slot]
            if f_idx is None:
                continue
            f_x_now = float(fx_arr[f_idx])
            f_y_now = float(fy_arr[f_idx])
            f_angle = float(fa_arr[f_idx])
            f_ships = int(fs_arr[f_idx])
            speed = max(0.5, 2.0 - 0.05 * math.sqrt(max(1, f_ships)))
            travel = speed * float(INBOUND_FLEET_FUTURE_TURNS)
            f_x_fut = f_x_now + math.cos(f_angle) * travel
            f_y_fut = f_y_now + math.sin(f_angle) * travel
            # planet position at the same future turn (orbit if applicable)
            p_x_now = float(raw_planets[slot][2])
            p_y_now = float(raw_planets[slot][3])
            p_x_fut, p_y_fut = _future_position(
                p_x_now, p_y_now, ang_vel, INBOUND_FLEET_FUTURE_TURNS
            )
            dx = (p_x_fut - f_x_fut) / BOARD_SIZE
            dy = (p_y_fut - f_y_fut) / BOARD_SIZE
            dist = math.sqrt(dx * dx + dy * dy)
            inbound_fleet_dx[slot] = dx
            inbound_fleet_dy[slot] = dy
            inbound_fleet_dist[slot] = dist
            inbound_fleet_ships_log[slot] = math.log1p(max(0, f_ships))
    else:
        # No fleets in the scene: inbound trajectory features stay default (no-inbound sentinel)
        inbound_fleet_dx = [0.0] * MAX_PLANETS
        inbound_fleet_dy = [0.0] * MAX_PLANETS
        inbound_fleet_dist = [-1.0] * MAX_PLANETS
        inbound_fleet_ships_log = [0.0] * MAX_PLANETS

    # === Predicted-distance columns: pre-compute future positions and centroids ===
    future_xy: list[tuple[float, float]] = []
    for slot in range(n):
        _, _, px, py, _, _, _ = raw_planets[slot]
        fx, fy = _future_position(float(px), float(py), ang_vel, PREDICT_TURNS)
        future_xy.append((fx, fy))

    my_future_pts: list[tuple[float, float]] = []
    enemy_future_pts: list[tuple[float, float]] = []
    for slot in range(n):
        owner_i = int(raw_planets[slot][1])
        if owner_i == player:
            my_future_pts.append(future_xy[slot])
        elif owner_i != -1:
            enemy_future_pts.append(future_xy[slot])

    if my_future_pts:
        my_cx = sum(p[0] for p in my_future_pts) / len(my_future_pts)
        my_cy = sum(p[1] for p in my_future_pts) / len(my_future_pts)
    else:
        my_cx, my_cy = SUN_X, SUN_Y
    if enemy_future_pts:
        en_cx = sum(p[0] for p in enemy_future_pts) / len(enemy_future_pts)
        en_cy = sum(p[1] for p in enemy_future_pts) / len(enemy_future_pts)
    else:
        en_cx, en_cy = SUN_X, SUN_Y

    # === iter2: production-weighted centroids (current xy, NOT future) ===
    my_prod_x = my_prod_y = 0.0
    my_prod_w = 0.0
    en_prod_x = en_prod_y = 0.0
    en_prod_w = 0.0
    for slot in range(n):
        _, owner_, px_, py_, _, _, prod_ = raw_planets[slot]
        owner_i = int(owner_)
        prod_i = float(prod_)
        if prod_i <= 0:
            continue
        if owner_i == player:
            my_prod_x += float(px_) * prod_i
            my_prod_y += float(py_) * prod_i
            my_prod_w += prod_i
        elif owner_i != -1:
            en_prod_x += float(px_) * prod_i
            en_prod_y += float(py_) * prod_i
            en_prod_w += prod_i
    if my_prod_w > 0:
        my_prod_cx = my_prod_x / my_prod_w
        my_prod_cy = my_prod_y / my_prod_w
    else:
        my_prod_cx, my_prod_cy = SUN_X, SUN_Y
    # iter4: en_prod_centroid + prod_centroid_dist は H5 削除に伴い未使用
    # (en_prod_w / en_prod_x / en_prod_y は computation を残しているが結果は使われない)

    # === iter2 carry-over: home planet pid を per-planet sparse mask J 用に取得 ===
    # iter4: home_planet_owner_flag global は H5 削除済み、home_x/y は home_dist_log 用に必要
    home_x: float | None = None
    home_y: float | None = None
    home_pid: int | None = None
    for hp in raw_initial_planets:
        if int(hp[1]) == player:
            home_pid = int(hp[0])
            home_x = float(hp[2])
            home_y = float(hp[3])
            break

    # iter4: next_comet_eta_norm / comet_active_flag は H4 削除済み
    # `is_comet` flag (per-planet col 8) で代替済み

    # === History (planet-level) — obs_{N-2} / obs_{N-3} only ===
    snap_t1: dict[int, tuple[int, int]] | None = None  # obs_{N-2}
    snap_t2: dict[int, tuple[int, int]] | None = None  # obs_{N-3}
    if history is not None:
        hist_snaps = list(history.prev_planet_snapshots)
        if len(hist_snaps) >= 2:
            snap_t1 = hist_snaps[-2]
        if len(hist_snaps) >= 3:
            snap_t2 = hist_snaps[-3]

    # === Launch event aggregation (per-planet enemy targeting + global counts) ===
    # `prev_fleets_{N-2}` -> `prev_fleets_{N-1}` diff is the launch event for turn N-1
    # (action-after-state for turn N-1, but causally *before* action_N).
    enemy_targeted_count = [0.0] * MAX_PLANETS
    enemy_targeted_ships = [0.0] * MAX_PLANETS
    enemy_launch_count_g = 0.0
    enemy_launch_ships_g = 0.0
    ally_launch_count_g = 0.0
    ally_launch_ships_g = 0.0

    if history is not None and len(history.prev_fleet_snapshots) >= 2:
        fleet_hist = list(history.prev_fleet_snapshots)
        # Collect all launches across the most-recent LAUNCH_HISTORY_WINDOW pairs
        # of (older, newer) snapshots into a single batch, then vectorize the
        # per-planet ETA reverse-resolution.
        all_launches: list[tuple[int, int, float, float, float, int, int]] = []
        pair_count = 0
        for i in range(len(fleet_hist) - 1, 0, -1):
            all_launches.extend(_diff_launches(fleet_hist[i - 1], fleet_hist[i]))
            pair_count += 1
            if pair_count >= LAUNCH_HISTORY_WINDOW:
                break

        if all_launches:
            n_lc = len(all_launches)
            lc_owner = np.zeros(n_lc, dtype=np.int64)
            lc_ships = np.zeros(n_lc, dtype=np.int64)
            lc_x = np.zeros(n_lc, dtype=np.float64)
            lc_y = np.zeros(n_lc, dtype=np.float64)
            lc_a = np.zeros(n_lc, dtype=np.float64)
            for i, fl in enumerate(all_launches):
                _, fowner_, fx_, fy_, fangle_, _, fships_ = fl
                lc_owner[i] = int(fowner_)
                lc_ships[i] = int(fships_)
                lc_x[i] = float(fx_)
                lc_y[i] = float(fy_)
                lc_a[i] = float(fangle_)

            # Global aggregates first (skip neutral)
            non_neutral = lc_owner != -1
            mine_l = (lc_owner == player) & non_neutral
            enemy_l = (~mine_l) & non_neutral
            ally_launch_count_g = float(mine_l.sum())
            ally_launch_ships_g = float(lc_ships[mine_l].sum())
            enemy_launch_count_g = float(enemy_l.sum())
            enemy_launch_ships_g = float(lc_ships[enemy_l].sum())

            # Per-planet attribution: vectorize ETA reverse-resolution for enemy launches only
            enemy_idx = np.where(enemy_l)[0]
            if enemy_idx.size > 0:
                eta_mat_lc = _batch_fleet_target_eta(
                    lc_x[enemy_idx],
                    lc_y[enemy_idx],
                    lc_a[enemy_idx],
                    lc_ships[enemy_idx],
                    planet_x_arr,
                    planet_y_arr,
                    planet_r_arr,
                )  # (E, P), inf for misses
                # For each enemy launch, pick its argmin planet (best ETA)
                best_slot = np.argmin(eta_mat_lc, axis=1)  # (E,)
                best_eta = eta_mat_lc[np.arange(eta_mat_lc.shape[0]), best_slot]
                hit = np.isfinite(best_eta)
                for k in range(enemy_idx.size):
                    if not hit[k]:
                        continue
                    tgt_slot = int(best_slot[k])
                    enemy_targeted_count[tgt_slot] += 1.0
                    enemy_targeted_ships[tgt_slot] += float(lc_ships[enemy_idx[k]])

    # === iter3 A: Pairwise Top-K pre-compute (P x K x 4 feats) ===
    # 各 source slot から見て、最近 K 個の other planet の (dist_log, ships_ratio_log,
    # owner_signed, prod_log) を numpy で一気に集約。
    pair_topk = np.zeros(
        (MAX_PLANETS, TOPK_NEIGHBORS, TOPK_FEATS_PER_NEIGHBOR), dtype=np.float32
    )
    if n >= 2:
        # planet_x_arr / planet_y_arr / planet_owner_arr / planet_ships_arr / planet_prod_arr を構築
        po_arr = np.zeros(n, dtype=np.int64)
        ps_arr = np.zeros(n, dtype=np.float64)
        pp_arr = np.zeros(n, dtype=np.float64)
        for s in range(n):
            po_arr[s] = int(raw_planets[s][1])
            ps_arr[s] = float(raw_planets[s][5])
            pp_arr[s] = float(raw_planets[s][6])
        # planet_x_arr / planet_y_arr は既に上で構築済 (size = n)
        dx_mat = planet_x_arr[None, :] - planet_x_arr[:, None]  # (n, n)
        dy_mat = planet_y_arr[None, :] - planet_y_arr[:, None]
        d2_mat = dx_mat * dx_mat + dy_mat * dy_mat
        # exclude self (set self distance to inf)
        np.fill_diagonal(d2_mat, np.inf)
        # top-K nearest (smallest d2) per source
        kk = min(TOPK_NEIGHBORS, n - 1)
        nearest_idx = np.argpartition(d2_mat, kth=kk - 1, axis=1)[:, :kk]
        # sort each row's k indices by actual distance ascending
        for s in range(n):
            row_d2 = d2_mat[s, nearest_idx[s]]
            order = np.argsort(row_d2)
            sorted_neighbors = nearest_idx[s, order]
            for k_pos in range(kk):
                t = int(sorted_neighbors[k_pos])
                d = float(np.sqrt(d2_mat[s, t]))
                src_ships = ps_arr[s]
                tgt_ships = ps_arr[t]
                tgt_owner = int(po_arr[t])
                # dist_log normalized
                pair_topk[s, k_pos, 0] = float(math.log1p(d) / math.log1p(BOARD_SIZE))
                # ships ratio log: log1p(tgt) - log1p(src)
                pair_topk[s, k_pos, 1] = float(
                    math.log1p(tgt_ships) - math.log1p(src_ships)
                )
                # owner signed: +1 mine, -1 enemy, 0 neutral
                if tgt_owner == player:
                    pair_topk[s, k_pos, 2] = 1.0
                elif tgt_owner == -1:
                    pair_topk[s, k_pos, 2] = 0.0
                else:
                    pair_topk[s, k_pos, 2] = -1.0
                pair_topk[s, k_pos, 3] = float(math.log1p(pp_arr[t]))
            # If kk < TOPK_NEIGHBORS, remaining slots are zero (sentinel: dist_log=0 only when no
            # neighbor; we use dist_log=-1 sentinel for "no slot" to distinguish). Set explicitly:
            for k_pos in range(kk, TOPK_NEIGHBORS):
                pair_topk[s, k_pos, 0] = -1.0  # no-neighbor sentinel
                # other 3 stay 0
    else:
        # n < 2: no neighbors at all, set sentinel for slot 0
        for k_pos in range(TOPK_NEIGHBORS):
            for s in range(MAX_PLANETS):
                pair_topk[s, k_pos, 0] = -1.0

    # iter6: K2 (outgoing_fleet_trajectory) は causal leak で削除済 (planet 53-56 列を抹消)。
    # 自軍 fleet (= action_N の結果) を input に入れていたため ships_per_src ラベルと
    # 完全に同情報源、policy が逆算可能だった (iter5 permutation で sum |Δ|=1.70 検出)。

    # === iter4 K3: Frontline distance (敵軍 nearest 1, 2 個との dist_log + ships_ratio_log)
    # source planet の視点で敵軍 (owner != player and != -1) の nearest 2 を抽出。
    # 不足 (敵 0 or 1 個) は sentinel: dist_log=-1, ships_ratio=0
    frontline_d1_log = [-1.0] * MAX_PLANETS
    frontline_r1_log = [0.0] * MAX_PLANETS
    frontline_d2_log = [-1.0] * MAX_PLANETS
    frontline_r2_log = [0.0] * MAX_PLANETS
    if n >= 2:
        # collect enemy slot indices once
        enemy_slots = [
            s
            for s in range(n)
            if int(raw_planets[s][1]) != player and int(raw_planets[s][1]) != -1
        ]
        if enemy_slots:
            for src in range(n):
                src_x = float(raw_planets[src][2])
                src_y = float(raw_planets[src][3])
                src_ships = float(raw_planets[src][5])
                # compute distances to all enemies, sort
                ed_pairs = []
                for es in enemy_slots:
                    if es == src:
                        continue
                    ex = float(raw_planets[es][2])
                    ey = float(raw_planets[es][3])
                    eships = float(raw_planets[es][5])
                    d = math.sqrt((ex - src_x) ** 2 + (ey - src_y) ** 2)
                    ed_pairs.append((d, eships))
                if not ed_pairs:
                    continue
                ed_pairs.sort(key=lambda t: t[0])
                d1, e1 = ed_pairs[0]
                frontline_d1_log[src] = math.log1p(d1) / math.log1p(BOARD_SIZE)
                frontline_r1_log[src] = math.log1p(e1) - math.log1p(src_ships)
                if len(ed_pairs) >= 2:
                    d2, e2 = ed_pairs[1]
                    frontline_d2_log[src] = math.log1p(d2) / math.log1p(BOARD_SIZE)
                    frontline_r2_log[src] = math.log1p(e2) - math.log1p(src_ships)

    # === Per-planet feature assembly ===
    for slot in range(n):
        pid, owner, px, py, radius, ships, production = raw_planets[slot]
        owner_i = int(owner)
        ships_i = int(ships)
        production_i = int(production)
        is_mine = owner_i == player
        is_neutral = owner_i == -1
        is_enemy = (not is_mine) and (not is_neutral)
        is_comet = int(pid) in raw_comet_ids

        eta_norm = min(nearest_eta[slot], HORIZON_TURNS + 1.0) / (HORIZON_TURNS + 1.0)

        tp = TimelinePlanet(
            id=int(pid), owner=owner_i, ships=ships_i, production=production_i
        )
        timeline = simulate_planet_timeline(
            tp, arrivals_by_slot[slot], player, horizon=TIMELINE_HORIZON
        )
        ts = summarize_timeline(timeline)
        # iter2: short/mid horizon summary from the same simulate (no extra cost)
        ts_multi = summarize_timeline_multi_horizon(timeline, horizons=(5, 15))

        # Predicted distance to centroids (future position)
        fx_p, fy_p = future_xy[slot]
        fut_dist_my = math.sqrt((fx_p - my_cx) ** 2 + (fy_p - my_cy) ** 2) / BOARD_SIZE
        fut_dist_enemy = (
            math.sqrt((fx_p - en_cx) ** 2 + (fy_p - en_cy) ** 2) / BOARD_SIZE
        )

        # History columns (delta_ships / owner_changed)
        delta_t1 = 0.0
        delta_t2 = 0.0
        owner_changed_t1 = 0.0
        if snap_t1 is not None and int(pid) in snap_t1:
            past_owner_t1, past_ships_t1 = snap_t1[int(pid)]
            denom = max(1, ships_i)
            delta_t1 = max(-1.0, min(1.0, (ships_i - past_ships_t1) / denom))
            if past_owner_t1 != owner_i:
                owner_changed_t1 = 1.0
        if snap_t2 is not None and int(pid) in snap_t2:
            _, past_ships_t2 = snap_t2[int(pid)]
            denom2 = max(1, ships_i)
            delta_t2 = max(-1.0, min(1.0, (ships_i - past_ships_t2) / denom2))

        # iter2: home distance + production-centroid distance per planet
        if home_x is not None and home_y is not None:
            home_dist_log = math.log1p(
                math.sqrt((float(px) - home_x) ** 2 + (float(py) - home_y) ** 2)
            ) / math.log1p(BOARD_SIZE)
        else:
            home_dist_log = 0.0
        prod_centroid_dist_log = math.log1p(
            math.sqrt((float(px) - my_prod_cx) ** 2 + (float(py) - my_prod_cy) ** 2)
        ) / math.log1p(BOARD_SIZE)

        # iter3 D: Defense surplus (per-planet 4 列)
        # 自軍 ships - 敵 incoming ships の予測値を h=5/15/30 + now
        # 自軍以外の planet では surplus は 0 で意味薄だが、neutral/enemy でも
        # incoming は意味がある (奪取後の defense margin) ので算出は同じ
        own_ships_now = float(ships_i) if is_mine else 0.0
        enemy_inbound = float(incoming[slot][1])
        prod_per_turn = float(production_i) if is_mine else 0.0
        margin_now = (own_ships_now - enemy_inbound) / 100.0
        margin_5 = (own_ships_now + 5.0 * prod_per_turn - enemy_inbound) / 100.0
        margin_15 = (own_ships_now + 15.0 * prod_per_turn - enemy_inbound) / 100.0
        margin_30 = (own_ships_now + 30.0 * prod_per_turn - enemy_inbound) / 100.0
        defense_margin_now = max(-1.0, min(1.0, margin_now))
        defense_surplus_5turn = max(-1.0, min(1.0, margin_5))
        defense_surplus_15turn = max(-1.0, min(1.0, margin_15))
        defense_surplus_30turn = max(-1.0, min(1.0, margin_30))

        # iter3 J: Sparse mask flags (per-planet 5 列)
        has_inbound_fleet_flag = 1.0 if inbound_fleet_dist[slot] != -1.0 else 0.0
        has_history_flag = 1.0 if snap_t1 is not None else 0.0
        has_enemy_targeted_flag = 1.0 if enemy_targeted_count[slot] > 0 else 0.0
        is_home_planet_flag = (
            1.0 if home_pid is not None and int(pid) == home_pid else 0.0
        )
        # is_neighbor_to_enemy: 半径 BOARD_SIZE/4 以内に敵 planet があるか
        is_neighbor_to_enemy_flag = 0.0
        nbr_radius = BOARD_SIZE / 4.0
        for s2 in range(n):
            if s2 == slot:
                continue
            other_owner = int(raw_planets[s2][1])
            if other_owner == player or other_owner == -1:
                continue
            dx2 = float(raw_planets[s2][2]) - float(px)
            dy2 = float(raw_planets[s2][3]) - float(py)
            if dx2 * dx2 + dy2 * dy2 <= nbr_radius * nbr_radius:
                is_neighbor_to_enemy_flag = 1.0
                break

        # iter3 A: Pairwise Top-K (5 neighbors × 4 feats = 20 列、flatten)
        topk_flat = pair_topk[slot].reshape(-1).tolist()

        feats = [
            float(px) / BOARD_SIZE,
            float(py) / BOARD_SIZE,
            float(radius) / 5.0,
            math.log1p(max(0, ships_i)),
            math.log1p(max(0, production_i)),
            1.0 if is_mine else 0.0,
            1.0 if is_enemy else 0.0,
            1.0 if is_neutral else 0.0,
            1.0 if is_comet else 0.0,
            math.log1p(incoming[slot][1]) - math.log1p(incoming[slot][0]),
            eta_norm,
            # ship-prediction 6 列 (case5)
            math.log1p(ts["loss_3turn"]),
            ts["ttf_norm"],
            math.log1p(ts["min_owned"]),
            math.log1p(ts["surplus"]),
            ts["fall_predicted"],
            math.log1p(ts["keep_needed"]),
            # case7 iter1: predicted-distance 2 列
            fut_dist_my,
            fut_dist_enemy,
            # case7 iter1: history 3 列 (obs_{N-2} / obs_{N-3})
            delta_t1,
            delta_t2,
            owner_changed_t1,
            # case7 iter4: G6 (enemy ship-event per-planet) 削除済 ← 元 idx 22-23
            # case7 iter2: fleet trajectory 4 列 (earliest enemy inbound, 5-turn future delta)
            inbound_fleet_dx[slot],
            inbound_fleet_dy[slot],
            inbound_fleet_dist[slot],
            inbound_fleet_ships_log[slot],
            # case7 iter2: multi-horizon ship-prediction 4 列 (h=5/15)
            math.log1p(ts_multi["loss_5turn"]),
            math.log1p(ts_multi["loss_15turn"]),
            math.log1p(ts_multi["min_owned_5turn"]),
            math.log1p(ts_multi["min_owned_15turn"]),
            # case7 iter2: home distance + production centroid distance 2 列
            home_dist_log,
            prod_centroid_dist_log,
            # case7 iter3 A (iter4 K1 で縮小): Pairwise Top-K (K=3 × 4 feat = 12 列)
            *topk_flat,
            # case7 iter3 D: Defense surplus (4 列、h=5/15/30 + now)
            defense_surplus_5turn,
            defense_surplus_15turn,
            defense_surplus_30turn,
            defense_margin_now,
            # case7 iter3 J: Sparse mask flags (5 列)
            has_inbound_fleet_flag,
            has_history_flag,
            has_enemy_targeted_flag,
            is_home_planet_flag,
            is_neighbor_to_enemy_flag,
            # iter6: K2 (outgoing fleet trajectory) は causal leak で削除済
            # case7 iter4 K3: Frontline distance (4 列、敵 nearest 1/2)
            frontline_d1_log[slot],
            frontline_r1_log[slot],
            frontline_d2_log[slot],
            frontline_r2_log[slot],
        ]
        for j in range(PLANET_FEAT_DIM):
            planet_feats[slot, j] = feats[j]
        planet_mask[slot] = True
        if is_mine:
            my_planet_mask[slot] = True
            my_planet_ids.append(int(pid))
            ctx = template_context_features(
                list(raw_planets[slot]), raw_planets, player, BOARD_SIZE
            )
            for j in range(TEMPLATE_CTX_DIM):
                template_ctx[slot, j] = ctx[j]
        if not is_mine:
            target_mask[slot] = True
        planet_ids.append(int(pid))

    # === Global features ===
    my_total_ships = 0.0
    enemy_total_ships = 0.0
    neutral_total_ships = 0.0
    my_total_prod = 0.0
    enemy_total_prod = 0.0
    for slot in range(n):
        _, owner, _, _, _, ships, production = raw_planets[slot]
        owner_i = int(owner)
        if owner_i == player:
            my_total_ships += float(ships)
            my_total_prod += float(production)
        elif owner_i == -1:
            neutral_total_ships += float(ships)
        else:
            enemy_total_ships += float(ships)
            enemy_total_prod += float(production)

    # iter6: K4 aux_multi_horizon_global は permutation importance 0.001 で完全 dead、削除済
    global_feats = torch.tensor(
        [
            # iter1 base 4 列 (H1 step/ang_vel は iter4 で削除)
            math.log1p(my_total_ships),
            math.log1p(enemy_total_ships),
            math.log1p(neutral_total_ships),
            math.log1p(my_total_prod) - math.log1p(enemy_total_prod),
            # case7 iter1: enemy/ally launch history 4 列 (H3)
            enemy_launch_count_g / 10.0,
            math.log1p(enemy_launch_ships_g) / 6.0,
            ally_launch_count_g / 10.0,
            math.log1p(ally_launch_ships_g) / 6.0,
            # iter6: K4 (aux_multi_horizon_global) 削除済
            # iter4: H1/H4/H5 削除済
        ],
        dtype=torch.float32,
    )

    batch = BatchFeatures(
        planet_feats=planet_feats.unsqueeze(0),
        planet_mask=planet_mask.unsqueeze(0),
        my_planet_mask=my_planet_mask.unsqueeze(0),
        target_mask=target_mask.unsqueeze(0),
        global_feats=global_feats.unsqueeze(0),
        template_ctx=template_ctx.unsqueeze(0),
    )
    snapshot = WorldSnapshot(
        planet_ids=tuple(planet_ids),
        my_planet_ids=tuple(my_planet_ids),
        player=player,
        step=step,
    )
    return batch, snapshot


def planet_snapshot_from_obs(obs: Any) -> dict[int, tuple[int, int]]:
    """Build {planet_id: (owner, ships)} for HistoryState.push()."""
    raw_planets = list(_read(obs, "planets", []) or [])
    out: dict[int, tuple[int, int]] = {}
    for row in raw_planets:
        pid, owner, _, _, _, ships, _ = row
        out[int(pid)] = (int(owner), int(ships))
    return out


def fleet_snapshot_from_obs(
    obs: Any,
) -> list[tuple[int, int, float, float, float, int, int]]:
    """Build a homogeneous list of fleet rows for HistoryState.push()."""
    raw_fleets = list(_read(obs, "fleets", []) or [])
    out: list[tuple[int, int, float, float, float, int, int]] = []
    for row in raw_fleets:
        fid, fowner, fx, fy, fangle, from_pid, fships = row
        out.append(
            (
                int(fid),
                int(fowner),
                float(fx),
                float(fy),
                float(fangle),
                int(from_pid),
                int(fships),
            )
        )
    return out
