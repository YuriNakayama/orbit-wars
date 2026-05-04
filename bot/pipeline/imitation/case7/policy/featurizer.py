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

import torch

from .templates import TEMPLATE_CTX_DIM, template_context_features
from .timeline import (
    DEFAULT_HORIZON as TIMELINE_HORIZON,
)
from .timeline import (
    TimelinePlanet,
    simulate_planet_timeline,
    summarize_timeline,
)
from .types import BatchFeatures, WorldSnapshot

PLANET_FEAT_DIM = 24  # case5 17 + 7 (predicted dist 2 + history 3 + enemy ship event 2)
GLOBAL_FEAT_DIM = 10  # case5 6 + 4 (enemy/ally launch count/ships last4)
MAX_PLANETS = 36
BOARD_SIZE = 100.0
HORIZON_TURNS = 30  # for incoming-fleet eta normalization
PREDICT_TURNS = 5  # future-position prediction horizon (orbit rotation)
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
    for slot in range(n):
        pid = int(raw_planets[slot][0])
        planet_index_by_id[pid] = slot

    for fleet_row in raw_fleets:
        fid, fowner, fx, fy, fangle, _from_pid, fships = fleet_row
        f_owner = int(fowner)
        f_ships = int(fships)
        f_x = float(fx)
        f_y = float(fy)
        f_angle = float(fangle)
        for slot in range(n):
            pid_, _, px, py, pradius, _, _ = raw_planets[slot]
            eta = _fleet_target_eta(
                f_x, f_y, f_angle, f_ships, float(px), float(py), float(pradius)
            )
            if eta is None or eta > HORIZON_TURNS:
                continue
            if f_owner == player:
                incoming[slot][0] += f_ships
            elif f_owner == -1:
                incoming[slot][2] += f_ships
            else:
                incoming[slot][1] += f_ships
            if eta < nearest_eta[slot]:
                nearest_eta[slot] = eta
            arrivals_by_slot[slot].append((eta, f_owner, f_ships))

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
        # Walk the most-recent LAUNCH_HISTORY_WINDOW pairs of (older, newer) snapshots.
        # Each pair contributes the launches between those two turns.
        # We use snapshots up to index -1 (prev_fleets_{N-1}); skip the latest pair if
        # we ever push current obs into history (we don't — agent pushes *after* featurize).
        pairs = []
        # Iterate from newest backwards; collect up to LAUNCH_HISTORY_WINDOW pairs.
        for i in range(len(fleet_hist) - 1, 0, -1):
            pairs.append((fleet_hist[i - 1], fleet_hist[i]))
            if len(pairs) >= LAUNCH_HISTORY_WINDOW:
                break
        for snap_a, snap_b in pairs:
            launches = _diff_launches(snap_a, snap_b)
            for fl in launches:
                fid_, fowner_, fx_, fy_, fangle_, _from_pid_, fships_ = fl
                fowner_i = int(fowner_)
                fships_i = int(fships_)
                if fowner_i == -1:
                    continue
                if fowner_i == player:
                    ally_launch_count_g += 1.0
                    ally_launch_ships_g += float(fships_i)
                else:
                    enemy_launch_count_g += 1.0
                    enemy_launch_ships_g += float(fships_i)
                    # Per-planet attribution (only enemy launches targeting my planets)
                    tgt_pid = _fleet_action_target(
                        float(fx_), float(fy_), float(fangle_), fships_i, raw_planets
                    )
                    if tgt_pid is None:
                        continue
                    tgt_slot = planet_index_by_id.get(tgt_pid)
                    if tgt_slot is None:
                        continue
                    enemy_targeted_count[tgt_slot] += 1.0
                    enemy_targeted_ships[tgt_slot] += float(fships_i)

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
            # case7 新規: predicted-distance 2 列
            fut_dist_my,
            fut_dist_enemy,
            # case7 新規: history 3 列 (obs_{N-2} / obs_{N-3})
            delta_t1,
            delta_t2,
            owner_changed_t1,
            # case7 新規: enemy ship-event 2 列 (per-planet)
            enemy_targeted_count[slot] / 5.0,
            math.log1p(enemy_targeted_ships[slot]) / 6.0,
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

    global_feats = torch.tensor(
        [
            float(step) / 500.0,
            ang_vel * 10.0,
            math.log1p(my_total_ships),
            math.log1p(enemy_total_ships),
            math.log1p(neutral_total_ships),
            math.log1p(my_total_prod) - math.log1p(enemy_total_prod),
            # case7 新規: enemy/ally launch history 4 列
            enemy_launch_count_g / 10.0,
            math.log1p(enemy_launch_ships_g) / 6.0,
            ally_launch_count_g / 10.0,
            math.log1p(ally_launch_ships_g) / 6.0,
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
