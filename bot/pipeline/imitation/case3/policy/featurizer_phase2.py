"""Phase 2 featurizer for imitation/case3.

Builds on Phase 1 final (case2 phase1 with group D = threat_pressure_long
removed) and adds short-term history features that require state outside the
single observation:

- Mapping #7 (per-planet ships history, K=2): planet cols
    delta_ships_t1, delta_ships_t2, owner_changed_t1
- Mapping #8 (per-frame launch history, last 4 turns): global cols
    enemy_launch_count_last4, enemy_launch_ships_last4,
    ally_launch_count_last4, ally_launch_ships_last4

History is supplied via `HistoryState`, owned by either preprocess (per
episode) or the inference agent (per match). When history is empty (first
turn), the new columns are zeroed.

Resulting dims:
  PLANET_FEAT_DIM = 35  (Phase 1 final 32 + 3 history)
  GLOBAL_FEAT_DIM = 20  (Phase 1 final 16 + 4 history)

Phase 1 column layout (cols 0..31, group D dropped):
  0..17  : same as case2 baseline featurizer
  18..19 : nearest_ally/neutral_dist                 (group A)
  20..27 : orbit prediction (dx, dy) at t={1, 2, 4, 8} (group B)
  28..31 : ally/enemy ETA & ships split              (group C)
  -- group D (threat_pressure_long) intentionally omitted --
Phase 2 additions (delays history by 1 turn to avoid action_N leakage —
see HistoryState docstring):
  32 : delta_ships_t1   = (ships_now - ships_{t-2}) / max(1, ships_now), clipped
  33 : delta_ships_t2   = (ships_now - ships_{t-3}) / max(1, ships_now), clipped
  34 : owner_changed_t1 = 1 if owner differed at t-2, else 0

Global cols 0..15 are the case2 phase1 16 dims (D was planet-only);
Phase 2 additions:
  16 : enemy_launch_count_last4 / 10
  17 : log1p(enemy_launch_ships_last4) / 6
  18 : ally_launch_count_last4 / 10
  19 : log1p(ally_launch_ships_last4) / 6
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from .geometry import (
    CENTER_X,
    CENTER_Y,
    Planet,
    predict_comet_position,
    predict_planet_position,
)
from .templates import TEMPLATE_CTX_DIM, template_context_features
from .types import BatchFeatures, WorldSnapshot

PLANET_FEAT_DIM = 35
GLOBAL_FEAT_DIM = 20
MAX_PLANETS = 36
BOARD_SIZE = 100.0
HORIZON_TURNS = 30
SUN_X = CENTER_X
SUN_Y = CENTER_Y
ROTATION_LIMIT = 50.0
DIAG = math.sqrt(2.0) * BOARD_SIZE
NEIGHBOR_RADIUS_SHORT = 8.0
NEIGHBOR_RADIUS_LONG = 25.0
COMET_WAVES = (50, 150, 250, 350, 450)
COMET_WINDOW = 30
ORBIT_HORIZONS = (1, 2, 4, 8)
LOG_NORM_DENOM = 6.0
HISTORY_TURNS = 4  # window for launch counts
LAUNCH_COUNT_NORM = 10.0


@dataclass
class PlanetSnapshot:
    """Compact per-planet snapshot retained across turns."""

    ships_by_id: dict[int, int]
    owner_by_id: dict[int, int]


@dataclass
class LaunchEvent:
    step: int
    owner: int
    ships: int


@dataclass
class HistoryState:
    """Mutable history buffer shared between preprocess and inference.

    `prev_planet_snapshots` keeps up to 3 prior turns (oldest first).
    The featurizer reads `[-2]` and `[-3]`, NOT `[-1]`, because
    `step[N].observation` in Kaggle replays reflects post-action-of-step-N
    state. Using `[-1]` (= obs of step N-1, which is post-action of N-1)
    is fine on its own, but combined with current obs (step N, post-action
    of N) the diff `obs_N - obs_{N-1}` directly contains action_N's ship
    count — a label leak. So we reach 2 turns back.

    `recent_launches` keeps launch events for the last HISTORY_TURNS turns.
    Reset (`clear`) is called between matches by the agent.
    """

    prev_planet_snapshots: deque[PlanetSnapshot] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    recent_launches: deque[LaunchEvent] = field(
        default_factory=lambda: deque(maxlen=200)
    )
    last_step: int | None = None

    def clear(self) -> None:
        self.prev_planet_snapshots.clear()
        self.recent_launches.clear()
        self.last_step = None


def _read(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships: int) -> float:
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


def _comet_active(step: int) -> bool:
    return any(0 <= step - w < COMET_WINDOW for w in COMET_WAVES)


def _next_comet_eta(step: int) -> int:
    if _comet_active(step):
        return 0
    upcoming = [w - step for w in COMET_WAVES if w >= step]
    if not upcoming:
        return 100
    return min(upcoming)


def _build_planet_obj(row: list[Any]) -> Planet:
    return Planet(
        id=int(row[0]),
        owner=int(row[1]),
        x=float(row[2]),
        y=float(row[3]),
        radius=float(row[4]),
        ships=int(row[5]),
        production=int(row[6]),
    )


def _orbit_predictions(
    planet: Planet,
    initial_by_id: dict[int, Planet],
    angular_velocity: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for turns in ORBIT_HORIZONS:
        if planet.id in comet_ids:
            pos = predict_comet_position(planet.id, comets, turns)
            if pos is None:
                out.append((0.0, 0.0))
                continue
            nx, ny = pos
        else:
            nx, ny = predict_planet_position(
                planet, initial_by_id, angular_velocity, turns
            )
        out.append(
            (
                (nx - planet.x) / BOARD_SIZE,
                (ny - planet.y) / BOARD_SIZE,
            )
        )
    return out


def featurize(
    obs: Any, history: HistoryState | None = None
) -> tuple[BatchFeatures, WorldSnapshot]:
    """Convert obs (+ optional history) to a BatchFeatures of batch_size=1.

    When `history` is None or empty, history columns (planet 32-34, global
    16-19) are zero-filled.
    """
    if history is None:
        history = HistoryState()

    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = list(_read(obs, "planets", []) or [])
    raw_fleets = list(_read(obs, "fleets", []) or [])
    raw_comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    raw_comets = list(_read(obs, "comets", []) or [])
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

    initial_by_id: dict[int, Planet] = {
        int(row[0]): _build_planet_obj(list(row)) for row in raw_initial_planets
    }

    incoming_ally_ships = [0.0] * MAX_PLANETS
    incoming_enemy_ships = [0.0] * MAX_PLANETS
    incoming_neutral_ships = [0.0] * MAX_PLANETS
    nearest_eta = [HORIZON_TURNS + 1.0] * MAX_PLANETS
    incoming_ally_eta_min = [HORIZON_TURNS + 1.0] * MAX_PLANETS
    incoming_enemy_eta_min = [HORIZON_TURNS + 1.0] * MAX_PLANETS

    for fleet_row in raw_fleets:
        _, fowner, fx, fy, fangle, _from_pid, fships = fleet_row
        f_owner = int(fowner)
        f_ships = int(fships)
        f_x = float(fx)
        f_y = float(fy)
        f_angle = float(fangle)
        for slot in range(n):
            _, _, px, py, pradius, _, _ = raw_planets[slot]
            eta = _fleet_target_eta(
                f_x, f_y, f_angle, f_ships, float(px), float(py), float(pradius)
            )
            if eta is None or eta > HORIZON_TURNS:
                continue
            if f_owner == player:
                incoming_ally_ships[slot] += f_ships
                if eta < incoming_ally_eta_min[slot]:
                    incoming_ally_eta_min[slot] = eta
            elif f_owner == -1:
                incoming_neutral_ships[slot] += f_ships
            else:
                incoming_enemy_ships[slot] += f_ships
                if eta < incoming_enemy_eta_min[slot]:
                    incoming_enemy_eta_min[slot] = eta
            if eta < nearest_eta[slot]:
                nearest_eta[slot] = eta

    nearest_ally_dist = [DIAG] * MAX_PLANETS
    nearest_enemy_dist = [DIAG] * MAX_PLANETS
    nearest_neutral_dist = [DIAG] * MAX_PLANETS
    support_density = [0.0] * MAX_PLANETS
    threat_pressure_short = [0.0] * MAX_PLANETS
    for i in range(n):
        _, _, px_i, py_i, _, _, _ = raw_planets[i]
        for j in range(n):
            if i == j:
                continue
            _, owner_j, px_j, py_j, _, ships_j, _ = raw_planets[j]
            owner_jj = int(owner_j)
            d = math.hypot(float(px_i) - float(px_j), float(py_i) - float(py_j))
            if owner_jj == player:
                if d < nearest_ally_dist[i]:
                    nearest_ally_dist[i] = d
                if d <= NEIGHBOR_RADIUS_LONG:
                    support_density[i] += float(ships_j)
            elif owner_jj == -1:
                if d < nearest_neutral_dist[i]:
                    nearest_neutral_dist[i] = d
            else:
                if d < nearest_enemy_dist[i]:
                    nearest_enemy_dist[i] = d
                if d <= NEIGHBOR_RADIUS_SHORT:
                    threat_pressure_short[i] += float(ships_j)
        for fleet_row in raw_fleets:
            _, fowner, fx, fy, _, _, fships = fleet_row
            f_owner = int(fowner)
            if f_owner == player or f_owner == -1:
                continue
            d = math.hypot(float(px_i) - float(fx), float(py_i) - float(fy))
            if d <= NEIGHBOR_RADIUS_SHORT:
                threat_pressure_short[i] += float(fships)

    my_count = 0
    enemy_count = 0
    neutral_count = 0

    # NOTE on leakage: `step[N].observation.ships` in Kaggle replays is
    # *post-action* of step N. If we used the most recent snapshot ([-1] =
    # obs_{N-1} = post-action of N-1) and diffed with current obs (post-action
    # of N), the difference would directly encode action_N's ship sent count
    # — a label leak. We reach 2 turns back instead so deltas are
    # (post-action of N) - (post-action of N-2), which does not include
    # action_N (only actions_{N-1} and earlier).
    snap_t1 = (
        history.prev_planet_snapshots[-2]
        if len(history.prev_planet_snapshots) >= 2
        else None
    )
    snap_t2 = (
        history.prev_planet_snapshots[-3]
        if len(history.prev_planet_snapshots) >= 3
        else None
    )

    for slot in range(n):
        pid, owner, px, py, radius, ships, production = raw_planets[slot]
        owner_i = int(owner)
        ships_i = int(ships)
        production_i = int(production)
        is_mine = owner_i == player
        is_neutral = owner_i == -1
        is_enemy = (not is_mine) and (not is_neutral)
        is_comet = int(pid) in raw_comet_ids
        if is_mine:
            my_count += 1
        elif is_neutral:
            neutral_count += 1
        else:
            enemy_count += 1

        eta_norm = min(nearest_eta[slot], HORIZON_TURNS + 1.0) / (HORIZON_TURNS + 1.0)
        sun_dist = math.hypot(float(px) - SUN_X, float(py) - SUN_Y)
        is_static = 1.0 if (sun_dist + float(radius) >= ROTATION_LIMIT) else 0.0
        prod_per_ship = float(production_i) / float(max(1, ships_i))
        net_signed = (incoming_enemy_ships[slot] - incoming_ally_ships[slot]) / float(
            max(1, ships_i)
        )

        planet_obj = _build_planet_obj(list(raw_planets[slot]))
        orbit = _orbit_predictions(
            planet_obj, initial_by_id, ang_vel, raw_comets, raw_comet_ids
        )
        ally_eta_norm = min(incoming_ally_eta_min[slot], HORIZON_TURNS + 1.0) / (
            HORIZON_TURNS + 1.0
        )
        enemy_eta_norm = min(incoming_enemy_eta_min[slot], HORIZON_TURNS + 1.0) / (
            HORIZON_TURNS + 1.0
        )

        # History columns
        delta_t1 = 0.0
        delta_t2 = 0.0
        owner_changed = 0.0
        if snap_t1 is not None and int(pid) in snap_t1.ships_by_id:
            prev_ships = snap_t1.ships_by_id[int(pid)]
            delta_t1 = float(ships_i - prev_ships) / float(max(1, ships_i))
            delta_t1 = max(-3.0, min(3.0, delta_t1)) / 3.0
            if int(pid) in snap_t1.owner_by_id:
                owner_changed = 1.0 if snap_t1.owner_by_id[int(pid)] != owner_i else 0.0
        if snap_t2 is not None and int(pid) in snap_t2.ships_by_id:
            prev_ships = snap_t2.ships_by_id[int(pid)]
            delta_t2 = float(ships_i - prev_ships) / float(max(1, ships_i))
            delta_t2 = max(-3.0, min(3.0, delta_t2)) / 3.0

        feats: list[float] = [
            float(px) / BOARD_SIZE,
            float(py) / BOARD_SIZE,
            float(radius) / 5.0,
            math.log1p(max(0, ships_i)),
            math.log1p(max(0, production_i)),
            1.0 if is_mine else 0.0,
            1.0 if is_enemy else 0.0,
            1.0 if is_neutral else 0.0,
            1.0 if is_comet else 0.0,
            math.log1p(incoming_enemy_ships[slot])
            - math.log1p(incoming_ally_ships[slot]),
            eta_norm,
            sun_dist / DIAG,
            is_static,
            min(prod_per_ship, 5.0) / 5.0,
            nearest_enemy_dist[slot] / DIAG,
            math.log1p(support_density[slot]) / LOG_NORM_DENOM,
            math.log1p(threat_pressure_short[slot]) / LOG_NORM_DENOM,
            max(-3.0, min(3.0, net_signed)) / 3.0,
            # ----- Phase 1 retained (groups A, B, C) -----
            nearest_ally_dist[slot] / DIAG,
            nearest_neutral_dist[slot] / DIAG,
            orbit[0][0],
            orbit[0][1],
            orbit[1][0],
            orbit[1][1],
            orbit[2][0],
            orbit[2][1],
            orbit[3][0],
            orbit[3][1],
            ally_eta_norm,
            enemy_eta_norm,
            math.log1p(incoming_ally_ships[slot]) / LOG_NORM_DENOM,
            math.log1p(incoming_enemy_ships[slot]) / LOG_NORM_DENOM,
            # ----- Phase 2 history -----
            delta_t1,
            delta_t2,
            owner_changed,
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

    total_planets = max(1, n)
    total_ships = my_total_ships + enemy_total_ships + neutral_total_ships
    total_prod = my_total_prod + enemy_total_prod
    phase_mid = 1.0 if 100 <= step < 300 else 0.0
    phase_late = 1.0 if step >= 300 else 0.0
    score_diff = math.log1p(my_total_ships) - math.log1p(enemy_total_ships)
    next_comet_eta_norm = float(_next_comet_eta(step)) / 100.0

    # Phase 2 launch history globals (last HISTORY_TURNS turns)
    enemy_launch_count = 0
    enemy_launch_ships = 0
    ally_launch_count = 0
    ally_launch_ships = 0
    threshold = step - HISTORY_TURNS
    for ev in history.recent_launches:
        if ev.step < threshold:
            continue
        if ev.owner == player:
            ally_launch_count += 1
            ally_launch_ships += ev.ships
        elif ev.owner != -1:
            enemy_launch_count += 1
            enemy_launch_ships += ev.ships

    global_feats = torch.tensor(
        [
            float(step) / 500.0,
            ang_vel * 10.0,
            math.log1p(my_total_ships),
            math.log1p(enemy_total_ships),
            math.log1p(neutral_total_ships),
            math.log1p(my_total_prod) - math.log1p(enemy_total_prod),
            float(my_count) / float(total_planets),
            float(enemy_count) / float(total_planets),
            1.0 if _comet_active(step) else 0.0,
            phase_mid,
            phase_late,
            min(1.0, next_comet_eta_norm),
            (my_total_ships / total_ships) if total_ships > 0 else 0.0,
            (enemy_total_ships / total_ships) if total_ships > 0 else 0.0,
            (my_total_prod / total_prod) if total_prod > 0 else 0.0,
            max(-3.0, min(3.0, score_diff)) / 3.0,
            # ----- Phase 2 launch history -----
            min(1.0, float(enemy_launch_count) / LAUNCH_COUNT_NORM),
            math.log1p(enemy_launch_ships) / LOG_NORM_DENOM,
            min(1.0, float(ally_launch_count) / LAUNCH_COUNT_NORM),
            math.log1p(ally_launch_ships) / LOG_NORM_DENOM,
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


def update_history(
    history: HistoryState,
    obs: Any,
    actions: list[list[Any]] | None = None,
) -> None:
    """Append the current obs's planet snapshot and any launch events.

    Call AFTER `featurize(obs, history)` so that `history` reflects state
    strictly prior to the current obs when fed in.
    """
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = list(_read(obs, "planets", []) or [])
    ships_by_id: dict[int, int] = {}
    owner_by_id: dict[int, int] = {}
    for row in raw_planets:
        ships_by_id[int(row[0])] = int(row[5])
        owner_by_id[int(row[0])] = int(row[1])
    history.prev_planet_snapshots.append(
        PlanetSnapshot(ships_by_id=ships_by_id, owner_by_id=owner_by_id)
    )
    history.last_step = step
    if actions:
        player = int(_read(obs, "player", 0) or 0)
        for act in actions:
            try:
                ships = int(act[2])
            except (IndexError, TypeError, ValueError):
                continue
            history.recent_launches.append(
                LaunchEvent(step=step, owner=player, ships=ships)
            )
    # Drop launches older than HISTORY_TURNS to keep the deque bounded.
    cutoff = step - HISTORY_TURNS
    while history.recent_launches and history.recent_launches[0].step < cutoff:
        history.recent_launches.popleft()
