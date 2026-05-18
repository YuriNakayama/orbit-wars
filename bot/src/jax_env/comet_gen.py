"""Byte-equal port of vendor `generate_comet_paths` + reset-time precomputation.

The vendor sim performs rejection sampling at runtime (every comet spawn step).
We pre-compute all 5 comets at reset time using the deterministic seed scheme
the vendor uses (``f"orbit_wars-comet-{episode_seed}-{step+1}"``), so step
remains pure JAX.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    MAX_COMETS,
    MAX_COMET_PATH_LEN,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_comet_paths(
    initial_planets: list[list[float | int]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: list[int] | None = None,
    comet_speed: float = COMET_SPEED,
    rng: random.Random | None = None,
) -> list[list[list[float]]] | None:
    """Mirror vendor `generate_comet_paths`. Returns 4 paths or None on failure."""
    if rng is None:
        rng = random.Random()
    comet_pid_set: set[int] = set(comet_planet_ids) if comet_planet_ids else set()

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense: list[tuple[float, float]] = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path: list[tuple[float, float]] = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start: int | None = None
        board_end: int | None = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        assert board_end is not None
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths: list[list[list[float]]] = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets: list[list[float | int]] = []
        orbiting_planets: list[list[float | int]] = []
        for planet in initial_planets:
            if planet[0] in comet_pid_set:
                continue
            pr = _distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def precompute_all_comets(
    initial_planets: list[list[float | int]],
    angular_velocity: float,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute every comet (spawn_step, paths, initial_ships) for the episode.

    Replicates vendor `interpreter` rows 422-462 RNG scheme: a fresh
    ``random.Random(f"orbit_wars-comet-{episode_seed}-{step+1}")`` per spawn.

    Vendor sim grows ``comet_planet_ids`` as comets spawn — only comets that
    spawned earlier are excluded from the next comet's overlap check. We must
    replicate that across spawns. Since the comet planet ids depend on the
    current ``max(p[0])`` in ``planets``, we tag them with synthetic ids
    derived from existing ``initial_planets``.
    """
    paths_arr = np.zeros((MAX_COMETS, 4, MAX_COMET_PATH_LEN, 2), dtype=np.float32)
    path_lens = np.zeros(MAX_COMETS, dtype=np.int32)
    initial_ships = np.zeros(MAX_COMETS, dtype=np.int32)

    # Vendor sim mutates a shared `initial_planets` list across spawns; each
    # successful comet appends 4 new planet rows that future comets must avoid.
    running_initial_planets: list[list[float | int]] = [p[:] for p in initial_planets]
    running_comet_planet_ids: list[int] = []
    next_id = max((p[0] for p in initial_planets), default=-1) + 1

    for c_idx, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
        comet_paths = generate_comet_paths(
            running_initial_planets,
            angular_velocity,
            spawn_step,
            running_comet_planet_ids,
            COMET_SPEED,
            rng=comet_rng,
        )
        if comet_paths is None:
            continue

        # Vendor: comet_ships = min(randint(1,99) four times)
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        initial_ships[c_idx] = comet_ships

        path_len = len(comet_paths[0])
        path_lens[c_idx] = path_len
        for q in range(4):
            for k in range(path_len):
                paths_arr[c_idx, q, k, 0] = comet_paths[q][k][0]
                paths_arr[c_idx, q, k, 1] = comet_paths[q][k][1]

        # Append synthetic planet rows so future comets reject overlaps.
        # Vendor uses planet row `[pid, -1, -99, -99, COMET_RADIUS, ships, 1]`,
        # but the relevant overlap fields in `generate_comet_paths` are
        # planet[2], planet[3] (y, x) = -99, -99 and planet[4] (radius). With
        # y=x=-99 these rows are far off-board and never trigger overlap
        # checks; they're added purely to update the running set membership.
        for _q in range(4):
            pid = int(next_id)
            running_comet_planet_ids.append(pid)
            running_initial_planets.append(
                [pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, 1]
            )
            next_id += 1

    return paths_arr, path_lens, initial_ships
