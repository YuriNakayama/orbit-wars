# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of case8's engine-accurate aim (`baseline/core/physics.py`).

case8's `aim_with_prediction` is fundamentally more precise than case2's: it
replays the fleet against the moving target tick-by-tick with the simulator's
swept-pair collision test (`_first_engine_hit_turn` / `_swept_pair_hit`) instead
of a geometric arrival estimate. This module mirrors *that* algorithm so the
JAX agent matches the case8 oracle's shot decisions (not case2's).

`aim_with_prediction` is called once per (my_planet, target) pair across every
mission generator, so for P planets it is O(P^2) calls per turn. Vectorizing it
into one `vmap` over the whole (src x target) grid is the point of the port.

The hard Python constructs and their JAX realizations:

* **`_first_engine_hit_turn` `for t in range(lo, hi+1)`** with early `return t`
  -> `lax.scan` over the fixed `[1 .. HORIZON]` turn grid carrying a `found`
  bool + `hit_turn`; turns outside `[turn_lo, turn_hi]` are masked off, and once
  a hit is found the carry freezes (first hit wins).

* **`aim_with_prediction` direct -> 5-iter lead refine -> sweep fallback** with
  early `return` per stage -> each stage runs unconditionally; a priority
  `jnp.where(direct, ..., where(refine, ..., sweep))` selects the winner exactly
  as the Python early-returns would.

* **`search_safe_intercept` variable sweep + argmin** -> `lax.scan` over the
  fixed `_CANDIDATE_TURNS` grid carrying the running lexicographic-best
  `(hit_turn, |hit_turn - candidate|)`; invalid candidates score `+inf`.

* **`None` short-circuit** -> a `valid` boolean threaded through; `valid == False`
  is the Python `None`.

Comet targets use host-resolved path arrays (the §12c pattern): the ragged
`comets` list is resolved on the host to a fixed `(MAX_COMET_PATH_LEN, 2)` path
+ `path_index` + `path_len`; `path_len == 0` selects the orbital/static branch.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    HORIZON,
    SAFE_INTERCEPT_HALF_STEP,
)
from .geometry_jax import safe_angle_and_distance_jax
from .physics_jax import (
    estimate_arrival_jax,
    fleet_speed_jax,
    predict_planet_position_jax,
)

# Fixed comet-path length for host-resolved comet target paths (vendor max ~40).
MAX_COMET_PATH_LEN: int = 40

# case8 constants.
_LAUNCH_OFFSET: float = 0.1  # engine starts fleets at planet surface + 0.1
_HIT_SEARCH_WINDOW: int = 4
REFINE_ITERS: int = 5

# Engine replay grid: integer turns 1..HORIZON. _first_engine_hit_turn walks
# integer turns; we scan the full grid and mask to [turn_lo, turn_hi] per call.
_TURN_GRID = jnp.arange(1, HORIZON + 1, dtype=jnp.int32)  # (HORIZON,)

# search_safe_intercept candidate-turns grid. case8 `_iter_candidate_turns`:
#   half-step:  {i / 2 for i in range(2, 2*max_turns + 1)} = {1.0, 1.5, .., max}
#   integer:    {t for t in range(1, max_turns + 1)}       = {1, 2, .., max}
# We build the full HORIZON-sized grid and mask candidates beyond max_turns.
if SAFE_INTERCEPT_HALF_STEP:
    _CANDIDATE_TURNS = jnp.arange(2, 2 * HORIZON + 1, dtype=jnp.float32) * 0.5
else:
    _CANDIDATE_TURNS = jnp.arange(1, HORIZON + 1, dtype=jnp.float32)
MAX_CANDIDATE_TURNS: int = int(_CANDIDATE_TURNS.shape[0])

_BIG: float = 1e18


def _swept_pair_hit_jax(
    ax: jax.Array,
    ay: jax.Array,
    bx: jax.Array,
    by: jax.Array,
    p0x: jax.Array,
    p0y: jax.Array,
    p1x: jax.Array,
    p1y: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    """Mirror `_swept_pair_hit`: do the two points come within `radius` over a tick?

    Branches (`quad_a < 1e-12`, `disc < 0`) become `jnp.where` masks.
    """
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    quad_a = dvx * dvx + dvy * dvy
    quad_b = 2.0 * (d0x * dvx + d0y * dvy)
    quad_c = d0x * d0x + d0y * d0y - radius * radius

    degenerate = quad_a < 1e-12
    degen_hit = quad_c <= 0.0

    safe_a = jnp.where(degenerate, 1.0, quad_a)
    disc = quad_b * quad_b - 4.0 * safe_a * quad_c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-quad_b - sq) / (2.0 * safe_a)
    t2 = (-quad_b + sq) / (2.0 * safe_a)
    quad_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)

    return jnp.where(degenerate, degen_hit, quad_hit)


def _predict_target_position_jax(
    t: jax.Array,
    tcx: jax.Array,
    tcy: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    ang_vel: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`predict_target_position(target, t, ...)` for integer/float turn `t`.

    Returns `(x, y, ok)`. `ok == False` mirrors Python `None` (comet index out
    of range). `path_len == 0` selects the orbital/static branch.
    """
    is_comet = path_len > 0

    orb_x, orb_y = predict_planet_position_jax(
        tcx, tcy, init_x, init_y, init_r, ang_vel, t
    )

    # predict_comet_position uses int(turns); replicate with floor.
    t_int = jnp.floor(t).astype(jnp.int32)
    future_idx = path_index + t_int
    in_range = is_comet & (future_idx >= 0) & (future_idx < path_len)
    safe_idx = jnp.clip(future_idx, 0, MAX_COMET_PATH_LEN - 1)
    com_x = path[safe_idx, 0]
    com_y = path[safe_idx, 1]

    x = jnp.where(is_comet, com_x, orb_x)
    y = jnp.where(is_comet, com_y, orb_y)
    ok = jnp.where(is_comet, in_range, True)
    return x, y, ok


def _first_engine_hit_turn_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    angle: jax.Array,
    ships: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    tcx: jax.Array,
    tcy: jax.Array,
    tr: jax.Array,
    ang_vel: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    turn_lo: jax.Array,
    turn_hi: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mirror `_first_engine_hit_turn`. Returns `(hit_turn, found)`.

    Scans the fixed `[1..HORIZON]` grid, masks to `[max(1,turn_lo), turn_hi]`,
    and freezes on the first swept-pair hit.
    """
    speed = fleet_speed_jax(jnp.maximum(jnp.int32(1), ships))
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    offset = sr + _LAUNCH_OFFSET
    start_x = sx + cos_a * offset
    start_y = sy + sin_a * offset

    start = jnp.maximum(jnp.int32(1), turn_lo)

    def fleet_at(t: jax.Array) -> tuple[jax.Array, jax.Array]:
        tf = t.astype(jnp.float32)
        return start_x + cos_a * speed * tf, start_y + sin_a * speed * tf

    def per_turn(t: jax.Array) -> jax.Array:
        active = (t >= start) & (t <= turn_hi)

        fx, fy = fleet_at(t)
        fpx, fpy = fleet_at(t - 1)
        px_now, py_now, ok_now = _predict_target_position_jax(
            t, tcx, tcy, init_x, init_y, init_r, ang_vel, path, path_index, path_len
        )
        px_prev, py_prev, ok_prev = _predict_target_position_jax(
            t - 1,
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )
        hit = _swept_pair_hit_jax(
            fpx, fpy, fx, fy, px_prev, py_prev, px_now, py_now, tr
        )
        return active & ok_now & ok_prev & hit

    # Each turn's hit predicate is independent; the old scan carry only encoded
    # "take the FIRST hit". vmap over the grid + argmax (first True index) is
    # byte-identical and removes a 110-step sequential kernel chain (GPU win).
    is_hit = jax.vmap(per_turn)(_TURN_GRID)  # (H,) bool
    found = jnp.any(is_hit)
    hit_turn = jnp.where(found, _TURN_GRID[jnp.argmax(is_hit)], jnp.int32(0))
    return hit_turn, found


def _hit_turn_for_target_position_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    aim_x: jax.Array,
    aim_y: jax.Array,
    ships: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    tcx: jax.Array,
    tcy: jax.Array,
    tr: jax.Array,
    ang_vel: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    max_turns: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mirror `_hit_turn_for_target_position`. Returns `(angle, hit_turn, ok)`.

    Aims at `(aim_x, aim_y)`, brackets the engine replay to
    `[est_turn +/- _HIT_SEARCH_WINDOW]`, reports the first hit. The fleet replay
    still tracks the real target center `(tcx, tcy)`, not the lead point.
    """
    angle, _hit_dist, safe_valid = safe_angle_and_distance_jax(
        sx, sy, sr, aim_x, aim_y, tr
    )
    _est_angle, est_turn, est_valid = estimate_arrival_jax(
        sx, sy, sr, aim_x, aim_y, tr, ships
    )

    lo = jnp.maximum(jnp.int32(1), est_turn - _HIT_SEARCH_WINDOW)
    hi = jnp.minimum(max_turns, est_turn + _HIT_SEARCH_WINDOW)
    hit_turn, found = _first_engine_hit_turn_jax(
        sx,
        sy,
        sr,
        angle,
        ships,
        init_x,
        init_y,
        init_r,
        tcx,
        tcy,
        tr,
        ang_vel,
        path,
        path_index,
        path_len,
        lo,
        hi,
    )
    ok = safe_valid & est_valid & found
    return angle, hit_turn, ok


def _search_safe_intercept_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    ships: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    tcx: jax.Array,
    tcy: jax.Array,
    tr: jax.Array,
    ang_vel: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    max_turns: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Mirror `search_safe_intercept`. Returns `(angle, hit_turn, ix, iy, valid)`.

    Sweeps the fixed candidate grid, predicts the (fractional) lead position,
    tests an engine hit, keeps the lexicographically-best
    `(hit_turn, |hit_turn - candidate|)` score.
    """

    def per_candidate(
        cand: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        in_range = cand <= max_turns.astype(jnp.float32)

        # predict_target_position_fractional(target, cand)
        lo_x, lo_y, lo_ok = _predict_target_position_jax(
            jnp.floor(cand),
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )
        frac = cand - jnp.floor(cand)
        hi_x, hi_y, hi_ok = _predict_target_position_jax(
            jnp.floor(cand) + 1.0,
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )
        use_interp = (frac > 1e-9) & hi_ok
        lead_x = jnp.where(use_interp, lo_x + (hi_x - lo_x) * frac, lo_x)
        lead_y = jnp.where(use_interp, lo_y + (hi_y - lo_y) * frac, lo_y)

        angle, hit_turn, hit_ok = _hit_turn_for_target_position_jax(
            sx,
            sy,
            sr,
            lead_x,
            lead_y,
            ships,
            init_x,
            init_y,
            init_r,
            tcx,
            tcy,
            tr,
            ang_vel,
            path,
            path_index,
            path_len,
            max_turns,
        )
        ax, ay, actual_ok = _predict_target_position_jax(
            hit_turn.astype(jnp.float32),
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )

        valid = in_range & lo_ok & hit_ok & actual_ok
        score_h = jnp.where(valid, hit_turn, jnp.int32(2**30))
        score_d = jnp.where(valid, jnp.abs(hit_turn.astype(jnp.float32) - cand), _BIG)
        return score_h, score_d, angle, hit_turn, ax, ay

    # The old 219-step scan carried a lexicographic-best (score_h, score_d) with
    # first-wins ties (strict `<`). Each candidate is independent — and the scan
    # NESTED the 110-step engine-hit scan per candidate (219x110 sequential
    # launches: the engine lineage's dominant GPU chain). vmap over candidates
    # collapses the outer chain to wide kernels; the two-stage argmin below picks
    # min score_h, then min score_d, then FIRST index — exactly the scan's order.
    score_h, score_d, angle_a, turn_a, ax_a, ay_a = jax.vmap(per_candidate)(
        _CANDIDATE_TURNS
    )
    min_h = jnp.min(score_h)
    d_masked = jnp.where(score_h == min_h, score_d, jnp.float32(jnp.inf))
    best = jnp.argmin(d_masked)  # first occurrence == scan's first-wins tie-break
    found = min_h < 2**30
    valid = found & (max_turns > 0)
    # Mask outputs to the scan's init values when nothing was taken (byte parity
    # on the invalid path too).
    best_angle = jnp.where(found, angle_a[best], jnp.float32(0.0))
    best_turn = jnp.where(found, turn_a[best], jnp.int32(0))
    best_ix = jnp.where(found, ax_a[best], jnp.float32(0.0))
    best_iy = jnp.where(found, ay_a[best], jnp.float32(0.0))
    return best_angle, best_turn, best_ix, best_iy, valid


def aim_with_prediction_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    tcx: jax.Array,
    tcy: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
    ang_vel: jax.Array,
    max_turns: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Mirror case8 `aim_with_prediction`. Returns `(angle, turns, ix, iy, valid)`.

    `(init_x, init_y, init_r)` is the target's orbital `initial` (or its current
    pose when unknown), `(path, path_index, path_len)` the host-resolved comet
    path (`path_len == 0` for non-comets), `max_turns` the per-call sweep cap.
    """
    max_turns = jnp.maximum(jnp.int32(0), max_turns)
    horizon_ok = max_turns > 0

    # --- direct aim: aim at the target's current position (tcx, tcy) ---
    d_angle, d_turn, d_ok = _hit_turn_for_target_position_jax(
        sx,
        sy,
        sr,
        tcx,
        tcy,
        ships,
        init_x,
        init_y,
        init_r,
        tcx,
        tcy,
        tr,
        ang_vel,
        path,
        path_index,
        path_len,
        max_turns,
    )
    da_x, da_y, da_actual_ok = _predict_target_position_jax(
        d_turn.astype(jnp.float32),
        tcx,
        tcy,
        init_x,
        init_y,
        init_r,
        ang_vel,
        path,
        path_index,
        path_len,
    )
    direct_hit = d_ok & da_actual_ok

    # --- lead-aim refinement: 5-iter scan, freeze once done ---
    _est_angle, est_turn0, est_valid0 = estimate_arrival_jax(
        sx, sy, sr, tcx, tcy, tr, ships
    )

    def refine_body(
        carry: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
        _i: jax.Array,
    ) -> tuple[
        tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
        None,
    ]:
        done, hit, turns_guess, r_angle, r_turn, r_ix, r_iy = carry

        pos_x, pos_y, pos_ok = _predict_target_position_jax(
            turns_guess.astype(jnp.float32),
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )
        a_angle, a_turn, a_ok = _hit_turn_for_target_position_jax(
            sx,
            sy,
            sr,
            pos_x,
            pos_y,
            ships,
            init_x,
            init_y,
            init_r,
            tcx,
            tcy,
            tr,
            ang_vel,
            path,
            path_index,
            path_len,
            max_turns,
        )
        act_x, act_y, act_ok = _predict_target_position_jax(
            a_turn.astype(jnp.float32),
            tcx,
            tcy,
            init_x,
            init_y,
            init_r,
            ang_vel,
            path,
            path_index,
            path_len,
        )
        success = (~done) & pos_ok & a_ok & act_ok
        new_hit = hit | success
        new_angle = jnp.where(success, a_angle, r_angle)
        new_turn = jnp.where(success, a_turn, r_turn)
        new_ix = jnp.where(success, act_x, r_ix)
        new_iy = jnp.where(success, act_y, r_iy)

        # next_est = estimate_arrival(src, pos)
        _na, next_turn, next_ok = estimate_arrival_jax(
            sx, sy, sr, pos_x, pos_y, tr, ships
        )
        # Python breaks when: pos None, success, next None, or new==guess.
        pos_break = (~done) & (~pos_ok)
        converged = (~next_ok) | (next_turn == turns_guess)
        new_done = done | success | pos_break | converged
        advance = (~done) & pos_ok & (~success) & next_ok & (~converged)
        new_guess = jnp.where(advance, next_turn, turns_guess)

        return (new_done, new_hit, new_guess, new_angle, new_turn, new_ix, new_iy), None

    # estimate_arrival None -> Python skips the loop entirely (straight to sweep).
    refine_init = (
        ~est_valid0,  # done
        jnp.bool_(False),  # hit
        est_turn0,  # turns_guess
        jnp.float32(0.0),
        jnp.int32(0),
        jnp.float32(0.0),
        jnp.float32(0.0),
    )
    (_rd, refine_hit, _g, r_angle, r_turn, r_ix, r_iy), _ = jax.lax.scan(
        refine_body, refine_init, jnp.arange(REFINE_ITERS)
    )

    # --- fallback sweep (search_safe_intercept) ---
    s_angle, s_turn, s_ix, s_iy, s_valid = _search_safe_intercept_jax(
        sx,
        sy,
        sr,
        ships,
        init_x,
        init_y,
        init_r,
        tcx,
        tcy,
        tr,
        ang_vel,
        path,
        path_index,
        path_len,
        max_turns,
    )

    # --- combine in Python priority order: direct -> refine -> sweep ---
    use_refine = (~direct_hit) & refine_hit
    use_sweep = (~direct_hit) & (~refine_hit)
    angle = jnp.where(direct_hit, d_angle, jnp.where(use_refine, r_angle, s_angle))
    turns = jnp.where(direct_hit, d_turn, jnp.where(use_refine, r_turn, s_turn))
    ix = jnp.where(direct_hit, da_x, jnp.where(use_refine, r_ix, s_ix))
    iy = jnp.where(direct_hit, da_y, jnp.where(use_refine, r_iy, s_iy))
    valid = horizon_ok & (direct_hit | refine_hit | (use_sweep & s_valid))

    return angle, turns, ix, iy, valid


def resolve_comet_path(
    target_id: int,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[list[list[float]], int, int]:
    """Host-side: resolve a target's comet path to a fixed `(MAX_COMET_PATH_LEN, 2)`.

    Returns `(path_arr, path_index, path_len)`. Non-comet -> zero path,
    `path_len == 0` (orbital branch). Mirrors `predict_comet_position` /
    `comet_remaining_life` lookups.
    """
    zeros = [[0.0, 0.0] for _ in range(MAX_COMET_PATH_LEN)]
    if target_id not in comet_ids:
        return zeros, 0, 0
    for group in comets:
        pids = group.get("planet_ids", [])
        if target_id not in pids:
            continue
        idx = pids.index(target_id)
        paths = group.get("paths", [])
        if idx >= len(paths):
            return zeros, 0, 0
        path = paths[idx]
        path_index = int(group.get("path_index", 0))
        out = list(zeros)
        n = min(len(path), MAX_COMET_PATH_LEN)
        for i in range(n):
            out[i] = [float(path[i][0]), float(path[i][1])]
        return out, path_index, len(path)
    return zeros, 0, 0


__all__ = [
    "MAX_COMET_PATH_LEN",
    "aim_with_prediction_jax",
    "resolve_comet_path",
]
