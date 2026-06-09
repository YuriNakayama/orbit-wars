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

from ._config_compat import (
    HORIZON,
    INTERCEPT_TOLERANCE,
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

    def body(
        carry: tuple[jax.Array, jax.Array], t: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        found, hit_turn = carry
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
        is_hit = active & ok_now & ok_prev & hit
        take = is_hit & (~found)
        return (found | take, jnp.where(take, t, hit_turn)), None

    (found, hit_turn), _ = jax.lax.scan(
        body, (jnp.bool_(False), jnp.int32(0)), _TURN_GRID
    )
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
    """Mirror case1 `search_safe_intercept` (GEOMETRIC, not engine-replay).

    case1's aim chain is geometric (`estimate_arrival` + `predict_target_position`),
    NOT case8's `_first_engine_hit_turn` collision replay. For each integer
    candidate turn `c` in `[1, max_turns]`:

      pos = predict_target_position(target, c)
      est_turns = estimate_arrival(src -> pos)            # geometric ceil(d/speed)
      if |est_turns - c| > tol: skip
      actual = max(est_turns, c)
      actual_pos = predict_target_position(target, actual)
      confirm = estimate_arrival(src -> actual_pos)
      if |confirm_turns - actual| > tol: skip
      score = (|confirm_turns - actual|, confirm_turns, c)  # lexicographic min

    Returns `(angle, hit_turn, ix, iy, valid)` for the best-scoring candidate,
    where `angle`/`hit_turn`/`(ix, iy)` come from the confirm pass.
    """
    tol = jnp.float32(INTERCEPT_TOLERANCE)

    def body(
        carry: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
        cand: jax.Array,
    ) -> tuple[
        tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
        None,
    ]:
        best_d, best_t, best_c, best_angle, best_turn, best_ix, best_iy = carry
        # cand is a (possibly fractional) candidate turn from `_CANDIDATE_TURNS`
        # (half-step grid when SAFE_INTERCEPT_HALF_STEP, else integers).
        in_range = cand <= max_turns.astype(jnp.float32)

        px, py, pos_ok = _predict_target_position_jax(
            cand,
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
        _ea_angle, est_turns, est_ok = estimate_arrival_jax(
            sx, sy, sr, px, py, tr, ships
        )
        gate1 = jnp.abs(est_turns.astype(jnp.float32) - cand) <= tol

        # Python: actual_turns = max(turns, int(ceil(candidate_turns))).
        cand_ceil = jnp.ceil(cand).astype(jnp.int32)
        actual = jnp.maximum(est_turns, cand_ceil)
        ax, ay, apos_ok = _predict_target_position_jax(
            actual.astype(jnp.float32),
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
        c_angle, c_turns, c_ok = estimate_arrival_jax(sx, sy, sr, ax, ay, tr, ships)
        delta = jnp.abs(c_turns.astype(jnp.float32) - actual.astype(jnp.float32))
        gate2 = delta <= tol

        valid = in_range & pos_ok & est_ok & gate1 & apos_ok & c_ok & gate2
        # lexicographic score (delta, confirm_turns, candidate_turns) minimization.
        d_i = jnp.where(valid, c_turns - actual, jnp.int32(2**30))  # == +delta or 0
        d_abs = jnp.abs(d_i)
        better = (
            (d_abs < best_d)
            | ((d_abs == best_d) & (c_turns < best_t))
            | ((d_abs == best_d) & (c_turns == best_t) & (cand < best_c))
        )
        take = valid & better
        return (
            jnp.where(take, d_abs, best_d),
            jnp.where(take, c_turns, best_t),
            jnp.where(take, cand, best_c),
            jnp.where(take, c_angle, best_angle),
            jnp.where(take, c_turns, best_turn),
            jnp.where(take, ax, best_ix),
            jnp.where(take, ay, best_iy),
        ), None

    init = (
        jnp.int32(2**30),  # best |delta|
        jnp.int32(2**30),  # best confirm_turns
        jnp.float32(1e18),  # best candidate (float — half-step grid)
        jnp.float32(0.0),
        jnp.int32(0),
        jnp.float32(0.0),
        jnp.float32(0.0),
    )
    cand_grid = _CANDIDATE_TURNS
    (best_d, _bt, _bc, best_angle, best_turn, best_ix, best_iy), _ = jax.lax.scan(
        body, init, cand_grid
    )
    valid = (best_d < 2**30) & (max_turns > 0)
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
    """Mirror case1 `aim_with_prediction` (GEOMETRIC lead-aim refine).

    case1 (NOT case8 engine-replay):

        est = estimate_arrival(src -> target.now)
        if est is None: return search_safe_intercept(...)
        tx, ty = target.x, target.y
        for _ in range(5):
            turns = est.turns
            pos = predict_target_position(target, turns)
            if pos is None: return None
            next_est = estimate_arrival(src -> pos)
            if next_est is None: return search_safe_intercept(...)
            if |pos - (tx,ty)| < 0.3 (both axes) and |next.turns - turns| <= tol:
                return next.angle, next.turns, pos            # converged
            tx, ty = pos; est = next_est
        final = estimate_arrival(src -> tx, ty)
        if final is None: return search_safe_intercept(...)
        return final.angle, final.turns, tx, ty

    `(init_x, init_y, init_r)` is the target orbital initial; `(path, path_index,
    path_len)` the comet path (`path_len == 0` for non-comets).
    """
    max_turns = jnp.maximum(jnp.int32(0), max_turns)
    horizon_ok = max_turns > 0
    tol = jnp.float32(INTERCEPT_TOLERANCE)

    # est = estimate_arrival(src -> target current pose)
    _e_angle, est_turns0, est_ok0 = estimate_arrival_jax(
        sx, sy, sr, tcx, tcy, tr, ships
    )

    # 5-iter geometric refine. carry: (done, hit, fail, turns, tx, ty, angle, rt)
    def refine_body(
        carry: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        _i: jax.Array,
    ) -> tuple[
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        None,
    ]:
        done, hit, fail, turns, tx, ty, r_angle, r_turn = carry
        px, py, pos_ok = _predict_target_position_jax(
            turns.astype(jnp.float32),
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
        # pos is None -> Python `return None` (hard fail).
        new_fail = fail | ((~done) & (~pos_ok))
        n_angle, n_turns, n_ok = estimate_arrival_jax(sx, sy, sr, px, py, tr, ships)
        # next_est is None -> fall through to sweep (treat as not-hit, stop refine).
        sweep_here = (~done) & pos_ok & (~n_ok)
        converged = (
            (~done)
            & pos_ok
            & n_ok
            & (jnp.abs(px - tx) < 0.3)
            & (jnp.abs(py - ty) < 0.3)
            & (jnp.abs(n_turns.astype(jnp.float32) - turns.astype(jnp.float32)) <= tol)
        )
        new_hit = hit | converged
        new_done = done | new_fail | sweep_here | converged
        # on convergence freeze (angle=next, turns=next, tx/ty=pos); else advance.
        out_angle = jnp.where(converged, n_angle, r_angle)
        out_turn = jnp.where(converged, n_turns, r_turn)
        advance = (~done) & pos_ok & n_ok & (~converged)
        out_tx = jnp.where(converged | advance, px, tx)
        out_ty = jnp.where(converged | advance, py, ty)
        out_turns = jnp.where(advance, n_turns, turns)
        return (
            new_done,
            new_hit,
            new_fail,
            out_turns,
            out_tx,
            out_ty,
            out_angle,
            out_turn,
        ), None

    refine_init = (
        ~est_ok0,  # done if est is None (skip straight to sweep)
        jnp.bool_(False),  # hit (converged)
        jnp.bool_(False),  # fail (pos None)
        est_turns0,  # turns
        tcx,  # tx = target.x
        tcy,  # ty = target.y
        jnp.float32(0.0),  # converged angle
        jnp.int32(0),  # converged turns
    )
    (r_done, r_hit, r_fail, _rt, r_tx, r_ty, r_angle, r_turn), _ = jax.lax.scan(
        refine_body, refine_init, jnp.arange(REFINE_ITERS)
    )

    # After 5 iters without convergence (and no fail / no early sweep): final_est
    # at (r_tx, r_ty). Python's loop-exit path.
    f_angle, f_turns, f_ok = estimate_arrival_jax(sx, sy, sr, r_tx, r_ty, tr, ships)
    # final used only when refine neither converged nor hard-failed nor needs sweep
    # purely from est-None; if est was valid and we fell out of the loop, use final.
    loop_exhausted = est_ok0 & (~r_hit) & (~r_fail)
    use_final = loop_exhausted & f_ok

    # fallback sweep (search_safe_intercept) — geometric.
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

    # Priority: hard-fail -> invalid; converged refine -> refine; loop-exhausted &
    # final ok -> final; else sweep. (Python returns None only on the pos-None
    # hard fail inside the loop.)
    use_refine = r_hit
    use_sweep = (~r_hit) & (~use_final) & (~r_fail)

    # confirm position for the refine/final branches (predict_target_position at
    # the chosen turns); Python returns (tx, ty) directly, so reuse r_tx/r_ty.
    angle = jnp.where(
        use_refine,
        r_angle,
        jnp.where(use_final, f_angle, s_angle),
    )
    turns = jnp.where(
        use_refine,
        r_turn,
        jnp.where(use_final, f_turns, s_turn),
    )
    ix = jnp.where(use_refine | use_final, r_tx, s_ix)
    iy = jnp.where(use_refine | use_final, r_ty, s_iy)

    valid = horizon_ok & (~r_fail) & (use_refine | use_final | (use_sweep & s_valid))
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
