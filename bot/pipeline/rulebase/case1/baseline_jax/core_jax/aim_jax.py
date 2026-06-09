"""JAX port of aim_with_prediction + search_safe_intercept (NON-COMET targets).

case1 baseline/core/physics.py:134-242. Faithful to the Python control flow:

aim_with_prediction:
  1. est0 = estimate_arrival to target's CURRENT position. If unsafe (sun) →
     fall back to search_safe_intercept.
  2. up to 5 refine iters: predict target at est.turns, re-estimate. Converge if
     |Δpos|<0.3 and |Δturns|<=INTERCEPT_TOLERANCE → return. If a re-estimate is
     unsafe → fall back to search. (predict_target_position never None for
     planets; the None branch is comet-only and handled elsewhere.)
  3. after 5 iters: final estimate to last predicted pos; if unsafe → search.

search_safe_intercept:
  sweep candidate_turns in 1..HORIZON, predict target there, estimate; keep
  candidates whose re-estimate turns are within INTERCEPT_TOLERANCE; pick the
  one minimizing (delta, confirm_turns, candidate_turns).

All fixed-shape: 5-iter refine via lax.scan w/ done flag; sweep via vmap over a
fixed HORIZON grid + masked argmin. Returns (angle, turns, ix, iy, valid).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics_jax import HORIZON_AIM as HORIZON
from .physics_jax import (
    INTERCEPT_TOLERANCE,
    estimate_arrival,
    predict_planet_position,
)

REFINE_ITERS = 5
_CANDIDATE_TURNS = jnp.arange(1, HORIZON + 1, dtype=jnp.int32)

Arr = jax.Array


def _predict(
    cur_x: Arr,
    cur_y: Arr,
    init_x: Arr,
    init_y: Arr,
    init_r: Arr,
    ang_vel: Arr,
    turns: Arr,
) -> tuple[Arr, Arr]:
    return predict_planet_position(cur_x, cur_y, init_x, init_y, init_r, ang_vel, turns)


def _search_safe_intercept(
    sx: Arr,
    sy: Arr,
    sr: Arr,
    cur_x: Arr,
    cur_y: Arr,
    init_x: Arr,
    init_y: Arr,
    init_r: Arr,
    tr: Arr,
    ships: Arr,
    ang_vel: Arr,
) -> tuple[Arr, Arr, Arr, Arr, Arr]:
    """Fixed HORIZON sweep + masked argmin, mirroring Python search_safe_intercept."""

    def score_candidate(
        cand_turns: Arr,
    ) -> tuple[Arr, Arr, Arr, Arr, Arr, Arr, Arr]:
        px, py = _predict(cur_x, cur_y, init_x, init_y, init_r, ang_vel, cand_turns)
        _a, turns, valid = estimate_arrival(sx, sy, sr, px, py, tr, ships)
        within = jnp.abs(turns - cand_turns) <= INTERCEPT_TOLERANCE
        ok1 = valid & within
        actual_turns = jnp.maximum(turns, cand_turns)
        apx, apy = _predict(cur_x, cur_y, init_x, init_y, init_r, ang_vel, actual_turns)
        cangle, cturns, cvalid = estimate_arrival(sx, sy, sr, apx, apy, tr, ships)
        delta = jnp.abs(cturns - actual_turns)
        ok = ok1 & cvalid & (delta <= INTERCEPT_TOLERANCE)
        return ok, delta, cturns, cand_turns, cangle, apx, apy

    ok, delta, cturns, cand, cangle, apx, apy = jax.vmap(score_candidate)(
        _CANDIDATE_TURNS
    )
    # Python picks min (delta, confirm_turns, candidate_turns) among ok candidates.
    # Encode lexicographic key; invalid → +inf. (*1.0 promotes to default float,
    # which respects jax_enable_x64 without an untyped config.read call.)
    key = delta * 1.0
    key = key * (HORIZON + 1) ** 2 + cturns * (HORIZON + 1) + cand
    key = jnp.where(ok, key, jnp.inf)
    best = jnp.argmin(key)
    any_ok = jnp.any(ok)
    return (
        cangle[best],
        cturns[best],
        apx[best],
        apy[best],
        any_ok,
    )


def aim_with_prediction(
    sx: Arr,
    sy: Arr,
    sr: Arr,
    cur_x: Arr,
    cur_y: Arr,
    init_x: Arr,
    init_y: Arr,
    init_r: Arr,
    tr: Arr,
    ships: Arr,
    ang_vel: Arr,
) -> tuple[Arr, Arr, Arr, Arr, Arr]:
    """Return (angle, turns, ix, iy, valid) for a NON-COMET target."""
    # Promote float inputs to the active default float (float64 under x64) so the
    # lax.scan carry (cur_x/cur_y/t0/init_fb) matches the body, which computes in
    # jnp.float_. Without this, a float32 caller + float64 body trip scan's
    # carry-dtype check; float32 path is unaffected (no-op cast).
    cur_x = cur_x.astype(jnp.float_)
    cur_y = cur_y.astype(jnp.float_)
    _a0, t0, safe0 = estimate_arrival(sx, sy, sr, cur_x, cur_y, tr, ships)
    t0 = t0.astype(jnp.float_)

    def refine_step(
        carry: tuple[Arr, Arr, Arr, Arr, Arr, tuple[Arr, Arr, Arr, Arr]], _i: Arr
    ) -> tuple[tuple[Arr, Arr, Arr, Arr, Arr, tuple[Arr, Arr, Arr, Arr]], None]:
        tx, ty, est_turns, done, fell_back, fb = carry
        # predict target at current est turns
        ntx, nty = _predict(cur_x, cur_y, init_x, init_y, init_r, ang_vel, est_turns)
        nangle, nturns, nvalid = estimate_arrival(sx, sy, sr, ntx, nty, tr, ships)
        converged = (
            (jnp.abs(ntx - tx) < 0.3)
            & (jnp.abs(nty - ty) < 0.3)
            & (jnp.abs(nturns - est_turns) <= INTERCEPT_TOLERANCE)
        )
        # if re-estimate unsafe and not already done → fall back to search
        need_fb = (~nvalid) & (~done)
        new_fb = jnp.where(need_fb & (~fell_back), True, fell_back)
        # record converged result the first time it happens
        first_converge = converged & nvalid & (~done) & (~new_fb)
        res_angle = jnp.where(first_converge, nangle, fb[0])
        res_turns = jnp.where(first_converge, nturns.astype(fb[1].dtype), fb[1])
        res_ix = jnp.where(first_converge, ntx, fb[2])
        res_iy = jnp.where(first_converge, nty, fb[3])
        new_done = done | first_converge | new_fb
        # advance tx,ty,est for next iter (only if not done)
        nx = jnp.where(new_done, tx, ntx)
        ny = jnp.where(new_done, ty, nty)
        ne = jnp.where(new_done, est_turns, nturns)
        return (
            nx,
            ny,
            ne,
            new_done,
            new_fb,
            (res_angle, res_turns, res_ix, res_iy),
        ), None

    # Bind scalar carry float dtype to cur_x (the upstream working float dtype)
    # so init and scan-body agree under both float32 and x64. A bare jnp.float_
    # baked to float32 at import while the body runs float64 under x64.
    _f = cur_x.dtype
    init_fb = (
        jnp.asarray(0.0, dtype=_f),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=_f),
        jnp.asarray(0.0, dtype=_f),
    )
    init = (cur_x, cur_y, t0, ~safe0, ~safe0, init_fb)
    (ftx, fty, fest, fdone, ffb, fres), _ = jax.lax.scan(
        refine_step, init, jnp.arange(REFINE_ITERS)
    )
    res_angle, res_turns, res_ix, res_iy = fres
    converged_done = fdone & (~ffb)

    # after 5 iters without convergence: final estimate to last (ftx, fty)
    fa, ftu, fvalid = estimate_arrival(sx, sy, sr, ftx, fty, tr, ships)
    used_final = (~converged_done) & (~ffb)
    final_needs_search = used_final & (~fvalid)

    # search fallback fires if: est0 unsafe, OR a refine re-estimate unsafe, OR
    # the post-loop final estimate unsafe.
    need_search = (~safe0) | ffb | final_needs_search
    s_angle, s_turns, s_ix, s_iy, s_valid = _search_safe_intercept(
        sx, sy, sr, cur_x, cur_y, init_x, init_y, init_r, tr, ships, ang_vel
    )

    # priority: search (if needed) > converged refine > final estimate
    angle = jnp.where(need_search, s_angle, jnp.where(converged_done, res_angle, fa))
    turns = jnp.where(
        need_search, s_turns, jnp.where(converged_done, res_turns, ftu)
    ).astype(jnp.int32)
    ix = jnp.where(need_search, s_ix, jnp.where(converged_done, res_ix, ftx))
    iy = jnp.where(need_search, s_iy, jnp.where(converged_done, res_iy, fty))
    valid = jnp.where(need_search, s_valid, jnp.where(converged_done, True, fvalid))
    return angle, turns, ix, iy, valid
