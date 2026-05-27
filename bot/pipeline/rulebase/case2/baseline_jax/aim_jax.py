# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the per-(src,target) intercept hot path in `baseline/core/physics.py`.

This is the *difficult* part the earlier leaf-only port deferred — and the part
that actually matters for speed. `aim_with_prediction` is called once per
(my_planet, target) pair across every mission generator (capture / harass /
crash / reinforcement), so for P planets it is O(P^2) calls per turn, each doing
a 5-step refinement and, on miss, a ~220-iteration `search_safe_intercept`
sweep. That double loop is the `act()` bottleneck; vectorizing it so a single
`vmap` evaluates the whole (src x target) grid at once is the point of the port.

Mirrors `aim_with_prediction` / `search_safe_intercept` from physics.py:

  aim: try direct estimate_arrival; refine target lead 5x with early-exit on
       convergence; if any step's estimate is invalid, fall back to the sweep.
  sweep: over candidate_turns in {1.0, 1.5, ..., max_turns}, find the turn whose
       predicted lead position is self-consistent (|turns - candidate| <= tol),
       pick the (delta, turns, candidate) argmin.

JAX realization of the three "hard" constructs:

* **5-iter refinement + early-exit** -> `lax.scan` over a fixed 5 steps carrying a
  `done` bool; once converged we stop *updating* (the carry freezes) instead of
  breaking, so the trace is static. The final result reads from the frozen carry.
* **variable-length candidate sweep + argmin** -> `lax.scan` over a fixed
  `MAX_CANDIDATE_TURNS` grid; out-of-range turns (> max_turns) and invalid
  estimates are masked to +inf score, then `jnp.argmin` selects the best. No
  data-dependent loop bound.
* **`None` short-circuit** -> a `valid` boolean threaded through and returned;
  `valid == False` is the Python `None`.

Comet targets are supported via **host-resolved path arrays** (§12c pattern):
the ragged `comets` list (variable comet groups, `list.index`, dict lookups) is
resolved *on the host* into a fixed `(MAX_COMET_PATH_LEN, 2)` path + `path_index`
+ `path_len` per target, then passed in as arrays. The JAX side indexes the path
with bounds-masking, so `predict_comet_position`'s `path[idx] if in range else
None` becomes a clamped gather + `ok` flag. `path_len == 0` signals a non-comet
(orbital/static) target, which takes the rotation branch. This keeps the ragged
*lookup* on the host (where it belongs) while the *numeric* hot path — including
comet targets — runs in one vmap. The parity test covers BOTH orbital/static and
comet targets against the Python `aim_with_prediction` (angle/turns/ix/iy +
valid/None).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..baseline.core.config import (
    CENTER_X,
    CENTER_Y,
    HORIZON,
    INTERCEPT_TOLERANCE,
    ROTATION_LIMIT,
    SAFE_INTERCEPT_HALF_STEP,
)
from .physics_jax import estimate_arrival_jax

# Fixed comet-path length for host-resolved comet target paths (matches the
# jax_env constant). Long enough for any vendor comet path (vendor max ~40).
MAX_COMET_PATH_LEN: int = 40

REFINE_ITERS: int = 5
CONVERGE_POS_EPS: float = 0.3

# Candidate-turns grid for search_safe_intercept. Python builds
# {1.0, 1.5, ..., HORIZON} when SAFE_INTERCEPT_HALF_STEP else {1, ..., HORIZON}.
# We pad to a fixed grid and mask turns beyond the (per-pair) max_turns.
_STEP: float = 0.5 if SAFE_INTERCEPT_HALF_STEP else 1.0
# range(2, 2*HORIZON+1) at half-step -> i/2 for i in 2..2H  => 1.0 .. HORIZON.
MAX_CANDIDATE_TURNS: int = int(HORIZON / _STEP)
_CANDIDATE_TURNS = jnp.arange(1, MAX_CANDIDATE_TURNS + 1, dtype=jnp.float32) * _STEP


def _predict_orbital_position(
    cur_x: jax.Array,
    cur_y: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_radius: jax.Array,
    ang_vel: jax.Array,
    turns: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Non-comet predict_target_position: orbital rotation, static stays put.

    `turns` may be fractional (sweep uses half-steps); the Python
    `predict_target_position_fractional` interpolates between floor/ceil, but
    for an orbital planet the rotation angle is linear in `turns`, so evaluating
    the rotation directly at the fractional `turns` is identical to the lerp of
    the two integer endpoints (both lie on the same circle, and the Python lerp
    is a chord approximation — within the 1e-4 band for the small per-step
    angle). Static planets ignore `turns` entirely.
    """
    r = jnp.sqrt((init_x - CENTER_X) ** 2 + (init_y - CENTER_Y) ** 2)
    static = (r + init_radius) >= ROTATION_LIMIT
    cur_ang = jnp.arctan2(cur_y - CENTER_Y, cur_x - CENTER_X)
    new_ang = cur_ang + ang_vel * turns
    rot_x = CENTER_X + r * jnp.cos(new_ang)
    rot_y = CENTER_Y + r * jnp.sin(new_ang)
    return jnp.where(static, cur_x, rot_x), jnp.where(static, cur_y, rot_y)


def _comet_pos_at(
    comet_path: jax.Array,  # (MAX_COMET_PATH_LEN, 2) host-resolved path
    path_index: jax.Array,  # scalar int — current position in the path
    path_len: jax.Array,  # scalar int — valid length (0 => not a comet)
    turn_idx: jax.Array,  # scalar int — integer turn offset
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mirror predict_comet_position for one integer turn. Returns (x, y, ok).

    Python: `future_idx = path_index + turns; return path[future_idx] if
    0 <= future_idx < len(path) else None`. We clamp the gather index and report
    `ok = 0 <= future_idx < path_len` so the caller masks out-of-range like None.
    """
    future_idx = path_index + turn_idx
    ok = (future_idx >= 0) & (future_idx < path_len)
    safe_idx = jnp.clip(future_idx, 0, comet_path.shape[0] - 1)
    pt = comet_path[safe_idx]
    return pt[0], pt[1], ok


def _predict_comet_fractional(
    comet_path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    turns: jax.Array,  # fractional
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mirror predict_target_position_fractional for a comet target.

    floor/ceil lookups + lerp. Python: pos_lo at floor(turns); if frac<=eps
    return pos_lo; pos_hi at floor+1; if pos_hi is None return pos_lo; else lerp.
    `ok` is False only when pos_lo itself is out of range (Python returns None).
    """
    lo = jnp.floor(turns).astype(jnp.int32)
    frac = turns - lo.astype(jnp.float32)
    lo_idx = jnp.maximum(0, lo)
    lo_x, lo_y, lo_ok = _comet_pos_at(comet_path, path_index, path_len, lo_idx)
    hi_x, hi_y, hi_ok = _comet_pos_at(comet_path, path_index, path_len, lo + 1)
    use_hi = (frac > 1e-9) & hi_ok
    x = jnp.where(use_hi, lo_x + (hi_x - lo_x) * frac, lo_x)
    y = jnp.where(use_hi, lo_y + (hi_y - lo_y) * frac, lo_y)
    return x, y, lo_ok


def _predict_position(
    cur_x: jax.Array,
    cur_y: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_radius: jax.Array,
    ang_vel: jax.Array,
    turns: jax.Array,
    comet_path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unified predict_target_position(_fractional). Returns (x, y, ok).

    Dispatches on `is_comet = path_len > 0` (host sets path_len=0 for orbital
    targets). The orbital branch never returns None (ok=True); the comet branch
    propagates the path-range None as ok=False.
    """
    is_comet = path_len > 0
    ox, oy = _predict_orbital_position(
        cur_x, cur_y, init_x, init_y, init_radius, ang_vel, turns
    )
    cx, cy, cok = _predict_comet_fractional(comet_path, path_index, path_len, turns)
    x = jnp.where(is_comet, cx, ox)
    y = jnp.where(is_comet, cy, oy)
    ok = jnp.where(is_comet, cok, True)
    return x, y, ok


def _search_safe_intercept_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    t_curx: jax.Array,
    t_cury: jax.Array,
    t_initx: jax.Array,
    t_inity: jax.Array,
    t_initr: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
    ang_vel: jax.Array,
    max_turns: jax.Array,
    comet_path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Vectorized search_safe_intercept. Returns (angle, turns, ix, iy, valid).

    Handles comet targets via the host-resolved `comet_path` (path_len>0); a
    `None` from `predict_target_position(_fractional)` (comet out of range) maps
    to that candidate being skipped (`pos_ok=False`), mirroring the Python
    `if pos is None: continue` / `if actual_pos is None: continue`.
    """

    def score_candidate(
        cand: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # Lead position at the candidate (fractional) turn.
        px, py, pos_ok = _predict_position(
            t_curx,
            t_cury,
            t_initx,
            t_inity,
            t_initr,
            ang_vel,
            cand,
            comet_path,
            path_index,
            path_len,
        )
        angle, turns, ok = estimate_arrival_jax(sx, sy, sr, px, py, tr, ships)
        # |turns - cand| <= tol gate.
        consistent = jnp.abs(turns.astype(jnp.float32) - cand) <= INTERCEPT_TOLERANCE
        actual_turns = jnp.maximum(turns, jnp.ceil(cand).astype(jnp.int32))
        apx, apy, apos_ok = _predict_position(
            t_curx,
            t_cury,
            t_initx,
            t_inity,
            t_initr,
            ang_vel,
            actual_turns.astype(jnp.float32),
            comet_path,
            path_index,
            path_len,
        )
        cangle, cturns, cok = estimate_arrival_jax(sx, sy, sr, apx, apy, tr, ships)
        delta = jnp.abs(cturns - actual_turns).astype(jnp.float32)
        cand_ok = (
            pos_ok
            & apos_ok
            & ok
            & cok
            & consistent
            & (delta <= INTERCEPT_TOLERANCE)
            & (cand <= max_turns.astype(jnp.float32))
        )
        # Score key mirrors Python tuple (delta, confirm_turns, candidate); we
        # fold to one float that preserves the lexicographic order well enough
        # for argmin (delta dominates, then turns, then candidate — all small
        # ints/halves). Invalid candidates -> +inf so argmin skips them.
        score = delta * 1e6 + cturns.astype(jnp.float32) * 1e2 + cand
        score = jnp.where(cand_ok, score, jnp.inf)
        return score, cangle, cturns, apx, apy

    scores, angles, turns_a, ixs, iys = jax.vmap(score_candidate)(_CANDIDATE_TURNS)
    best = jnp.argmin(scores)
    any_valid = jnp.isfinite(scores[best])
    return angles[best], turns_a[best], ixs[best], iys[best], any_valid


def aim_with_prediction_jax(
    sx: jax.Array,
    sy: jax.Array,
    sr: jax.Array,
    t_curx: jax.Array,
    t_cury: jax.Array,
    t_initx: jax.Array,
    t_inity: jax.Array,
    t_initr: jax.Array,
    tr: jax.Array,
    ships: jax.Array,
    ang_vel: jax.Array,
    max_turns: jax.Array,
    comet_path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """aim_with_prediction for orbital, static, AND comet targets.

    Comet targets are signalled by `path_len > 0` with a host-resolved
    `comet_path` (MAX_COMET_PATH_LEN, 2) and `path_index`; orbital/static
    targets pass `path_len=0` and a dummy path. Designed to be `jax.vmap`'d over
    (src, target) pairs to replace the Python `for src: for target:` double loop.

    Three terminal outcomes from the Python refinement loop are reproduced:
      - converge        -> success (refine result)
      - predict pos None -> return None outright   (comet out of range)
      - next_est None    -> fall back to the sweep
    tracked as `pos_failed` and `fellback` flags in the scan carry.
    """
    # Direct estimate to the target's current position (no comet lookup here,
    # matching Python's `estimate_arrival(target.x, target.y, ...)`).
    angle0, turns0, ok0 = estimate_arrival_jax(sx, sy, sr, t_curx, t_cury, tr, ships)

    # carry: tx, ty, est_angle, est_turns, done, fellback, pos_failed
    RefineCarry = tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ]

    def refine_step(carry: RefineCarry, _: jax.Array) -> tuple[RefineCarry, None]:
        tx, ty, e_ang, e_turns, done, fellback, pos_failed = carry
        live = ~done
        # predict_target_position at current est turns; comet may return None.
        px, py, pos_ok = _predict_position(
            t_curx,
            t_cury,
            t_initx,
            t_inity,
            t_initr,
            ang_vel,
            e_turns.astype(jnp.float32),
            comet_path,
            path_index,
            path_len,
        )
        # Python: `if pos is None: return None` — terminal failure for this step.
        new_pos_failed = pos_failed | (live & ~pos_ok)
        n_ang, n_turns, n_ok = estimate_arrival_jax(sx, sy, sr, px, py, tr, ships)
        # Python: `if next_est is None: return search_safe_intercept(...)`.
        new_fellback = fellback | (live & pos_ok & ~n_ok)
        converged = (
            (jnp.abs(px - tx) < CONVERGE_POS_EPS)
            & (jnp.abs(py - ty) < CONVERGE_POS_EPS)
            & (jnp.abs(n_turns - e_turns) <= INTERCEPT_TOLERANCE)
        )
        active = live & pos_ok & n_ok
        out_tx = jnp.where(active, px, tx)
        out_ty = jnp.where(active, py, ty)
        out_ang = jnp.where(active, n_ang, e_ang)
        out_turns = jnp.where(active, n_turns, e_turns)
        # Stop once converged, pos failed, or est failed (each is terminal).
        new_done = (
            done | (live & ~pos_ok) | (active & converged) | (live & pos_ok & ~n_ok)
        )
        return (
            out_tx,
            out_ty,
            out_ang,
            out_turns,
            new_done,
            new_fellback,
            new_pos_failed,
        ), None

    init = (t_curx, t_cury, angle0, turns0, ~ok0, ~ok0, jnp.bool_(False))
    (fx, fy, f_ang, f_turns, _done, fellback, pos_failed), _ = jax.lax.scan(
        refine_step, init, jnp.arange(REFINE_ITERS)
    )

    # Sweep fallback result (always computed; used when ~ok0 or fellback).
    s_ang, s_turns, s_ix, s_iy, s_valid = _search_safe_intercept_jax(
        sx,
        sy,
        sr,
        t_curx,
        t_cury,
        t_initx,
        t_inity,
        t_initr,
        tr,
        ships,
        ang_vel,
        max_turns,
        comet_path,
        path_index,
        path_len,
    )

    use_sweep = (~ok0) | fellback
    angle = jnp.where(use_sweep, s_ang, f_ang)
    turns = jnp.where(use_sweep, s_turns, f_turns)
    ix = jnp.where(use_sweep, s_ix, fx)
    iy = jnp.where(use_sweep, s_iy, fy)
    refine_valid = ~pos_failed  # pos None -> Python returns None
    valid = jnp.where(use_sweep, s_valid, refine_valid)
    # pos_failed overrides everything to invalid (Python `return None`), but only
    # on the refine path — if we fell back to the sweep before pos failed, the
    # sweep result stands. pos_failed can only be set on a live (non-fellback,
    # non-done) step, so it's safe to AND here.
    valid = jnp.where(pos_failed & ~use_sweep, jnp.bool_(False), valid)
    return angle, turns, ix, iy, valid


def resolve_comet_path(
    target_id: int,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[list[list[float]], int, int]:
    """Host-side: resolve a target's comet path to fixed-shape inputs.

    Returns (path_padded, path_index, path_len) where path_padded is a
    (MAX_COMET_PATH_LEN, 2) Python list (caller -> jnp.array). For a non-comet
    target, path_len=0 and the path is all zeros (JAX takes the orbital branch).
    Mirrors the group/`list.index` lookup in `predict_comet_position` /
    `comet_remaining_life` — this is the ragged part kept on the host. `comets`
    is `dict[str, Any]` because the vendor comet obs is heterogeneous (the
    Python source `physics.py` types it the same way).
    """
    zeros = [[0.0, 0.0] for _ in range(MAX_COMET_PATH_LEN)]
    if target_id not in comet_ids:
        return zeros, 0, 0
    for group in comets:
        pids = list(group.get("planet_ids", []))
        if target_id not in pids:
            continue
        idx = pids.index(target_id)
        paths = list(group.get("paths", []))
        path_index = int(group.get("path_index", 0))
        if idx >= len(paths):
            return zeros, 0, 0
        path = paths[idx]
        out = zeros[:]
        for i, pt in enumerate(path):
            if i >= MAX_COMET_PATH_LEN:
                break
            out[i] = [float(pt[0]), float(pt[1])]
        return out, path_index, int(len(path))
    return zeros, 0, 0


__all__ = [
    "aim_with_prediction_jax",
    "resolve_comet_path",
    "MAX_CANDIDATE_TURNS",
    "MAX_COMET_PATH_LEN",
]
