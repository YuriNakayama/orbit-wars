# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of `WorldModel.plan_shot` + its 4 post-aim safety filters.

`plan_shot(src, target, ships)` = `aim_with_prediction` followed by 4 boolean
gates from `baseline/core/safety.py`; failing any gate turns the shot into
`None`. This module ports each gate as a pure mask and combines them with the
aim into `plan_shot_jax`, which returns `(angle, turns, ix, iy, valid)` — the
unit the Phase-2 (src x target) grid will `vmap` over.

The Python gates are all bounded loops over `[0..turns]` (turns <= HORIZON) or
small comet sets, so they become `lax.scan`/`vmap` over fixed grids with masks:

* `is_trajectory_sun_safe`      -> sample the fleet line at integer turns 0..H,
  mask t > turns, sun-hit if any active sample is within SUN_R + safety.
* `intercept_holds_within_tolerance` -> check turns +/- INTERCEPT_TOLERANCE that
  the target stays within `target.radius * miss_radius_factor` of the intercept.
* `target_reachable_before_comet_expiry` -> scalar comet-life comparison.
* `fleet_crosses_other_comet`   -> swept-pair sweep of the fleet line vs every
  *other* active comet planet's path; host passes a fixed
  `(MAX_OTHER_COMETS, MAX_COMET_PATH_LEN, 2)` array + lengths + ids.

Comet inputs follow the host-resolved §12c pattern already used by `aim_jax`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._config_compat import (
    CENTER_X,
    CENTER_Y,
    INTERCEPT_TOLERANCE,
    LAUNCH_CLEARANCE,
    SUN_R,
    SUN_SAFETY,
)
from .aim_jax import (
    MAX_COMET_PATH_LEN,
    _predict_target_position_jax,
    _swept_pair_hit_jax,
    aim_with_prediction_jax,
)
from .physics_jax import fleet_speed_jax

# Max comet planets the fleet-crossing check sweeps against. 5 waves * 4
# quadrants = 20, but only one wave is active at a time; 20 is a safe upper cap.
MAX_OTHER_COMETS: int = 20

_MISS_RADIUS_FACTOR: float = 0.7
_COMET_CROSS_SAFETY: float = 1.0
_COMET_RADIUS_DEFAULT: float = 2.0
_COMET_EXPIRY_SAFETY: int = 1

# Fleet sun-safety sample grid: integer turns 0..HORIZON (Python samples
# range(turns + 1)). Imported HORIZON via aim_jax's grid would be cleaner but
# we re-derive to keep the dependency explicit.
from ._config_compat import HORIZON  # noqa: E402

_SUN_SAMPLE_TURNS = jnp.arange(0, HORIZON + 1, dtype=jnp.float32)  # (H+1,)
_CROSS_TURNS = jnp.arange(1, HORIZON + 1, dtype=jnp.int32)  # (H,) — t in 1..turns


def is_trajectory_sun_safe_jax(
    launch_x: jax.Array,
    launch_y: jax.Array,
    angle: jax.Array,
    turns: jax.Array,
    ships: jax.Array,
    safety: float = SUN_SAFETY,
) -> jax.Array:
    """Mirror `is_trajectory_sun_safe`. Returns True when the line clears the sun.

    Samples integer turns 0..turns; turns<0 -> False (Python early return).
    """
    speed = fleet_speed_jax(jnp.maximum(jnp.int32(1), ships))
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)
    threshold = SUN_R + safety

    px = launch_x + dx * speed * _SUN_SAMPLE_TURNS
    py = launch_y + dy * speed * _SUN_SAMPLE_TURNS
    d = jnp.hypot(px - CENTER_X, py - CENTER_Y)
    active = _SUN_SAMPLE_TURNS <= turns.astype(jnp.float32)
    hits_sun = jnp.any(active & (d < threshold))
    return (turns >= 0) & (~hits_sun)


def intercept_holds_within_tolerance_jax(
    tcx: jax.Array,
    tcy: jax.Array,
    tr: jax.Array,
    predicted_turns: jax.Array,
    pred_x: jax.Array,
    pred_y: jax.Array,
    init_x: jax.Array,
    init_y: jax.Array,
    init_r: jax.Array,
    ang_vel: jax.Array,
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    miss_radius_factor: float = _MISS_RADIUS_FACTOR,
    tolerance: int = INTERCEPT_TOLERANCE,
) -> jax.Array:
    """Mirror `intercept_holds_within_tolerance`.

    Checks turns in `[predicted_turns - tol, predicted_turns + tol]` (t >= 0)
    that the target stays within `tr * miss_radius_factor` of `(pred_x, pred_y)`.
    A comet index out of range -> False (Python returns False on pos None).
    """
    keep_within = jnp.maximum(1e-3, tr * miss_radius_factor)
    deltas = jnp.arange(-tolerance, tolerance + 1, dtype=jnp.int32)

    def body(_carry: jax.Array, delta: jax.Array) -> tuple[jax.Array, None]:
        t = predicted_turns + delta
        x, y, ok = _predict_target_position_jax(
            t.astype(jnp.float32),
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
        is_comet = path_len > 0
        # t < 0 -> Python `continue` (skip, not fail).
        skip = t < 0
        # comet + out of range -> fail; orbital is always ok.
        comet_fail = is_comet & (~ok) & (~skip)
        within = _dist(x, y, pred_x, pred_y) <= keep_within
        ok_here = skip | (within & ~comet_fail)
        return _carry & ok_here, None

    holds, _ = jax.lax.scan(body, jnp.bool_(True), deltas)
    return jnp.asarray(holds)


def target_reachable_before_comet_expiry_jax(
    predicted_turns: jax.Array,
    is_comet: jax.Array,
    comet_remaining_life: jax.Array,
    safety_margin: int = _COMET_EXPIRY_SAFETY,
) -> jax.Array:
    """Mirror `target_reachable_before_comet_expiry`. Non-comet -> always True."""
    reachable = comet_remaining_life > (predicted_turns + safety_margin)
    return jnp.where(is_comet, reachable, True)


def fleet_crosses_other_comet_jax(
    launch_x: jax.Array,
    launch_y: jax.Array,
    angle: jax.Array,
    turns: jax.Array,
    ships: jax.Array,
    exclude_planet_id: jax.Array,
    other_paths: jax.Array,
    other_path_index: jax.Array,
    other_path_len: jax.Array,
    other_planet_id: jax.Array,
    comet_radius: jax.Array,
    safety: float = _COMET_CROSS_SAFETY,
) -> jax.Array:
    """Mirror `fleet_crosses_other_comet`. Returns True if the fleet line crosses
    any *other* active comet planet's swept path within `comet_radius + safety`.

    `other_*` are host-resolved fixed arrays over MAX_OTHER_COMETS comet planets
    (`other_path_len == 0` marks an empty slot). The target itself is excluded by
    matching `other_planet_id == exclude_planet_id`.
    """
    speed = fleet_speed_jax(jnp.maximum(jnp.int32(1), ships))
    dx = jnp.cos(angle)
    dy = jnp.sin(angle)
    threshold = comet_radius + safety

    def per_comet(
        cpath: jax.Array, cidx: jax.Array, clen: jax.Array, cid: jax.Array
    ) -> jax.Array:
        active_comet = (clen > 0) & (cid != exclude_planet_id)

        def cstep(
            prev: tuple[jax.Array, jax.Array, jax.Array], t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            prev_fx, prev_fy, prev_cidx_valid = prev
            # current comet pos at turn t: path[cidx + t]
            fut = cidx + t
            in_range = (fut >= 0) & (fut < clen)
            safe_i = jnp.clip(fut, 0, MAX_COMET_PATH_LEN - 1)
            cx = cpath[safe_i, 0]
            cy = cpath[safe_i, 1]
            # previous comet pos at t-1
            fut_p = cidx + (t - 1)
            in_range_p = (fut_p >= 0) & (fut_p < clen)
            safe_ip = jnp.clip(fut_p, 0, MAX_COMET_PATH_LEN - 1)
            cpx = cpath[safe_ip, 0]
            cpy = cpath[safe_ip, 1]

            tf = t.astype(jnp.float32)
            fx = launch_x + dx * speed * tf
            fy = launch_y + dy * speed * tf

            within_turns = t <= turns
            # Python breaks the t-loop when comet pos None: once out of range we
            # stop counting hits (mirror via a running "still valid" flag).
            still_valid = prev_cidx_valid & in_range & in_range_p
            hit = _swept_pair_hit_jax(
                prev_fx, prev_fy, fx, fy, cpx, cpy, cx, cy, threshold
            )
            crossed = active_comet & within_turns & still_valid & hit
            return (fx, fy, still_valid), crossed

        init = (launch_x, launch_y, jnp.bool_(True))
        _final, crossed_seq = jax.lax.scan(cstep, init, _CROSS_TURNS)
        return jnp.any(crossed_seq)

    crossed_any = jax.vmap(per_comet)(
        other_paths, other_path_index, other_path_len, other_planet_id
    )
    return (turns >= 0) & jnp.any(crossed_any)


def _dist(ax: jax.Array, ay: jax.Array, bx: jax.Array, by: jax.Array) -> jax.Array:
    return jnp.hypot(ax - bx, ay - by)


def plan_shot_jax(
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
    # target comet path (for aim + intercept + reachability)
    path: jax.Array,
    path_index: jax.Array,
    path_len: jax.Array,
    target_id: jax.Array,
    target_is_comet: jax.Array,
    target_comet_life: jax.Array,
    # other comets (for fleet-crossing)
    other_paths: jax.Array,
    other_path_index: jax.Array,
    other_path_len: jax.Array,
    other_planet_id: jax.Array,
    comet_radius: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Mirror `WorldModel.plan_shot`. Returns `(angle, turns, ix, iy, valid)`.

    `valid == False` is the Python `None` (aim failed OR a safety gate rejected).
    """
    angle, turns, ix, iy, aim_valid = aim_with_prediction_jax(
        sx,
        sy,
        sr,
        tcx,
        tcy,
        init_x,
        init_y,
        init_r,
        tr,
        ships,
        ang_vel,
        max_turns,
        path,
        path_index,
        path_len,
    )

    clearance = sr + LAUNCH_CLEARANCE
    launch_x = sx + jnp.cos(angle) * clearance
    launch_y = sy + jnp.sin(angle) * clearance

    sun_safe = is_trajectory_sun_safe_jax(launch_x, launch_y, angle, turns, ships)
    intercept_ok = intercept_holds_within_tolerance_jax(
        tcx,
        tcy,
        tr,
        turns,
        ix,
        iy,
        init_x,
        init_y,
        init_r,
        ang_vel,
        path,
        path_index,
        path_len,
    )
    reachable = target_reachable_before_comet_expiry_jax(
        turns, target_is_comet, target_comet_life
    )
    crosses = fleet_crosses_other_comet_jax(
        launch_x,
        launch_y,
        angle,
        turns,
        ships,
        target_id,
        other_paths,
        other_path_index,
        other_path_len,
        other_planet_id,
        comet_radius,
    )

    valid = aim_valid & sun_safe & intercept_ok & reachable & (~crosses)
    return angle, turns, ix, iy, valid


__all__ = [
    "MAX_OTHER_COMETS",
    "fleet_crosses_other_comet_jax",
    "intercept_holds_within_tolerance_jax",
    "is_trajectory_sun_safe_jax",
    "plan_shot_jax",
    "target_reachable_before_comet_expiry_jax",
]
