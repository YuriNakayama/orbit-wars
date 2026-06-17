# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Host-side adapter: call `aim_with_prediction_jax` from the Python WorldModel.

This is the wiring that makes the agent's hot path actually run in JAX. The
Python `WorldModel.plan_shot` calls `aim_jax(...)` here instead of the pure-
Python `aim_with_prediction`; we marshal the Python `Planet` objects + comet
dicts into the JAX function's array arguments (resolving the ragged comet path
on the host via `resolve_comet_path`), invoke the jitted solver, and marshal the
result back into the same `(angle, turns, ix, iy) | None` tuple the Python
version returns.

`aim_with_prediction_jax` is wrapped in `jax.jit` once at import (its argument
shapes are static — all scalars except the fixed `(MAX_COMET_PATH_LEN, 2)`
path), so repeated calls reuse one compiled kernel.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ...baseline.core.physics import comet_remaining_life
from ...baseline.core.types import Planet
from ._config_compat import HORIZON
from .aim_jax import MAX_COMET_PATH_LEN, aim_with_prediction_jax, resolve_comet_path

_jitted_aim = jax.jit(aim_with_prediction_jax)


def aim_jax(
    src: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Planet],
    ang_vel: float,
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[float, int, float, float] | None:
    """Drop-in for `aim_with_prediction`, backed by the JAX solver.

    Resolves the target's orbital `initial` (the host `dict[int, Planet]` lookup
    that `predict_planet_position` does) and its comet path, then calls the
    jitted JAX aim. Returns `(angle, turns, ix, iy)` or None — identical contract
    to the Python `aim_with_prediction`.
    """
    init = initial_by_id.get(target.id)
    # predict_planet_position uses `init` for the rotation radius; when the
    # target is unknown the Python falls back to the current position, which we
    # reproduce by passing the current x/y/radius as the "initial".
    init_x = float(init.x) if init is not None else float(target.x)
    init_y = float(init.y) if init is not None else float(target.y)
    init_r = float(init.radius) if init is not None else float(target.radius)

    path_arr, path_index, path_len = resolve_comet_path(target.id, comets, comet_ids)
    # Comet max_turns clamp: Python caps the sweep at comet_remaining_life-1.
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    angle, turns, ix, iy, valid = _jitted_aim(
        jnp.float32(src.x),
        jnp.float32(src.y),
        jnp.float32(src.radius),
        jnp.float32(target.x),
        jnp.float32(target.y),
        jnp.float32(init_x),
        jnp.float32(init_y),
        jnp.float32(init_r),
        jnp.float32(target.radius),
        jnp.int32(ships),
        jnp.float32(ang_vel),
        jnp.int32(max_turns),
        jnp.asarray(path_arr, dtype=jnp.float32),
        jnp.int32(path_index),
        jnp.int32(path_len),
    )
    if not bool(valid):
        return None
    return float(angle), int(turns), float(ix), float(iy)


_grid_aim = jax.jit(
    jax.vmap(
        aim_with_prediction_jax,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0),
    )
)


def aim_jax_grid(
    rows: list[tuple[float, ...]],
    paths: list[list[list[float]]],
    ang_vel: float,
) -> list[tuple[float, int, float, float] | None]:
    """Batched aim over many rows in ONE vmap call.

    Each row is the flat tuple
    (sx, sy, sr, tcx, tcy, init_x, init_y, init_r, tr, ships, max_turns,
    path_index, path_len) already host-resolved by the caller; `paths[i]` is the
    matching (MAX_COMET_PATH_LEN, 2) comet path. Returns one
    `(angle, turns, ix, iy) | None` per row, identical to calling `aim_jax` per
    row — but a single jit-dispatched kernel for the whole grid instead of N
    round-trips. This is what makes the agent's O(P^2) probe sweep fast: collect
    the grid, resolve comets/init on the host, vmap once.
    """
    if not rows:
        return []
    cols = [jnp.asarray([r[i] for r in rows], dtype=jnp.float32) for i in range(9)]
    sx, sy, sr, tcx, tcy, ix0, iy0, ir, tr = cols
    ships = jnp.asarray([r[9] for r in rows], dtype=jnp.int32)
    max_turns = jnp.asarray([r[10] for r in rows], dtype=jnp.int32)
    pidx = jnp.asarray([r[11] for r in rows], dtype=jnp.int32)
    plen = jnp.asarray([r[12] for r in rows], dtype=jnp.int32)
    paths_arr = jnp.asarray(paths, dtype=jnp.float32)
    angle, turns, ix, iy, valid = _grid_aim(
        sx,
        sy,
        sr,
        tcx,
        tcy,
        ix0,
        iy0,
        ir,
        tr,
        ships,
        jnp.float32(ang_vel),
        max_turns,
        paths_arr,
        pidx,
        plen,
    )
    out: list[tuple[float, int, float, float] | None] = []
    for i in range(len(rows)):
        if bool(valid[i]):
            out.append((float(angle[i]), int(turns[i]), float(ix[i]), float(iy[i])))
        else:
            out.append(None)
    return out


__all__ = ["aim_jax", "aim_jax_grid", "MAX_COMET_PATH_LEN"]
