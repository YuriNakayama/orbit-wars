"""Parity: case8 `plan_shot_jax` vs Python `WorldModel.plan_shot`.

`plan_shot` = aim + 4 post-aim safety gates (sun-safe / intercept tolerance /
comet expiry / other-comet crossing). The aim itself is covered by
`test_aim_jax_parity`; this asserts the *combined* gate decision matches on
realistic boards (reset jax_env states), across a (src x target x ships) grid.

Tolerance: the boolean `valid` (Python None <=> JAX valid==False) must match on
the large majority; angle/turns/ix/iy compared with float32 tol when both valid.
A handful of near-boundary numeric flips are allowed (the same band the aim test
uses), since a 1e-3 intercept-tolerance flip can change a downstream gate.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.rulebase.case8.baseline.agent import build_world
from pipeline.rulebase.case8.baseline.core.config import HORIZON
from pipeline.rulebase.case8.baseline.core.physics import (
    comet_remaining_life,
    reset_predict_cache,
)
from pipeline.rulebase.case8.baseline_jax.aim_jax import (
    MAX_COMET_PATH_LEN,
    resolve_comet_path,
)
from pipeline.rulebase.case8.baseline_jax.plan_shot_jax import (
    MAX_OTHER_COMETS,
    plan_shot_jax,
)

pytestmark = pytest.mark.slow

ANGLE_TOL = 1e-2
POS_TOL = 1e-2
_ZEROS_PATH = [[0.0, 0.0] for _ in range(MAX_COMET_PATH_LEN)]

_jitted = jax.jit(plan_shot_jax)


def _resolve_other_comets(
    comets: list[dict], comet_ids: set[int]
) -> tuple[list[list[list[float]]], list[int], list[int], list[int]]:
    """Host-resolve every active comet planet into fixed MAX_OTHER_COMETS slots."""
    paths: list[list[list[float]]] = []
    idxs: list[int] = []
    lens: list[int] = []
    ids: list[int] = []
    for cid in sorted(comet_ids):
        p, pi, pl = resolve_comet_path(cid, comets, comet_ids)
        paths.append(p)
        idxs.append(pi)
        lens.append(pl)
        ids.append(cid)
    while len(paths) < MAX_OTHER_COMETS:
        paths.append([row[:] for row in _ZEROS_PATH])
        idxs.append(0)
        lens.append(0)
        ids.append(-1)
    return (
        paths[:MAX_OTHER_COMETS],
        idxs[:MAX_OTHER_COMETS],
        lens[:MAX_OTHER_COMETS],
        ids[:MAX_OTHER_COMETS],
    )


def _call_jax(world, src, target, ships: int):
    init = world.initial_by_id.get(target.id)
    init_x = float(init.x) if init is not None else float(target.x)
    init_y = float(init.y) if init is not None else float(target.y)
    init_r = float(init.radius) if init is not None else float(target.radius)

    is_comet = target.id in world.comet_ids
    path, pidx, plen = resolve_comet_path(target.id, world.comets, world.comet_ids)
    max_turns = HORIZON
    comet_life = 0
    if is_comet:
        comet_life = comet_remaining_life(target.id, world.comets)
        max_turns = min(max_turns, max(0, comet_life - 1))

    o_paths, o_idx, o_len, o_id = _resolve_other_comets(world.comets, world.comet_ids)

    return _jitted(
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
        jnp.float32(world.ang_vel),
        jnp.int32(max_turns),
        jnp.asarray(path, dtype=jnp.float32),
        jnp.int32(pidx),
        jnp.int32(plen),
        jnp.int32(target.id),
        jnp.bool_(is_comet),
        jnp.int32(comet_life),
        jnp.asarray(o_paths, dtype=jnp.float32),
        jnp.asarray(o_idx, dtype=jnp.int32),
        jnp.asarray(o_len, dtype=jnp.int32),
        jnp.asarray(o_id, dtype=jnp.int32),
        jnp.float32(2.0),  # _COMET_RADIUS_DEFAULT
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_plan_shot_parity(seed: int) -> None:
    state = reset(seed=seed, num_agents=2)
    valid_disagree = 0
    value_disagree = 0
    compared = 0

    # Advance a few steps so the board has fleets / varied positions.
    from orbit_wars_jax.step import empty_actions, step

    for _ in range(6):
        obs = state_to_obs(state, player=0)
        reset_predict_cache()
        world = build_world(obs)
        ships_grid = [3, 12, 40]
        for src in world.my_planets[:4]:
            for target in world.planets[:8]:
                if target.id == src.id:
                    continue
                for ships in ships_grid:
                    reset_predict_cache()
                    py = world.plan_shot(src.id, target.id, ships)
                    angle, turns, ix, iy, valid = _call_jax(world, src, target, ships)
                    jax_valid = bool(valid)
                    compared += 1
                    if (py is None) != (not jax_valid):
                        valid_disagree += 1
                        continue
                    if py is None:
                        continue
                    pa, pt, pix, piy = py
                    da = abs((float(angle) - pa + math.pi) % (2 * math.pi) - math.pi)
                    ok = (
                        da < ANGLE_TOL
                        and int(turns) == pt
                        and abs(float(ix) - pix) < POS_TOL
                        and abs(float(iy) - piy) < POS_TOL
                    )
                    if not ok:
                        value_disagree += 1
        state, _r, term = step(state, empty_actions())
        if bool(term):
            break

    assert compared > 0
    vfrac = valid_disagree / compared
    dfrac = value_disagree / max(1, compared)
    # valid/None decision is the gate that matters; allow <=2% boundary flips.
    assert vfrac <= 0.02, (
        f"seed={seed}: valid/None disagreed on {valid_disagree}/{compared} "
        f"({100 * vfrac:.1f}%)"
    )
    assert dfrac <= 0.02, (
        f"seed={seed}: value disagreed on {value_disagree}/{compared} "
        f"({100 * dfrac:.1f}%)"
    )


def _quiet() -> None:
    _ = np
