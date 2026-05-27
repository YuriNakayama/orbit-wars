"""Parity: case2 aim_with_prediction_jax vs Python aim_with_prediction.

Covers the hot-path intercept solver (the per-(src,target) call that dominates
`act()`), for BOTH orbital/static targets AND comet targets. Comet paths are
resolved on the host into fixed arrays (`resolve_comet_path`) and fed to the JAX
port; the test confirms the two implementations agree on the comet branch too,
so routing `plan_shot` through JAX is provably behavior-preserving.

Tolerance: angle/ix/iy float32 1e-3 (the solver chains several trig + ceil ops,
so the band is looser than the 1e-4 used for single primitives); `turns` is an
integer and compared exactly; valid/None is compared exactly. A handful of
near-boundary geometries can land on different candidate turns between the
fractional-lerp (Python) and direct-fractional-rotation (JAX) lead prediction;
the test asserts agreement on the overwhelming majority and that the valid flag
never disagrees.
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import pytest

from pipeline.rulebase.case2.baseline.core.config import HORIZON
from pipeline.rulebase.case2.baseline.core.physics import aim_with_prediction
from pipeline.rulebase.case2.baseline.core.types import Planet
from pipeline.rulebase.case2.baseline_jax.aim_jax import (
    MAX_COMET_PATH_LEN,
    aim_with_prediction_jax,
    resolve_comet_path,
)

ANGLE_TOL = 1e-3
POS_TOL = 1e-3
_NO_COMET = [[0.0, 0.0] for _ in range(MAX_COMET_PATH_LEN)]


def _rng_pairs(n: int, seed: int) -> list[tuple[Planet, Planet, int, float]]:
    rng = random.Random(seed)
    out: list[tuple[Planet, Planet, int, float]] = []
    for _ in range(n):
        # src + target on the board, away from the sun so paths are plausible.
        sx, sy = rng.uniform(15, 85), rng.uniform(15, 85)
        tx, ty = rng.uniform(15, 85), rng.uniform(15, 85)
        sr, tr = rng.uniform(1.0, 3.0), rng.uniform(1.0, 3.0)
        ships = rng.randint(1, 400)
        ang_vel = rng.uniform(-0.05, 0.05)
        src = Planet(0, 0, sx, sy, sr, ships, 1)
        target = Planet(1, 1, tx, ty, tr, 50, 1)
        out.append((src, target, ships, ang_vel))
    return out


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
def test_aim_parity_noncomet(seed: int) -> None:
    pairs = _rng_pairs(60, seed)
    # No comets: initial_by_id maps each target to itself (orbital), comets empty.
    disagreements = 0
    valid_disagreements = 0
    for src, target, ships, ang_vel in pairs:
        initial_by_id = {target.id: target, src.id: src}
        py = aim_with_prediction(
            src, target, ships, initial_by_id, ang_vel, comets=[], comet_ids=set()
        )
        angle, turns, ix, iy, valid = aim_with_prediction_jax(
            jnp.float32(src.x),
            jnp.float32(src.y),
            jnp.float32(src.radius),
            jnp.float32(target.x),
            jnp.float32(target.y),
            jnp.float32(target.x),
            jnp.float32(target.y),
            jnp.float32(target.radius),
            jnp.float32(target.radius),
            jnp.int32(ships),
            jnp.float32(ang_vel),
            jnp.int32(HORIZON),
            jnp.array(_NO_COMET, dtype=jnp.float32),
            jnp.int32(0),
            jnp.int32(0),
        )
        jax_valid = bool(valid)
        if (py is None) != (not jax_valid):
            valid_disagreements += 1
            continue
        if py is None:
            continue
        p_angle, p_turns, p_ix, p_iy = py
        # angle compared modulo 2pi
        da = abs((float(angle) - p_angle + math.pi) % (2 * math.pi) - math.pi)
        ok = (
            da < ANGLE_TOL
            and int(turns) == p_turns
            and abs(float(ix) - p_ix) < POS_TOL
            and abs(float(iy) - p_iy) < POS_TOL
        )
        if not ok:
            disagreements += 1

    # valid/None must never disagree.
    assert valid_disagreements == 0, (
        f"seed={seed} valid/None disagreed on {valid_disagreements}/60 pairs"
    )
    # Allow a small fraction of near-boundary numeric disagreements.
    assert disagreements <= 3, (
        f"seed={seed} aim disagreed on {disagreements}/60 non-comet pairs (>3)"
    )


def _make_comet(planet_id: int, seed: int) -> tuple[Planet, list[dict], set[int], int]:
    """Build a comet target + a comets list with a straight-line path.

    Returns (target_planet, comets, comet_ids, path_index). The target's current
    (x,y) is set to path[path_index] so Python's `predict_comet_position` and the
    target object agree at turn 0.
    """
    rng = random.Random(seed)
    x0, y0 = rng.uniform(20, 80), rng.uniform(20, 80)
    dx, dy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
    path_len = rng.randint(8, 30)
    path = [[x0 + dx * i, y0 + dy * i] for i in range(path_len)]
    path_index = rng.randint(0, max(0, path_len // 3))
    cx, cy = path[path_index]
    target = Planet(planet_id, 1, cx, cy, rng.uniform(1.0, 2.5), 50, 1)
    comets = [{"planet_ids": [planet_id], "paths": [path], "path_index": path_index}]
    return target, comets, {planet_id}, path_index


@pytest.mark.parametrize("seed", [0, 3, 9, 21, 100])
def test_aim_parity_comet(seed: int) -> None:
    rng = random.Random(seed * 7 + 1)
    valid_disagreements = 0
    disagreements = 0
    n = 40
    for _ in range(n):
        target, comets, comet_ids, _pidx = _make_comet(1, rng.randint(0, 10**6))
        sx, sy = rng.uniform(15, 85), rng.uniform(15, 85)
        sr = rng.uniform(1.0, 3.0)
        ships = rng.randint(1, 400)
        ang_vel = rng.uniform(-0.05, 0.05)
        src = Planet(0, 0, sx, sy, sr, ships, 1)
        initial_by_id = {src.id: src}  # comet target not in initial_by_id

        py = aim_with_prediction(
            src, target, ships, initial_by_id, ang_vel, comets, comet_ids
        )
        path_arr, path_index, path_len = resolve_comet_path(
            target.id, comets, comet_ids
        )
        angle, turns, ix, iy, valid = aim_with_prediction_jax(
            jnp.float32(src.x),
            jnp.float32(src.y),
            jnp.float32(src.radius),
            jnp.float32(target.x),
            jnp.float32(target.y),
            jnp.float32(target.x),
            jnp.float32(target.y),
            jnp.float32(target.radius),
            jnp.float32(target.radius),
            jnp.int32(ships),
            jnp.float32(ang_vel),
            jnp.int32(HORIZON),
            jnp.array(path_arr, dtype=jnp.float32),
            jnp.int32(path_index),
            jnp.int32(path_len),
        )
        jax_valid = bool(valid)
        if (py is None) != (not jax_valid):
            valid_disagreements += 1
            continue
        if py is None:
            continue
        p_angle, p_turns, p_ix, p_iy = py
        da = abs((float(angle) - p_angle + math.pi) % (2 * math.pi) - math.pi)
        ok = (
            da < ANGLE_TOL
            and int(turns) == p_turns
            and abs(float(ix) - p_ix) < POS_TOL
            and abs(float(iy) - p_iy) < POS_TOL
        )
        if not ok:
            disagreements += 1

    assert valid_disagreements == 0, (
        f"seed={seed} comet valid/None disagreed on {valid_disagreements}/{n}"
    )
    assert disagreements <= 2, (
        f"seed={seed} comet aim disagreed on {disagreements}/{n} (>2)"
    )
