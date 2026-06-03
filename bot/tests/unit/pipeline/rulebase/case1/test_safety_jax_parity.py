"""x64 parity: safety_jax guards vs baseline/core/safety.py (over-fire fix lock-in).

These two guards (is_trajectory_sun_safe + intercept_holds_within_tolerance) are
what WorldModel.plan_shot applies after aim_with_prediction; porting them was the
fix that eliminated the JAX agent's over-fire (the ~0-win failure mode). This
unit test locks them at parity so the fix can never silently regress (currently
only the slow 10-game e2e guards it).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline.core import safety as spy
from pipeline.rulebase.case1.baseline.core.types import Planet
from pipeline.rulebase.case1.baseline_jax.core_jax import safety_jax as sjax


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_is_trajectory_sun_safe_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mism = []
    n = 400
    for _ in range(n):
        lx, ly = rng.uniform(0, 100, 2)
        angle = rng.uniform(-np.pi, np.pi)
        turns = int(rng.integers(0, 60))
        ships = int(rng.integers(1, 300))
        ref = spy.is_trajectory_sun_safe(lx, ly, angle, turns, ships)
        got = bool(
            sjax.is_trajectory_sun_safe(
                jnp.asarray(lx),
                jnp.asarray(ly),
                jnp.asarray(angle),
                jnp.asarray(turns),
                jnp.asarray(ships),
            )
        )
        if ref != got:
            mism.append((lx, ly, angle, turns, ships, ref, got))
    assert not mism, f"seed={seed}: {len(mism)}/{n}: {mism[:5]}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_intercept_holds_within_tolerance_parity(seed: int) -> None:
    rng = np.random.default_rng(seed + 30)
    ang_vel = 0.04
    mism = []
    n = 300
    for _ in range(n):
        # target near center → rotating (exercises the tolerance check)
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(12, 38)
        tx, ty = 50 + r * np.cos(a), 50 + r * np.sin(a)
        tr = rng.uniform(1, 4)
        turns = int(rng.integers(1, 50))
        # predicted pos: the true future pos (so it usually holds) or a jittered
        # one (so it sometimes fails) — mix both.
        cur = Planet(id=1, owner=-1, x=tx, y=ty, radius=tr, ships=5, production=2)
        true_pos = spy._predict_planet_position(cur, cur, ang_vel, turns)
        jitter = rng.uniform(-1.5, 1.5, 2)
        px, py = true_pos[0] + jitter[0], true_pos[1] + jitter[1]
        ref = spy.intercept_holds_within_tolerance(
            cur, turns, (px, py), {1: cur}, ang_vel, [], set()
        )
        got = bool(
            sjax.intercept_holds_within_tolerance(
                jnp.asarray(tx),
                jnp.asarray(ty),
                jnp.asarray(tx),
                jnp.asarray(ty),
                jnp.asarray(tr),
                jnp.asarray(tr),
                jnp.asarray(turns),
                jnp.asarray(px),
                jnp.asarray(py),
                jnp.asarray(ang_vel),
            )
        )
        if ref != got:
            mism.append((turns, round(px, 2), round(py, 2), ref, got))
    assert not mism, f"seed={seed}: {len(mism)}/{n}: {mism[:5]}"
