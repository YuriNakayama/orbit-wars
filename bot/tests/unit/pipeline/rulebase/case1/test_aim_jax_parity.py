"""x64 parity: aim_jax.aim_with_prediction vs Python (NON-COMET targets).

The intricate core piece: 5-iter refine + search_safe_intercept fallback.
Covers static and rotating planets (comet paths are host-resolved separately).
Equivalence required: angle/intercept-pos float64-close, turns exact, valid flag
matches Python None-vs-result.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline.core import physics as ppy
from pipeline.rulebase.case1.baseline.core.types import Planet
from pipeline.rulebase.case1.baseline_jax.core_jax import aim_jax

RTOL = 1e-7
ATOL = 1e-7


def _call_jax(src: Planet, tgt: Planet, init: Planet, ships: int, ang_vel: float):
    return aim_jax.aim_with_prediction(
        jnp.asarray(src.x),
        jnp.asarray(src.y),
        jnp.asarray(src.radius),
        jnp.asarray(tgt.x),
        jnp.asarray(tgt.y),
        jnp.asarray(init.x),
        jnp.asarray(init.y),
        jnp.asarray(init.radius),
        jnp.asarray(tgt.radius),
        jnp.asarray(ships),
        jnp.asarray(ang_vel),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_aim_with_prediction_parity_planets(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ang_vel = 0.035
    mismatches = []
    n = 80
    for _ in range(n):
        sx, sy, tx, ty = rng.uniform(5, 95, 4)
        sr, tr = rng.uniform(1, 4, 2)
        ships = int(rng.integers(1, 300))
        # target's "initial" position == current (turn snapshot); rotating if
        # within ROTATION_LIMIT. Use current pos as init so predict is exact.
        src = Planet(id=0, owner=0, x=sx, y=sy, radius=sr, ships=ships, production=2)
        tgt = Planet(id=1, owner=-1, x=tx, y=ty, radius=tr, ships=5, production=2)
        init = Planet(id=1, owner=-1, x=tx, y=ty, radius=tr, ships=5, production=2)

        ref = ppy.aim_with_prediction(src, tgt, ships, {1: init}, ang_vel, [], set())
        angle, turns, ix, iy, valid = _call_jax(src, tgt, init, ships, ang_vel)

        if ref is None:
            if bool(valid):
                mismatches.append(("py=None jax=valid", float(angle), int(turns)))
            continue
        if not bool(valid):
            mismatches.append(("py=valid jax=None", ref))
            continue
        ok = (
            np.isclose(ref[0], float(angle), rtol=RTOL, atol=ATOL)
            and int(ref[1]) == int(turns)
            and np.isclose(ref[2], float(ix), rtol=1e-4, atol=1e-4)
            and np.isclose(ref[3], float(iy), rtol=1e-4, atol=1e-4)
        )
        if not ok:
            mismatches.append(
                (
                    f"ref={ref}",
                    f"jax=({float(angle):.5f},{int(turns)},{float(ix):.3f},{float(iy):.3f})",
                )
            )

    assert not mismatches, (
        f"seed={seed}: {len(mismatches)}/{n} mismatches: {mismatches[:5]}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aim_with_prediction_parity_rotating(seed: int) -> None:
    """Rotating targets (within ROTATION_LIMIT) exercise the refine loop."""
    rng = np.random.default_rng(seed + 100)
    ang_vel = 0.04
    mismatches = []
    n = 60
    refine_used = 0
    for _ in range(n):
        sx, sy = rng.uniform(5, 95, 2)
        # place target near center so it rotates (r + radius < 50)
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(15, 35)
        tx, ty = 50 + r * np.cos(ang), 50 + r * np.sin(ang)
        sr, tr = rng.uniform(1, 4, 2)
        ships = int(rng.integers(1, 300))
        src = Planet(id=0, owner=0, x=sx, y=sy, radius=sr, ships=ships, production=2)
        tgt = Planet(id=1, owner=-1, x=tx, y=ty, radius=tr, ships=5, production=2)
        init = Planet(id=1, owner=-1, x=tx, y=ty, radius=tr, ships=5, production=2)
        if not ppy.is_static_planet(tgt):
            refine_used += 1

        ref = ppy.aim_with_prediction(src, tgt, ships, {1: init}, ang_vel, [], set())
        angle, turns, ix, iy, valid = _call_jax(src, tgt, init, ships, ang_vel)

        if ref is None:
            if bool(valid):
                mismatches.append(("py=None jax=valid",))
            continue
        if not bool(valid):
            mismatches.append(("py=valid jax=None", ref))
            continue
        ok = (
            np.isclose(ref[0], float(angle), rtol=RTOL, atol=ATOL)
            and int(ref[1]) == int(turns)
            and np.isclose(ref[2], float(ix), rtol=1e-4, atol=1e-4)
            and np.isclose(ref[3], float(iy), rtol=1e-4, atol=1e-4)
        )
        if not ok:
            mismatches.append((f"ref={ref}", f"jax_turns={int(turns)}"))

    assert refine_used > 0, "test did not exercise any rotating targets"
    assert not mismatches, f"seed={seed}: {len(mismatches)}/{n} ({mismatches[:5]})"
