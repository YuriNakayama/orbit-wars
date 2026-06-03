"""Parity: swarm_jax.allocate_2 vs Python process_multi_source_mission allocation.

Standalone check of the ordered 2-source allocation before wiring into the agent
(a naive split regressed the win-rate). Reference replicates mission_resolver.py
:108-118 for 2 sources.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from pipeline.rulebase.case1.baseline_jax.core_jax import swarm_jax as sw


def _py_allocate_2(
    need: int,
    lim_a: int,
    turns_a: int,
    src_a: int,
    lim_b: int,
    turns_b: int,
    src_b: int,
) -> tuple[int, int, bool]:
    # (limit, turns, src, key) tuples — homogeneous so mypy infers int/str cleanly.
    opts: list[tuple[int, int, int, str]] = [
        (lim_a, turns_a, src_a, "a"),
        (lim_b, turns_b, src_b, "b"),
    ]
    ordered = sorted(opts, key=lambda o: (o[1], -o[0], o[2]))
    remaining = need
    sends: dict[str, int] = {}
    for idx, o in enumerate(ordered):
        remaining_other = sum(x[0] for x in ordered[idx + 1 :])
        send = min(o[0], max(0, remaining - remaining_other))
        sends[o[3]] = send
        remaining -= send
    if remaining > 0:
        return 0, 0, False
    return sends["a"], sends["b"], True


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_allocate_2_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mism: list[Any] = []
    n = 600
    for _ in range(n):
        need = int(rng.integers(1, 80))
        la, lb = int(rng.integers(0, 60)), int(rng.integers(0, 60))
        ta, tb = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        sa, sb = int(rng.integers(0, 48)), int(rng.integers(0, 48))
        if sa == sb:
            sb = (sb + 1) % 48
        ra, rb, rok = _py_allocate_2(need, la, ta, sa, lb, tb, sb)
        ja, jb, jok = sw.allocate_2(
            jnp.asarray(need),
            jnp.asarray(la),
            jnp.asarray(ta),
            jnp.asarray(sa),
            jnp.asarray(lb),
            jnp.asarray(tb),
            jnp.asarray(sb),
        )
        if rok != bool(jok) or (rok and (ra != int(ja) or rb != int(jb))):
            mism.append(
                (
                    need,
                    la,
                    ta,
                    sa,
                    lb,
                    tb,
                    sb,
                    (ra, rb, rok),
                    (int(ja), int(jb), bool(jok)),
                )
            )
    assert not mism, f"seed={seed}: {len(mism)}/{n}: {mism[:3]}"
