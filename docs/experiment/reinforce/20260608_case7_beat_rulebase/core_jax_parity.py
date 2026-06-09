"""core_jax agent_full parity vs real v1, under float32 vs x64."""

from __future__ import annotations

import os
import sys

# x64 toggle via env BEFORE jax import
if os.environ.get("X64") == "1":
    os.environ["JAX_ENABLE_X64"] = "1"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from orbit_wars_jax.constants import NUM_AGENTS_MAX  # noqa: E402
from orbit_wars_jax.observation import state_to_obs  # noqa: E402
from orbit_wars_jax.reset import reset  # noqa: E402
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step  # noqa: E402

from pipeline.rulebase.case1.baseline.agent import agent as v1_py  # noqa: E402
from pipeline.rulebase.case1.baseline_jax.core_jax.agent_full_jax import (  # noqa: E402
    compute_actions_jax_jit as core_jax,
)


def p(*a):
    print(*a, flush=True)


def _nj(arr):
    out = set()
    for row in np.asarray(arr):
        pid = int(round(float(row[0])))
        if pid >= 0:
            out.add((pid, int(round(float(row[1]))), int(round(float(row[2])))))
    return out


def _np_(moves):
    out = set()
    for mv in moves or []:
        pid = int(round(float(mv[0])))
        if pid >= 0:
            out.add((pid, int(round(float(mv[1]))), int(round(float(mv[2])))))
    return out


def main():
    mode = "x64" if os.environ.get("X64") == "1" else "float32"
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    seat, n = 1, 30
    exact = src = fire_a = fire_b = a_tot = b_tot = shared = 0
    for s in range(n):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 3):
            state, _, term = step(state, noop)
            if bool(term):
                break
        A = _nj(core_jax(state, seat))
        B = _np_(v1_py(state_to_obs(state, player=seat)))
        if A == B:
            exact += 1
        if {x[0] for x in A} == {x[0] for x in B}:
            src += 1
        fire_a += 1 if A else 0
        fire_b += 1 if B else 0
        a_tot += len(A)
        b_tot += len(B)
        shared += len(A & B)
    p(f"=== core_jax vs real v1 [{mode}] ===")
    p(f"  full exact:   {exact}/{n} ({100*exact/n:.0f}%)")
    p(f"  source match: {src}/{n} ({100*src/n:.0f}%)")
    p(f"  fire-rate:    JAX {fire_a}/{n}  Python {fire_b}/{n}")
    p(f"  launches:     JAX={a_tot} Python={b_tot} shared={shared}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
