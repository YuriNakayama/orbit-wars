"""Measure case8 JAX rule parity vs real Python baseline_v8 on the same JAX states.

Mirrors env_agent_split: source-planet match, fire-rate, shared launches.
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case8.baseline.agent import agent as v8_py
from pipeline.rulebase.case8.baseline_jax.agent_jax import (
    _modes_from_features,
    compute_actions,
)
from pipeline.rulebase.case8.baseline_jax.world_features import (
    build_world_features_from_state,
)


def p(*a):
    print(*a, flush=True)


def jax8(state, seat):
    feats = build_world_features_from_state(state, seat)
    return compute_actions(feats, _modes_from_features(feats))


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
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    seat = 1
    n = 30
    exact = src_match = fire_a = fire_b = a_tot = b_tot = shared = 0
    p("case8 JAX rule vs real baseline_v8 on", n, "JAX states")
    for s in range(n):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 3):
            state, _, term = step(state, noop)
            if bool(term):
                break
        A = _nj(jax8(state, seat))
        B = _np_(v8_py(state_to_obs(state, player=seat)))
        if A == B:
            exact += 1
        if {x[0] for x in A} == {x[0] for x in B}:
            src_match += 1
        fire_a += 1 if A else 0
        fire_b += 1 if B else 0
        a_tot += len(A)
        b_tot += len(B)
        shared += len(A & B)
        if s < 4:
            p(f"  s{s}: jax={sorted(A)[:3]} py={sorted(B)[:3]}")
    p("")
    p(f"=== case8 JAX vs real v8 ===")
    p(f"  full exact:    {exact}/{n} ({100*exact/n:.0f}%)")
    p(f"  source match:  {src_match}/{n} ({100*src_match/n:.0f}%)")
    p(f"  fire-rate:     JAX {fire_a}/{n}  Python {fire_b}/{n}")
    p(f"  launches:      JAX={a_tot} Python={b_tot} shared={shared}")
    p("")
    p("compare to case1 baseline_jax_full vs v1: src 63%, fire 16vs27, shared 0")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
