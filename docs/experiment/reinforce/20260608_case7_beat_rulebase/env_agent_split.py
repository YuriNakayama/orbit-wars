"""Isolate whether the JAX parity gap is in the ENV (obs layer) or the AGENT (rule).

Three quantities on the SAME JAX states (seat 1), printed unbuffered:

  A = compute_actions_jax(state)              [JAX env-state + JAX rule]
  B = py_v1(state_to_obs(state))              [JAX env-state -> obs -> REAL rule]
  (earlier probe = A vs B = 10% match)

A vs B isolates the AGENT layer IF state_to_obs is faithful, because BOTH read the
same JAX state; the only difference is JAX-rule vs Python-rule logic.

To also check the ENV/obs layer, we sanity-check that state_to_obs produces a
self-consistent obs the real agent can act on (no crash, sensible planet counts),
and we report how often the real rule even FIRES on JAX obs vs the JAX rule —
a large fire-rate gap points to obs differences the agent reacts to.
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline_jax_full import compute_actions_jax as jax_full


def p(*a):
    print(*a, flush=True)


def _norm_jax(arr):
    out = set()
    for row in np.asarray(arr):
        pid = int(round(float(row[0])))
        if pid >= 0:
            out.add((pid, int(round(float(row[1]))), int(round(float(row[2])))))
    return out


def _norm_py(moves):
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
    a_set_total = b_set_total = inter_total = 0
    a_src = []  # source-planet-only sets, to test "same target, diff angle/ships"
    b_src = []
    src_match = 0
    fire_a = fire_b = 0
    p("checking obs faithfulness + agent-layer parity on", n, "JAX states")
    for s in range(n):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 3):
            state, _, term = step(state, noop)
            if bool(term):
                break
        obs = state_to_obs(state, player=seat)
        # ENV sanity: obs has the planets the JAX state has
        n_planets_state = int(np.asarray(state.planet_valid).sum())
        n_planets_obs = len(obs.get("planets", []))
        if s < 3:
            p(f"  s{s}: state_planets={n_planets_state} obs_planets={n_planets_obs} step={obs.get('step')}")
        A = _norm_jax(jax_full(state, seat))
        B = _norm_py(v1_py(obs))
        if A:
            fire_a += 1
        if B:
            fire_b += 1
        a_set_total += len(A)
        b_set_total += len(B)
        inter_total += len(A & B)
        # source-planet-only comparison (ignore angle/ships)
        Asrc = {x[0] for x in A}
        Bsrc = {x[0] for x in B}
        if Asrc == Bsrc:
            src_match += 1
        a_src.append(Asrc)
        b_src.append(Bsrc)
    p("")
    p("=== AGENT-LAYER (same JAX state, JAX-rule vs Python-rule) ===")
    p(f"  full-action exact match was ~10% (earlier probe)")
    p(f"  source-planet-set match: {src_match}/{n} ({100*src_match/n:.0f}%)")
    p(f"  fire-rate: JAX-rule {fire_a}/{n}, Python-rule {fire_b}/{n}")
    p(f"  launches: JAX={a_set_total} Python={b_set_total} shared={inter_total}")
    p("")
    p("interpretation:")
    p("  - if source-planet match is HIGH but full match LOW => agent picks same")
    p("    targets but differs on angle/ships (aim/allocation rule divergence).")
    p("  - if source-planet match is also LOW => agent picks different targets")
    p("    (scoring/target-selection rule divergence).")
    p("  - obs_planets == state_planets each row => state_to_obs (env layer) faithful.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
