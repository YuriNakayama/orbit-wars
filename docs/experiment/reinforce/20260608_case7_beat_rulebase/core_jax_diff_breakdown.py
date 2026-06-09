"""Where does core_jax diverge from real v1? Break the 37% mismatch into
source / angle / ships components, on the launches that DO share a source planet."""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline_jax.core_jax.agent_full_jax import (
    compute_actions_jax_jit as core_jax,
)


def p(*a):
    print(*a, flush=True)


def _rows_jax(arr):
    out = []
    for row in np.asarray(arr):
        pid = int(round(float(row[0])))
        if pid >= 0:
            out.append((pid, float(row[1]), int(round(float(row[2])))))
    return out


def _rows_py(moves):
    out = []
    for mv in moves or []:
        pid = int(round(float(mv[0])))
        if pid >= 0:
            out.append((pid, float(mv[1]), int(round(float(mv[2])))))
    return out


def main():
    noop = jnp.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0)
    seat, n = 1, 30
    # for launches sharing (source), how often angle and ships match
    shared_src = 0
    angle_match = 0
    ships_match = 0
    both_match = 0
    angle_diffs = []
    ships_diffs = []
    for s in range(n):
        state = reset(seed=s, num_agents=2)
        for _ in range((s % 12) * 3):
            state, _, term = step(state, noop)
            if bool(term):
                break
        J = _rows_jax(core_jax(state, seat))
        P = _rows_py(v1_py(state_to_obs(state, player=seat)))
        # index python launches by source pid (may be multiple per source)
        from collections import defaultdict

        pj = defaultdict(list)
        for pid, ang, sh in P:
            pj[pid].append((ang, sh))
        for pid, ang, sh in J:
            if pid in pj and pj[pid]:
                # match against closest python launch from same source
                cands = pj[pid]
                best = min(cands, key=lambda c: abs(c[0] - ang) + abs(c[1] - sh))
                shared_src += 1
                da = abs(best[0] - ang)
                ds = abs(best[1] - sh)
                angle_diffs.append(da)
                ships_diffs.append(ds)
                am = da < 1.0
                sm = ds == 0
                angle_match += am
                ships_match += sm
                both_match += am and sm
    p("=== core_jax vs v1: per-source-launch breakdown ===")
    p(f"  shared-source launches: {shared_src}")
    if shared_src:
        p(f"  angle match (<1deg): {angle_match}/{shared_src} ({100*angle_match/shared_src:.0f}%)")
        p(f"  ships match (exact): {ships_match}/{shared_src} ({100*ships_match/shared_src:.0f}%)")
        p(f"  both match:          {both_match}/{shared_src} ({100*both_match/shared_src:.0f}%)")
        p(f"  mean angle diff: {np.mean(angle_diffs):.2f}deg  max {np.max(angle_diffs):.1f}")
        p(f"  mean ships diff: {np.mean(ships_diffs):.2f}  max {np.max(ships_diffs):.0f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
