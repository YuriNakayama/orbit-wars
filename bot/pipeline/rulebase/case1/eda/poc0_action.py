"""PoC0b: can we reproduce the Python turn-0 LAUNCH action, not just available?

This is the real test. available parity at turn0 is trivial (reserve always 0).
The substance is: target selection + send sizing + angle. We instrument the
real agent's turn-0 decision across seeds to understand the decision surface
before committing to the JAX mission/greedy port.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from pipeline.rulebase.case1.baseline.agent import agent as v1_py, build_world

def run(seeds):
    n_one = n_multi = n_zero = 0
    for seed in seeds:
        state = reset(seed=seed, num_agents=2)
        obs = state_to_obs(state, player=0)
        moves = v1_py(obs)
        w = build_world(obs)
        if len(moves) == 0:
            n_zero += 1
        elif len(moves) == 1:
            n_one += 1
        else:
            n_multi += 1
        if seed < 8:
            # show the decision detail
            srcs = {p.id: (int(p.ships), p.production) for p in w.my_planets}
            print(f"seed={seed}: {len(moves)} moves={[(int(m[0]), round(float(m[1]),3), int(m[2])) for m in moves]} my_planets(ships,prod)={srcs}")
    print(f"\nseeds={len(seeds)}: 0-move={n_zero}, 1-move={n_one}, multi-move={n_multi}")

if __name__ == "__main__":
    run(range(0, 50))
