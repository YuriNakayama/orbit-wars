"""No-degradation sweep: each ported JAX case vs its own Python agent.

10 games/case (5 seeds x 2 seats), foreground (background hangs JAX). Confirms no
case collapsed to ~0-win. Bounded per loop discipline (not a full 300-game eval).
"""
from __future__ import annotations
import importlib
import logging
import jax.numpy as jnp
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def _row(m):
    r = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, x in enumerate(m[:MAX_LAUNCHES_PER_AGENT]):
        r = r.at[i].set(jnp.asarray([x[0], x[1], x[2]], dtype=jnp.float32))
    return r

def load(case):
    py = importlib.import_module(f"pipeline.rulebase.case{case}.baseline.agent").agent
    cj = importlib.import_module(
        f"pipeline.rulebase.case{case}.baseline_jax.core_jax.agent_full_jax"
    ).compute_actions_jax_jit
    return py, cj

def play(py, cj, seed, js):
    st = reset(seed=seed, num_agents=2); ps = 1 - js; rw = None
    for _ in range(500):
        st, rw, term = step(
            st,
            empty_actions().at[js].set(cj(st, seat=js)).at[ps].set(_row(py(state_to_obs(st, player=ps)))),
        )
        if bool(term): break
    if rw is None: return -1
    rj, rp = float(rw[js]), float(rw[ps])
    return js if rj > rp else (ps if rp > rj else -1)

results = {}
for case in (1, 2, 3, 4, 6, 7, 8, 9):
    py, cj = load(case)
    w = g = 0
    for seed in range(5):
        for js in (0, 1):
            win = play(py, cj, seed, js); g += 1; w += (win == js)
    rate = 100.0 * w / g
    results[case] = (w, g, rate)
    log.info("case%d: JAX %d/%d = %.0f%%", case, w, g, rate)
log.info("=== SWEEP DONE ===")
for c, (w, g, r) in results.items():
    flag = "DEGRADED" if r < 20 else "ok"
    log.info("case%d %d/%d %.0f%% [%s]", c, w, g, r, flag)
