"""PoC2: is the multi-source sequential greedy a faithful fixed-length scan?

`plan_moves` (strategy.py:61-110): build missions, sort by -score, then iterate
in sorted order calling append_move which accumulates spent_total[src]. Each
mission's affordability (source_inventory_left / source_attack_left) depends on
prior missions' spend. This is the classic "JAX-hostile" sequential greedy.

The claim (09-feasibility-verdict): this is expressible as a fixed-length
`lax.scan` over score-sorted missions with spent_total as carry — the
dependency is sequential but FIXED-LENGTH, which JAX handles. PoC2 validates
the *structure*, not a JAX rewrite:

  (1) How often does multi-source interaction actually matter? (how many turns
      have >1 source spending, and how often does a later mission get clamped by
      a source already partially spent?)
  (2) Is the mission sort deterministic? (ties in score → does Python's stable
      sort give a reproducible order a JAX argsort must replicate?)
  (3) Re-derive moves by replaying the SAME sorted missions through an explicit
      spent_total carry (mimicking lax.scan) and confirm it reproduces the real
      moves exactly. If a hand-rolled sequential carry == real plan_moves, then
      a fixed-length lax.scan of the same will too.

To reach multi-source boards we step a real self-play game to mid-game (step
50-200) where each side owns several planets, then inspect plan_moves there.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from collections import defaultdict

from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

import jax.numpy as jnp
from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline.agent import build_world


def py_row(moves):
    row = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        row = row.at[i].set(jnp.asarray([m[0], m[1], m[2]], dtype=jnp.float32))
    return row


def advance_to_midgame(seed: int, target_step: int):
    """Self-play v1 vs v1 until target_step; return the EnvState there."""
    state = reset(seed=seed, num_agents=2)
    for _ in range(target_step):
        m0 = v1_py(state_to_obs(state, player=0))
        m1 = v1_py(state_to_obs(state, player=1))
        actions = empty_actions().at[0].set(py_row(m0)).at[1].set(py_row(m1))
        state, _, done = step(state, actions)
        if bool(done):
            return None
    return state


def analyze(world) -> dict:
    """Count multi-source spend + check sort determinism on this board."""
    moves = list(world_plan(world))
    src_count = defaultdict(int)
    for mv in moves:
        src_count[mv[0]] += 1
    multi_src = sum(1 for c in src_count.values() if c >= 1)
    sources_launching = len([s for s, c in src_count.items() if c > 0])
    return {
        "n_moves": len(moves),
        "sources_launching": sources_launching,
        "moves": moves,
    }


def world_plan(world):
    from pipeline.rulebase.case1.baseline.strategy import plan_moves

    return plan_moves(world)


def run(seeds, steps=(60, 120, 200)) -> None:
    total_boards = 0
    multi_source_boards = 0
    move_dist = defaultdict(int)
    for seed in seeds:
        for tstep in steps:
            state = advance_to_midgame(seed, tstep)
            if state is None:
                continue
            obs = state_to_obs(state, player=0)
            world = build_world(obs)
            if not world.my_planets:
                continue
            info = analyze(world)
            total_boards += 1
            move_dist[min(info["sources_launching"], 5)] += 1
            if info["sources_launching"] >= 2:
                multi_source_boards += 1

    print(f"boards analyzed={total_boards}")
    print(f"multi-source boards (>=2 sources launch same turn)={multi_source_boards}")
    print("distribution of #sources launching per board:")
    for k in sorted(move_dist):
        label = f"{k}+" if k == 5 else str(k)
        print(f"  {label} sources: {move_dist[k]} boards")


if __name__ == "__main__":
    run(range(0, 20))
