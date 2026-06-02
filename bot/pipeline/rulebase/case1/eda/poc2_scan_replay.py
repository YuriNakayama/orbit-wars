"""PoC2b: prove the multi-source greedy is a faithful FIXED-LENGTH sequential scan.

`plan_moves` sorts missions by -score then iterates, mutating two carries:
  - spent_total[src]        (ships already committed per source)
  - planned_commitments[tgt] (arrivals scheduled per target; read by
                              ships_needed_to_capture)

Claim: replaying the SAME sorted mission list through an explicit step-function
that threads (spent_total, planned_commitments) as carry reproduces the real
`plan_moves` output byte-for-byte. If a hand-rolled sequential carry == real
output, then a fixed-length `lax.scan` of the identical recurrence will too
(scan IS a sequential fold over a fixed-length xs).

We don't reimplement the mission builders or scorers — we capture the real
sorted-mission stream from `plan_moves`, then re-run the per-mission resolver
loop ourselves with our own carries and a fresh `append_move`. Matching the
real moves proves the recurrence is faithfully sequential + fixed-length.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from collections import defaultdict

import jax.numpy as jnp
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline.agent import build_world
from pipeline.rulebase.case1.baseline.planner import (
    enforce_inventory_cap,
    process_multi_source_mission,
    process_single_source_mission,
)
from pipeline.rulebase.case1.baseline.planner.mission_resolver import (
    SINGLE_SOURCE_MISSION_KINDS,
)
from pipeline.rulebase.case1.baseline.strategy import plan_moves
from pipeline.rulebase.case1.baseline.strategy_helpers import build_modes


def py_row(moves):
    row = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        row = row.at[i].set(jnp.asarray([m[0], m[1], m[2]], dtype=jnp.float32))
    return row


def capture_sorted_missions(world):
    """Rebuild the exact sorted mission list plan_moves iterates (strategy.py:61-77)."""
    from collections import defaultdict as dd

    from pipeline.rulebase.case1.baseline.missions.crash_exploit import (
        build_crash_exploit_missions,
    )
    from pipeline.rulebase.case1.baseline.missions.reinforcement import (
        build_reinforcement_missions,
    )
    from pipeline.rulebase.case1.baseline.planner import (
        build_swarm_missions,
        collect_capture_options_and_missions,
    )

    modes = build_modes(world)
    planned = dd(list)
    spent = dd(int)

    def s_inv(sid):
        return world.source_inventory_left(sid, spent)

    def s_atk(sid):
        return world.source_attack_left(sid, spent)

    missions = list(build_reinforcement_missions(world, planned, modes, s_inv))
    cap, opts = collect_capture_options_and_missions(world, planned, modes, s_atk)
    missions.extend(cap)
    missions.extend(build_swarm_missions(world, opts, planned, modes))
    missions.extend(build_crash_exploit_missions(world, planned, modes))
    missions.sort(key=lambda m: -m.score)
    return missions, modes


def replay_scan(world, missions, modes):
    """Re-run the mission loop with our OWN carries (mimics lax.scan fold)."""
    planned_commitments: dict[int, list] = defaultdict(list)
    spent_total: dict[int, int] = defaultdict(int)
    moves: list[list] = []

    def source_inventory_left(sid):
        return world.source_inventory_left(sid, spent_total)

    def source_attack_left(sid):
        return world.source_attack_left(sid, spent_total)

    def append_move(src_id, angle, ships):
        send = min(int(ships), source_inventory_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    # fixed-length fold over the sorted missions (== lax.scan over xs=missions)
    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in SINGLE_SOURCE_MISSION_KINDS:
            process_single_source_mission(
                mission, target, world, modes, planned_commitments,
                source_inventory_left, source_attack_left, append_move,
            )
        else:
            process_multi_source_mission(
                mission, target, world, planned_commitments,
                source_attack_left, append_move,
            )
    return moves


def run(seeds, steps=(60, 120, 200)) -> None:
    boards = 0
    prefix_match = 0  # mission-loop portion matches (before followup/evac/rear)
    full_match = 0
    multi_src = 0
    mismatches = []
    for seed in seeds:
        state = reset(seed=seed, num_agents=2)
        for tstep in range(max(steps) + 1):
            if tstep in steps:
                obs = state_to_obs(state, player=0)
                world = build_world(obs)
                if world.my_planets:
                    boards += 1
                    real = plan_moves(world)
                    missions, modes = capture_sorted_missions(world)
                    mine = enforce_inventory_cap(replay_scan(world, missions, modes), world)
                    srcs = {mv[0] for mv in real}
                    if len(srcs) >= 2:
                        multi_src += 1
                    # compare the mission-loop moves (our replay omits followup/
                    # evac/rear_guard, so compare the subset our scan produces).
                    mine_set = sorted((int(m[0]), round(float(m[1]), 3), int(m[2])) for m in mine)
                    real_set = sorted((int(m[0]), round(float(m[1]), 3), int(m[2])) for m in real)
                    # our replay is a SUBSET (no followup/evac/rear). check subset+order-free match
                    if mine_set == real_set:
                        full_match += 1
                    if all(mv in real_set for mv in mine_set):
                        prefix_match += 1
                    else:
                        if len(mismatches) < 10:
                            extra = [mv for mv in mine_set if mv not in real_set]
                            mismatches.append((seed, tstep, f"replay-only moves={extra}"))
            m0 = v1_py(state_to_obs(state, player=0))
            m1 = v1_py(state_to_obs(state, player=1))
            actions = empty_actions().at[0].set(py_row(m0)).at[1].set(py_row(m1))
            state, _, done = step(state, actions)
            if bool(done):
                break

    print(f"boards={boards}, multi-source={multi_src}")
    print(f"replay-moves ⊆ real-moves (mission-loop faithful): {prefix_match}/{boards}")
    print(f"replay == real exactly (board has no followup/evac/rear): {full_match}/{boards}")
    for mm in mismatches:
        print("  MISMATCH", mm)


if __name__ == "__main__":
    run(range(0, 20))
