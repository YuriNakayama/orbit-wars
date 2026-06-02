"""PoC1: is the turn-0 launch/hold + target selection a pure per-target score+mask?

The real `collect_capture_options_and_missions` enumerates (src, target) pairs,
computes a per-target `score` (a long chain of pure comparisons × constants),
vetoes via `opening_filter`, and `plan_moves` picks the highest-score mission.

PoC1 reproduces the turn-0 decision as: for each target, compute
(score, vetoed, send) via the SAME helper math, take argmax over non-vetoed
targets, and compare the chosen (target, send) + launch/hold against the real
agent's actual move. This proves the decision surface is a fixed-shape
per-target score + mask + argmax — the exact pattern a JAX port needs — WITHOUT
yet hand-porting the ~40 config constants (that is Step 3; here we validate the
STRUCTURE by reusing the Python helpers per-target).

If argmax-over-scores reproduces the real choice, the JAX port is just
"vectorize these helpers" (mechanical). If it does NOT, the decision depends on
something beyond per-target score (e.g. multi-source interaction) and we learn
that now, cheaply.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import math

from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline.agent import build_world
from pipeline.rulebase.case1.baseline.core.config import (
    ATTACK_COST_TURN_WEIGHT,
    PARTIAL_SOURCE_MIN_SHIPS,
)
from pipeline.rulebase.case1.baseline.strategy_helpers import (
    apply_score_modifiers,
    build_modes,
    opening_filter,
    preferred_send,
    target_value,
)


def best_capture_for_source(world, src, modes) -> tuple[int, float, float, int] | None:
    """Replicate option_collector's per-target score; return best (tgt, score, angle, send).

    Mirrors option_collector.collect_capture_options_and_missions for a single
    source, single-mission (capture/single) path. No snipe/swarm here — turn-0
    decisions are dominated by the single capture mission.
    """
    src_available = world.available.get(src.id, 0)
    if src_available <= 0:
        return None

    def empty_commit():
        from collections import defaultdict

        return defaultdict(list)

    planned = empty_commit()
    best: tuple[int, float, float, int] | None = None
    for target in world.planets:
        if target.id == src.id or target.owner == world.player:
            continue
        rough_ships = max(
            1, min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1))
        )
        rough_aim = world.plan_shot(src.id, target.id, rough_ships)
        if rough_aim is None:
            continue
        rough_turns = rough_aim[1]
        rough_needed = world.ships_needed_to_capture(target.id, rough_turns, planned)
        if rough_needed <= 0:
            continue
        if opening_filter(target, rough_turns, rough_needed, src_available, world):
            continue
        send_guess = preferred_send(
            target, rough_needed, rough_turns, src_available, world, modes
        )
        aim = world.plan_shot(src.id, target.id, max(1, send_guess))
        if aim is None:
            continue
        angle, turns, _, _ = aim
        needed = world.ships_needed_to_capture(target.id, turns, planned)
        if needed <= 0:
            continue
        if opening_filter(target, turns, needed, src_available, world):
            continue
        send_cap = min(
            src_available, preferred_send(target, needed, turns, src_available, world, modes)
        )
        if send_cap < 1 or (send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS):
            continue
        value = target_value(target, turns, "capture", world, modes)
        if value <= 0:
            continue
        expected_send = max(
            needed,
            min(send_cap, preferred_send(target, needed, turns, send_cap, world, modes)),
        )
        score = apply_score_modifiers(
            value / (expected_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
            target,
            "capture",
            world,
        )
        if send_cap < needed:
            continue  # turn0: single mission requires send_cap >= needed
        cand = (target.id, score, angle, send_cap)
        # tie-break: highest score; on tie, lowest target id (match sort stability)
        if best is None or score > best[1]:
            best = cand
    return best


def run(seeds) -> None:
    match = 0
    mismatch = 0
    hold_match = 0
    details = []
    for seed in seeds:
        state = reset(seed=seed, num_agents=2)
        obs = state_to_obs(state, player=0)
        world = build_world(obs)
        modes = build_modes(world)

        real_moves = v1_py(obs)
        # turn0: at most one source (single home planet)
        best = None
        for src in world.my_planets:
            cand = best_capture_for_source(world, src, modes)
            if cand is not None and (best is None or cand[1] > best[1]):
                best = cand

        if not real_moves:
            # real agent holds; PoC should also produce no capture mission
            if best is None:
                hold_match += 1
                match += 1
            else:
                mismatch += 1
                details.append((seed, "real=HOLD", f"poc=launch tgt{best[0]}"))
            continue

        rm = real_moves[0]
        real_tgt_angle = round(float(rm[1]), 3)
        real_send = int(rm[2])
        if best is None:
            mismatch += 1
            details.append((seed, f"real=launch a{real_tgt_angle} s{real_send}", "poc=HOLD"))
            continue
        # compare angle (proxy for target) + send
        poc_angle = round(float(best[2]), 3)
        poc_send = int(best[3])
        if abs(poc_angle - real_tgt_angle) < 1e-2 and poc_send == real_send:
            match += 1
        else:
            mismatch += 1
            details.append(
                (seed, f"real a{real_tgt_angle} s{real_send}", f"poc a{poc_angle} s{poc_send} tgt{best[0]}")
            )

    print(f"seeds={len(seeds)}: match={match} (hold_match={hold_match}), mismatch={mismatch}")
    for d in details[:15]:
        print("  MISMATCH", d)


if __name__ == "__main__":
    run(range(0, 50))
