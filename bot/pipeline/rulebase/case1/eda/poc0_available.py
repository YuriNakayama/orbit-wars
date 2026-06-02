"""PoC0: replicate Python WorldModel.available at turn 0 in JAX-faithful logic.

Goal: confirm the home planet gets available == ships (so it can launch all
10 ships on turn 0), matching the real Python baseline_v1, across many seeds.

At turn 0 there are NO fleets, so every planet's arrival ledger is empty and
`keep_needed == 0`. The only non-trivial reserve term is
`_multi_enemy_proactive_keep`, which depends on enemy-planet ETAs computed via
`travel_time` (= estimate_arrival, the intercept solver).

This PoC uses the REAL Python `travel_time` for now (the intercept solver is
already JAX-proven in case2/aim_jax.py — porting it is Step 1, not PoC0). The
point of PoC0 is to validate the AVAILABLE formula + proactive_keep logic in a
clean, vectorizable form and confirm turn-0 parity.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.rulebase.case1.baseline.agent import build_world
from pipeline.rulebase.case1.baseline.core.config import (
    MULTI_ENEMY_PROACTIVE_HORIZON,
    MULTI_ENEMY_PROACTIVE_RATIO,
    MULTI_ENEMY_STACK_WINDOW,
    PROACTIVE_DEFENSE_HORIZON,
    PROACTIVE_DEFENSE_RATIO,
)
from pipeline.rulebase.case1.baseline.core.physics import travel_time


def proactive_keep_faithful(planet, enemy_planets) -> int:
    """Reimplementation of WorldModel._multi_enemy_proactive_keep, decoupled.

    Same operations as world_model.py:438-476, written as a standalone fn so a
    JAX version can be diffed against it term-by-term. Uses real travel_time.
    """
    if not enemy_planets:
        return 0
    threats: list[tuple[int, int]] = []
    for enemy in enemy_planets:
        eta = travel_time(
            enemy.x, enemy.y, enemy.radius,
            planet.x, planet.y, planet.radius,
            max(1, enemy.ships),
        )
        if eta > MULTI_ENEMY_PROACTIVE_HORIZON:
            continue
        threats.append((eta, int(enemy.ships)))
    if not threats:
        return 0
    threats.sort()
    best_stacked = 0
    left = 0
    running = 0
    for right in range(len(threats)):
        running += threats[right][1]
        while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
            running -= threats[left][1]
            left += 1
        best_stacked = max(best_stacked, running)
    proactive = int(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO)
    legacy = 0
    for eta, ships in threats:
        if eta <= PROACTIVE_DEFENSE_HORIZON:
            legacy = max(legacy, int(ships * PROACTIVE_DEFENSE_RATIO))
    return max(proactive, legacy)


def available_faithful(world) -> dict[int, int]:
    """available[pid] for my planets at turn 0 (keep_needed==0, no fleets)."""
    out: dict[int, int] = {}
    for planet in world.my_planets:
        keep_needed = world.base_timeline[planet.id]["keep_needed"]
        proactive = proactive_keep_faithful(planet, world.enemy_planets)
        reserve = min(int(planet.ships), max(keep_needed, proactive))
        out[planet.id] = max(0, int(planet.ships) - reserve)
    return out


def run(seeds: range) -> None:
    mismatches = 0
    home_full = 0
    total_planets = 0
    for seed in seeds:
        state = reset(seed=seed, num_agents=2)
        obs = state_to_obs(state, player=0)
        world = build_world(obs)

        py_avail = dict(world.available)  # ground truth
        my_avail = available_faithful(world)

        for pid in py_avail:
            total_planets += 1
            if py_avail[pid] != my_avail.get(pid):
                mismatches += 1
                print(
                    f"seed={seed} pid={pid}: py={py_avail[pid]} mine={my_avail.get(pid)} "
                    f"ships={world.planet_by_id[pid].ships} prod={world.planet_by_id[pid].production}"
                )
        # check home planet launches all its ships (keep_needed==0 at turn0)
        for planet in world.my_planets:
            if py_avail[planet.id] == int(planet.ships):
                home_full += 1

    print(
        f"\nseeds={len(seeds)}: planets compared={total_planets}, "
        f"available mismatches={mismatches}, "
        f"planets with available==ships (turn0 full release)={home_full}"
    )


if __name__ == "__main__":
    run(range(0, 50))


def diagnostic(seeds: range) -> None:
    """How often is reserve non-zero at turn 0? (avoid trivial all-zero test)."""
    nonzero_keep = 0
    nonzero_proactive = 0
    for seed in seeds:
        state = reset(seed=seed, num_agents=2)
        world = build_world(state_to_obs(state, player=0))
        for planet in world.my_planets:
            if world.base_timeline[planet.id]["keep_needed"] != 0:
                nonzero_keep += 1
            if proactive_keep_faithful(planet, world.enemy_planets) != 0:
                nonzero_proactive += 1
    print(f"diagnostic over {len(seeds)} seeds: nonzero keep_needed={nonzero_keep}, nonzero proactive={nonzero_proactive}")
