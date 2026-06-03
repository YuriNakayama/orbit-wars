"""x64 parity: featurize_jax.reaction_times vs WorldModel.reaction_times.

Uses real reset boards (realistic planet layouts) so the min-travel-time over
my/enemy planets is exercised on true geometry. Safe/contested-neutral classes
follow from the same reaction times.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.rulebase.case1.baseline.agent import build_world
from pipeline.rulebase.case1.baseline_jax.core_jax import featurize_jax as fjax

jax.config.update("jax_enable_x64", True)


def _planet_arrays(world):
    xs, ys, rs, ships, mine, enemy = [], [], [], [], [], []
    for p in world.planets:
        xs.append(p.x)
        ys.append(p.y)
        rs.append(p.radius)
        ships.append(p.ships)
        mine.append(p.owner == world.player)
        enemy.append(p.owner not in (-1, world.player))
    to = lambda a, dt: jnp.asarray(np.asarray(a, dtype=dt))  # noqa: E731
    return (
        to(xs, np.float64),
        to(ys, np.float64),
        to(rs, np.float64),
        to(ships, np.float64),
        to(mine, np.bool_),
        to(enemy, np.bool_),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 7, 11])
def test_reaction_times_parity(seed: int) -> None:
    state = reset(seed=seed, num_agents=2)
    world = build_world(state_to_obs(state, player=0))
    px, py, pr, ps, mine, enemy = _planet_arrays(world)

    mismatches = []
    for target in world.planets:
        ref_my, ref_en = world.reaction_times(target.id)
        my_t, en_t = fjax.reaction_times(
            jnp.asarray(target.x),
            jnp.asarray(target.y),
            jnp.asarray(target.radius),
            px,
            py,
            pr,
            ps,
            mine,
            enemy,
        )
        if int(ref_my) != int(my_t) or int(ref_en) != int(en_t):
            mismatches.append((target.id, (ref_my, ref_en), (int(my_t), int(en_t))))

    assert not mismatches, (
        f"seed={seed}: {len(mismatches)} mismatches: {mismatches[:5]}"
    )


from pipeline.rulebase.case1.baseline.core.types import Fleet  # noqa: E402
from pipeline.rulebase.case1.baseline.core.world_model import (  # noqa: E402
    fleet_target_planet as ftp_py,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fleet_target_planet_parity(seed: int) -> None:
    """Synthetic fleets vs real planets; ray-circle first-hit must match."""
    state = reset(seed=seed, num_agents=2)
    world = build_world(state_to_obs(state, player=0))
    px, py, pr, _ps, _m, _e = _planet_arrays(world)
    pv = jnp.asarray([True] * len(world.planets))
    planets = world.planets

    rng = np.random.default_rng(seed + 200)
    mism = []
    for _ in range(150):
        fx, fy = rng.uniform(0, 100, 2)
        fangle = rng.uniform(-np.pi, np.pi)
        fships = int(rng.integers(1, 100))
        fleet = Fleet(
            id=0, owner=1, x=fx, y=fy, angle=fangle, from_planet_id=-1, ships=fships
        )
        ref_planet, ref_eta = ftp_py(fleet, planets)
        slot, eta = fjax.fleet_target_planet(
            jnp.asarray(fx),
            jnp.asarray(fy),
            jnp.asarray(fangle),
            jnp.asarray(fships),
            px,
            py,
            pr,
            pv,
        )
        ref_id = ref_planet.id if ref_planet is not None else None
        got_id = planets[int(slot)].id if int(slot) >= 0 else None
        if ref_id != got_id or (ref_eta if ref_eta is not None else -1) != int(eta):
            mism.append((ref_id, got_id, ref_eta, int(eta)))
    assert not mism, f"seed={seed}: {len(mism)}/150: {mism[:5]}"


def test_reaction_times_parity_midgame() -> None:
    """reaction_times must stay parity on MID-GAME boards, not just turn-0.

    A ships-only over-send diagnosis suspected reaction_times diverging mid-game
    (the other tests only cover reset boards). This advances a real self-play
    game and checks parity at several mid points — it confirmed 0 mismatch, so
    this locks reaction_times faithfulness across the game, not just the opening.
    """
    from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, empty_actions, step

    from pipeline.rulebase.case1.baseline.agent import agent as v1_py

    def _pyrow(m: list) -> jnp.ndarray:
        r = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
        for i, mv in enumerate(m[:MAX_LAUNCHES_PER_AGENT]):
            r = r.at[i].set(jnp.asarray([mv[0], mv[1], mv[2]], dtype=jnp.float32))
        return r

    state = reset(seed=0, num_agents=2)
    checkpoints = {60, 120, 152, 180}
    mism = []
    for t in range(181):
        if t in checkpoints:
            w = build_world(state_to_obs(state, player=0))
            px, py, pr, ps, mine, enemy = _planet_arrays(w)
            for tp in w.planets:
                rm, re = w.reaction_times(tp.id)
                jm, je = fjax.reaction_times(
                    jnp.asarray(tp.x),
                    jnp.asarray(tp.y),
                    jnp.asarray(tp.radius),
                    px,
                    py,
                    pr,
                    ps,
                    mine,
                    enemy,
                )
                if int(rm) != int(jm) or int(re) != int(je):
                    mism.append((t, tp.id, (rm, re), (int(jm), int(je))))
        o0 = state_to_obs(state, player=0)
        o1 = state_to_obs(state, player=1)
        actions = (
            empty_actions().at[0].set(_pyrow(v1_py(o0))).at[1].set(_pyrow(v1_py(o1)))
        )
        state, _r, term = step(state, actions)
        if bool(term):
            break
    assert not mism, f"mid-game reaction_times mismatch: {mism[:5]}"
