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
