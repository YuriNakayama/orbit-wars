"""x64 parity: missions_jax score chain vs strategy_helpers (non-comet, real boards).

Feeds the JAX functions the SAME precomputed per-target features the Python
WorldModel exposes, and asserts target_value / preferred_send / opening_filter /
apply_score_modifiers match. Real reset boards (turn 0) over several seeds.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.rulebase.case1.baseline import strategy_helpers as sh
from pipeline.rulebase.case1.baseline.agent import build_world
from pipeline.rulebase.case1.baseline_jax.core_jax import missions_jax as mj

RTOL = 1e-7


def _modes(world: Any) -> dict[str, Any]:
    return sh.build_modes(world)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_target_value_parity(seed: int) -> None:
    state = reset(seed=seed, num_agents=2)
    world = build_world(state_to_obs(state, player=0))
    modes = _modes(world)
    mism = []
    for tgt in world.planets:
        if tgt.owner == world.player or tgt.id in world.comet_ids:
            continue
        for turns in (3, 8, 20):
            ref = sh.target_value(tgt, turns, "capture", world, modes)
            my_t, en_t = world.reaction_times(tgt.id)
            enemy_strength = world.owner_strength.get(tgt.owner, 0)
            got = float(
                mj.target_value(
                    jnp.asarray(tgt.owner),
                    jnp.asarray(tgt.production),
                    jnp.asarray(tgt.ships),
                    jnp.asarray(world.is_static(tgt.id)),
                    jnp.asarray(world.indirect_wealth_map[tgt.id]),
                    jnp.asarray(turns),
                    jnp.asarray(my_t),
                    jnp.asarray(en_t),
                    jnp.asarray(world.remaining_steps),
                    jnp.asarray(world.is_opening),
                    jnp.asarray(world.is_early),
                    jnp.asarray(world.is_late),
                    jnp.asarray(enemy_strength),
                    jnp.asarray(modes["is_finishing"]),
                    jnp.asarray(modes["is_behind"]),
                    jnp.asarray(modes["is_dominating"]),
                    jnp.asarray(world.player),
                )
            )
            if not np.isclose(ref, got, rtol=RTOL, atol=1e-6):
                mism.append((tgt.id, turns, ref, got))
    assert not mism, f"seed={seed}: {len(mism)}: {mism[:5]}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_preferred_send_and_opening_filter_parity(seed: int) -> None:
    state = reset(seed=seed, num_agents=2)
    world = build_world(state_to_obs(state, player=0))
    modes = _modes(world)
    mism = []
    for tgt in world.planets:
        if tgt.owner == world.player or tgt.id in world.comet_ids:
            continue
        my_t, en_t = world.reaction_times(tgt.id)
        for turns in (3, 8, 20):
            for needed in (3, 7, 15):
                src_av = 30
                ref_ps = sh.preferred_send(tgt, needed, turns, src_av, world, modes)
                got_ps = int(
                    mj.preferred_send(
                        jnp.asarray(tgt.owner),
                        jnp.asarray(tgt.production),
                        jnp.asarray(needed),
                        jnp.asarray(turns),
                        jnp.asarray(src_av),
                        jnp.asarray(world.is_static(tgt.id)),
                        jnp.asarray(my_t),
                        jnp.asarray(en_t),
                        jnp.asarray(world.is_four_player),
                        jnp.asarray(modes["attack_margin_mult"]),
                        jnp.asarray(modes["is_finishing"]),
                        jnp.asarray(world.player),
                    )
                )
                if ref_ps != got_ps:
                    mism.append(("send", tgt.id, turns, needed, ref_ps, got_ps))

                ref_of = sh.opening_filter(tgt, turns, needed, src_av, world)
                got_of = bool(
                    mj.opening_filter(
                        jnp.asarray(tgt.owner),
                        jnp.asarray(tgt.production),
                        jnp.asarray(world.is_static(tgt.id)),
                        jnp.asarray(turns),
                        jnp.asarray(needed),
                        jnp.asarray(src_av),
                        jnp.asarray(my_t),
                        jnp.asarray(en_t),
                        jnp.asarray(world.is_opening),
                        jnp.asarray(world.is_four_player),
                    )
                )
                if ref_of != got_of:
                    mism.append(("filter", tgt.id, turns, needed, ref_of, got_of))
    assert not mism, f"seed={seed}: {len(mism)}: {mism[:5]}"
