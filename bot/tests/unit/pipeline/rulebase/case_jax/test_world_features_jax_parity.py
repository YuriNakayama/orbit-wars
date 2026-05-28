"""Parity: `baseline_jax/world_features.py` vs the Python `WorldModel`.

`build_world_features(obs)` is a HOST-side resolver that mirrors what
`WorldModel.__init__` + `build_modes` compute, but as fixed-shape JAX arrays
(MAX_PLANETS padded + a `planet_valid` mask). This test builds realistic boards
via `orbit_wars_jax.reset` / `step` (so fleets and comets appear), constructs the
Python `WorldModel` through `build_world`, and compares EVERY scalar + per-planet
feature.

Tolerance: owner / int / bool fields match EXACTLY; float fields within float32
`1e-4`. The Python predict cache is reset before each Python call.

Relied-on config flags (all dead branches for this agent):
* OPPONENT_MODEL_ENABLED == False  -> predicted_arrivals / threat empty
* LOOKAHEAD_ENABLED      == False  -> no probe injection
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.core.world_model import WorldModel
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.timeline_jax import (
    state_at_timeline_jax,
)
from pipeline.rulebase.case_jax.baseline_jax.world_features import (
    WorldFeatures,
    build_world_features,
)

pytestmark = pytest.mark.slow

PARITY_TOL = 1e-4
BIG = 1e9

Obs = dict[str, Any]


def _boards() -> list[tuple[Obs, int]]:
    """Realistic obs dicts at several steps across a couple of seeds."""
    boards: list[tuple[Obs, int]] = []
    for seed in (0, 1, 7):
        state = reset(seed=seed, num_agents=2)
        for stp in range(0, 70):
            state, _r, term = step(state, empty_actions())
            if stp in (5, 12, 55, 60):
                boards.append((state_to_obs(state, player=0), seed))
            if bool(term):
                break
    # A 4-player board to exercise is_four_player / four-player margins.
    state4 = reset(seed=3, num_agents=4)
    for _ in range(8):
        state4, _r, _t = step(state4, empty_actions())
    boards.append((state_to_obs(state4, player=0), 3))
    return boards


def _build_py(obs: Obs) -> tuple[WorldModel, dict[str, Any]]:
    reset_predict_cache()
    world = build_world(obs)
    modes = build_modes(world)
    return world, modes


def _id_to_idx(feats: WorldFeatures) -> dict[int, int]:
    ids = np.asarray(feats.planet_id)
    valid = np.asarray(feats.planet_valid)
    return {int(ids[i]): i for i in range(len(ids)) if bool(valid[i])}


def test_world_features_scalar_parity() -> None:
    max_div = 0.0
    for obs, _seed in _boards():
        world, modes = _build_py(obs)
        feats = build_world_features(obs)

        assert int(feats.step) == world.step
        assert int(feats.player) == world.player
        assert int(feats.num_players) == world.num_players
        assert int(feats.remaining_steps) == world.remaining_steps
        assert bool(feats.is_early) == world.is_early
        assert bool(feats.is_opening) == world.is_opening
        assert bool(feats.is_late) == world.is_late
        assert bool(feats.is_very_late) == world.is_very_late
        assert bool(feats.is_four_player) == world.is_four_player
        assert int(feats.my_total) == world.my_total
        assert int(feats.enemy_total) == world.enemy_total
        assert int(feats.max_enemy_strength) == world.max_enemy_strength
        assert int(feats.my_prod) == world.my_prod
        assert int(feats.enemy_prod) == world.enemy_prod
        assert int(feats.static_neutral_count) == len(world.static_neutral_planets)

        max_div = max(max_div, abs(float(feats.ang_vel) - world.ang_vel))

        # modes
        assert bool(feats.is_behind) == modes["is_behind"]
        assert bool(feats.is_ahead) == modes["is_ahead"]
        assert bool(feats.is_dominating) == modes["is_dominating"]
        assert bool(feats.is_finishing) == modes["is_finishing"]
        max_div = max(max_div, abs(float(feats.domination) - modes["domination"]))
        max_div = max(
            max_div,
            abs(float(feats.attack_margin_mult) - modes["attack_margin_mult"]),
        )
    assert max_div < PARITY_TOL, f"scalar float divergence {max_div}"


def test_world_features_per_planet_parity() -> None:
    max_div = 0.0
    for obs, _seed in _boards():
        world, _modes = _build_py(obs)
        feats = build_world_features(obs)
        idx_map = _id_to_idx(feats)

        assert len(idx_map) == len(world.planets)

        owner_a = np.asarray(feats.owner)
        xy_a = np.asarray(feats.xy)
        radius_a = np.asarray(feats.radius)
        ships_a = np.asarray(feats.ships)
        prod_a = np.asarray(feats.prod)
        static_a = np.asarray(feats.is_static)
        comet_a = np.asarray(feats.is_comet)
        comet_life_a = np.asarray(feats.comet_life)
        wealth_a = np.asarray(feats.indirect_wealth)
        init_xy_a = np.asarray(feats.initial_xy)
        init_r_a = np.asarray(feats.initial_radius)
        rmy_a = np.asarray(feats.reaction_my_t)
        ren_a = np.asarray(feats.reaction_enemy_t)

        for planet in world.planets:
            i = idx_map[planet.id]
            assert int(owner_a[i]) == planet.owner, f"owner {planet.id}"
            assert int(ships_a[i]) == int(planet.ships), f"ships {planet.id}"
            assert int(prod_a[i]) == int(planet.production), f"prod {planet.id}"
            assert bool(static_a[i]) == world.is_static(planet.id), (
                f"static {planet.id}"
            )
            assert bool(comet_a[i]) == (planet.id in world.comet_ids), (
                f"comet {planet.id}"
            )
            assert int(comet_life_a[i]) == world.comet_life(planet.id), (
                f"comet_life {planet.id}"
            )
            max_div = max(max_div, abs(float(xy_a[i, 0]) - planet.x))
            max_div = max(max_div, abs(float(xy_a[i, 1]) - planet.y))
            max_div = max(max_div, abs(float(radius_a[i]) - planet.radius))
            max_div = max(
                max_div,
                abs(float(wealth_a[i]) - world.indirect_wealth_map[planet.id]),
            )

            init = world.initial_by_id.get(planet.id)
            exp_ix = init.x if init is not None else planet.x
            exp_iy = init.y if init is not None else planet.y
            exp_ir = init.radius if init is not None else planet.radius
            max_div = max(max_div, abs(float(init_xy_a[i, 0]) - exp_ix))
            max_div = max(max_div, abs(float(init_xy_a[i, 1]) - exp_iy))
            max_div = max(max_div, abs(float(init_r_a[i]) - exp_ir))

            ref_my, ref_en = world.reaction_times(planet.id)
            got_my = float(rmy_a[i])
            got_en = float(ren_a[i])
            # Sentinel 1e9 (no planet of that owner) -> compare as "large".
            if ref_my >= BIG:
                assert got_my >= BIG, f"reaction_my sentinel {planet.id}"
            else:
                assert int(got_my) == ref_my, f"reaction_my {planet.id}"
            if ref_en >= BIG:
                assert got_en >= BIG, f"reaction_enemy sentinel {planet.id}"
            else:
                assert int(got_en) == ref_en, f"reaction_enemy {planet.id}"
    assert max_div < PARITY_TOL, f"per-planet float divergence {max_div}"


def test_world_features_arrival_table_parity() -> None:
    """The bucketed arrival table reproduces WorldModel.base_timeline per planet.

    We feed each padded planet's table slice to `simulate_planet_timeline_jax`
    (the same call the downstream agent makes) and compare keep_needed /
    min_owned / fall_turn / first_enemy / projected ships against the Python
    `base_timeline[planet.id]`.
    """
    from pipeline.rulebase.case_jax.baseline_jax.timeline_jax import (
        simulate_planet_timeline_jax,
    )

    max_div = 0.0
    for obs, _seed in _boards():
        world, _modes = _build_py(obs)
        feats = build_world_features(obs)
        idx_map = _id_to_idx(feats)
        horizon = int(feats.horizon)

        for planet in world.planets:
            i = idx_map[planet.id]
            tl = simulate_planet_timeline_jax(
                jnp.int32(planet.owner),
                jnp.float32(planet.ships),
                jnp.float32(planet.production),
                feats.arr_owner[i],
                feats.arr_ships[i],
                feats.arr_valid[i],
                jnp.int32(world.player),
                horizon,
            )
            ref = world.base_timeline[planet.id]
            assert int(tl.keep_needed) == ref["keep_needed"], f"keep_needed {planet.id}"
            assert int(tl.min_owned) == ref["min_owned"], f"min_owned {planet.id}"
            assert bool(tl.holds_full) == ref["holds_full"], f"holds {planet.id}"
            ref_first = -1 if ref["first_enemy"] is None else ref["first_enemy"]
            ref_fall = -1 if ref["fall_turn"] is None else ref["fall_turn"]
            assert int(tl.first_enemy) == ref_first, f"first_enemy {planet.id}"
            assert int(tl.fall_turn) == ref_fall, f"fall_turn {planet.id}"

            from pipeline.rulebase.case_jax.baseline.core.world_model import (
                state_at_timeline,
            )

            for at in (1.0, 5.0, 20.0, 60.0, 109.0):
                ref_o, ref_s = state_at_timeline(ref, at)
                g_o, g_s = state_at_timeline_jax(
                    tl.owner_at, tl.ships_at, jnp.int32(horizon), jnp.float32(at)
                )
                assert int(g_o) == ref_o, f"state owner {planet.id}@{at}"
                max_div = max(max_div, abs(float(g_s) - ref_s))
    assert max_div < PARITY_TOL, f"arrival-table ships divergence {max_div}"
