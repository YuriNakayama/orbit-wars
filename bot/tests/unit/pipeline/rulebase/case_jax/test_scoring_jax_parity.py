"""Parity: `baseline_jax/scoring_jax.py` vs `baseline/strategy_helpers.py`.

Compares the pure-JAX scoring ops over `WorldFeatures` arrays against the Python
oracle (`target_value` / `score_attack` / `preferred_send` /
`is_safe_neutral` / `is_contested_neutral`) called through a real `WorldModel`
plus `build_modes`. Boards come from `orbit_wars_jax.reset` / `step`.

Tolerance: `score_attack` / `target_value` are combined-tolerance
(`atol=1e-4, rtol=1e-5`) because the value chain (`production * turns_profit`)
reaches the thousands, where the float64 Python oracle and the float32 JAX port
differ only at float32 representation level (measured worst *relative* error
~4e-7). Predicate booleans and `preferred_send` ints match EXACTLY. This is the
same absolute+relative convention the rest of the JAX parity suite uses for
large-magnitude quantities (see `tests/unit/jax_env/test_geometry.py`); it still
catches any real logic divergence, which would show as a large relative error.

Mission set: capture / snipe / swarm / reinforce (per spec). Missions are encoded
as int constants (`MISSION_*`). Relied-on config flags:
* OPPONENT_MODEL_ENABLED == False   -> OM block omitted from target_value
* FINISHING_TIE_GUARD    == False   -> _avoid_tie_total guarded off in preferred_send
* COMET_NPV_ENABLED      == False   -> comet value uses the flat COMET_VALUE_MULT
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.config import (
    ATTACK_COST_TURN_WEIGHT,
    SNIPE_COST_TURN_WEIGHT,
)
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.core.world_model import WorldModel
from pipeline.rulebase.case_jax.baseline.strategy_helpers import (
    build_modes,
    is_contested_neutral,
    is_safe_neutral,
    preferred_send,
    score_attack,
    target_value,
)
from pipeline.rulebase.case_jax.baseline_jax.scoring_jax import (
    MISSION_CAPTURE,
    MISSION_REINFORCE,
    MISSION_SNIPE,
    MISSION_SWARM,
    ModesArrays,
    is_contested_neutral_jax,
    is_safe_neutral_jax,
    preferred_send_jax,
    score_attack_jax,
    target_value_jax,
)
from pipeline.rulebase.case_jax.baseline_jax.world_features import (
    WorldFeatures,
    build_world_features,
)

pytestmark = pytest.mark.slow

# Combined tolerance for thousand-scale scores: float32 representation error at
# value ~4000 is ~5e-4 absolute but ~4e-7 relative. atol catches small scores,
# rtol catches large ones; a real logic bug breaks both.
SCORE_ATOL = 1e-4
SCORE_RTOL = 1e-5


def _close(got: float, ref: float) -> bool:
    return abs(got - ref) <= SCORE_ATOL + SCORE_RTOL * abs(ref)


Obs = dict[str, Any]

MISSIONS: list[tuple[str, int]] = [
    ("capture", MISSION_CAPTURE),
    ("snipe", MISSION_SNIPE),
    ("swarm", MISSION_SWARM),
    ("reinforce", MISSION_REINFORCE),
]


def _boards() -> list[Obs]:
    boards: list[Obs] = []
    for seed in (0, 1, 7):
        state = reset(seed=seed, num_agents=2)
        for stp in range(0, 70):
            state, _r, term = step(state, empty_actions())
            if stp in (5, 12, 55, 60):
                boards.append(state_to_obs(state, player=0))
            if bool(term):
                break
    state4 = reset(seed=3, num_agents=4)
    for _ in range(8):
        state4, _r, _t = step(state4, empty_actions())
    boards.append(state_to_obs(state4, player=0))
    return boards


def _build(obs: Obs) -> tuple[WorldModel, dict[str, Any], WorldFeatures]:
    reset_predict_cache()
    world = build_world(obs)
    modes = build_modes(world)
    feats = build_world_features(obs)
    return world, modes, feats


def _modes_arr(modes: dict[str, Any]) -> ModesArrays:
    return ModesArrays(
        domination=jnp.float32(modes["domination"]),
        is_behind=jnp.bool_(modes["is_behind"]),
        is_ahead=jnp.bool_(modes["is_ahead"]),
        is_dominating=jnp.bool_(modes["is_dominating"]),
        is_finishing=jnp.bool_(modes["is_finishing"]),
        attack_margin_mult=jnp.float32(modes["attack_margin_mult"]),
    )


def _idx_map(feats: WorldFeatures) -> dict[int, int]:
    ids = np.asarray(feats.planet_id)
    valid = np.asarray(feats.planet_valid)
    return {int(ids[i]): i for i in range(len(ids)) if bool(valid[i])}


def test_safe_contested_neutral_parity() -> None:
    for obs in _boards():
        world, _modes, feats = _build(obs)
        idx_map = _idx_map(feats)
        for planet in world.planets:
            i = idx_map[planet.id]
            ref_safe = is_safe_neutral(planet, world)
            ref_cont = is_contested_neutral(planet, world)
            got_safe = bool(
                is_safe_neutral_jax(
                    feats.owner[i], feats.reaction_my_t[i], feats.reaction_enemy_t[i]
                )
            )
            got_cont = bool(
                is_contested_neutral_jax(
                    feats.owner[i], feats.reaction_my_t[i], feats.reaction_enemy_t[i]
                )
            )
            assert got_safe == ref_safe, f"safe {planet.id}"
            assert got_cont == ref_cont, f"contested {planet.id}"


def test_target_value_parity() -> None:
    worst_rel = 0.0
    arrival_grid = [1, 3, 8, 25, 70]
    for obs in _boards():
        world, modes, feats = _build(obs)
        modes_arr = _modes_arr(modes)
        idx_map = _idx_map(feats)
        for planet in world.planets:
            i = idx_map[planet.id]
            for at in arrival_grid:
                for name, code in MISSIONS:
                    ref = target_value(planet, at, name, world, modes)
                    got = float(
                        target_value_jax(
                            jnp.int32(i), jnp.int32(at), code, feats, modes_arr
                        )
                    )
                    assert _close(got, ref), (
                        f"target_value planet={planet.id} at={at} m={name}: "
                        f"{got}!={ref}"
                    )
                    worst_rel = max(worst_rel, abs(got - ref) / max(1e-6, abs(ref)))
    # Sanity: divergence is float32-representation level, not a logic bug.
    assert worst_rel < 1e-5, f"target_value worst relative divergence {worst_rel}"


def test_score_attack_parity() -> None:
    worst_rel = 0.0
    arrival_grid = [1, 4, 12, 40]
    cost_grid = [1, 10, 50]
    for obs in _boards():
        world, modes, feats = _build(obs)
        modes_arr = _modes_arr(modes)
        idx_map = _idx_map(feats)
        for planet in world.planets:
            i = idx_map[planet.id]
            for at in arrival_grid:
                for cost in cost_grid:
                    for name, code in MISSIONS:
                        tw = (
                            SNIPE_COST_TURN_WEIGHT
                            if name == "snipe"
                            else ATTACK_COST_TURN_WEIGHT
                        )
                        ref = score_attack(planet, cost, at, name, tw, world, modes)
                        got = float(
                            score_attack_jax(
                                jnp.int32(i),
                                jnp.int32(cost),
                                jnp.int32(at),
                                code,
                                jnp.float32(tw),
                                feats,
                                modes_arr,
                            )
                        )
                        assert _close(got, ref), (
                            f"score_attack planet={planet.id} at={at} cost={cost} "
                            f"m={name}: {got}!={ref}"
                        )
                        worst_rel = max(worst_rel, abs(got - ref) / max(1e-6, abs(ref)))
    assert worst_rel < 1e-5, f"score_attack worst relative divergence {worst_rel}"


def test_preferred_send_parity() -> None:
    arrival_grid = [1, 8, 20, 55]
    src_grid = [5, 16, 40, 80]
    for obs in _boards():
        world, modes, feats = _build(obs)
        modes_arr = _modes_arr(modes)
        idx_map = _idx_map(feats)
        for planet in world.planets:
            i = idx_map[planet.id]
            for at in arrival_grid:
                for src in src_grid:
                    base_needed = world.ships_needed_to_capture(planet.id, at)
                    ref = preferred_send(planet, base_needed, at, src, world, modes)
                    got = int(
                        preferred_send_jax(
                            jnp.int32(i),
                            jnp.int32(base_needed),
                            jnp.int32(at),
                            jnp.int32(src),
                            feats,
                            modes_arr,
                        )
                    )
                    assert got == ref, (
                        f"preferred_send planet={planet.id} at={at} src={src}: "
                        f"{got}!={ref}"
                    )


def test_vmap_score_attack_grid() -> None:
    """vmap score_attack_jax over the whole (src x target) cost grid vmaps cleanly."""
    obs = _boards()[2]
    world, modes, feats = _build(obs)
    modes_arr = _modes_arr(modes)
    idx_map = _idx_map(feats)

    n = int(np.asarray(feats.planet_valid).sum())
    target_idx = jnp.arange(n, dtype=jnp.int32)
    cost = jnp.full((n,), 10, dtype=jnp.int32)
    turns = jnp.full((n,), 8, dtype=jnp.int32)

    batched = jax.vmap(
        score_attack_jax,
        in_axes=(0, 0, 0, None, None, None, None),
    )
    got = np.asarray(
        batched(
            target_idx,
            cost,
            turns,
            MISSION_CAPTURE,
            jnp.float32(ATTACK_COST_TURN_WEIGHT),
            feats,
            modes_arr,
        )
    )

    for planet in world.planets:
        i = idx_map[planet.id]
        ref = score_attack(
            planet, 10, 8, "capture", ATTACK_COST_TURN_WEIGHT, world, modes
        )
        assert _close(float(got[i]), ref), f"vmap score_attack {planet.id}"
