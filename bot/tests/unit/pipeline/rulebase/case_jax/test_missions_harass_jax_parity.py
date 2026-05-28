"""Parity: case_jax HARASS grid vs the Python `build_harass_missions` builder.

`build_harass_grid` mirrors `baseline/missions/build_harass_missions` over the
full (src x target) grid for ENEMY targets, keeping a single best-scoring source
per enemy target (best-per-target argmax), exactly like the Python builder.

Reaching combat boards
----------------------
Harass only fires once enemies own multiple high-production planets, which
self-play noop boards never reach (each side keeps a single home planet). So we
drive the REAL Python agent on BOTH seats through `orbit_wars_jax`'s env — the
agent emits `[from_planet_id, angle, num_ships]` launches that encode directly
into the `(NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3)` action tensor — and
capture mid-game obs once captures have produced several enemy planets. On those
boards the Python builder emits 0/1 harass `Mission` per enemy target.

Asserts (per the spec):
* valid cells match (Python harass for target exists <=> JAX cell valid); allow
  <=2% boundary mismatch (plan_shot float flips), reported as a %.
* where both valid: score (rtol 1e-4), turns EXACT, needed EXACT, angle (1e-2).
* best-per-target argmax: each enemy target column has at most one JAX valid
  cell, matching Python's single best-source Mission per target.

`reset_predict_cache()` is called before every Python build. A synthetic
positive case (`test_harass_grid_positive_case`) guarantees the harass branch is
exercised even if the sampled self-play boards yield zero harass missions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.constants import NUM_AGENTS_MAX
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case_jax.baseline.agent import agent as py_agent
from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.core.types import Mission
from pipeline.rulebase.case_jax.baseline.core.world_model import WorldModel
from pipeline.rulebase.case_jax.baseline.missions.harass import build_harass_missions
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.missions_capture_jax import (
    HarassGrid,
    build_harass_grid,
)
from pipeline.rulebase.case_jax.baseline_jax.scoring_jax import ModesArrays
from pipeline.rulebase.case_jax.baseline_jax.world_features import (
    WorldFeatures,
    build_world_features,
)

pytestmark = pytest.mark.slow

SCORE_RTOL = 1e-4
SCORE_ATOL = 1e-4
ANGLE_TOL = 1e-2
VALID_MISMATCH_CAP = 0.02

# Steps at which to snapshot a self-play board (late enough for several captures).
_CAPTURE_STEPS: frozenset[int] = frozenset({40, 60, 80, 100, 120, 140})
_MAX_STEP: int = 145
_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 7)

Obs = dict[str, Any]


def _encode(moves_by_agent: dict[int, list[list[float]]]) -> jnp.ndarray:
    """Pack per-agent Python launch lists into the jax env action tensor."""
    acts = np.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=np.float32)
    for owner, moves in moves_by_agent.items():
        for c, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
            acts[owner, c, 0] = float(mv[0])
            acts[owner, c, 1] = float(mv[1])
            acts[owner, c, 2] = float(mv[2])
    return jnp.asarray(acts)


def _combat_boards() -> list[Obs]:
    """Self-play the Python agent on both seats; snapshot mid-game combat obs."""
    boards: list[Obs] = []
    for seed in _SEEDS:
        state = reset(seed=seed, num_agents=2)
        for stp in range(_MAX_STEP):
            obs0 = state_to_obs(state, player=0)
            obs1 = state_to_obs(state, player=1)
            reset_predict_cache()
            moves0 = py_agent(obs0)
            reset_predict_cache()
            moves1 = py_agent(obs1)
            state, _r, term = step(state, _encode({0: moves0, 1: moves1}))
            if stp in _CAPTURE_STEPS:
                boards.append(state_to_obs(state, player=0))
            if bool(term):
                break
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


def _py_harass(
    world: WorldModel, modes: dict[str, Any]
) -> dict[tuple[int, int], Mission]:
    """Python harass missions keyed by (src_id, target_id) (collection-time)."""
    planned: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    missions = build_harass_missions(world, planned, modes)
    out: dict[tuple[int, int], Mission] = {}
    for m in missions:
        opt = m.options[0]
        out[(opt.src_id, opt.target_id)] = m
    return out


def _close(got: float, ref: float) -> bool:
    return abs(got - ref) <= SCORE_ATOL + SCORE_RTOL * abs(ref)


def _angle_close(got: float, ref: float) -> bool:
    return abs((got - ref + math.pi) % (2 * math.pi) - math.pi) < ANGLE_TOL


def test_harass_grid_parity() -> None:
    boards = _combat_boards()
    assert boards, "self-play produced no combat boards"

    valid_mismatch = 0
    compared_cells = 0
    value_fail: list[str] = []
    argmax_fail: list[str] = []
    py_harass_total = 0
    jax_valid_total = 0

    for b, obs in enumerate(boards):
        world, modes, feats = _build(obs)
        idx_map = _idx_map(feats)
        grid: HarassGrid = build_harass_grid(feats, _modes_arr(modes))

        gv = np.asarray(grid.valid)
        gscore = np.asarray(grid.score)
        gangle = np.asarray(grid.angle)
        gturns = np.asarray(grid.turns)
        gneed = np.asarray(grid.needed)

        py = _py_harass(world, modes)
        py_harass_total += len(py)
        jax_valid_total += int(gv.sum())

        # best-per-target argmax: at most one JAX valid cell per enemy target.
        for target in world.enemy_planets:
            ti = idx_map[target.id]
            col_valid = int(gv[:, ti].sum())
            if col_valid > 1:
                argmax_fail.append(
                    f"b{b} target={target.id} has {col_valid} valid JAX sources"
                )

        # Compare every (my-planet src, enemy target) cell.
        for src in world.my_planets:
            si = idx_map[src.id]
            for target in world.enemy_planets:
                if target.id == src.id:
                    continue
                ti = idx_map[target.id]
                compared_cells += 1
                jax_valid = bool(gv[si, ti])
                py_m = py.get((src.id, target.id))
                py_valid = py_m is not None

                if jax_valid != py_valid:
                    valid_mismatch += 1
                    continue
                if not py_valid:
                    continue

                assert py_m is not None
                opt = py_m.options[0]
                if int(gturns[si, ti]) != opt.turns:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) turns "
                        f"jax={int(gturns[si, ti])} py={opt.turns}"
                    )
                if int(gneed[si, ti]) != opt.needed:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) needed "
                        f"jax={int(gneed[si, ti])} py={opt.needed}"
                    )
                if not _close(float(gscore[si, ti]), py_m.score):
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) score "
                        f"jax={float(gscore[si, ti])} py={py_m.score}"
                    )
                if not _angle_close(float(gangle[si, ti]), opt.angle):
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) angle "
                        f"jax={float(gangle[si, ti])} py={opt.angle}"
                    )

    assert compared_cells > 0
    vfrac = valid_mismatch / compared_cells
    assert not argmax_fail, "best-per-target violated: " + "; ".join(argmax_fail[:8])
    assert vfrac <= VALID_MISMATCH_CAP, (
        f"harass valid mismatch {valid_mismatch}/{compared_cells} "
        f"({100 * vfrac:.2f}%) exceeds {100 * VALID_MISMATCH_CAP:.0f}% "
        f"(py_harass={py_harass_total} jax_valid={jax_valid_total})"
    )
    assert not value_fail, (
        f"harass value divergence on {len(value_fail)} cells "
        f"(valid mismatch {100 * vfrac:.2f}%): " + "; ".join(value_fail[:12])
    )


def _harass_scenario_obs() -> Obs:
    """A board with a high-production enemy planet within harass range.

    My home (id0) has plenty of ships (>= reserve) to send a need-sized wave; the
    enemy planet (id1) sits a short, sun-clear hop away with high production and a
    small garrison, so the Python `build_harass_missions` fires a harass Mission
    (turns within HARASS_MAX_TRAVEL_TURNS, need below the reserve cap).
    """
    planets = [
        [0, 0, 20.0, 30.0, 2.0, 80, 1],  # mine (home), large garrison
        [1, 1, 40.0, 15.0, 2.0, 4, 3],  # enemy, high prod, small garrison, near
        [2, -1, 90.0, 90.0, 2.0, 5, 1],  # neutral filler (far)
    ]
    return {
        "player": 0,
        "step": 60,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comets": [],
        "comet_planet_ids": [],
    }


def test_harass_grid_positive_case() -> None:
    """The JAX harass grid reproduces a Python harass Mission on a synthetic board.

    Guarantees the harass branch is exercised with EXACT turns / needed and
    rtol-1e-4 score / 1e-2 angle, independent of whether the sampled self-play
    boards happen to produce harass missions.
    """
    obs = _harass_scenario_obs()
    world, modes, feats = _build(obs)
    idx_map = _idx_map(feats)

    py = _py_harass(world, modes)
    assert py, "scenario should produce at least one Python harass mission"

    grid: HarassGrid = build_harass_grid(feats, _modes_arr(modes))
    gv = np.asarray(grid.valid)
    gscore = np.asarray(grid.score)
    gangle = np.asarray(grid.angle)
    gturns = np.asarray(grid.turns)
    gneed = np.asarray(grid.needed)

    for (sid, tid), m in py.items():
        opt = m.options[0]
        si = idx_map[sid]
        ti = idx_map[tid]
        assert bool(gv[si, ti]), f"jax harass missing for ({sid},{tid})"
        assert int(gturns[si, ti]) == opt.turns
        assert int(gneed[si, ti]) == opt.needed
        assert _close(float(gscore[si, ti]), m.score)
        assert _angle_close(float(gangle[si, ti]), opt.angle)
