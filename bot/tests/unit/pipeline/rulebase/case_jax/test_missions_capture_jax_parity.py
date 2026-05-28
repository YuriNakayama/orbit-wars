"""Parity: case_jax CAPTURE + SNIPE grids vs the Python mission builders.

`build_capture_grid` / `build_snipe_grid` mirror
`baseline/missions/build_capture_and_snipe_missions` over the full (src x target)
grid. On several reset boards (advanced a few steps) we build the Python
`WorldModel` via `build_world(obs)` and call `build_capture_and_snipe_missions`
with EMPTY `planned_commitments` / `spent_total` (the collection-time state), then
compare the JAX grids per (src, target) cell.

The Python builder produces, per (src, target): 0/1 capture `ShotOption`
(recorded in `source_options_by_target[target_id]`), 0/1 "single" `Mission`
(when `send_cap >= needed`), and 0/1 snipe `Mission`.

Asserts (per the spec):
* capture valid cells match (Python option exists <=> JAX cell valid); allow
  <=2% boundary mismatch (plan_shot float flips), reported as a %.
* where both valid: score (rtol 1e-4), turns EXACT, needed EXACT, send_cap EXACT,
  angle (1e-2); `is_single` flag matches.
* snipe valid cells match (same 2% boundary band); where both valid: score
  (rtol 1e-4), turns EXACT, needed EXACT.

`reset_predict_cache()` is called before every Python build.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.core.types import Mission, ShotOption
from pipeline.rulebase.case_jax.baseline.core.world_model import WorldModel
from pipeline.rulebase.case_jax.baseline.missions.capture import (
    build_capture_and_snipe_missions,
)
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.missions_capture_jax import (
    CaptureGrid,
    SnipeGrid,
    build_capture_grid,
    build_snipe_grid,
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

Obs = dict[str, Any]


def _boards() -> list[Obs]:
    boards: list[Obs] = []
    for seed in (0, 1, 7):
        state = reset(seed=seed, num_agents=2)
        for stp in range(0, 70):
            state, _r, term = step(state, empty_actions())
            if stp in (4, 10, 30, 55):
                boards.append(state_to_obs(state, player=0))
            if bool(term):
                break
    state4 = reset(seed=3, num_agents=4)
    for _ in range(10):
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


def _run_py_builder(
    world: WorldModel, modes: dict[str, Any]
) -> tuple[list[Mission], dict[int, list[ShotOption]]]:
    """Invoke the Python builder with EMPTY commitments (collection-time state)."""
    planned: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    spent: dict[int, int] = defaultdict(int)
    return build_capture_and_snipe_missions(
        world,
        planned,
        modes,
        lambda sid: world.source_attack_left(sid, spent),
    )


def _close(got: float, ref: float) -> bool:
    return abs(got - ref) <= SCORE_ATOL + SCORE_RTOL * abs(ref)


def _angle_close(got: float, ref: float) -> bool:
    return abs((got - ref + math.pi) % (2 * math.pi) - math.pi) < ANGLE_TOL


def test_capture_grid_parity() -> None:
    valid_mismatch = 0
    compared_cells = 0
    value_fail: list[str] = []

    for b, obs in enumerate(_boards()):
        world, modes, feats = _build(obs)
        modes_arr = _modes_arr(modes)
        idx_map = _idx_map(feats)
        grid: CaptureGrid = build_capture_grid(feats, modes_arr)

        gv = np.asarray(grid.valid)
        gsingle = np.asarray(grid.is_single)
        gscore = np.asarray(grid.score)
        gangle = np.asarray(grid.angle)
        gturns = np.asarray(grid.turns)
        gneed = np.asarray(grid.needed)
        gsend = np.asarray(grid.send_cap)

        missions, options_by_target = _run_py_builder(world, modes)
        py_opts: dict[tuple[int, int], ShotOption] = {}
        for opts in options_by_target.values():
            for opt in opts:
                py_opts[(opt.src_id, opt.target_id)] = opt
        py_single: set[tuple[int, int]] = set()
        for m in missions:
            if m.kind == "single" and m.options:
                opt = m.options[0]
                py_single.add((opt.src_id, opt.target_id))

        # Compare every (my-planet src, target) cell.
        for src in world.my_planets:
            si = idx_map[src.id]
            for target in world.planets:
                if target.id == src.id:
                    continue
                ti = idx_map[target.id]
                compared_cells += 1
                jax_valid = bool(gv[si, ti])
                py_opt = py_opts.get((src.id, target.id))
                py_valid = py_opt is not None

                if jax_valid != py_valid:
                    valid_mismatch += 1
                    continue
                if not py_valid:
                    continue

                assert py_opt is not None
                # is_single flag
                jax_single = bool(gsingle[si, ti])
                py_is_single = (src.id, target.id) in py_single
                if jax_single != py_is_single:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) single jax={jax_single} "
                        f"py={py_is_single}"
                    )
                # exact int fields
                if int(gturns[si, ti]) != py_opt.turns:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) turns "
                        f"jax={int(gturns[si, ti])} py={py_opt.turns}"
                    )
                if int(gneed[si, ti]) != py_opt.needed:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) needed "
                        f"jax={int(gneed[si, ti])} py={py_opt.needed}"
                    )
                if int(gsend[si, ti]) != py_opt.send_cap:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) send_cap "
                        f"jax={int(gsend[si, ti])} py={py_opt.send_cap}"
                    )
                if not _close(float(gscore[si, ti]), py_opt.score):
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) score "
                        f"jax={float(gscore[si, ti])} py={py_opt.score}"
                    )
                if not _angle_close(float(gangle[si, ti]), py_opt.angle):
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) angle "
                        f"jax={float(gangle[si, ti])} py={py_opt.angle}"
                    )

    assert compared_cells > 0
    vfrac = valid_mismatch / compared_cells
    assert vfrac <= VALID_MISMATCH_CAP, (
        f"capture valid mismatch {valid_mismatch}/{compared_cells} "
        f"({100 * vfrac:.2f}%) exceeds {100 * VALID_MISMATCH_CAP:.0f}%"
    )
    assert not value_fail, (
        f"capture value divergence on {len(value_fail)} cells "
        f"(valid mismatch {100 * vfrac:.2f}%): " + "; ".join(value_fail[:12])
    )


def test_snipe_grid_parity() -> None:
    valid_mismatch = 0
    compared_cells = 0
    value_fail: list[str] = []

    for b, obs in enumerate(_boards()):
        world, modes, feats = _build(obs)
        modes_arr = _modes_arr(modes)
        idx_map = _idx_map(feats)
        grid: SnipeGrid = build_snipe_grid(feats, modes_arr)

        gv = np.asarray(grid.valid)
        gscore = np.asarray(grid.score)
        gturns = np.asarray(grid.turns)
        gneed = np.asarray(grid.needed)

        missions, _options = _run_py_builder(world, modes)
        py_snipe: dict[tuple[int, int], Mission] = {}
        for m in missions:
            if m.kind == "snipe" and m.options:
                opt = m.options[0]
                py_snipe[(opt.src_id, opt.target_id)] = m

        for src in world.my_planets:
            si = idx_map[src.id]
            for target in world.planets:
                if target.id == src.id or target.owner != -1:
                    continue
                ti = idx_map[target.id]
                compared_cells += 1
                jax_valid = bool(gv[si, ti])
                py_m = py_snipe.get((src.id, target.id))
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
                        f"b{b} ({src.id},{target.id}) snipe turns "
                        f"jax={int(gturns[si, ti])} py={opt.turns}"
                    )
                if int(gneed[si, ti]) != opt.needed:
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) snipe needed "
                        f"jax={int(gneed[si, ti])} py={opt.needed}"
                    )
                if not _close(float(gscore[si, ti]), py_m.score):
                    value_fail.append(
                        f"b{b} ({src.id},{target.id}) snipe score "
                        f"jax={float(gscore[si, ti])} py={py_m.score}"
                    )

    assert compared_cells > 0
    vfrac = valid_mismatch / compared_cells
    assert vfrac <= VALID_MISMATCH_CAP, (
        f"snipe valid mismatch {valid_mismatch}/{compared_cells} "
        f"({100 * vfrac:.2f}%) exceeds {100 * VALID_MISMATCH_CAP:.0f}%"
    )
    assert not value_fail, (
        f"snipe value divergence on {len(value_fail)} cells "
        f"(valid mismatch {100 * vfrac:.2f}%): " + "; ".join(value_fail[:12])
    )


def _snipe_scenario_obs() -> Obs:
    """A hand-built board where an enemy wave is inbound on a neutral target.

    Self-play noop boards never launch enemy fleets at neutrals, so the snipe
    branch needs a synthetic positive case: my home (id0) and an enemy (id2)
    both within sync range of neutral target (id1), with an enemy fleet timed so
    `abs(my_eta - enemy_eta) <= 1`. The Python builder fires a snipe Mission here.
    """
    ex, ey, tx, ty = 88.0, 50.0, 50.0, 10.0
    ang = math.atan2(ty - ey, tx - ex)
    fx = ex + (tx - ex) * 0.6
    fy = ey + (ty - ey) * 0.6
    planets = [
        [0, 0, 30.0, 20.0, 2.0, 40, 2],  # mine (home)
        [1, -1, 50.0, 10.0, 2.0, 5, 3],  # neutral target
        [2, 1, 88.0, 50.0, 2.0, 40, 2],  # enemy
    ]
    return {
        "player": 0,
        "step": 30,
        "planets": planets,
        "fleets": [[99, 1, fx, fy, ang, 2, 12]],
        "angular_velocity": 0.0,
        "initial_planets": planets,
        "comets": [],
        "comet_planet_ids": [],
    }


def test_snipe_grid_positive_case() -> None:
    """The JAX snipe grid reproduces a Python snipe Mission on a synthetic board.

    The self-play boards above produce zero snipe missions (no enemy waves on
    neutrals), so this exercises the snipe branch on a positive case and asserts
    EXACT turns / sync_turn / needed and rtol-1e-4 score / 1e-2 angle.
    """
    obs = _snipe_scenario_obs()
    world, modes, feats = _build(obs)
    modes_arr = _modes_arr(modes)
    idx_map = _idx_map(feats)

    missions, _options = _run_py_builder(world, modes)
    py_snipes = [m for m in missions if m.kind == "snipe" and m.options]
    assert py_snipes, "scenario should produce at least one Python snipe mission"

    grid: SnipeGrid = build_snipe_grid(feats, modes_arr)
    gv = np.asarray(grid.valid)
    gscore = np.asarray(grid.score)
    gangle = np.asarray(grid.angle)
    gturns = np.asarray(grid.turns)
    gsync = np.asarray(grid.sync_turn)
    gneed = np.asarray(grid.needed)

    for m in py_snipes:
        opt = m.options[0]
        si = idx_map[opt.src_id]
        ti = idx_map[opt.target_id]
        assert bool(gv[si, ti]), f"jax snipe missing for ({opt.src_id},{opt.target_id})"
        assert int(gturns[si, ti]) == opt.turns
        assert int(gsync[si, ti]) == m.turns
        assert int(gneed[si, ti]) == opt.needed
        assert _close(float(gscore[si, ti]), m.score)
        assert _angle_close(float(gangle[si, ti]), opt.angle)
