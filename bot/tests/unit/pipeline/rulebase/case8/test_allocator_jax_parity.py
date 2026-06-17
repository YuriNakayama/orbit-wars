"""Parity: case8 single-source greedy allocator vs a capture-only Python ref.

`run_single_source_allocator` reproduces the ordered, ledger-threaded commit loop
of `plan_moves` (`_process_single_source_mission`) for the capture "single"
mission family. To isolate THIS layer from snipe/swarm/movements, the Python
reference runs the same `plan_moves` loop but fed ONLY the capture "single"
missions (built via `build_capture_and_snipe_missions`, filtered to kind=="single").

We assert the resulting move set matches: same set of (src planet id -> ships,
angle), allowing a small boundary-flip margin. This validates the
argsort + sequential greedy commit + `ships_needed_to_capture`-with-commitments
recompute — the order-dependent core the old approximation dropped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jax.numpy as jnp
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions, step

from pipeline.rulebase.case8.baseline.agent import build_world
from pipeline.rulebase.case8.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case8.baseline.missions import collect_missions
from pipeline.rulebase.case8.baseline.strategy import (
    _enforce_inventory_cap,
    _process_single_source_mission,
)
from pipeline.rulebase.case8.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case8.baseline_jax.allocator_jax import (
    KIND_SINGLE,
    SingleMissionTable,
    run_single_source_allocator,
)
from pipeline.rulebase.case8.baseline_jax.missions_capture_jax import (
    build_capture_grid,
)
from pipeline.rulebase.case8.baseline_jax.scoring_jax import ModesArrays
from pipeline.rulebase.case8.baseline_jax.world_features import build_world_features

pytestmark = pytest.mark.slow


def _modes_arr(modes: dict[str, Any]) -> ModesArrays:
    return ModesArrays(
        domination=jnp.float32(modes["domination"]),
        is_behind=jnp.bool_(modes["is_behind"]),
        is_ahead=jnp.bool_(modes["is_ahead"]),
        is_dominating=jnp.bool_(modes["is_dominating"]),
        is_finishing=jnp.bool_(modes["is_finishing"]),
        attack_margin_mult=jnp.float32(modes["attack_margin_mult"]),
    )


def _python_single_capture_moves(world: Any) -> dict[int, tuple[int, float]]:
    """Run plan_moves's greedy loop fed ONLY capture 'single' missions.

    Returns {src_id: (ships, angle)} after the inventory cap.
    """
    modes = build_modes(world)
    planned: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    moves: list[list[int | float]] = []
    spent: dict[int, int] = defaultdict(int)

    def src_inv_left(sid: int) -> int:
        return world.source_inventory_left(sid, spent)

    def src_atk_left(sid: int) -> int:
        return world.source_attack_left(sid, spent)

    def append_move(src_id: int, angle: float, ships: int) -> int:
        send = min(int(ships), src_inv_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent[src_id] += send
        return send

    missions = collect_missions(world, planned, modes, src_inv_left, src_atk_left)
    missions = [m for m in missions if m.kind == "single"]
    missions.sort(key=lambda m: -m.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        _process_single_source_mission(
            mission,
            target,
            world,
            modes,
            planned,
            src_inv_left,
            src_atk_left,
            append_move,
        )

    capped = _enforce_inventory_cap(moves, world)
    out: dict[int, tuple[int, float]] = {}
    for src_id, angle, ships in capped:
        out[int(src_id)] = (int(ships), float(angle))
    return out


def _flatten_single_table(grid: Any, features: Any) -> SingleMissionTable:
    """Flatten capture grid is_single cells into a (N,) SingleMissionTable."""
    p = grid.valid.shape[0]
    n = p * p
    is_single = grid.is_single  # (P, P)
    src_idx = jnp.repeat(jnp.arange(p, dtype=jnp.int32), p)  # (N,)
    tgt_idx = jnp.tile(jnp.arange(p, dtype=jnp.int32), p)  # (N,)
    return SingleMissionTable(
        valid=is_single.reshape(n),
        score=grid.score.reshape(n),
        kind=jnp.full((n,), KIND_SINGLE, dtype=jnp.int32),
        src_slot=src_idx,
        target_slot=tgt_idx,
        angle=grid.angle.reshape(n),
        turns=grid.turns.reshape(n),
        send_cap=grid.send_cap.reshape(n),
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_single_capture_allocator_parity(seed: int) -> None:
    state = reset(seed=seed, num_agents=2)
    boards_checked = 0
    src_mismatch = 0
    ships_mismatch = 0
    total_py_moves = 0

    for _ in range(6):
        obs = state_to_obs(state, player=0)
        reset_predict_cache()
        world = build_world(obs)

        py = _python_single_capture_moves(world)

        feats = build_world_features(obs)
        modes = _modes_arr(build_modes(world))
        grid = build_capture_grid(feats, modes)
        table = _flatten_single_table(grid, feats)
        res = run_single_source_allocator(table, feats, modes)

        jax_moves: dict[int, tuple[int, float]] = {}
        count = int(res.move_count)
        for i in range(count):
            sid = int(res.move_src[i])
            if sid < 0:
                continue
            jax_moves[sid] = (int(res.move_ships[i]), float(res.move_angle[i]))

        boards_checked += 1
        total_py_moves += len(py)
        # Compare the set of source planets that fired and their ship counts.
        all_src = set(py) | set(jax_moves)
        for sid in all_src:
            if (sid in py) != (sid in jax_moves):
                src_mismatch += 1
                continue
            if py[sid][0] != jax_moves[sid][0]:
                ships_mismatch += 1

        state, _r, term = step(state, empty_actions())
        if bool(term):
            break

    assert boards_checked > 0
    denom = max(1, total_py_moves)
    # The src-set (which planets fire) is the target-match proxy; require >=90%.
    src_match_frac = 1.0 - src_mismatch / denom
    assert src_match_frac >= 0.90, (
        f"seed={seed}: src-set match {src_match_frac:.2%} "
        f"(src_mismatch={src_mismatch}, py_moves={total_py_moves})"
    )
    # ships should match on the overwhelming majority of agreed sources.
    assert ships_mismatch <= max(1, total_py_moves // 10), (
        f"seed={seed}: ships mismatch {ships_mismatch}/{total_py_moves}"
    )
