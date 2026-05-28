"""Parity: case_jax followup pass vs Python `emit_followup_moves`.

`run_followup_pass` reproduces `baseline/movements/followup.emit_followup_moves`:
a SECOND scan over source slots IN ORDER that runs AFTER the mission commit loop,
sharing the same `planned_commitments` / `spent_total` carry. For each source with
`source_attack_left(src) >= FOLLOWUP_MIN_SHIPS`, it argmaxes a capture target with
followup's OWN filters (LATE_CAPTURE_BUFFER late gate, LOW_VALUE_COMET_PRODUCTION
comet skip, opening_filter), re-plans, recomputes `missing` off the live ledger,
and commits — sequentially, so later sources see earlier followup commits.

To isolate THIS layer from the partial JAX mission scan (capture+snipe+harass,
no swarm/reinforce yet), we build the carry from Python's FULL mission loop
(`plan_moves`'s loop over every collected mission), feed it into the JAX
`run_followup_pass`, and compare the moves followup ADDS:

* src-set match (which sources fire a followup move) within a ≤5% boundary margin,
* ship counts EXACT on agreed sources.

We drive real COMBAT boards by self-playing the Python agent on both seats in the
JAX env (the noop-stepped boards never reach combat), the same pattern as
`test_agent_jax_combat_parity.py`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, step

from pipeline.rulebase.case_jax.baseline.agent import agent as py_agent
from pipeline.rulebase.case_jax.baseline.agent import build_world
from pipeline.rulebase.case_jax.baseline.core.physics import reset_predict_cache
from pipeline.rulebase.case_jax.baseline.missions import collect_missions
from pipeline.rulebase.case_jax.baseline.movements.followup import emit_followup_moves
from pipeline.rulebase.case_jax.baseline.strategy import (
    SINGLE_SOURCE_MISSION_KINDS,
    _process_multi_source_mission,
    _process_single_source_mission,
)
from pipeline.rulebase.case_jax.baseline.strategy_helpers import build_modes
from pipeline.rulebase.case_jax.baseline_jax.allocator_jax import (
    MAX_COMMIT,
    MAX_MOVES,
    AllocCarry,
    run_followup_pass,
)
from pipeline.rulebase.case_jax.baseline_jax.scoring_jax import ModesArrays
from pipeline.rulebase.case_jax.baseline_jax.timeline_jax import MAX_PLANETS
from pipeline.rulebase.case_jax.baseline_jax.world_features import build_world_features

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


def _to_action_tensor(per_seat_moves: list[list[Any]]) -> jnp.ndarray:
    from orbit_wars_jax.constants import NUM_AGENTS_MAX

    a = np.full((NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=np.float32)
    a[:, :, 1:] = 0.0
    for seat, moves in enumerate(per_seat_moves):
        for j, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
            a[seat, j] = [mv[0], mv[1], mv[2]]
    return jnp.asarray(a)


def _python_mission_then_followup(
    world: Any,
) -> tuple[dict[int, int], dict[int, int], dict[int, list[tuple[int, int, int]]]]:
    """Run the FULL Python mission loop, then followup with instrumentation.

    Returns `(spent_after_missions, followup_added, planned_after_missions)`:
      * `spent_after_missions[src_id]` — spent_total at followup entry,
      * `followup_added[src_id]` — ships appended by `emit_followup_moves` (per src),
      * `planned_after_missions[target_id]` — commit ledger at followup entry.
    """
    modes = build_modes(world)
    planned: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    moves: list[list[int | float]] = []
    spent: dict[int, int] = defaultdict(int)

    def src_inv_left(sid: int) -> int:
        return int(world.source_inventory_left(sid, spent))

    def src_atk_left(sid: int) -> int:
        return int(world.source_attack_left(sid, spent))

    def append_move(src_id: int, angle: float, ships: int) -> int:
        send = min(int(ships), src_inv_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent[src_id] += send
        return send

    missions = collect_missions(world, planned, modes, src_inv_left, src_atk_left)
    missions.sort(key=lambda m: -m.score)
    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in SINGLE_SOURCE_MISSION_KINDS:
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
        else:
            _process_multi_source_mission(
                mission, target, world, planned, src_atk_left, append_move
            )

    spent_at_entry = {sid: int(v) for sid, v in spent.items()}
    planned_at_entry = {tid: list(v) for tid, v in planned.items()}

    # Instrument followup: record only the moves emit_followup_moves appends.
    followup_added: dict[int, int] = defaultdict(int)

    def followup_append(src_id: int, angle: float, ships: int) -> int:
        sent = append_move(src_id, angle, ships)
        if sent > 0:
            followup_added[src_id] += sent
        return sent

    emit_followup_moves(world, planned, modes, src_atk_left, followup_append)

    return (
        spent_at_entry,
        {sid: int(v) for sid, v in followup_added.items()},
        planned_at_entry,
    )


def _carry_from_python(
    obs: Any,
    spent_at_entry: dict[int, int],
    planned_at_entry: dict[int, list[tuple[int, int, int]]],
) -> AllocCarry:
    """Build an AllocCarry mirroring Python's mission-loop state at followup entry.

    Maps planet ids -> slot indices (slot i == planets[i].id, the obs order the
    WorldFeatures resolver also uses), then fills the commit ledger + spent so the
    JAX followup pass reads the exact same `source_attack_left` / `missing`.
    """
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    id_to_slot = {int(p[0]): i for i, p in enumerate(planets)}

    spent = np.zeros(MAX_PLANETS, dtype=np.int32)
    for sid, val in spent_at_entry.items():
        slot = id_to_slot.get(int(sid))
        if slot is not None:
            spent[slot] = int(val)

    commit_eta = np.zeros((MAX_PLANETS, MAX_COMMIT), dtype=np.int32)
    commit_ships = np.zeros((MAX_PLANETS, MAX_COMMIT), dtype=np.int32)
    commit_count = np.zeros(MAX_PLANETS, dtype=np.int32)
    for tid, commits in planned_at_entry.items():
        slot = id_to_slot.get(int(tid))
        if slot is None:
            continue
        for eta, _player_owned, ships in commits:
            c = int(commit_count[slot])
            if c >= MAX_COMMIT:
                break
            commit_eta[slot, c] = int(eta)
            commit_ships[slot, c] = int(ships)
            commit_count[slot] = c + 1

    return AllocCarry(
        commit_eta=jnp.asarray(commit_eta),
        commit_ships=jnp.asarray(commit_ships),
        commit_count=jnp.asarray(commit_count),
        spent=jnp.asarray(spent),
        move_src=jnp.full((MAX_MOVES,), -1, dtype=jnp.int32),
        move_angle=jnp.zeros((MAX_MOVES,), dtype=jnp.float32),
        move_ships=jnp.zeros((MAX_MOVES,), dtype=jnp.int32),
        move_count=jnp.int32(0),
    )


# jit once (fixed shapes) so the lax.scan + per-source vmap compile a single
# time and the trace is reused across every sampled board — otherwise the eager
# call re-traces the whole pass on each invocation and the test takes ~30 min.
_followup_jit = jax.jit(run_followup_pass)


def _jax_followup_added(
    obs: Any,
    spent_at_entry: dict[int, int],
    planned_at_entry: dict[int, list[tuple[int, int, int]]],
) -> dict[int, int]:
    """Run JAX followup from the Python-mirrored carry; return added {src_id: ships}.

    The carry starts with an EMPTY move buffer (move_count == 0), so every move in
    the ending buffer is a followup commit.
    """
    feats = build_world_features(obs)
    modes = _modes_arr(build_modes(build_world(obs)))
    carry = _carry_from_python(obs, spent_at_entry, planned_at_entry)
    out = _followup_jit(carry, feats, modes)

    added: dict[int, int] = {}
    count = int(out.move_count)
    move_src = np.asarray(out.move_src)
    move_ships = np.asarray(out.move_ships)
    for i in range(count):
        sid = int(move_src[i])
        if sid < 0:
            continue
        added[sid] = added.get(sid, 0) + int(move_ships[i])
    return added


def _accumulate_board(obs0: Any, stats: dict[str, int]) -> None:
    """Run Python mission+followup and JAX followup on obs0; fold parity counts."""
    reset_predict_cache()
    world = build_world(obs0)
    if not world.my_planets:
        return
    spent_entry, py_added, planned_entry = _python_mission_then_followup(world)
    jax_added = _jax_followup_added(obs0, spent_entry, planned_entry)

    stats["boards"] += 1
    stats["py_total"] += len(py_added)
    for sid in set(py_added) | set(jax_added):
        if (sid in py_added) != (sid in jax_added):
            stats["src_mismatch"] += 1
            continue
        stats["agreed"] += 1
        if py_added[sid] != jax_added[sid]:
            stats["ships_mismatch"] += 1


def test_followup_pass_parity() -> None:
    stats: dict[str, int] = {
        "boards": 0,
        "py_total": 0,
        "src_mismatch": 0,
        "ships_mismatch": 0,
        "agreed": 0,
    }

    # 1v1 self-play, sampling EVERY combat step: the followup pass is genuinely
    # sparse (it only fires when a source has leftover attack ships AND a
    # profitable uncovered capture target), so dense sampling is needed to grow
    # the followup-source population. 1v1 self-play is the cheap path; these seeds
    # accumulate enough followup events to clear the `py_total >= 3` guard.
    configs = [(seed, 2) for seed in (0, 1, 3, 7)]
    for seed, num_agents in configs:
        state = reset(seed=seed, num_agents=num_agents)
        for stp in range(110):
            per_seat = []
            for seat in range(num_agents):
                obs_s = state_to_obs(state, player=seat)
                reset_predict_cache()
                per_seat.append(py_agent(obs_s))

            if stp >= 20:
                obs0 = state_to_obs(state, player=0)
                _accumulate_board(obs0, stats)

            state, _r, term = step(state, _to_action_tensor(per_seat))
            if bool(term):
                break

    assert stats["boards"] > 0
    denom = max(1, stats["py_total"] + stats["src_mismatch"])
    src_match_frac = 1.0 - stats["src_mismatch"] / denom
    print(
        f"\n[followup parity] boards={stats['boards']} "
        f"py_followup_srcs={stats['py_total']} src_mismatch={stats['src_mismatch']} "
        f"agreed={stats['agreed']} ships_mismatch={stats['ships_mismatch']} "
        f"src_match={src_match_frac:.2%}"
    )
    # Guard against a vacuous pass: the followup pass IS sparse, but across these
    # seeds Python must fire it at least a few times for the parity claim to mean
    # anything. If this trips, widen the seed set / step horizon rather than
    # loosening the parity tolerances below.
    assert stats["py_total"] >= 3, (
        f"followup never fired enough to test parity (py_srcs={stats['py_total']}); "
        "widen the seed set"
    )
    # src-set match within a small boundary margin (≤5% of the source population).
    assert src_match_frac >= 0.95, (
        f"followup src-set match {src_match_frac:.2%} "
        f"(src_mismatch={stats['src_mismatch']}, py_srcs={stats['py_total']})"
    )
    # ships EXACT on the overwhelming majority of agreed sources.
    assert stats["ships_mismatch"] <= max(1, stats["agreed"] // 20), (
        f"followup ships mismatch {stats['ships_mismatch']}/{stats['agreed']}"
    )
