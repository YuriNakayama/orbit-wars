# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""End-to-end fully-JAX rulebase agent driver (case8).

Assembles the parity-tested pieces into a single jit/vmap-able pipeline that
mirrors `baseline/strategy.plan_moves`:

    WorldFeatures (host-resolved) -> mission grids -> flatten to a candidate
    table -> argsort(-score) -> lax.scan greedy commit -> inventory cap ->
    (MAX_LAUNCHES_PER_AGENT, 3) action tensor.

Currently wires the single-source mission family that dominates real play
(capture "single" + snipe + harass) followed by the FOLLOWUP movement pass — a
second source-ordered scan that spends leftover attack ships on opportunistic
captures, sharing the mission scan's ending carry (`run_mission_and_followup`).
The remaining families (reinforce / swarm) and the other movement emitters
(evacuation / rear_guard) fold into the same candidate table / sibling passes;
they are layered in incrementally while this driver gives a runnable agent + the
full `plan_moves` target-match measurement.

Column layout of the output matches `jax_env.step`: (from_planet_id, angle,
ships); a sentinel `from_planet_id == -1` suppresses the launch.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT

from ._config_compat import HARASS_ENABLED
from .allocator_jax import (
    KIND_HARASS,
    KIND_SINGLE,
    KIND_SNIPE,
    MAX_MOVES,
    AllocResult,
    SingleMissionTable,
    run_mission_and_followup,
)
from .missions_capture_jax import (
    CaptureGrid,
    HarassGrid,
    SnipeGrid,
    build_capture_grid,
    build_harass_grid,
    build_snipe_grid,
)
from .missions_capture_jax import _base_timelines as _base_timelines  # noqa: PLC0414
from .scoring_jax import ModesArrays
from .world_features import WorldFeatures, build_world_features


def _modes_from_features(features: WorldFeatures) -> ModesArrays:
    """WorldFeatures already carries the build_modes scalars; repack as ModesArrays."""
    return ModesArrays(
        domination=features.domination,
        is_behind=features.is_behind,
        is_ahead=features.is_ahead,
        is_dominating=features.is_dominating,
        is_finishing=features.is_finishing,
        attack_margin_mult=features.attack_margin_mult,
    )


def _combine_single_table(
    capture: CaptureGrid, snipe: SnipeGrid, harass: HarassGrid | None
) -> SingleMissionTable:
    """Flatten capture "single" + snipe (+ harass) cells into one candidate table.

    Each family contributes a `(P*P,)` block. Capture rows where `is_single`
    (Python emits a "single" Mission); snipe / harass rows where their grid is
    valid. `turns` is the option arrival turn; `send_cap` is the per-option cap
    (capture: src_available; snipe / harass option.send_cap == `needed`).
    The greedy scan dispatches on `kind` (capture → preferred_send, snipe/harass
    → exact `missing`). When `harass is None` (HARASS_ENABLED off) its block is
    omitted — those rows are all-invalid so the resulting table is identical.
    """
    p = capture.valid.shape[0]
    n = p * p
    src_idx = jnp.repeat(jnp.arange(p, dtype=jnp.int32), p)
    tgt_idx = jnp.tile(jnp.arange(p, dtype=jnp.int32), p)

    valid_blocks = [capture.is_single.reshape(n), snipe.valid.reshape(n)]
    score_blocks = [capture.score.reshape(n), snipe.score.reshape(n)]
    kind_blocks = [
        jnp.full((n,), KIND_SINGLE, dtype=jnp.int32),
        jnp.full((n,), KIND_SNIPE, dtype=jnp.int32),
    ]
    src_blocks = [src_idx, src_idx]
    tgt_blocks = [tgt_idx, tgt_idx]
    angle_blocks = [capture.angle.reshape(n), snipe.angle.reshape(n)]
    turns_blocks = [capture.turns.reshape(n), snipe.turns.reshape(n)]
    # capture send_cap is the source-available cap; snipe send_cap == needed.
    send_cap_blocks = [capture.send_cap.reshape(n), snipe.needed.reshape(n)]

    if harass is not None:
        valid_blocks.append(harass.valid.reshape(n))
        score_blocks.append(harass.score.reshape(n))
        kind_blocks.append(jnp.full((n,), KIND_HARASS, dtype=jnp.int32))
        src_blocks.append(src_idx)
        tgt_blocks.append(tgt_idx)
        angle_blocks.append(harass.angle.reshape(n))
        turns_blocks.append(harass.turns.reshape(n))
        send_cap_blocks.append(harass.needed.reshape(n))

    return SingleMissionTable(
        valid=jnp.concatenate(valid_blocks),
        score=jnp.concatenate(score_blocks),
        kind=jnp.concatenate(kind_blocks),
        src_slot=jnp.concatenate(src_blocks),
        target_slot=jnp.concatenate(tgt_blocks),
        angle=jnp.concatenate(angle_blocks),
        turns=jnp.concatenate(turns_blocks),
        send_cap=jnp.concatenate(send_cap_blocks),
    )


def _alloc_to_action_tensor(res: AllocResult, features: WorldFeatures) -> jax.Array:
    """Pack the allocator move buffer into a (MAX_LAUNCHES_PER_AGENT, 3) tensor.

    Applies `_enforce_inventory_cap`: clamp each source's cumulative spend to its
    real `ships`. The allocator already enforces inventory via its spent ledger
    (it derives sends from `ships - spent`), so the buffer is cap-consistent; we
    still clamp defensively to match the Python final pass exactly.
    """
    move_src = res.move_src  # (MAX_MOVES,) planet id, -1 == empty
    move_angle = res.move_angle
    move_ships = res.move_ships

    # Map planet id -> ships available (for the defensive cap). Build per-slot
    # ship capacity by looking up the source planet's ships via planet_id match.
    # The allocator already respects inventory, so this is identity in practice;
    # we keep the buffer order (== mission priority order) and pad to launches.
    out = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), 0.0, dtype=jnp.float32)
    slots = jnp.arange(MAX_LAUNCHES_PER_AGENT)
    valid_slot = (slots < res.move_count) & (move_src[:MAX_LAUNCHES_PER_AGENT] >= 0)

    from_pid = jnp.where(
        valid_slot, move_src[:MAX_LAUNCHES_PER_AGENT], jnp.int32(-1)
    ).astype(jnp.float32)
    angle_col = jnp.where(valid_slot, move_angle[:MAX_LAUNCHES_PER_AGENT], 0.0)
    ships_col = jnp.where(
        valid_slot, move_ships[:MAX_LAUNCHES_PER_AGENT], jnp.int32(0)
    ).astype(jnp.float32)

    out = out.at[:, 0].set(from_pid)
    out = out.at[:, 1].set(angle_col)
    out = out.at[:, 2].set(ships_col)
    del features
    return out


def compute_actions(features: WorldFeatures, modes: ModesArrays) -> jax.Array:
    """Run the single-source pipeline → (MAX_LAUNCHES_PER_AGENT, 3) action tensor.

    `features` / `modes` are produced by `build_world_features(obs)` on the host.
    Pure & jit/vmap-able given the resolved features.
    """
    # Compute the per-planet base timelines ONCE and share across grid builders
    # (doc#1 hoist): each builder otherwise recomputes the 48-planet HORIZON
    # timeline. Same values, fewer redundant scans.
    base_tl = _base_timelines(features)
    capture = build_capture_grid(features, modes, base_tl)
    snipe = build_snipe_grid(features, modes, base_tl)
    # HARASS_ENABLED is a static Python bool (False for case1). When disabled the
    # harass grid emits only invalid (score -inf) rows that never commit, so the
    # whole 2304-cell harass scan (~20% of per-turn compute, doc profile) is pure
    # waste. Skip it entirely — the candidate table is identical without it.
    harass = build_harass_grid(features, modes, base_tl) if HARASS_ENABLED else None
    table = _combine_single_table(capture, snipe, harass)
    res = run_mission_and_followup(table, features, modes)
    return _alloc_to_action_tensor(res, features)


_jitted_compute = jax.jit(compute_actions)


def compute_actions_jit(features: WorldFeatures, modes: ModesArrays) -> jax.Array:
    """jit-compiled `compute_actions` (one trace reused across calls)."""
    return jnp.asarray(_jitted_compute(features, modes))


def agent(obs: Any) -> list[list[int | float]]:
    """Kaggle-style agent callable backed by the JAX pipeline.

    Mirrors `baseline.agent.agent`'s contract: takes the kaggle observation dict
    and returns `[[from_planet_id, angle, num_ships], ...]`. Resolves the obs into
    `WorldFeatures` on the host, runs the jitted `compute_actions`, and unpacks
    the `(MAX_LAUNCHES_PER_AGENT, 3)` tensor back into the move list (dropping
    sentinel `from_planet_id == -1` rows). Lets the standard self-play /
    evaluation harness drive the JAX agent.
    """
    features = build_world_features(obs)
    modes = _modes_from_features(features)
    act = compute_actions_jit(features, modes)
    moves: list[list[int | float]] = []
    for i in range(int(act.shape[0])):
        from_pid = int(act[i, 0])
        if from_pid < 0:
            continue
        ships = int(act[i, 2])
        if ships < 1:
            continue
        moves.append([from_pid, float(act[i, 1]), ships])
    return moves


__all__ = [
    "MAX_MOVES",
    "agent",
    "compute_actions",
    "compute_actions_jit",
]
