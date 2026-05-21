"""Parity tests for the W1 JAX featurizer.

Compares `featurize_jax_w1(state)` against the existing PyTorch
`featurize(obs)` for the columns implemented in W1. Other columns are
zero-filled in W1 and excluded from this test; they will be covered as
W2/W3 ships.

Run path: reset the JAX env at a few seeds, step through a fixed number
of no-op turns (so history-dependent columns and fleet-dependent columns
stay zero), then assert column-by-column.

Tolerance: 1e-4 (float32 + slightly different operation orders between
NumPy and JAX). Per the W1 plan in `docs/plans/jax-env/01-design.md`,
this is the BC-compatibility budget.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from jax_env.observation import state_to_obs
from jax_env.reset import reset
from jax_env.state import EnvState
from jax_env.step import empty_actions, step
from pipeline.reinforce.case1.policy.featurizer import (
    HistoryState,
    featurize,
    update_history,
)
from pipeline.reinforce.case1.policy.featurizer_jax import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize_jax_w1,
    init_history_jax,
    update_history_jax,
)

PARITY_TOL = 1e-4

# Columns implemented through W2a (planet×planet) + W2b (fleet ETA) +
# W2c (orbit predictions). Remaining: 32-34 (delta_t1/t2/owner_changed
# history), 35-40 (timeline).
W1_PLANET_COLS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    35,
    36,
    37,
    38,
    39,
    40,
)
W1_GLOBAL_COLS = tuple(i for i in range(GLOBAL_FEAT_DIM) if i not in (16, 17, 18, 19))


def _build_fire_actions(state: EnvState) -> jnp.ndarray:
    """Action tensor that fires 1 ship at angle 0 from each seat's home.

    Sentinel: from_planet_id == -1 means no-op. We populate row 0 of seats
    0 and 1 with their first own-planet (ships >= 2) so both sides spawn
    a fleet each turn, exercising the W2-fleet code path.
    """
    actions = np.full(empty_actions().shape, -1, dtype=np.int32)
    pid = np.asarray(state.planet_id)
    owner = np.asarray(state.planet_owner)
    valid = np.asarray(state.planet_valid)
    ships = np.asarray(state.planet_ships)
    for seat in (0, 1):
        for i in range(pid.shape[0]):
            if bool(valid[i]) and int(owner[i]) == seat and int(ships[i]) >= 2:
                actions[seat, 0, 0] = int(pid[i])
                actions[seat, 0, 1] = 0
                actions[seat, 0, 2] = 1
                break
    return jnp.asarray(actions)


def _featurize_both(
    seed: int, turns: int, with_fleets: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Run both featurizers from a fresh state with `turns` steps.

    When `with_fleets=True`, both seats fire 1 ship from their home each
    turn so the fleet ETA / incoming columns are exercised.
    """
    js = reset(seed=seed, num_agents=2)
    ea = empty_actions()
    for _ in range(turns):
        if with_fleets:
            actions = _build_fire_actions(js)
            js, _, _ = step(js, actions)
        else:
            js, _, _ = step(js, ea)
    obs = state_to_obs(js, player=0)
    jax_batch = featurize_jax_w1(js, player=0)
    torch_batch, _snap = featurize(obs, history=HistoryState())
    jax_planet = np.asarray(jax_batch.planet_feats[0])
    torch_planet = torch_batch.planet_feats[0].cpu().numpy()
    jax_global = np.asarray(jax_batch.global_feats[0])
    torch_global = torch_batch.global_feats[0].cpu().numpy()
    return (
        jax_planet,
        torch_planet,
        {
            "valid": np.asarray(jax_batch.planet_mask[0]),
            "global": jax_global,
        },
        {
            "valid": torch_batch.planet_mask[0].cpu().numpy(),
            "global": torch_global,
        },
    )


@pytest.mark.parametrize("seed", [0, 7, 13, 42])
@pytest.mark.parametrize("turns", [0, 1, 10, 30])
def test_global_feats_w1_columns(seed: int, turns: int) -> None:
    """Global feature columns 0..15 must match within 1e-4."""
    _, _, jx, tx = _featurize_both(seed, turns)
    for col in W1_GLOBAL_COLS:
        diff = abs(float(jx["global"][col]) - float(tx["global"][col]))
        assert diff < PARITY_TOL, (
            f"seed={seed} turns={turns} global col {col}: "
            f"jax={jx['global'][col]:.6f} torch={tx['global'][col]:.6f} "
            f"diff={diff:.3e}"
        )


@pytest.mark.parametrize("seed", [0, 7, 13, 42])
@pytest.mark.parametrize("turns", [0, 1, 10, 30])
def test_planet_feats_w1_columns(seed: int, turns: int) -> None:
    """Planet feature columns implemented in W1 must match within 1e-4."""
    jax_planet, torch_planet, jx, _tx = _featurize_both(seed, turns)
    valid = jx["valid"]
    for col in W1_PLANET_COLS:
        # Only compare valid slots; invalid slots are zero in both.
        for slot in range(MAX_PLANETS):
            if not bool(valid[slot]):
                continue
            j = float(jax_planet[slot, col])
            t = float(torch_planet[slot, col])
            diff = abs(j - t)
            assert diff < PARITY_TOL, (
                f"seed={seed} turns={turns} slot={slot} col={col}: "
                f"jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
            )


def test_w1_zero_filled_cols_are_zero_in_jax() -> None:
    """W1 leaves the un-implemented columns as zeros (sanity check).

    Documents the W1→W2 contract: any column not in W1_PLANET_COLS /
    W1_GLOBAL_COLS must read as zero from `featurize_jax_w1`.
    """
    js = reset(seed=0, num_agents=2)
    jax_batch = featurize_jax_w1(js, player=0)
    planet_arr = np.asarray(jax_batch.planet_feats[0])
    global_arr = np.asarray(jax_batch.global_feats[0])
    for col in range(PLANET_FEAT_DIM):
        if col in W1_PLANET_COLS:
            continue
        assert np.allclose(planet_arr[:, col], 0.0), (
            f"planet col {col} should be zero in W1, got {planet_arr[:, col]}"
        )
    for col in range(GLOBAL_FEAT_DIM):
        if col in W1_GLOBAL_COLS:
            continue
        assert float(global_arr[col]) == 0.0, (
            f"global col {col} should be zero in W1, got {global_arr[col]}"
        )


# W2b: fleet ETA columns. Run with active fleets so the cols actually carry
# information and parity is meaningful.
W2B_PLANET_COLS = (9, 10, 17, 28, 29, 30, 31)
# idx 16 in the with-fleets case also includes the fleet contribution.
W2B_PLANET_COLS_FLEET_TOUCHED = W2B_PLANET_COLS + (16,)


# W2c parity past the first comet spawn (step 50) so the comet orbit
# branch (cols 20-27 for comet planets) is exercised. Restricted to
# seeds where Rust matches Python vendor in comet rejection sampling.
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("turns", [60, 80])
def test_planet_feats_w2c_comet_orbit(seed: int, turns: int) -> None:
    """Orbit columns (20-27) must match for comet planets after step 50."""
    jax_planet, torch_planet, jx, _tx = _featurize_both(seed, turns)
    valid = jx["valid"]
    orbit_cols = tuple(range(20, 28))
    for col in orbit_cols:
        for slot in range(MAX_PLANETS):
            if not bool(valid[slot]):
                continue
            j = float(jax_planet[slot, col])
            t = float(torch_planet[slot, col])
            diff = abs(j - t)
            assert diff < PARITY_TOL, (
                f"seed={seed} turns={turns} slot={slot} col={col} "
                f"(orbit): jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
            )


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("turns", [10, 30])
def test_planet_feats_fleet_columns(seed: int, turns: int) -> None:
    """Fleet-dependent columns must match within tolerance with active fleets."""
    jax_planet, torch_planet, jx, _tx = _featurize_both(seed, turns, with_fleets=True)
    valid = jx["valid"]
    for col in W2B_PLANET_COLS_FLEET_TOUCHED:
        for slot in range(MAX_PLANETS):
            if not bool(valid[slot]):
                continue
            j = float(jax_planet[slot, col])
            t = float(torch_planet[slot, col])
            diff = abs(j - t)
            assert diff < PARITY_TOL, (
                f"seed={seed} turns={turns} slot={slot} col={col} "
                f"(fleets): jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
            )


# W2d: history-dependent cols (delta_t1/t2/owner_changed = 32-34;
# global launch counts = 16-19). Run both featurizers in lockstep,
# updating histories each turn so they reflect the same state.
W2D_PLANET_COLS = (32, 33, 34)
W2D_GLOBAL_COLS = (16, 17, 18, 19)


def _seat0_action_rows(
    state: EnvState,
) -> tuple[list[list[int | float]], np.ndarray, np.ndarray, np.ndarray]:
    """Build seat 0's PyTorch-style action list and matching JAX arrays.

    Returns:
      torch_actions: list of [from_pid, angle, ships] (vendor format)
      pid: int32 (L,) — JAX history input
      ships: int32 (L,)
      valid: bool (L,)
    """
    pid = np.asarray(state.planet_id)
    owner = np.asarray(state.planet_owner)
    valid = np.asarray(state.planet_valid)
    ships = np.asarray(state.planet_ships)
    torch_actions: list[list[int | float]] = []
    pid_buf = []
    ships_buf = []
    valid_buf = []
    for i in range(pid.shape[0]):
        if bool(valid[i]) and int(owner[i]) == 0 and int(ships[i]) >= 2:
            torch_actions.append([int(pid[i]), 0.0, 1])
            pid_buf.append(int(pid[i]))
            ships_buf.append(1)
            valid_buf.append(True)
            break
    # Pad to fixed length 4 (small upper bound — seat fires at most 1 / turn).
    while len(pid_buf) < 4:
        pid_buf.append(-1)
        ships_buf.append(0)
        valid_buf.append(False)
    return (
        torch_actions,
        np.asarray(pid_buf, dtype=np.int32),
        np.asarray(ships_buf, dtype=np.int32),
        np.asarray(valid_buf, dtype=np.bool_),
    )


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("turns", [3, 6, 10])
def test_planet_feats_w2d_history(seed: int, turns: int) -> None:
    """delta_t1/t2/owner_changed + launch-count globals match after N turns.

    Drives both featurizers from the same JAX env trajectory with seat 0
    firing 1 ship from its first ready home each turn. Both histories
    are updated in lockstep; at turn N we assert cols 32-34 (planet)
    and 16-19 (global) match within tol.
    """
    js = reset(seed=seed, num_agents=2)
    torch_history = HistoryState()
    jax_history = init_history_jax()
    for _ in range(turns):
        # Mimic the reinforce/case1 rollout loop: featurize then push
        # this turn's snapshot + launches into history.
        obs = state_to_obs(js, player=0)
        torch_acts, jp, js_, jv = _seat0_action_rows(js)
        update_history(torch_history, obs, actions=torch_acts)
        jax_history = update_history_jax(
            jax_history,
            js,
            jnp.asarray(jp),
            jnp.asarray(js_),
            jnp.asarray(jv),
            player=0,
        )
        # Step the env (both seats fire so the trajectory has fleets).
        actions = _build_fire_actions(js)
        js, _, _ = step(js, actions)

    # Now compare final-turn featurize output.
    obs = state_to_obs(js, player=0)
    torch_batch, _ = featurize(obs, history=torch_history)
    jax_batch = featurize_jax_w1(js, player=0, history=jax_history)
    torch_p = torch_batch.planet_feats[0].cpu().numpy()
    jax_p = np.asarray(jax_batch.planet_feats[0])
    torch_g = torch_batch.global_feats[0].cpu().numpy()
    jax_g = np.asarray(jax_batch.global_feats[0])
    mask = torch_batch.planet_mask[0].cpu().numpy()

    for col in W2D_PLANET_COLS:
        for slot in range(MAX_PLANETS):
            if not bool(mask[slot]):
                continue
            j = float(jax_p[slot, col])
            t = float(torch_p[slot, col])
            diff = abs(j - t)
            assert diff < PARITY_TOL, (
                f"seed={seed} turns={turns} slot={slot} col={col} "
                f"(history): jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
            )
    for col in W2D_GLOBAL_COLS:
        j = float(jax_g[col])
        t = float(torch_g[col])
        diff = abs(j - t)
        assert diff < PARITY_TOL, (
            f"seed={seed} turns={turns} global col {col}: "
            f"jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
        )


# W2f: template_ctx (40-dim per own planet). Verified at empty-history /
# no-fleet path (simpler) and with active fleets.
@pytest.mark.parametrize("seed", [0, 7, 13, 42])
@pytest.mark.parametrize("turns", [0, 5, 20])
def test_template_ctx_matches_torch(seed: int, turns: int) -> None:
    """template_ctx must match torch for every own planet × every col."""
    jax_p_unused, _, _, _ = _featurize_both(seed, turns)  # warmup
    js = reset(seed=seed, num_agents=2)
    ea = empty_actions()
    for _ in range(turns):
        js, _, _ = step(js, ea)
    obs = state_to_obs(js, player=0)
    torch_batch, _ = featurize(obs, history=HistoryState())
    jax_batch = featurize_jax_w1(js, player=0)
    torch_ctx = torch_batch.template_ctx[0].cpu().numpy()
    jax_ctx = np.asarray(jax_batch.template_ctx[0])
    mask_mine = np.asarray(jax_batch.my_planet_mask[0])

    for slot in range(MAX_PLANETS):
        if not bool(mask_mine[slot]):
            # Non-own slots are zero in both; skip.
            continue
        for col in range(40):
            j = float(jax_ctx[slot, col])
            t = float(torch_ctx[slot, col])
            diff = abs(j - t)
            assert diff < PARITY_TOL, (
                f"seed={seed} turns={turns} slot={slot} col={col} "
                f"(template_ctx): jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
            )


# W2-final: candidate block (per own src × CAND_K=8 candidates × 14 feats).
@pytest.mark.parametrize("seed", [0, 7, 13, 42])
@pytest.mark.parametrize("turns", [0, 5, 20])
def test_candidate_block_matches_torch(seed: int, turns: int) -> None:
    """candidate_feats / candidate_mask / candidate_pid must match torch."""
    js = reset(seed=seed, num_agents=2)
    ea = empty_actions()
    for _ in range(turns):
        js, _, _ = step(js, ea)
    obs = state_to_obs(js, player=0)
    torch_batch, _ = featurize(obs, history=HistoryState())
    jax_batch = featurize_jax_w1(js, player=0)
    t_feats = torch_batch.candidate_feats[0].cpu().numpy()
    j_feats = np.asarray(jax_batch.candidate_feats[0])
    t_mask = torch_batch.candidate_mask[0].cpu().numpy()
    j_mask = np.asarray(jax_batch.candidate_mask[0])
    t_pid = torch_batch.candidate_pid[0].cpu().numpy()
    j_pid = np.asarray(jax_batch.candidate_pid[0])
    mine = np.asarray(jax_batch.my_planet_mask[0])

    for slot in range(MAX_PLANETS):
        if not bool(mine[slot]):
            continue
        # mask / pid are integer — exact match required.
        assert (j_mask[slot] == t_mask[slot]).all(), (
            f"seed={seed} turns={turns} slot={slot}: "
            f"candidate_mask jax={j_mask[slot]} torch={t_mask[slot]}"
        )
        assert (j_pid[slot] == t_pid[slot]).all(), (
            f"seed={seed} turns={turns} slot={slot}: "
            f"candidate_pid jax={j_pid[slot]} torch={t_pid[slot]}"
        )
        # feats — tol=1e-4.
        for k in range(8):
            for c in range(14):
                j = float(j_feats[slot, k, c])
                t = float(t_feats[slot, k, c])
                diff = abs(j - t)
                assert diff < PARITY_TOL, (
                    f"seed={seed} turns={turns} slot={slot} k={k} c={c}: "
                    f"jax={j:.6f} torch={t:.6f} diff={diff:.3e}"
                )


def test_masks_match_torch() -> None:
    """planet_mask / my_planet_mask / target_mask must agree with PyTorch."""
    js = reset(seed=0, num_agents=2)
    obs = state_to_obs(js, player=0)
    jax_batch = featurize_jax_w1(js, player=0)
    torch_batch, _ = featurize(obs, history=HistoryState())

    j_pm = np.asarray(jax_batch.planet_mask[0])
    t_pm = torch_batch.planet_mask[0].cpu().numpy()
    assert (j_pm == t_pm).all(), f"planet_mask differs:\n jax={j_pm}\n torch={t_pm}"

    j_mm = np.asarray(jax_batch.my_planet_mask[0])
    t_mm = torch_batch.my_planet_mask[0].cpu().numpy()
    assert (j_mm == t_mm).all(), f"my_planet_mask differs:\n jax={j_mm}\n torch={t_mm}"

    j_tm = np.asarray(jax_batch.target_mask[0])
    t_tm = torch_batch.target_mask[0].cpu().numpy()
    assert (j_tm == t_tm).all(), f"target_mask differs:\n jax={j_tm}\n torch={t_tm}"
