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

import numpy as np
import pytest

from jax_env.observation import state_to_obs
from jax_env.reset import reset
from jax_env.step import empty_actions, step
from pipeline.reinforce.case1.policy.featurizer import (
    HistoryState,
    featurize,
)
from pipeline.reinforce.case1.policy.featurizer_jax import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize_jax_w1,
)

PARITY_TOL = 1e-4

# Columns implemented in W1. Anything outside this set is allowed to drift
# in W1 (they are zero-filled in the JAX version) and will be unblocked
# in W2.
W1_PLANET_COLS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13)
W1_GLOBAL_COLS = tuple(i for i in range(GLOBAL_FEAT_DIM) if i not in (16, 17, 18, 19))


def _featurize_both(seed: int, turns: int) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Run both featurizers from a fresh state with `turns` no-op steps."""
    js = reset(seed=seed, num_agents=2)
    ea = empty_actions()
    for _ in range(turns):
        js, _, _ = step(js, ea)
    obs = state_to_obs(js, player=0)
    jax_batch = featurize_jax_w1(js, player=0)
    torch_batch, _snap = featurize(obs, history=HistoryState())
    jax_planet = np.asarray(jax_batch.planet_feats[0])
    torch_planet = torch_batch.planet_feats[0].cpu().numpy()
    jax_global = np.asarray(jax_batch.global_feats[0])
    torch_global = torch_batch.global_feats[0].cpu().numpy()
    return jax_planet, torch_planet, {
        "valid": np.asarray(jax_batch.planet_mask[0]),
        "global": jax_global,
    }, {
        "valid": torch_batch.planet_mask[0].cpu().numpy(),
        "global": torch_global,
    }


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
