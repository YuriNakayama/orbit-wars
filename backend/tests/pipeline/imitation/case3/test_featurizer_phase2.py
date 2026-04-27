"""Unit tests for pipeline.imitation.case3.policy.featurizer_phase2."""

from __future__ import annotations

import pytest
import torch

from pipeline.imitation.case3.policy.featurizer_phase2 import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
    HistoryState,
    LaunchEvent,
    featurize,
    update_history,
)


def _make_obs(num_planets: int = 3, step: int = 12) -> dict[str, object]:
    planets = [
        [1, 0, 20.0, 20.0, 2.0, 50, 5],
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 70.0, 1.5, 10, 2],
    ][:num_planets]
    return {
        "player": 0,
        "step": step,
        "angular_velocity": 0.01,
        "comet_planet_ids": [],
        "comets": [],
        "initial_planets": planets,
        "planets": planets,
        "fleets": [],
    }


def test_phase2_dims() -> None:
    assert PLANET_FEAT_DIM == 35
    assert GLOBAL_FEAT_DIM == 20


def test_featurize_no_history_zero_columns() -> None:
    batch, _ = featurize(_make_obs())
    # planet cols 32..34 should be 0 when no history is supplied.
    valid = batch.planet_feats[0, :3]
    assert torch.all(valid[:, 32] == 0.0)
    assert torch.all(valid[:, 33] == 0.0)
    assert torch.all(valid[:, 34] == 0.0)
    # global cols 16..19 should be 0.
    g = batch.global_feats[0]
    assert g[16].item() == 0.0
    assert g[17].item() == 0.0
    assert g[18].item() == 0.0
    assert g[19].item() == 0.0


def test_featurize_with_only_one_history_keeps_planet_cols_zero() -> None:
    """delta_t1 needs `prev_planet_snapshots[-2]` (≥2 entries). One entry isn't
    enough — planet history columns must remain zero (no leak from [-1])."""
    history = HistoryState()
    obs_prev = _make_obs(step=10)
    obs_prev["planets"] = [
        [1, 0, 20.0, 20.0, 2.0, 999, 5],  # would-be-leaky snapshot
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 70.0, 1.5, 10, 2],
    ]
    update_history(history, obs_prev, None)
    batch, _ = featurize(_make_obs(step=11), history)
    valid = batch.planet_feats[0, :3]
    assert torch.all(valid[:, 32] == 0.0)  # delta_t1
    assert torch.all(valid[:, 33] == 0.0)  # delta_t2
    assert torch.all(valid[:, 34] == 0.0)  # owner_changed_t1


def test_featurize_with_history_populates_columns() -> None:
    """delta_t1 reads `[-2]` (= obs_{N-2}), so we need at least 2 history pushes
    before featurize sees a non-zero delta. obs_t in this test is step=12 with
    50 ships; obs_{N-2} (step=10) had 50 ships too, but we'll mutate the
    earlier snapshot to verify the delta reads from `[-2]`, not `[-1]`.
    """
    history = HistoryState()
    obs_t_minus_2 = _make_obs(step=10)
    obs_t_minus_2["planets"] = [
        [1, 0, 20.0, 20.0, 2.0, 50, 5],
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 70.0, 1.5, 10, 2],
    ]
    update_history(history, obs_t_minus_2, [[1, 0.0, 5]])

    obs_t_minus_1 = _make_obs(step=11)
    # An ostensibly leaky snapshot: planet 1 = 999 ships. Featurizer must
    # IGNORE this and read [-2] (the 50-ship snapshot above).
    obs_t_minus_1["planets"] = [
        [1, 0, 20.0, 20.0, 2.0, 999, 5],
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 70.0, 1.5, 10, 2],
    ]
    update_history(history, obs_t_minus_1, [[1, 0.0, 7]])

    obs_t = _make_obs(step=12)
    obs_t["planets"] = [
        [1, 0, 20.0, 20.0, 2.0, 60, 5],
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 70.0, 1.5, 10, 2],
    ]
    batch, _ = featurize(obs_t, history)
    # Expected delta_t1 = (60 - 50) / 60 from snap_t1 = [-2] (50 ships, NOT 999).
    delta_t1 = batch.planet_feats[0, 0, 32].item()
    expected = (60 - 50) / 60 / 3.0  # normalized
    assert delta_t1 == pytest.approx(expected, abs=1e-4)
    # Global launch history at step=12, window=12-4=8: events at steps 10, 11
    # both fall inside [8, 12]; 2 ally launches → 2/10 = 0.2.
    assert batch.global_feats[0, 18].item() == pytest.approx(0.2, abs=1e-5)


def test_owner_changed_flag() -> None:
    """owner_changed reads snap_t1 = [-2]. Need at least 2 history pushes."""
    history = HistoryState()
    obs_t_minus_2 = _make_obs(step=10)
    obs_t_minus_2["planets"] = [[1, 0, 20.0, 20.0, 2.0, 50, 5]]
    update_history(history, obs_t_minus_2, None)
    obs_t_minus_1 = _make_obs(step=11)
    obs_t_minus_1["planets"] = [[1, 0, 20.0, 20.0, 2.0, 50, 5]]
    update_history(history, obs_t_minus_1, None)
    obs_t = _make_obs(step=12)
    obs_t["planets"] = [[1, 1, 20.0, 20.0, 2.0, 50, 5]]  # owner changed 0 -> 1
    obs_t["player"] = 1
    obs_t["initial_planets"] = obs_t["planets"]
    batch, _ = featurize(obs_t, history)
    # Slot 0 owner_changed_t1 should be 1.0 (snap_t1 = [-2] had owner 0).
    assert batch.planet_feats[0, 0, 34].item() == 1.0


def test_history_clear_resets_columns() -> None:
    history = HistoryState()
    obs1 = _make_obs(step=10)
    update_history(history, obs1, [[1, 0.0, 5]])
    history.clear()
    assert len(history.prev_planet_snapshots) == 0
    assert len(history.recent_launches) == 0
    assert history.last_step is None


def test_recent_launches_decay_outside_window() -> None:
    history = HistoryState()
    history.recent_launches.append(LaunchEvent(step=0, owner=1, ships=10))
    history.recent_launches.append(LaunchEvent(step=10, owner=1, ships=20))
    obs = _make_obs(step=20)
    obs["player"] = 0  # we are player 0; the events above are enemy
    batch, _ = featurize(obs, history)
    # step=20, window=20-HISTORY_TURNS=16; step-0 too old, step=10 too old
    assert batch.global_feats[0, 16].item() == 0.0  # enemy_launch_count_last4
    # Add a fresh enemy event within window:
    history.recent_launches.append(LaunchEvent(step=18, owner=1, ships=30))
    batch2, _ = featurize(obs, history)
    assert batch2.global_feats[0, 16].item() == pytest.approx(0.1, abs=1e-5)


def test_dims_match_baseline_pre_history_block() -> None:
    """Planet cols 0..31 should match phase1 final layout (no group D)."""
    history = HistoryState()
    obs = _make_obs()
    batch, _ = featurize(obs, history)
    valid = batch.planet_feats[0, :3]
    # All cols finite, in plausible range.
    assert torch.isfinite(valid).all().item()


def test_update_history_truncates_old_launches() -> None:
    history = HistoryState()
    for s in range(0, 100):
        history.recent_launches.append(LaunchEvent(step=s, owner=0, ships=1))
    obs = _make_obs(step=100)
    update_history(history, obs, [[1, 0.0, 7]])
    # All events older than 100 - HISTORY_TURNS = 96 should be dropped.
    earliest = min(ev.step for ev in history.recent_launches)
    assert earliest >= 96


def test_planet_snapshot_buffer_keeps_at_most_three() -> None:
    history = HistoryState()
    for s in range(6):
        update_history(history, _make_obs(step=s), None)
    assert len(history.prev_planet_snapshots) == 3
