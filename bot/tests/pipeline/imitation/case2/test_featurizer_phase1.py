"""Unit tests for pipeline.imitation.case2.policy.featurizer_phase1."""

from __future__ import annotations

import torch

from pipeline.imitation.case2.policy.featurizer_phase1 import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize,
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
        "initial_planets": [
            [1, 0, 20.0, 20.0, 2.0, 50, 5],
            [2, 1, 80.0, 80.0, 2.0, 40, 4],
            [3, -1, 50.0, 70.0, 1.5, 10, 2],
        ][:num_planets],
        "planets": planets,
        "fleets": [[100, 1, 60.0, 60.0, 3.926, 2, 12]],
    }


def test_phase1_dims_are_extended() -> None:
    assert PLANET_FEAT_DIM == 33
    assert GLOBAL_FEAT_DIM == 16


def test_featurize_returns_expected_shapes() -> None:
    batch, snap = featurize(_make_obs())
    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)
    assert snap.player == 0
    assert snap.my_planet_ids == (1,)


def test_featurize_all_finite() -> None:
    batch, _ = featurize(_make_obs())
    assert torch.isfinite(batch.planet_feats).all().item()
    assert torch.isfinite(batch.global_feats).all().item()


def test_padding_zero_and_mask() -> None:
    batch, _ = featurize(_make_obs(num_planets=2))
    assert torch.equal(
        batch.planet_feats[0, 2],
        torch.zeros(PLANET_FEAT_DIM, dtype=torch.float32),
    )
    assert not bool(batch.planet_mask[0, 2].item())


def test_empty_obs_handled() -> None:
    batch, snap = featurize({"player": 0, "step": 0})
    assert int(batch.planet_mask.sum().item()) == 0
    assert snap.my_planet_ids == ()
    assert torch.isfinite(batch.global_feats).all().item()


def test_distance_columns_in_unit_range() -> None:
    """Cols 18..19 (nearest ally/neutral) and 20..27 (orbit dx,dy) sanity."""
    batch, _ = featurize(_make_obs())
    valid = batch.planet_feats[0, :3]
    nearest_ally = valid[:, 18]
    nearest_neutral = valid[:, 19]
    assert torch.all((nearest_ally >= 0.0) & (nearest_ally <= 1.0))
    assert torch.all((nearest_neutral >= 0.0) & (nearest_neutral <= 1.0))
    # Orbit dx/dy are normalized by board size; |delta| should be much < 1.
    orbit_block = valid[:, 20:28]
    assert torch.all(orbit_block.abs() < 1.0)


def test_static_planet_orbit_predictions_zero() -> None:
    """Static planets (sun_dist + radius >= ROTATION_LIMIT) shouldn't drift."""
    obs = _make_obs(num_planets=1)
    obs["planets"] = [[1, 0, 5.0, 5.0, 60.0, 50, 5]]  # large radius -> static
    obs["initial_planets"] = obs["planets"]
    batch, _ = featurize(obs)
    orbit_block = batch.planet_feats[0, 0, 20:28]
    assert torch.all(orbit_block.abs() < 1e-5)


def test_incoming_eta_split() -> None:
    """Cols 28-29 (ally/enemy ETA) should be in [0, 1]."""
    batch, _ = featurize(_make_obs())
    valid = batch.planet_feats[0, :3]
    ally_eta = valid[:, 28]
    enemy_eta = valid[:, 29]
    assert torch.all((ally_eta >= 0.0) & (ally_eta <= 1.0))
    assert torch.all((enemy_eta >= 0.0) & (enemy_eta <= 1.0))


def test_global_phase_and_comet_flags() -> None:
    early = featurize(_make_obs(step=10))[0].global_feats[0]
    mid_pre_comet = featurize(_make_obs(step=200))[0].global_feats[0]
    late = featurize(_make_obs(step=400))[0].global_feats[0]
    # phase_mid at idx 9, phase_late at idx 10 (unchanged from baseline)
    assert early[9].item() == 0.0
    assert early[10].item() == 0.0
    assert mid_pre_comet[9].item() == 1.0
    assert late[10].item() == 1.0
    # next_comet_eta_norm at idx 11, in [0, 1]
    assert 0.0 <= early[11].item() <= 1.0
    assert 0.0 <= late[11].item() <= 1.0


def test_global_total_fractions_sum_consistent() -> None:
    batch, _ = featurize(_make_obs())
    g = batch.global_feats[0]
    # idx 12: my_ships_frac, idx 13: enemy_ships_frac
    my_frac = g[12].item()
    enemy_frac = g[13].item()
    assert 0.0 <= my_frac <= 1.0
    assert 0.0 <= enemy_frac <= 1.0
    assert my_frac + enemy_frac <= 1.0 + 1e-6  # neutral takes the remainder


def test_score_diff_in_clipped_range() -> None:
    batch, _ = featurize(_make_obs())
    g = batch.global_feats[0]
    assert -1.0 <= g[15].item() <= 1.0
