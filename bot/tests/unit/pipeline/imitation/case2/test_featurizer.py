"""Unit tests for pipeline.imitation.case2.policy.featurizer."""

from __future__ import annotations

import torch

from pipeline.imitation.case2.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    featurize,
)


def _make_obs(num_planets: int = 3) -> dict[str, object]:
    planets = [
        [1, 0, 20.0, 20.0, 2.0, 50, 5],
        [2, 1, 80.0, 80.0, 2.0, 40, 4],
        [3, -1, 50.0, 50.0, 1.5, 10, 2],
    ][:num_planets]
    return {
        "player": 0,
        "step": 12,
        "angular_velocity": 0.01,
        "comet_planet_ids": [33, 34, 35, 36],
        "planets": planets,
        "fleets": [[100, 1, 60.0, 60.0, 3.926, 2, 12]],
        "initial_planets": [],
    }


def test_feature_dims_are_extended() -> None:
    assert PLANET_FEAT_DIM == 18
    assert GLOBAL_FEAT_DIM == 11


def test_featurize_returns_expected_shapes() -> None:
    obs = _make_obs()
    batch, snap = featurize(obs)
    assert batch.planet_feats.shape == (1, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.planet_mask.shape == (1, MAX_PLANETS)
    assert batch.my_planet_mask.shape == (1, MAX_PLANETS)
    assert batch.target_mask.shape == (1, MAX_PLANETS)
    assert batch.global_feats.shape == (1, GLOBAL_FEAT_DIM)
    assert snap.player == 0
    assert snap.my_planet_ids == (1,)
    assert snap.planet_ids == (1, 2, 3)


def test_featurize_all_finite() -> None:
    batch, _ = featurize(_make_obs())
    assert torch.isfinite(batch.planet_feats).all().item()
    assert torch.isfinite(batch.global_feats).all().item()


def test_masks_count_correctly() -> None:
    batch, _ = featurize(_make_obs())
    assert int(batch.planet_mask.sum().item()) == 3
    assert int(batch.my_planet_mask.sum().item()) == 1
    assert int(batch.target_mask.sum().item()) == 2


def test_padding_slots_are_zero() -> None:
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


def test_engineered_columns_in_range() -> None:
    """Engineered columns 11..17 should be finite and lie in plausible ranges."""
    batch, _ = featurize(_make_obs())
    valid_rows = batch.planet_feats[0, :3]
    sun_dist = valid_rows[:, 11]
    is_static = valid_rows[:, 12]
    prod_per_ship = valid_rows[:, 13]
    nearest_enemy = valid_rows[:, 14]
    support = valid_rows[:, 15]
    threat = valid_rows[:, 16]
    net_signed = valid_rows[:, 17]
    assert torch.all((sun_dist >= 0.0) & (sun_dist <= 1.0))
    assert torch.all((is_static == 0.0) | (is_static == 1.0))
    assert torch.all((prod_per_ship >= 0.0) & (prod_per_ship <= 1.0))
    assert torch.all((nearest_enemy >= 0.0) & (nearest_enemy <= 1.0))
    assert torch.all(support >= 0.0)
    assert torch.all(threat >= 0.0)
    assert torch.all((net_signed >= -1.0) & (net_signed <= 1.0))


def test_global_phase_flags_consistent() -> None:
    """phase_mid / phase_late should be mutually exclusive 0/1 flags."""
    obs = _make_obs()
    obs["step"] = 50  # early
    early_batch, _ = featurize(obs)
    obs["step"] = 200  # mid
    mid_batch, _ = featurize(obs)
    obs["step"] = 400  # late
    late_batch, _ = featurize(obs)
    # phase_mid at index 9, phase_late at index 10
    assert early_batch.global_feats[0, 9].item() == 0.0
    assert early_batch.global_feats[0, 10].item() == 0.0
    assert mid_batch.global_feats[0, 9].item() == 1.0
    assert mid_batch.global_feats[0, 10].item() == 0.0
    assert late_batch.global_feats[0, 9].item() == 0.0
    assert late_batch.global_feats[0, 10].item() == 1.0
