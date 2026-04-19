"""Unit tests for pipeline.case3.policy.featurizer."""

from __future__ import annotations

import torch

from pipeline.case3.policy.featurizer import (
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
    assert int(batch.my_planet_mask.sum().item()) == 1  # planet id=1 owned by player
    # target_mask covers everything that is NOT mine
    assert int(batch.target_mask.sum().item()) == 2


def test_padding_slots_are_zero() -> None:
    batch, _ = featurize(_make_obs(num_planets=2))
    # Slot index 2 (third) is padding now
    assert torch.equal(
        batch.planet_feats[0, 2],
        torch.zeros(PLANET_FEAT_DIM, dtype=torch.float32),
    )
    assert not bool(batch.planet_mask[0, 2].item())


def test_empty_obs_handled() -> None:
    batch, snap = featurize({"player": 0, "step": 0})
    assert int(batch.planet_mask.sum().item()) == 0
    assert snap.my_planet_ids == ()
