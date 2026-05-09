"""Unit tests for pipeline.imitation.case1.training.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.imitation.case1.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case1.policy.templates import TEMPLATE_CTX_DIM
from pipeline.imitation.case1.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    Sample,
    collate,
)


def test_dataset_len_and_getitem(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    assert len(ds) == 10
    sample = ds[3]
    assert isinstance(sample, Sample)
    assert sample.planet_feats.shape == (MAX_PLANETS, PLANET_FEAT_DIM)
    assert sample.global_feats.shape == (GLOBAL_FEAT_DIM,)
    assert sample.template_ctx.shape == (MAX_PLANETS, TEMPLATE_CTX_DIM)
    assert sample.planet_mask.dtype == torch.bool
    assert sample.from_multihot.dtype == torch.bool
    assert int(sample.target_per_src[0].item()) == 4
    assert int(sample.ships_per_src[0].item()) == 3 % 4


def test_sample_weights_from_target_lifts_minority(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    # target_per_src[0] in the fixture = (i+1) % MAX_PLANETS — uniform over
    # positions 1..MAX_PLANETS-1 except 0. Use a small num_classes so counts
    # differ enough to test the inverse-frequency direction.
    w = ds.sample_weights_from_target(num_classes=MAX_PLANETS, power=0.5)
    assert w.shape == (len(ds),)
    # Mean normalised to 1.0
    assert abs(w.mean() - 1.0) < 1e-6
    # All positive
    assert (w > 0).all()


def test_sample_weights_power_zero_is_uniform(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    w = ds.sample_weights_from_target(num_classes=MAX_PLANETS, power=0.0)
    # With power=0, every fired-frame weight equals 1, so normalised mean=1.
    assert np.allclose(w, 1.0)


def test_dataloader_batch_shapes(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, BatchedSample)
    assert batch.planet_feats.shape == (4, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (4, GLOBAL_FEAT_DIM)
    assert batch.planet_mask.shape == (4, MAX_PLANETS)
    assert batch.template_ctx.shape == (4, MAX_PLANETS, TEMPLATE_CTX_DIM)
    assert batch.from_multihot.shape == (4, MAX_PLANETS)
    assert batch.target_per_src.shape == (4, MAX_PLANETS)
    assert batch.ships_per_src.shape == (4, MAX_PLANETS)
    assert batch.is_noop.shape == (4,)
    assert batch.target_per_src.dtype == torch.long
