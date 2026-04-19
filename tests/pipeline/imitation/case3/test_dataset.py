"""Unit tests for pipeline.imitation.case3.training.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.imitation.case3.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case3.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    Sample,
    collate,
)


def _make_fixture_parquet(path: Path, n: int = 10) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "planet_feats": rng.standard_normal(MAX_PLANETS * PLANET_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": [True] * MAX_PLANETS,
                "my_planet_mask": [bool(j == 0) for j in range(MAX_PLANETS)],
                "target_mask": [bool(j != 0) for j in range(MAX_PLANETS)],
                "from_label": int(i % MAX_PLANETS),
                "target_label": int((i + 1) % MAX_PLANETS),
                "ships_label": int(i % 5),
                "is_noop": bool(i % 3 == 0),
            }
        )
    pl.DataFrame(rows).write_parquet(path)


@pytest.fixture
def fixture_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "fixture.parquet"
    _make_fixture_parquet(p, n=10)
    return p


def test_dataset_len_and_getitem(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    assert len(ds) == 10
    sample = ds[3]
    assert isinstance(sample, Sample)
    assert sample.planet_feats.shape == (MAX_PLANETS, PLANET_FEAT_DIM)
    assert sample.global_feats.shape == (GLOBAL_FEAT_DIM,)
    assert sample.planet_mask.dtype == torch.bool
    assert sample.from_label == 3
    assert sample.target_label == 4


def test_dataloader_batch_shapes(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, BatchedSample)
    assert batch.planet_feats.shape == (4, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (4, GLOBAL_FEAT_DIM)
    assert batch.planet_mask.shape == (4, MAX_PLANETS)
    assert batch.from_label.shape == (4,)
    assert batch.target_label.shape == (4,)
    assert batch.ships_label.shape == (4,)
    assert batch.is_noop.shape == (4,)
    assert batch.from_label.dtype == torch.long


def test_dataset_handles_noop_label(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    # row 0 is_noop=True (i%3==0)
    s = ds[0]
    assert s.is_noop is True
