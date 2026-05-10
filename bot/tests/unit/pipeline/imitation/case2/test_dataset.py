"""Unit tests for pipeline.imitation.case2.training.dataset."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline.imitation.case2.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case2.policy.templates import TEMPLATE_CTX_DIM
from pipeline.imitation.case2.training.dataset import (
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


def test_dataloader_batch_shapes(fixture_parquet: Path) -> None:
    ds = CaseThreeDataset(fixture_parquet)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, BatchedSample)
    assert batch.planet_feats.shape == (4, MAX_PLANETS, PLANET_FEAT_DIM)
    assert batch.global_feats.shape == (4, GLOBAL_FEAT_DIM)
    assert batch.template_ctx.shape == (4, MAX_PLANETS, TEMPLATE_CTX_DIM)
    assert batch.from_multihot.shape == (4, MAX_PLANETS)
    assert batch.target_per_src.dtype == torch.long
