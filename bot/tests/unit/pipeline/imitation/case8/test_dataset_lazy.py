"""Tests for iter13 in-memory CaseFourDataset (pyarrow zero-copy load).

Filename retained from iter5 lazy refactor for git history continuity;
tests now exercise the iter13 in-memory implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pipeline.imitation.case8.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case8.training.dataset import CaseFourDataset
from pipeline.imitation.case8.training.preprocess import _arrow_schema


def _build_synthetic_parquet(
    path: Path, n_rows: int = 12, row_group_size: int = 5
) -> None:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        rows.append(
            {
                "planet_feats": rng.standard_normal(MAX_PLANETS * PLANET_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": [True] * 4 + [False] * (MAX_PLANETS - 4),
                "my_planet_mask": [True] * 2 + [False] * (MAX_PLANETS - 2),
                "target_mask": [True] * 4 + [False] * (MAX_PLANETS - 4),
                "candidate_feats": rng.standard_normal(
                    MAX_PLANETS * CAND_K * CAND_FEAT_DIM
                )
                .astype(np.float32)
                .tolist(),
                "candidate_mask": [True] * (MAX_PLANETS * CAND_K),
                "candidate_pid": [int(i)] * (MAX_PLANETS * CAND_K),
                "cand_slot_per_src": [int(i % CAND_K), 0] + [-1] * (MAX_PLANETS - 2),
                "ship_label_per_src": [int(20 + i), -1] + [-1] * (MAX_PLANETS - 2),
                "is_noop": False,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_arrow_schema())
    pq.write_table(table, str(path), compression="zstd", row_group_size=row_group_size)


def test_dataset_len_matches_n_rows(tmp_path: Path) -> None:
    pq_path = tmp_path / "ds.parquet"
    _build_synthetic_parquet(pq_path, n_rows=12, row_group_size=5)
    ds = CaseFourDataset(pq_path)
    assert len(ds) == 12


def test_dataset_getitem_returns_correct_row(tmp_path: Path) -> None:
    pq_path = tmp_path / "ds.parquet"
    _build_synthetic_parquet(pq_path, n_rows=12, row_group_size=5)
    ds = CaseFourDataset(pq_path)

    for idx in range(12):
        sample = ds[idx]
        assert sample.planet_feats.shape == (MAX_PLANETS, PLANET_FEAT_DIM)
        assert sample.candidate_feats.shape == (MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
        assert sample.cand_slot_per_src.shape == (MAX_PLANETS,)
        assert int(sample.candidate_pid[0, 0]) == idx


def test_dataset_class_weight_uses_full_scan(tmp_path: Path) -> None:
    pq_path = tmp_path / "ds.parquet"
    _build_synthetic_parquet(pq_path, n_rows=16, row_group_size=4)
    ds = CaseFourDataset(pq_path)
    weights = ds.class_weight_on_slots(num_classes=CAND_K)
    assert weights.shape == (CAND_K,)
    nonzero = weights[weights > 0]
    assert torch.isclose(nonzero.mean(), torch.tensor(1.0), atol=1e-5)


def test_dataset_random_access_works(tmp_path: Path) -> None:
    """DataLoader shuffle does random index access — must work without I/O."""
    pq_path = tmp_path / "ds.parquet"
    _build_synthetic_parquet(pq_path, n_rows=20, row_group_size=5)
    ds = CaseFourDataset(pq_path)
    rng = np.random.default_rng(42)
    indices = rng.permutation(20).tolist()
    samples = [ds[int(i)] for i in indices]
    for s, idx in zip(samples, indices, strict=True):
        assert int(s.candidate_pid[0, 0]) == idx


def test_dataset_mask_planet_cols_zeros_columns(tmp_path: Path) -> None:
    """`mask_planet_cols=[i]` zeros that column on `planet_feats`."""
    pq_path = tmp_path / "ds.parquet"
    _build_synthetic_parquet(pq_path, n_rows=4, row_group_size=2)
    ds = CaseFourDataset(pq_path, mask_planet_cols=[0, 3])
    sample = ds[0]
    assert torch.all(sample.planet_feats[:, 0] == 0)
    assert torch.all(sample.planet_feats[:, 3] == 0)
    # Untouched column should not be all zeros (probabilistically)
    assert not torch.all(sample.planet_feats[:, 1] == 0)
