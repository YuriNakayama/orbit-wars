"""Tests for iter5 lazy CaseFourDataset (chunk-by-chunk row group reading)."""

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
    # Force multiple row groups by using a smaller batch size.
    pq.write_table(table, str(path), compression="zstd", row_group_size=row_group_size)


def test_lazy_dataset_len_matches_n_rows(tmp_path: Path) -> None:
    pq_path = tmp_path / "lazy.parquet"
    _build_synthetic_parquet(pq_path, n_rows=12, row_group_size=5)
    ds = CaseFourDataset(pq_path)
    assert len(ds) == 12
    # 12 rows / 5 per group = 3 row groups
    assert ds._num_row_groups == 3


def test_lazy_dataset_getitem_returns_correct_row(tmp_path: Path) -> None:
    pq_path = tmp_path / "lazy.parquet"
    _build_synthetic_parquet(pq_path, n_rows=12, row_group_size=5)
    ds = CaseFourDataset(pq_path)

    # Walk every row across multiple row groups
    for idx in range(12):
        sample = ds[idx]
        assert sample.planet_feats.shape == (MAX_PLANETS, PLANET_FEAT_DIM)
        assert sample.candidate_feats.shape == (MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
        assert sample.cand_slot_per_src.shape == (MAX_PLANETS,)
        # Check row identity via candidate_pid (we set it to idx for each row)
        assert int(sample.candidate_pid[0, 0]) == idx


def test_lazy_dataset_class_weight_uses_full_scan(tmp_path: Path) -> None:
    pq_path = tmp_path / "lazy.parquet"
    _build_synthetic_parquet(pq_path, n_rows=16, row_group_size=4)
    ds = CaseFourDataset(pq_path)
    # 16 rows × 2 valid src per row = 32 labeled slots
    weights = ds.class_weight_on_slots(num_classes=CAND_K)
    assert weights.shape == (CAND_K,)
    # Average over present classes should be 1.0 (inverse-freq normalization)
    nonzero = weights[weights > 0]
    assert torch.isclose(nonzero.mean(), torch.tensor(1.0), atol=1e-5)


def test_lazy_dataset_lru_cache_evicts(tmp_path: Path) -> None:
    """LRU cache size is fixed; old row groups are evicted on demand."""
    pq_path = tmp_path / "lazy.parquet"
    _build_synthetic_parquet(pq_path, n_rows=30, row_group_size=5)  # 6 row groups
    ds = CaseFourDataset(pq_path)
    # Touch all groups in order
    for idx in range(0, 30, 5):
        _ = ds[idx]
    # Cache size is 4; we touched 6 groups, so first 2 should be evicted
    cache_keys = list(ds._cache.keys())
    assert len(cache_keys) == 4
    assert 0 not in cache_keys
    assert 1 not in cache_keys
    assert 5 in cache_keys  # last touched (group idx 5)


def test_lazy_dataset_random_access_works(tmp_path: Path) -> None:
    """DataLoader shuffle does random index access; LRU cache must absorb."""
    pq_path = tmp_path / "lazy.parquet"
    _build_synthetic_parquet(pq_path, n_rows=20, row_group_size=5)
    ds = CaseFourDataset(pq_path)
    rng = np.random.default_rng(42)
    indices = rng.permutation(20).tolist()
    samples = [ds[int(i)] for i in indices]
    # Each sample's candidate_pid[0, 0] should equal its global index
    for s, idx in zip(samples, indices, strict=True):
        assert int(s.candidate_pid[0, 0]) == idx
