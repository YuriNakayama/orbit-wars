"""Parquet → torch Dataset/DataLoader for imitation/case8.

iter5: lazy chunk-by-chunk read using pyarrow.parquet.ParquetFile.
__init__ does not load row data into RAM (only metadata + cand_slot_per_src
for class_weight). __getitem__ on-demand-loads the row group containing the
requested index, with a small LRU cache to absorb sequential / random access
from DataLoader. This avoids the RAM peak that caused iter4-v1..v4 OOM.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case8.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)


@dataclass(frozen=True)
class Sample:
    planet_feats: torch.Tensor  # (MAX_PLANETS, PLANET_FEAT_DIM)
    global_feats: torch.Tensor  # (GLOBAL_FEAT_DIM,)
    planet_mask: torch.Tensor  # (MAX_PLANETS,) bool
    my_planet_mask: torch.Tensor  # (MAX_PLANETS,) bool
    target_mask: torch.Tensor  # (MAX_PLANETS,) bool
    candidate_feats: torch.Tensor  # (MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (MAX_PLANETS, CAND_K) bool
    candidate_pid: torch.Tensor  # (MAX_PLANETS, CAND_K) int64
    cand_slot_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    ship_label_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    is_noop: bool


@dataclass(frozen=True)
class BatchedSample:
    planet_feats: torch.Tensor
    global_feats: torch.Tensor
    planet_mask: torch.Tensor
    my_planet_mask: torch.Tensor
    target_mask: torch.Tensor
    candidate_feats: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_pid: torch.Tensor
    cand_slot_per_src: torch.Tensor
    ship_label_per_src: torch.Tensor
    is_noop: torch.Tensor


_DATA_COLUMNS = (
    "planet_feats",
    "global_feats",
    "planet_mask",
    "my_planet_mask",
    "target_mask",
    "candidate_feats",
    "candidate_mask",
    "candidate_pid",
    "cand_slot_per_src",
    "ship_label_per_src",
    "is_noop",
)


@dataclass
class _RowGroupArrays:
    """Numpy arrays for a single row group, indexed by row offset within the group."""

    planet_feats: np.ndarray
    global_feats: np.ndarray
    planet_mask: np.ndarray
    my_planet_mask: np.ndarray
    target_mask: np.ndarray
    candidate_feats: np.ndarray
    candidate_mask: np.ndarray
    candidate_pid: np.ndarray
    cand_slot_per_src: np.ndarray
    ship_label_per_src: np.ndarray
    is_noop: np.ndarray


def _list_col_to_numpy(col: object, dtype: np.dtype) -> np.ndarray:
    """pyarrow ListArray (chunked) → flat numpy. Avoids Python list intermediate
    for inner values; only the per-row Python-list materialisation is unavoidable
    when going through `to_pylist()`."""
    pylist = col.to_pylist()  # type: ignore[attr-defined]
    arr = np.asarray(pylist, dtype=dtype)
    del pylist
    return arr


class CaseFourDataset(Dataset[Sample]):
    """Lazy parquet-backed Dataset for case8 (iter5 refactor).

    init time: O(num_rows) for cand_slot_per_src scan only; other columns are
    loaded on demand via row-group cache.

    RAM peak: ~one row group (≈ 5000 rows × ~40k float = ~800MB) plus
    cand_slot_per_src array (≈ n_rows × 48 × 8 bytes ≈ a few MB). Compare to
    iter4 in-memory load that needed ~20-30 GB peak.
    """

    _CACHE_SIZE = 4  # LRU row-group cache; absorbs DataLoader shuffle access

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
    ) -> None:
        self._path = str(parquet_path)
        self._pf = pq.ParquetFile(self._path)
        meta = self._pf.metadata
        self._num_rows = int(meta.num_rows)
        self._num_row_groups = int(meta.num_row_groups)
        # Compute row-group row offsets for binary search.
        offsets = [0]
        for rg_idx in range(self._num_row_groups):
            offsets.append(offsets[-1] + int(meta.row_group(rg_idx).num_rows))
        self._row_group_offsets = offsets

        # iter4 hyperparams use class_weight, so we need a full-pass scan of
        # cand_slot_per_src. Load only this one column.
        cand_slot_table = self._pf.read(columns=["cand_slot_per_src"])
        self._cand_slot_per_src_full = _list_col_to_numpy(
            cand_slot_table["cand_slot_per_src"].combine_chunks(),
            np.dtype(np.int64),
        ).reshape(-1, MAX_PLANETS)
        del cand_slot_table
        gc.collect()

        # Inferred dims (consistent with iter1-3; we don't load planet_feats yet).
        self._planet_feat_dim = PLANET_FEAT_DIM
        self._global_feat_dim = GLOBAL_FEAT_DIM

        self._mask_planet_cols = list(mask_planet_cols or [])
        self._mask_global_cols = list(mask_global_cols or [])

        # LRU cache: row_group_idx → _RowGroupArrays
        self._cache: OrderedDict[int, _RowGroupArrays] = OrderedDict()
        self._has_ship_label = "ship_label_per_src" in self._pf.schema_arrow.names

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Inverse-frequency class weights — see iter1-3 for rationale."""
        del beta
        flat = self._cand_slot_per_src_full.reshape(-1)
        flat = flat[flat != ignore_index]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        present_mask = counts > 0
        if not present_mask.any():
            return torch.ones(num_classes, dtype=torch.float32)
        raw = np.zeros(num_classes, dtype=np.float64)
        raw[present_mask] = 1.0 / counts[present_mask]
        mean_raw = float(raw[present_mask].mean())
        weights = raw / max(mean_raw, 1e-12)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return self._num_rows

    def _locate(self, idx: int) -> tuple[int, int]:
        """Return (row_group_idx, offset_within_group) for the global index."""
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(idx)
        # Linear scan is fine: num_row_groups is O(num_rows / 5000) ≈ 60 for
        # our largest preprocess output. Bisect is overkill.
        offsets = self._row_group_offsets
        for rg_idx in range(self._num_row_groups):
            if idx < offsets[rg_idx + 1]:
                return rg_idx, idx - offsets[rg_idx]
        raise AssertionError("unreachable")

    def _load_row_group(self, rg_idx: int) -> _RowGroupArrays:
        if rg_idx in self._cache:
            self._cache.move_to_end(rg_idx)
            return self._cache[rg_idx]

        table = self._pf.read_row_group(rg_idx, columns=list(_DATA_COLUMNS))
        n = table.num_rows

        planet_arr = _list_col_to_numpy(
            table["planet_feats"].combine_chunks(), np.dtype(np.float32)
        )
        if planet_arr.size > 0:
            self._planet_feat_dim = int(planet_arr.size // (n * MAX_PLANETS))
        planet_arr = planet_arr.reshape(n, MAX_PLANETS, self._planet_feat_dim)
        for col in self._mask_planet_cols:
            if 0 <= col < self._planet_feat_dim:
                planet_arr[:, :, col] = 0.0

        global_arr = _list_col_to_numpy(
            table["global_feats"].combine_chunks(), np.dtype(np.float32)
        )
        if global_arr.size > 0:
            self._global_feat_dim = int(global_arr.size // n)
        global_arr = global_arr.reshape(n, self._global_feat_dim)
        for col in self._mask_global_cols:
            if 0 <= col < self._global_feat_dim:
                global_arr[:, col] = 0.0

        rg = _RowGroupArrays(
            planet_feats=planet_arr,
            global_feats=global_arr,
            planet_mask=_list_col_to_numpy(
                table["planet_mask"].combine_chunks(), np.dtype(np.bool_)
            ).reshape(n, MAX_PLANETS),
            my_planet_mask=_list_col_to_numpy(
                table["my_planet_mask"].combine_chunks(), np.dtype(np.bool_)
            ).reshape(n, MAX_PLANETS),
            target_mask=_list_col_to_numpy(
                table["target_mask"].combine_chunks(), np.dtype(np.bool_)
            ).reshape(n, MAX_PLANETS),
            candidate_feats=_list_col_to_numpy(
                table["candidate_feats"].combine_chunks(), np.dtype(np.float32)
            ).reshape(n, MAX_PLANETS, CAND_K, CAND_FEAT_DIM),
            candidate_mask=_list_col_to_numpy(
                table["candidate_mask"].combine_chunks(), np.dtype(np.bool_)
            ).reshape(n, MAX_PLANETS, CAND_K),
            candidate_pid=_list_col_to_numpy(
                table["candidate_pid"].combine_chunks(), np.dtype(np.int64)
            ).reshape(n, MAX_PLANETS, CAND_K),
            cand_slot_per_src=_list_col_to_numpy(
                table["cand_slot_per_src"].combine_chunks(), np.dtype(np.int64)
            ).reshape(n, MAX_PLANETS),
            ship_label_per_src=(
                _list_col_to_numpy(
                    table["ship_label_per_src"].combine_chunks(),
                    np.dtype(np.int64),
                ).reshape(n, MAX_PLANETS)
                if self._has_ship_label
                else np.full((n, MAX_PLANETS), -1, dtype=np.int64)
            ),
            is_noop=table["is_noop"].to_numpy().astype(np.bool_),
        )
        del table

        self._cache[rg_idx] = rg
        if len(self._cache) > self._CACHE_SIZE:
            self._cache.popitem(last=False)
        return rg

    def __getitem__(self, idx: int) -> Sample:
        rg_idx, offset = self._locate(idx)
        rg = self._load_row_group(rg_idx)
        return Sample(
            planet_feats=torch.from_numpy(rg.planet_feats[offset]),
            global_feats=torch.from_numpy(rg.global_feats[offset]),
            planet_mask=torch.from_numpy(rg.planet_mask[offset]),
            my_planet_mask=torch.from_numpy(rg.my_planet_mask[offset]),
            target_mask=torch.from_numpy(rg.target_mask[offset]),
            candidate_feats=torch.from_numpy(rg.candidate_feats[offset]),
            candidate_mask=torch.from_numpy(rg.candidate_mask[offset]),
            candidate_pid=torch.from_numpy(rg.candidate_pid[offset]),
            cand_slot_per_src=torch.from_numpy(rg.cand_slot_per_src[offset]),
            ship_label_per_src=torch.from_numpy(rg.ship_label_per_src[offset]),
            is_noop=bool(rg.is_noop[offset]),
        )


def collate(samples: list[Sample]) -> BatchedSample:
    return BatchedSample(
        planet_feats=torch.stack([s.planet_feats for s in samples]),
        global_feats=torch.stack([s.global_feats for s in samples]),
        planet_mask=torch.stack([s.planet_mask for s in samples]),
        my_planet_mask=torch.stack([s.my_planet_mask for s in samples]),
        target_mask=torch.stack([s.target_mask for s in samples]),
        candidate_feats=torch.stack([s.candidate_feats for s in samples]),
        candidate_mask=torch.stack([s.candidate_mask for s in samples]),
        candidate_pid=torch.stack([s.candidate_pid for s in samples]),
        cand_slot_per_src=torch.stack([s.cand_slot_per_src for s in samples]),
        ship_label_per_src=torch.stack([s.ship_label_per_src for s in samples]),
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
