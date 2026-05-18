"""Parquet → torch Dataset/DataLoader for imitation/case11.

case11 mart は ~5.94M rows / ~19 GB (ZSTD parquet)。in-memory zero-copy
(case8 iter13 流) では peak ~100-140 GB の host RAM が要求され、RunPod
の典型 host RAM (24-32 GB) で OOM (exit 137) する。

このモジュールは **row-group lazy load + LRU cache** で再実装:

- `__init__` は `pq.ParquetFile` を保持し、row → row_group mapping を
  precompute するだけ。peak RAM は ~50 MB。
- `__getitem__(idx)` は idx を含む row_group を fetch → numpy slice → torch。
  LRU cache (default size 4) で連続 idx の再読み込みを抑止。
- `class_weight_on_*` は 1pass で全 row_group を scan して bincount。
  peak は row_group 1 つ分のみ。

shuffle=True との相性: idx がランダムに飛ぶと cache ヒット率が下がるが、
batch=512 でほとんどの batch が 1-2 row group に分散する想定 (1073 groups,
5000 rows/group)。worker=0 推奨 (process fork で cache 共有不可)。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case11.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case11.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case11.policy.templates import TEMPLATE_CTX_DIM


@dataclass(frozen=True)
class Sample:
    planet_feats: torch.Tensor
    global_feats: torch.Tensor
    planet_mask: torch.Tensor
    my_planet_mask: torch.Tensor
    target_mask: torch.Tensor
    template_ctx: torch.Tensor
    candidate_feats: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_pid: torch.Tensor
    cand_slot_per_src: torch.Tensor
    ship_label_per_src: torch.Tensor
    ships_bucket_per_src: torch.Tensor
    from_multihot: torch.Tensor
    target_per_src: torch.Tensor
    ships_per_src: torch.Tensor
    target_pid_per_src: torch.Tensor
    ship_pred_label: torch.Tensor
    is_noop: bool


@dataclass(frozen=True)
class BatchedSample:
    planet_feats: torch.Tensor
    global_feats: torch.Tensor
    planet_mask: torch.Tensor
    my_planet_mask: torch.Tensor
    target_mask: torch.Tensor
    template_ctx: torch.Tensor
    candidate_feats: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_pid: torch.Tensor
    cand_slot_per_src: torch.Tensor
    ship_label_per_src: torch.Tensor
    ships_bucket_per_src: torch.Tensor
    from_multihot: torch.Tensor
    target_per_src: torch.Tensor
    ships_per_src: torch.Tensor
    target_pid_per_src: torch.Tensor
    ship_pred_label: torch.Tensor
    is_noop: torch.Tensor


def _flatten_arrow(arr: pa.Array) -> pa.Array:
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        arr = arr.flatten()
    return arr


@dataclass
class _GroupArrays:
    """Numpy arrays for a single row_group, lazily materialised on first access."""

    planet_feats: np.ndarray
    global_feats: np.ndarray
    planet_mask: np.ndarray
    my_planet_mask: np.ndarray
    target_mask: np.ndarray
    template_ctx: np.ndarray
    candidate_feats: np.ndarray
    candidate_mask: np.ndarray
    candidate_pid: np.ndarray
    cand_slot_per_src: np.ndarray
    ship_label_per_src: np.ndarray
    ships_bucket_per_src: np.ndarray
    from_multihot: np.ndarray
    target_per_src: np.ndarray
    ships_per_src: np.ndarray
    target_pid_per_src: np.ndarray
    ship_pred_label: np.ndarray
    is_noop: np.ndarray


def _list_col_to_numpy(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    """Convert a list-typed column to a standalone numpy buffer.

    `to_numpy(zero_copy_only=False)` may return a view into the arrow
    buffer; without an explicit copy the underlying arrow Table cannot be
    GC'd even after we drop our Python reference, leading to RAM growth
    that defeats lazy loading. We force a contiguous copy here so the
    arrow buffer can be freed as soon as the caller's `table` reference
    is gone.
    """
    col = table[name]
    chunks = col.chunks if col.num_chunks > 0 else [col.combine_chunks()]
    parts: list[np.ndarray] = []
    for ch in chunks:
        flat = _flatten_arrow(ch)
        parts.append(flat.to_numpy(zero_copy_only=False).astype(dtype, copy=True))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _primitive_col(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.array(table[name].to_numpy(zero_copy_only=False), dtype=dtype, copy=True)


def _materialise_group(
    table: pa.Table, schema_names: set[str], n_rows: int
) -> _GroupArrays:
    """row_group の pyarrow Table を `_GroupArrays` (numpy) に変換する。"""
    pf_dim = PLANET_FEAT_DIM
    gf_dim = GLOBAL_FEAT_DIM
    pa_dtype_f32 = np.dtype(np.float32)
    pa_dtype_bool = np.dtype(np.bool_)
    pa_dtype_i64 = np.dtype(np.int64)

    planet_feats = _list_col_to_numpy(table, "planet_feats", pa_dtype_f32).reshape(
        n_rows, MAX_PLANETS, pf_dim
    )
    global_feats = _list_col_to_numpy(table, "global_feats", pa_dtype_f32).reshape(
        n_rows, gf_dim
    )
    planet_mask = _list_col_to_numpy(table, "planet_mask", pa_dtype_bool).reshape(
        n_rows, MAX_PLANETS
    )
    my_planet_mask = _list_col_to_numpy(table, "my_planet_mask", pa_dtype_bool).reshape(
        n_rows, MAX_PLANETS
    )
    target_mask = _list_col_to_numpy(table, "target_mask", pa_dtype_bool).reshape(
        n_rows, MAX_PLANETS
    )

    if "template_ctx" in schema_names:
        template_ctx = _list_col_to_numpy(table, "template_ctx", pa_dtype_f32).reshape(
            n_rows, MAX_PLANETS, TEMPLATE_CTX_DIM
        )
    else:
        template_ctx = np.zeros(
            (n_rows, MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=np.float32
        )

    candidate_feats = _list_col_to_numpy(
        table, "candidate_feats", pa_dtype_f32
    ).reshape(n_rows, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask = _list_col_to_numpy(table, "candidate_mask", pa_dtype_bool).reshape(
        n_rows, MAX_PLANETS, CAND_K
    )
    candidate_pid = _list_col_to_numpy(table, "candidate_pid", pa_dtype_i64).reshape(
        n_rows, MAX_PLANETS, CAND_K
    )
    cand_slot_per_src = _list_col_to_numpy(
        table, "cand_slot_per_src", pa_dtype_i64
    ).reshape(n_rows, MAX_PLANETS)

    if "ship_label_per_src" in schema_names:
        ship_label_per_src = _list_col_to_numpy(
            table, "ship_label_per_src", pa_dtype_i64
        ).reshape(n_rows, MAX_PLANETS)
    else:
        ship_label_per_src = np.full((n_rows, MAX_PLANETS), -1, dtype=np.int64)

    if "ships_bucket_per_src" in schema_names:
        ships_bucket_per_src = _list_col_to_numpy(
            table, "ships_bucket_per_src", pa_dtype_i64
        ).reshape(n_rows, MAX_PLANETS)
    else:
        ships_bucket_per_src = np.full((n_rows, MAX_PLANETS), -1, dtype=np.int64)

    if "from_multihot" in schema_names:
        from_multihot = _list_col_to_numpy(
            table, "from_multihot", pa_dtype_bool
        ).reshape(n_rows, MAX_PLANETS)
    else:
        from_multihot = np.zeros((n_rows, MAX_PLANETS), dtype=np.bool_)

    if "target_per_src" in schema_names:
        target_per_src = _list_col_to_numpy(
            table, "target_per_src", pa_dtype_i64
        ).reshape(n_rows, MAX_PLANETS)
    else:
        target_per_src = np.full((n_rows, MAX_PLANETS), -1, dtype=np.int64)

    if "ships_per_src" in schema_names:
        ships_per_src = _list_col_to_numpy(
            table, "ships_per_src", pa_dtype_i64
        ).reshape(n_rows, MAX_PLANETS)
    else:
        ships_per_src = np.full((n_rows, MAX_PLANETS), -1, dtype=np.int64)

    if "target_pid_per_src" in schema_names:
        target_pid_per_src = _list_col_to_numpy(
            table, "target_pid_per_src", pa_dtype_i64
        ).reshape(n_rows, MAX_PLANETS)
    else:
        target_pid_per_src = np.full((n_rows, MAX_PLANETS), MAX_PLANETS, dtype=np.int64)

    if "ship_pred_label" in schema_names:
        ship_pred_label = _list_col_to_numpy(
            table, "ship_pred_label", pa_dtype_f32
        ).reshape(n_rows, MAX_PLANETS)
    else:
        ship_pred_label = np.full((n_rows, MAX_PLANETS), -1.0, dtype=np.float32)

    is_noop = _primitive_col(table, "is_noop", pa_dtype_bool)

    return _GroupArrays(
        planet_feats=planet_feats,
        global_feats=global_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        template_ctx=template_ctx,
        candidate_feats=candidate_feats,
        candidate_mask=candidate_mask,
        candidate_pid=candidate_pid,
        cand_slot_per_src=cand_slot_per_src,
        ship_label_per_src=ship_label_per_src,
        ships_bucket_per_src=ships_bucket_per_src,
        from_multihot=from_multihot,
        target_per_src=target_per_src,
        ships_per_src=ships_per_src,
        target_pid_per_src=target_pid_per_src,
        ship_pred_label=ship_pred_label,
        is_noop=is_noop,
    )


class CaseFourDataset(Dataset[Sample]):
    """Lazy row-group-based parquet Dataset for case11.

    Holds an open `pq.ParquetFile` and reads one row_group at a time on demand,
    caching the most recent groups in an LRU. Memory peak is ~ (cache_size
    × row_group_size × per-row-bytes); with cache=4 and ~5000 rows/group this
    is roughly 0.5-1 GB even for the 19 GB mart, well under the 24-32 GB host
    RAM cap.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
        *,
        cache_size: int = 4,
    ) -> None:
        if mask_planet_cols or mask_global_cols:
            # ablation masking would require in-memory rewrite; case11 does not
            # need it, so reject early instead of silently ignoring.
            raise NotImplementedError(
                "mask_planet_cols/mask_global_cols not supported in lazy mode"
            )
        path = str(parquet_path)
        self._pf = pq.ParquetFile(path)
        self._schema_names: set[str] = set(self._pf.schema_arrow.names)
        self._planet_feat_dim = PLANET_FEAT_DIM
        self._global_feat_dim = GLOBAL_FEAT_DIM

        num_groups = self._pf.num_row_groups
        self._group_row_counts = np.empty(num_groups, dtype=np.int64)
        for g in range(num_groups):
            self._group_row_counts[g] = self._pf.metadata.row_group(g).num_rows
        # Cumulative offsets: row_offsets[g] = first row index in row_group g.
        self._row_offsets = np.concatenate(([0], np.cumsum(self._group_row_counts)))
        self._n = int(self._row_offsets[-1])
        self._num_groups = num_groups

        self._cache: OrderedDict[int, _GroupArrays] = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._lock = Lock()

    def _group_for_row(self, idx: int) -> tuple[int, int]:
        # binary search; right-inclusive: find largest g with row_offsets[g] <= idx.
        g = int(np.searchsorted(self._row_offsets, idx, side="right") - 1)
        return g, idx - int(self._row_offsets[g])

    def _get_group(self, g: int) -> _GroupArrays:
        with self._lock:
            cached = self._cache.get(g)
            if cached is not None:
                self._cache.move_to_end(g)
                return cached
        # Read outside the lock to avoid serialising IO.
        table = self._pf.read_row_group(g)
        arrays = _materialise_group(
            table, self._schema_names, int(self._group_row_counts[g])
        )
        # Drop the arrow Table explicitly so the buffer it owns can be freed
        # as soon as Python decrements the refcount. `_list_col_to_numpy`
        # copies every column, so `arrays` does not retain any view into it.
        del table
        with self._lock:
            self._cache[g] = arrays
            self._cache.move_to_end(g)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return arrays

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        counts = np.zeros(num_classes, dtype=np.float64)
        for g in range(self._num_groups):
            arr = self._get_group(g).cand_slot_per_src.reshape(-1)
            arr = arr[arr != ignore_index]
            if arr.size:
                counts += np.bincount(arr, minlength=num_classes)
        return _inverse_freq_weights(counts)

    def class_weight_on_templates_including_noop(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        counts = np.zeros(num_classes, dtype=np.float64)
        for g in range(self._num_groups):
            grp = self._get_group(g)
            labels = grp.target_per_src.copy()
            labels[labels == ignore_index] = num_classes - 1
            valid = grp.my_planet_mask.reshape(-1)
            flat = labels.reshape(-1)[valid]
            if flat.size:
                counts += np.bincount(flat, minlength=num_classes)
        return _inverse_freq_weights(counts)

    def class_weight_on_ships(
        self, num_classes: int = 4, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        counts = np.zeros(num_classes, dtype=np.float64)
        for g in range(self._num_groups):
            grp = self._get_group(g)
            flat = grp.ships_per_src.reshape(-1)
            if np.all(flat == ignore_index):
                flat = grp.ships_bucket_per_src.reshape(-1)
            flat = flat[flat != ignore_index]
            if flat.size:
                counts += np.bincount(flat, minlength=num_classes)
        return _inverse_freq_weights(counts)

    def __len__(self) -> int:
        return int(self._n)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        g, local = self._group_for_row(idx)
        grp = self._get_group(g)
        return Sample(
            planet_feats=torch.from_numpy(grp.planet_feats[local]),
            global_feats=torch.from_numpy(grp.global_feats[local]),
            planet_mask=torch.from_numpy(grp.planet_mask[local]),
            my_planet_mask=torch.from_numpy(grp.my_planet_mask[local]),
            target_mask=torch.from_numpy(grp.target_mask[local]),
            template_ctx=torch.from_numpy(grp.template_ctx[local]),
            candidate_feats=torch.from_numpy(grp.candidate_feats[local]),
            candidate_mask=torch.from_numpy(grp.candidate_mask[local]),
            candidate_pid=torch.from_numpy(grp.candidate_pid[local]),
            cand_slot_per_src=torch.from_numpy(grp.cand_slot_per_src[local]),
            ship_label_per_src=torch.from_numpy(grp.ship_label_per_src[local]),
            ships_bucket_per_src=torch.from_numpy(grp.ships_bucket_per_src[local]),
            from_multihot=torch.from_numpy(grp.from_multihot[local]),
            target_per_src=torch.from_numpy(grp.target_per_src[local]),
            ships_per_src=torch.from_numpy(grp.ships_per_src[local]),
            target_pid_per_src=torch.from_numpy(grp.target_pid_per_src[local]),
            ship_pred_label=torch.from_numpy(grp.ship_pred_label[local]),
            is_noop=bool(grp.is_noop[local]),
        )


def _inverse_freq_weights(counts: np.ndarray) -> torch.Tensor:
    present_mask = counts > 0
    if not present_mask.any():
        return torch.ones(counts.size, dtype=torch.float32)
    raw = np.zeros_like(counts)
    raw[present_mask] = 1.0 / counts[present_mask]
    mean_raw = float(raw[present_mask].mean())
    weights = raw / max(mean_raw, 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def collate(samples: list[Sample]) -> BatchedSample:
    return BatchedSample(
        planet_feats=torch.stack([s.planet_feats for s in samples]),
        global_feats=torch.stack([s.global_feats for s in samples]),
        planet_mask=torch.stack([s.planet_mask for s in samples]),
        my_planet_mask=torch.stack([s.my_planet_mask for s in samples]),
        target_mask=torch.stack([s.target_mask for s in samples]),
        template_ctx=torch.stack([s.template_ctx for s in samples]),
        candidate_feats=torch.stack([s.candidate_feats for s in samples]),
        candidate_mask=torch.stack([s.candidate_mask for s in samples]),
        candidate_pid=torch.stack([s.candidate_pid for s in samples]),
        cand_slot_per_src=torch.stack([s.cand_slot_per_src for s in samples]),
        ship_label_per_src=torch.stack([s.ship_label_per_src for s in samples]),
        ships_bucket_per_src=torch.stack([s.ships_bucket_per_src for s in samples]),
        from_multihot=torch.stack([s.from_multihot for s in samples]),
        target_per_src=torch.stack([s.target_per_src for s in samples]),
        ships_per_src=torch.stack([s.ships_per_src for s in samples]),
        target_pid_per_src=torch.stack([s.target_pid_per_src for s in samples]),
        ship_pred_label=torch.stack([s.ship_pred_label for s in samples]),
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
