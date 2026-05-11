"""Parquet → torch Dataset/DataLoader for imitation/case10.

iter13: in-memory load via pyarrow zero-copy. iter5-12 lazy + per-row-group
to_pylist refactor caused epoch=0 hang on RunPod RTX 4090 (likely host-RAM
swap thrash from cache + arrow buffer pool overhead under shuffle=True).

Reverts to the iter1-3 in-memory shape but **avoids the polars+pylist+numpy
triple-allocation that triggered the iter4 OOM**: list-typed columns are
flattened via pyarrow `flatten().to_numpy(zero_copy_only=False)` so only one
numpy buffer is materialised per column.

RAM peak estimate: ~6-8 GB for 300k rows (vs. 30 GB for the iter1
read_parquet→to_list→np.array path that OOM'd, and vs. iter5-12 lazy that
hung at epoch_start). Comfortably under the 24-32 GB host of an RTX 4090
pod.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case10.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case10.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case10.policy.templates import TEMPLATE_CTX_DIM


@dataclass(frozen=True)
class Sample:
    planet_feats: torch.Tensor  # (MAX_PLANETS, PLANET_FEAT_DIM)
    global_feats: torch.Tensor  # (GLOBAL_FEAT_DIM,)
    planet_mask: torch.Tensor  # (MAX_PLANETS,) bool
    my_planet_mask: torch.Tensor  # (MAX_PLANETS,) bool
    target_mask: torch.Tensor  # (MAX_PLANETS,) bool
    template_ctx: torch.Tensor  # (MAX_PLANETS, TEMPLATE_CTX_DIM)
    candidate_feats: torch.Tensor  # (MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (MAX_PLANETS, CAND_K) bool
    candidate_pid: torch.Tensor  # (MAX_PLANETS, CAND_K) int64
    cand_slot_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    ship_label_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    ships_bucket_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    from_multihot: torch.Tensor  # (MAX_PLANETS,) bool — legacy fire label
    target_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
    ships_per_src: torch.Tensor  # (MAX_PLANETS,) int64; -1 = unused
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
    is_noop: torch.Tensor


def _flatten_arrow_array(arr: pa.Array) -> pa.Array:
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        arr = arr.flatten()
    return arr


def _list_to_numpy(col: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    """Flatten a chunked list column to numpy without Arrow combine_chunks().

    Large case10 parquet files can exceed the 32-bit offset limit of Arrow
    ListArray when `combine_chunks()` is called. Flatten each row-group chunk
    independently, then concatenate numpy buffers. This preserves the exact
    flattened values while avoiding offset overflow.
    """
    arrays = [
        np.asarray(
            _flatten_arrow_array(chunk).to_numpy(zero_copy_only=False),
            dtype=dtype,
        )
        for chunk in col.chunks
    ]
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=True)


def _take(
    table_box: list[pa.Table],
    name: str,
    dtype: np.dtype,
    *,
    full_n: int = 0,
    keep_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a list-typed column to numpy, then drop it from the table to
    free its arrow buffer. The table is held in a single-element list so this
    helper can mutate it in place.

    When ``keep_idx`` is provided, the column is reshaped to ``(full_n, -1)``
    and rows are sliced before being flattened again. This keeps the
    keep_row_indices feature working without going through pyarrow.take(),
    which overflows on 5M-row list-typed columns.
    """
    arr = _list_to_numpy(table_box[0][name], dtype)
    table_box[0] = table_box[0].drop_columns([name])
    if keep_idx is not None and full_n > 0 and arr.size > 0:
        per_row = arr.size // full_n
        arr = arr.reshape(full_n, per_row)[keep_idx].reshape(-1)
    return arr


def _take_primitive(
    table_box: list[pa.Table],
    name: str,
    dtype: np.dtype,
    *,
    keep_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Like `_take` but for primitive (non-list) columns."""
    arr = np.array(
        table_box[0][name].to_numpy(zero_copy_only=False), dtype=dtype, copy=True
    )
    table_box[0] = table_box[0].drop_columns([name])
    if keep_idx is not None:
        arr = arr[keep_idx]
    return arr


class CaseTenDataset(Dataset[Sample]):
    """In-memory parquet-backed Dataset for case10 (iter13 refactor).

    `__init__` reads the entire parquet once via pyarrow and converts each
    list-typed column to a single numpy buffer. `__getitem__` is a pure
    numpy slice, so DataLoader workers can fan out without per-batch I/O.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
        keep_row_indices: np.ndarray | None = None,
    ) -> None:
        # Drop columns from the arrow Table one at a time after converting to
        # numpy: peak RAM is then ~ (largest column copy) + (table - that
        # column), not (full table + all numpy copies). The largest column
        # is `candidate_feats` at ~6 GB for 300k rows.
        # NOTE: do NOT call `pa.Table.take` on this table — list-typed columns
        # exceed the 32-bit offset limit when concatenated. Instead, slice each
        # column at the numpy stage (post `_list_to_numpy`) using
        # ``self._keep_idx``. When keep_row_indices is None we keep the full
        # row range; otherwise this becomes a fancy index applied per column.
        keep_idx = (
            np.asarray(keep_row_indices, dtype=np.int64)
            if keep_row_indices is not None
            else None
        )
        self._keep_idx = keep_idx
        tbl: list[pa.Table] = [pq.read_table(str(parquet_path))]
        full_n = tbl[0].num_rows
        n = int(keep_idx.size) if keep_idx is not None else full_n
        has_ship_label = "ship_label_per_src" in tbl[0].schema.names
        take_kw = {"full_n": full_n, "keep_idx": keep_idx}

        planet_arr = _take(tbl, "planet_feats", np.dtype(np.float32), **take_kw)
        if n > 0:
            self._planet_feat_dim = int(planet_arr.size // (n * MAX_PLANETS))
        else:
            self._planet_feat_dim = PLANET_FEAT_DIM
        # Store in fp16 (recast in __getitem__) to halve RAM. The training
        # loop already runs the forward in fp32 so accuracy is preserved.
        planet_feats_f32 = planet_arr.reshape(
            n, MAX_PLANETS, self._planet_feat_dim
        )
        for col in mask_planet_cols or []:
            if 0 <= col < self._planet_feat_dim:
                planet_feats_f32[:, :, col] = 0.0
        self._planet_feats = planet_feats_f32.astype(np.float16, copy=False)
        del planet_feats_f32

        global_arr = _take(tbl, "global_feats", np.dtype(np.float32), **take_kw)
        if n > 0:
            self._global_feat_dim = int(global_arr.size // n)
        else:
            self._global_feat_dim = GLOBAL_FEAT_DIM
        self._global_feats = global_arr.reshape(n, self._global_feat_dim)
        for col in mask_global_cols or []:
            if 0 <= col < self._global_feat_dim:
                self._global_feats[:, col] = 0.0

        self._planet_mask = _take(
            tbl, "planet_mask", np.dtype(np.bool_), **take_kw
        ).reshape(n, MAX_PLANETS)
        self._my_planet_mask = _take(
            tbl, "my_planet_mask", np.dtype(np.bool_), **take_kw
        ).reshape(n, MAX_PLANETS)
        self._target_mask = _take(
            tbl, "target_mask", np.dtype(np.bool_), **take_kw
        ).reshape(n, MAX_PLANETS)
        if "template_ctx" in tbl[0].schema.names:
            self._template_ctx = _take(
                tbl, "template_ctx", np.dtype(np.float32), **take_kw
            ).reshape(n, MAX_PLANETS, TEMPLATE_CTX_DIM)
        else:
            self._template_ctx = np.zeros(
                (n, MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=np.float32
            )
        # candidate_feats is the largest list-typed column (~15 GB at 5M rows
        # in fp32). Store in fp16 to halve in-memory footprint; cast back to
        # fp32 in __getitem__ where downstream model expects float.
        cand_feats_f32 = _take(
            tbl, "candidate_feats", np.dtype(np.float32), **take_kw
        ).reshape(n, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
        self._candidate_feats = cand_feats_f32.astype(np.float16, copy=False)
        del cand_feats_f32
        self._candidate_mask = _take(
            tbl, "candidate_mask", np.dtype(np.bool_), **take_kw
        ).reshape(n, MAX_PLANETS, CAND_K)
        self._candidate_pid = _take(
            tbl, "candidate_pid", np.dtype(np.int64), **take_kw
        ).reshape(n, MAX_PLANETS, CAND_K)
        self._cand_slot_per_src = _take(
            tbl, "cand_slot_per_src", np.dtype(np.int64), **take_kw
        ).reshape(n, MAX_PLANETS)
        if has_ship_label:
            self._ship_label_per_src = _take(
                tbl, "ship_label_per_src", np.dtype(np.int64), **take_kw
            ).reshape(n, MAX_PLANETS)
        else:
            self._ship_label_per_src = np.full((n, MAX_PLANETS), -1, dtype=np.int64)
        if "ships_bucket_per_src" in tbl[0].schema.names:
            self._ships_bucket_per_src = _take(
                tbl, "ships_bucket_per_src", np.dtype(np.int64), **take_kw
            ).reshape(n, MAX_PLANETS)
        else:
            self._ships_bucket_per_src = np.full((n, MAX_PLANETS), -1, dtype=np.int64)
        if "from_multihot" in tbl[0].schema.names:
            self._from_multihot = _take(
                tbl, "from_multihot", np.dtype(np.bool_), **take_kw
            ).reshape(n, MAX_PLANETS)
        else:
            self._from_multihot = np.zeros((n, MAX_PLANETS), dtype=np.bool_)
        if "target_per_src" in tbl[0].schema.names:
            self._target_per_src = _take(
                tbl, "target_per_src", np.dtype(np.int64), **take_kw
            ).reshape(n, MAX_PLANETS)
        else:
            self._target_per_src = np.full((n, MAX_PLANETS), -1, dtype=np.int64)
        if "ships_per_src" in tbl[0].schema.names:
            self._ships_per_src = _take(
                tbl, "ships_per_src", np.dtype(np.int64), **take_kw
            ).reshape(n, MAX_PLANETS)
        else:
            self._ships_per_src = np.full((n, MAX_PLANETS), -1, dtype=np.int64)
        self._is_noop = _take_primitive(
            tbl, "is_noop", np.dtype(np.bool_), keep_idx=keep_idx
        )
        self._n = n

        del tbl

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Inverse-frequency class weights — see iter1-3 for rationale."""
        del beta
        flat = self._cand_slot_per_src.reshape(-1)
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

    def class_weight_on_templates_including_noop(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Inverse-frequency weights for template_ships, including no-op class."""
        del beta
        labels = self._target_per_src.copy()
        labels[labels == ignore_index] = num_classes - 1
        valid = self._my_planet_mask.reshape(-1)
        flat = labels.reshape(-1)[valid]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        present_mask = counts > 0
        if not present_mask.any():
            return torch.ones(num_classes, dtype=torch.float32)
        raw = np.zeros(num_classes, dtype=np.float64)
        raw[present_mask] = 1.0 / counts[present_mask]
        mean_raw = float(raw[present_mask].mean())
        weights = raw / max(mean_raw, 1e-12)
        return torch.tensor(weights, dtype=torch.float32)

    def class_weight_on_ships(
        self, num_classes: int = 4, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Inverse-frequency weights for fired-source ship buckets."""
        del beta
        flat = self._ships_per_src.reshape(-1)
        if np.all(flat == ignore_index):
            flat = self._ships_bucket_per_src.reshape(-1)
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
        return int(self._n)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        return Sample(
            planet_feats=torch.from_numpy(
                self._planet_feats[idx].astype(np.float32, copy=False)
            ),
            global_feats=torch.from_numpy(self._global_feats[idx]),
            planet_mask=torch.from_numpy(self._planet_mask[idx]),
            my_planet_mask=torch.from_numpy(self._my_planet_mask[idx]),
            target_mask=torch.from_numpy(self._target_mask[idx]),
            template_ctx=torch.from_numpy(self._template_ctx[idx]),
            candidate_feats=torch.from_numpy(
                self._candidate_feats[idx].astype(np.float32, copy=False)
            ),
            candidate_mask=torch.from_numpy(self._candidate_mask[idx]),
            candidate_pid=torch.from_numpy(self._candidate_pid[idx]),
            cand_slot_per_src=torch.from_numpy(self._cand_slot_per_src[idx]),
            ship_label_per_src=torch.from_numpy(self._ship_label_per_src[idx]),
            ships_bucket_per_src=torch.from_numpy(self._ships_bucket_per_src[idx]),
            from_multihot=torch.from_numpy(self._from_multihot[idx]),
            target_per_src=torch.from_numpy(self._target_per_src[idx]),
            ships_per_src=torch.from_numpy(self._ships_per_src[idx]),
            is_noop=bool(self._is_noop[idx]),
        )


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
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
