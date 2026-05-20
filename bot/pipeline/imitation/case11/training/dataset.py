"""Per-column .npy mmap Dataset/DataLoader for imitation/case11.

retry #14 までの経緯:
- case8 iter13 流の in-memory zero-copy: peak RAM ~100-140GB で OOM (#11)
- row-group lazy load (LRU cache): RAM 安定だが 1 epoch ~2-3h (#13/#14, IO 律速)

このモジュールは **per-column .npy + mmap_mode="r"** で再実装:

- `parquet_to_npy.py` で事前に train/val parquet を per-column .npy に変換
  (ZSTD 解除、shape を flatten 済 ndarray として保存)。1 回限りのコスト。
- Dataset は `np.load(mmap_mode="r")` で各 .npy を mmap open。RSS は数十 MB
  のみ。__getitem__ は np.array() で row slice を copy し、torch.from_numpy
  に渡す (mmap view を切って GPU 転送後の解放を保証)。
- shuffle=True でも OS page cache が頻繁 access する row を hot に保つので
  実質 in-memory アクセスと同等速度。
- DataLoader workers=2-4 推奨: mmap は fork-safe で各 worker が独立 view を
  持てる (parquet ParquetFile と違って共有不可問題なし)。

mart layout:
  data/mart/imitation/case11/train_npy/{planet_feats,...}.npy
  data/mart/imitation/case11/val_npy/{planet_feats,...}.npy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case11.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)


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


def _npy_dir_for(parquet_path: Path | str) -> Path:
    """`<dir>/<split>.parquet` -> `<dir>/<split>_npy/`."""
    p = Path(parquet_path)
    return p.with_name(p.stem + "_npy")


class CaseFourDataset(Dataset[Sample]):
    """Per-column .npy + mmap_mode="r" Dataset for case11.

    Expects mart layout produced by `parquet_to_npy.py`:
        <stem>_npy/{planet_feats, ..., is_noop}.npy

    Each column is mmap-opened (zero RAM cost, OS page-cache backed).
    `__getitem__(idx)` returns a per-row torch.Tensor with `np.array` copy
    to ensure the resulting tensor does not retain a view into the memmap
    (avoids the retry #12-style accumulation observed under pin_memory).
    """

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
        max_rows: int | None = None,
    ) -> None:
        if mask_planet_cols or mask_global_cols:
            raise NotImplementedError(
                "mask_planet_cols/mask_global_cols not supported in mmap mode"
            )
        npy_dir = _npy_dir_for(parquet_path)
        if not npy_dir.is_dir():
            raise FileNotFoundError(
                f"npy mart not found: {npy_dir}. "
                "Run `python -m pipeline.imitation.case11.training.parquet_to_npy`"
            )
        self._planet_feat_dim = PLANET_FEAT_DIM
        self._global_feat_dim = GLOBAL_FEAT_DIM

        def _mm(name: str) -> np.ndarray:
            arr: np.ndarray = np.load(npy_dir / f"{name}.npy", mmap_mode="r")
            return arr

        self._planet_feats = _mm("planet_feats")
        self._global_feats = _mm("global_feats")
        self._planet_mask = _mm("planet_mask")
        self._my_planet_mask = _mm("my_planet_mask")
        self._target_mask = _mm("target_mask")
        self._template_ctx = _mm("template_ctx")
        self._candidate_feats = _mm("candidate_feats")
        self._candidate_mask = _mm("candidate_mask")
        self._candidate_pid = _mm("candidate_pid")
        self._cand_slot_per_src = _mm("cand_slot_per_src")
        self._ship_label_per_src = _mm("ship_label_per_src")
        self._ships_bucket_per_src = _mm("ships_bucket_per_src")
        self._from_multihot = _mm("from_multihot")
        self._target_per_src = _mm("target_per_src")
        self._ships_per_src = _mm("ships_per_src")
        self._target_pid_per_src = _mm("target_pid_per_src")
        self._ship_pred_label = _mm("ship_pred_label")
        self._is_noop = _mm("is_noop")

        self._n = int(self._planet_feats.shape[0])
        if max_rows is not None and max_rows > 0:
            self._n = min(self._n, max_rows)

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        flat = np.asarray(self._cand_slot_per_src).reshape(-1)
        flat = flat[flat != ignore_index]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        return _inverse_freq_weights(counts)

    def class_weight_on_templates_including_noop(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        labels = np.asarray(self._target_per_src).copy()
        labels[labels == ignore_index] = num_classes - 1
        valid = np.asarray(self._my_planet_mask).reshape(-1)
        flat = labels.reshape(-1)[valid]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        return _inverse_freq_weights(counts)

    def class_weight_on_ships(
        self, num_classes: int = 4, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        del beta
        flat = np.asarray(self._ships_per_src).reshape(-1)
        if np.all(flat == ignore_index):
            flat = np.asarray(self._ships_bucket_per_src).reshape(-1)
        flat = flat[flat != ignore_index]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        return _inverse_freq_weights(counts)

    def __len__(self) -> int:
        return int(self._n)

    def __getitem__(self, idx: int) -> Sample:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        # np.array(...) forces a copy out of the memmap so the resulting
        # torch.Tensor owns a contiguous buffer independent of the OS page
        # cache. This sidesteps the view-accumulation pattern that caused
        # retry #12 to leak ~6 GB / 5 min into 138 GB OOM.
        return Sample(
            planet_feats=torch.from_numpy(np.array(self._planet_feats[idx])),
            global_feats=torch.from_numpy(np.array(self._global_feats[idx])),
            planet_mask=torch.from_numpy(np.array(self._planet_mask[idx])),
            my_planet_mask=torch.from_numpy(np.array(self._my_planet_mask[idx])),
            target_mask=torch.from_numpy(np.array(self._target_mask[idx])),
            template_ctx=torch.from_numpy(np.array(self._template_ctx[idx])),
            candidate_feats=torch.from_numpy(np.array(self._candidate_feats[idx])),
            candidate_mask=torch.from_numpy(np.array(self._candidate_mask[idx])),
            candidate_pid=torch.from_numpy(np.array(self._candidate_pid[idx])),
            cand_slot_per_src=torch.from_numpy(np.array(self._cand_slot_per_src[idx])),
            ship_label_per_src=torch.from_numpy(
                np.array(self._ship_label_per_src[idx])
            ),
            ships_bucket_per_src=torch.from_numpy(
                np.array(self._ships_bucket_per_src[idx])
            ),
            from_multihot=torch.from_numpy(np.array(self._from_multihot[idx])),
            target_per_src=torch.from_numpy(np.array(self._target_per_src[idx])),
            ships_per_src=torch.from_numpy(np.array(self._ships_per_src[idx])),
            target_pid_per_src=torch.from_numpy(
                np.array(self._target_pid_per_src[idx])
            ),
            ship_pred_label=torch.from_numpy(np.array(self._ship_pred_label[idx])),
            is_noop=bool(self._is_noop[idx]),
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
