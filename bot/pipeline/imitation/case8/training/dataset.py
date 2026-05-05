"""Parquet → torch Dataset/DataLoader for imitation/case8."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
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


class CaseFourDataset(Dataset[Sample]):
    """In-memory parquet-backed Dataset for case8."""

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
    ) -> None:
        # iter4 OOM fix: column ごとに to_list → numpy 化 → 中間 list と polars
        # series を即座に削除する。`pl.read_parquet` 全体を抱えたまま全列を
        # 順次 to_list() すると、polars buffer (~10GB) + Python list (~3-5GB
        # per col の中間) + numpy (~7GB peak) が三重で乗り、328k row で
        # 24-32GB の RTX 4090 host RAM を超過 OOM (iter4-v2/v3 で観測)。
        df_local = pl.read_parquet(str(parquet_path))
        n_rows = df_local.height
        has_ship_label = "ship_label_per_src" in df_local.columns

        def pop_to_numpy(df: pl.DataFrame, col: str, dtype: np.dtype) -> np.ndarray:
            """polars Series → list → numpy 即変換、polars 側から column 削除。"""
            series_list = df[col].to_list()
            arr = np.array(series_list, dtype=dtype)
            del series_list
            df.drop_in_place(col)
            gc.collect()
            return arr

        planet_arr = pop_to_numpy(df_local, "planet_feats", np.dtype(np.float32))
        planet_dim = (
            planet_arr.size // (n_rows * MAX_PLANETS) if n_rows > 0 else PLANET_FEAT_DIM
        )
        self._planet_feat_dim = int(planet_dim)
        self._planet_feats = planet_arr.reshape(-1, MAX_PLANETS, self._planet_feat_dim)
        del planet_arr
        gc.collect()

        global_arr = pop_to_numpy(df_local, "global_feats", np.dtype(np.float32))
        global_dim = global_arr.size // n_rows if n_rows > 0 else GLOBAL_FEAT_DIM
        self._global_feat_dim = int(global_dim)
        self._global_feats = global_arr.reshape(-1, self._global_feat_dim)
        del global_arr
        gc.collect()

        if mask_planet_cols:
            for col in mask_planet_cols:
                if 0 <= col < self._planet_feat_dim:
                    self._planet_feats[:, :, col] = 0.0
        if mask_global_cols:
            for col in mask_global_cols:
                if 0 <= col < self._global_feat_dim:
                    self._global_feats[:, col] = 0.0

        self._planet_mask = pop_to_numpy(
            df_local, "planet_mask", np.dtype(np.bool_)
        ).reshape(-1, MAX_PLANETS)
        self._my_planet_mask = pop_to_numpy(
            df_local, "my_planet_mask", np.dtype(np.bool_)
        ).reshape(-1, MAX_PLANETS)
        self._target_mask = pop_to_numpy(
            df_local, "target_mask", np.dtype(np.bool_)
        ).reshape(-1, MAX_PLANETS)
        self._candidate_feats = pop_to_numpy(
            df_local, "candidate_feats", np.dtype(np.float32)
        ).reshape(-1, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
        self._candidate_mask = pop_to_numpy(
            df_local, "candidate_mask", np.dtype(np.bool_)
        ).reshape(-1, MAX_PLANETS, CAND_K)
        self._candidate_pid = pop_to_numpy(
            df_local, "candidate_pid", np.dtype(np.int64)
        ).reshape(-1, MAX_PLANETS, CAND_K)
        self._cand_slot_per_src = pop_to_numpy(
            df_local, "cand_slot_per_src", np.dtype(np.int64)
        ).reshape(-1, MAX_PLANETS)
        if has_ship_label:
            self._ship_label_per_src = pop_to_numpy(
                df_local, "ship_label_per_src", np.dtype(np.int64)
            ).reshape(-1, MAX_PLANETS)
        else:
            self._ship_label_per_src = np.full(
                (n_rows, MAX_PLANETS), -1, dtype=np.int64
            )
        # is_noop は値型 (List ではない) なので to_numpy() で直接 OK
        self._is_noop = df_local["is_noop"].to_numpy().astype(np.bool_)
        df_local.drop_in_place("is_noop")
        del df_local
        gc.collect()

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Effective-number-of-samples class weights over cand_slot labels.

        Effective-number 公式 `(1 - beta^n) / (1 - beta)` は n が大きいと
        beta^n → 0 で飽和し、すべての class について `~ (1-beta)` という
        ほぼ同じ値を返す。本番 (n_slot ~ 数百万) で実際にこれが起き、
        weights が `[1, 1, ..., 1]` に縮退して loss が `cross_entropy` から
        Inf を吐き、backprop で全 param が NaN/Inf になっていた。

        修正: 数値安定 + 縮退回避のため inverse-frequency weight を採用。
        各 class i について `w_i = total_present / (counts[i] * num_present)`
        を計算する。これは:
          - counts が大きい class は w が小さく
          - counts が小さい (rare) class は w が大きい
          - counts==0 (= train data に未出現) の class は 0 で除外
          - 全 present class 平均が 1.0 になるよう正規化
        beta は API 互換性のため受け取るが現状未使用 (将来の effective-number
        ハイブリッドで再活用余地あり)。
        """
        del beta  # unused; kept for backward compat
        flat = self._cand_slot_per_src.reshape(-1)
        flat = flat[flat != ignore_index]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        present_mask = counts > 0
        if not present_mask.any():
            return torch.ones(num_classes, dtype=torch.float32)
        # inverse-frequency, then normalize so present classes average to 1.0
        raw = np.zeros(num_classes, dtype=np.float64)
        raw[present_mask] = 1.0 / counts[present_mask]
        mean_raw = float(raw[present_mask].mean())
        weights = raw / max(mean_raw, 1e-12)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._planet_feats.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            planet_feats=torch.from_numpy(self._planet_feats[idx]),
            global_feats=torch.from_numpy(self._global_feats[idx]),
            planet_mask=torch.from_numpy(self._planet_mask[idx]),
            my_planet_mask=torch.from_numpy(self._my_planet_mask[idx]),
            target_mask=torch.from_numpy(self._target_mask[idx]),
            candidate_feats=torch.from_numpy(self._candidate_feats[idx]),
            candidate_mask=torch.from_numpy(self._candidate_mask[idx]),
            candidate_pid=torch.from_numpy(self._candidate_pid[idx]),
            cand_slot_per_src=torch.from_numpy(self._cand_slot_per_src[idx]),
            ship_label_per_src=torch.from_numpy(self._ship_label_per_src[idx]),
            is_noop=bool(self._is_noop[idx]),
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
