"""Parquet → torch Dataset/DataLoader for imitation/case4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case4.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case4.policy.featurizer import (
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
    is_noop: torch.Tensor


class CaseFourDataset(Dataset[Sample]):
    """In-memory parquet-backed Dataset for case4."""

    def __init__(
        self,
        parquet_path: Path | str,
        mask_planet_cols: list[int] | None = None,
        mask_global_cols: list[int] | None = None,
    ) -> None:
        self._df = pl.read_parquet(str(parquet_path))
        n_rows = self._df.height

        planet_arr = np.array(self._df["planet_feats"].to_list(), dtype=np.float32)
        if n_rows > 0:
            planet_dim = planet_arr.size // (n_rows * MAX_PLANETS)
        else:
            planet_dim = PLANET_FEAT_DIM
        self._planet_feat_dim = int(planet_dim)
        self._planet_feats = planet_arr.reshape(-1, MAX_PLANETS, self._planet_feat_dim)

        global_arr = np.array(self._df["global_feats"].to_list(), dtype=np.float32)
        if n_rows > 0:
            global_dim = global_arr.size // n_rows
        else:
            global_dim = GLOBAL_FEAT_DIM
        self._global_feat_dim = int(global_dim)
        self._global_feats = global_arr.reshape(-1, self._global_feat_dim)

        if mask_planet_cols:
            for col in mask_planet_cols:
                if 0 <= col < self._planet_feat_dim:
                    self._planet_feats[:, :, col] = 0.0
        if mask_global_cols:
            for col in mask_global_cols:
                if 0 <= col < self._global_feat_dim:
                    self._global_feats[:, col] = 0.0

        self._planet_mask = np.array(
            self._df["planet_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._my_planet_mask = np.array(
            self._df["my_planet_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._target_mask = np.array(
            self._df["target_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._candidate_feats = np.array(
            self._df["candidate_feats"].to_list(), dtype=np.float32
        ).reshape(-1, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
        self._candidate_mask = np.array(
            self._df["candidate_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS, CAND_K)
        self._candidate_pid = np.array(
            self._df["candidate_pid"].to_list(), dtype=np.int64
        ).reshape(-1, MAX_PLANETS, CAND_K)
        self._cand_slot_per_src = np.array(
            self._df["cand_slot_per_src"].to_list(), dtype=np.int64
        ).reshape(-1, MAX_PLANETS)
        self._is_noop = self._df["is_noop"].to_numpy().astype(np.bool_)

    def class_weight_on_slots(
        self, num_classes: int, beta: float = 0.999, ignore_index: int = -1
    ) -> torch.Tensor:
        """Effective-number-of-samples class weights over cand_slot labels."""
        flat = self._cand_slot_per_src.reshape(-1)
        flat = flat[flat != ignore_index]
        counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
        eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        eff_num = np.clip(eff_num, 1.0, None)
        weights = (1.0 - beta) / np.clip(1.0 - np.power(beta, counts), 1e-12, None)
        weights = np.where(counts > 0, weights, 0.0)
        present = np.sum(counts > 0)
        denom = max(weights.sum(), 1e-12)
        weights = weights * (max(present, 1) / denom)
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
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
