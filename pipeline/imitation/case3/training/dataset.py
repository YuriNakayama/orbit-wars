"""Parquet → torch Dataset/DataLoader for case3 IL baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from pipeline.imitation.case3.policy.featurizer import (
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
    from_label: int
    target_label: int
    ships_label: int
    is_noop: bool


@dataclass(frozen=True)
class BatchedSample:
    planet_feats: torch.Tensor  # (B, MAX_PLANETS, PLANET_FEAT_DIM)
    global_feats: torch.Tensor  # (B, GLOBAL_FEAT_DIM)
    planet_mask: torch.Tensor  # (B, MAX_PLANETS)
    my_planet_mask: torch.Tensor  # (B, MAX_PLANETS)
    target_mask: torch.Tensor  # (B, MAX_PLANETS)
    from_label: torch.Tensor  # (B,)
    target_label: torch.Tensor  # (B,)
    ships_label: torch.Tensor  # (B,)
    is_noop: torch.Tensor  # (B,) bool


class CaseThreeDataset(Dataset[Sample]):
    """In-memory parquet-backed Dataset.

    Loads the entire parquet via polars then materializes per-row tensors
    on demand. Memory: ~100k rows × ~1.6KB ≈ 160MB which fits comfortably.
    """

    def __init__(self, parquet_path: Path | str) -> None:
        self._df = pl.read_parquet(str(parquet_path))
        self._planet_feats = np.array(
            self._df["planet_feats"].to_list(), dtype=np.float32
        ).reshape(-1, MAX_PLANETS, PLANET_FEAT_DIM)
        self._global_feats = np.array(
            self._df["global_feats"].to_list(), dtype=np.float32
        ).reshape(-1, GLOBAL_FEAT_DIM)
        self._planet_mask = np.array(
            self._df["planet_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._my_planet_mask = np.array(
            self._df["my_planet_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._target_mask = np.array(
            self._df["target_mask"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._from_label = self._df["from_label"].to_numpy().astype(np.int64)
        self._target_label = self._df["target_label"].to_numpy().astype(np.int64)
        self._ships_label = self._df["ships_label"].to_numpy().astype(np.int64)
        self._is_noop = self._df["is_noop"].to_numpy().astype(np.bool_)

    def __len__(self) -> int:
        return int(self._planet_feats.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            planet_feats=torch.from_numpy(self._planet_feats[idx]),
            global_feats=torch.from_numpy(self._global_feats[idx]),
            planet_mask=torch.from_numpy(self._planet_mask[idx]),
            my_planet_mask=torch.from_numpy(self._my_planet_mask[idx]),
            target_mask=torch.from_numpy(self._target_mask[idx]),
            from_label=int(self._from_label[idx]),
            target_label=int(self._target_label[idx]),
            ships_label=int(self._ships_label[idx]),
            is_noop=bool(self._is_noop[idx]),
        )


def collate(samples: list[Sample]) -> BatchedSample:
    return BatchedSample(
        planet_feats=torch.stack([s.planet_feats for s in samples]),
        global_feats=torch.stack([s.global_feats for s in samples]),
        planet_mask=torch.stack([s.planet_mask for s in samples]),
        my_planet_mask=torch.stack([s.my_planet_mask for s in samples]),
        target_mask=torch.stack([s.target_mask for s in samples]),
        from_label=torch.tensor([s.from_label for s in samples], dtype=torch.long),
        target_label=torch.tensor([s.target_label for s in samples], dtype=torch.long),
        ships_label=torch.tensor([s.ships_label for s in samples], dtype=torch.long),
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
