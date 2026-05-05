"""Parquet → torch Dataset/DataLoader for imitation/case5 IL baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

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
    from_multihot: torch.Tensor  # (MAX_PLANETS,) bool — True for fired-from sources
    target_per_src: (
        torch.Tensor
    )  # (MAX_PLANETS,) int64 — target slot or NO_OP, -1 if unused
    ships_per_src: torch.Tensor  # (MAX_PLANETS,) int64 — bucket index, -1 if unused
    is_noop: bool


@dataclass(frozen=True)
class BatchedSample:
    planet_feats: torch.Tensor  # (B, MAX_PLANETS, PLANET_FEAT_DIM)
    global_feats: torch.Tensor  # (B, GLOBAL_FEAT_DIM)
    planet_mask: torch.Tensor  # (B, MAX_PLANETS)
    my_planet_mask: torch.Tensor  # (B, MAX_PLANETS)
    target_mask: torch.Tensor  # (B, MAX_PLANETS)
    template_ctx: torch.Tensor  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM)
    from_multihot: torch.Tensor  # (B, MAX_PLANETS) bool
    target_per_src: torch.Tensor  # (B, MAX_PLANETS) int64
    ships_per_src: torch.Tensor  # (B, MAX_PLANETS) int64
    is_noop: torch.Tensor  # (B,) bool


class CaseThreeDataset(Dataset[Sample]):
    """In-memory parquet-backed Dataset (rows ≈ 1.6 KB each)."""

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
        self._template_ctx = np.array(
            self._df["template_ctx"].to_list(), dtype=np.float32
        ).reshape(-1, MAX_PLANETS, TEMPLATE_CTX_DIM)
        self._from_multihot = np.array(
            self._df["from_multihot"].to_list(), dtype=np.bool_
        ).reshape(-1, MAX_PLANETS)
        self._target_per_src = np.array(
            self._df["target_per_src"].to_list(), dtype=np.int64
        ).reshape(-1, MAX_PLANETS)
        self._ships_per_src = np.array(
            self._df["ships_per_src"].to_list(), dtype=np.int64
        ).reshape(-1, MAX_PLANETS)
        self._is_noop = self._df["is_noop"].to_numpy().astype(np.bool_)

    def sample_weights_from_target(
        self,
        num_classes: int,
        power: float = 0.5,
        ignore_index: int = -1,
        aggregate: str = "mean",
    ) -> np.ndarray:
        """Per-sample weight ∝ mean (1 / class_freq)^power across fired target labels.

        A frame may fire multiple sources with different target templates. We
        aggregate per-source class_weight via `aggregate` (`mean` or `max`).
        `mean` is the default — it lifts minority frames proportionally to
        their minority content rather than letting a single rare label in a
        mostly-majority frame dominate. `power=0.5` is half-strength 1/freq.
        """
        flat = self._target_per_src.reshape(-1)
        valid = flat[flat != ignore_index]
        counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
        freq = np.where(counts > 0, counts / counts.sum(), 1.0)
        class_weight = np.power(1.0 / np.clip(freq, 1e-12, None), power)
        weights = np.zeros(self._target_per_src.shape[0], dtype=np.float64)
        for i in range(self._target_per_src.shape[0]):
            row = self._target_per_src[i]
            fired = row[row != ignore_index]
            if fired.size == 0:
                weights[i] = 1.0
            elif aggregate == "max":
                weights[i] = float(class_weight[fired].max())
            else:
                weights[i] = float(class_weight[fired].mean())
        weights *= len(weights) / weights.sum()
        return weights

    def __len__(self) -> int:
        return int(self._planet_feats.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            planet_feats=torch.from_numpy(self._planet_feats[idx]),
            global_feats=torch.from_numpy(self._global_feats[idx]),
            planet_mask=torch.from_numpy(self._planet_mask[idx]),
            my_planet_mask=torch.from_numpy(self._my_planet_mask[idx]),
            target_mask=torch.from_numpy(self._target_mask[idx]),
            template_ctx=torch.from_numpy(self._template_ctx[idx]),
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
        from_multihot=torch.stack([s.from_multihot for s in samples]),
        target_per_src=torch.stack([s.target_per_src for s in samples]),
        ships_per_src=torch.stack([s.ships_per_src for s in samples]),
        is_noop=torch.tensor([s.is_noop for s in samples], dtype=torch.bool),
    )
