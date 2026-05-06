"""TensorDataset wrapper for the case0 dummy parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CaseZeroDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads a parquet of (f0..f{N-1}, label) and exposes (features, label)."""

    def __init__(self, parquet_path: Path, in_dim: int) -> None:
        df = pd.read_parquet(parquet_path)
        feature_cols = [f"f{i}" for i in range(in_dim)]
        missing = [c for c in (*feature_cols, "label") if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns in {parquet_path}: {missing}")
        self._features = torch.from_numpy(
            df[feature_cols].to_numpy(dtype="float32", copy=True)
        )
        self._labels = torch.from_numpy(df["label"].to_numpy(dtype="int64", copy=True))

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[idx], self._labels[idx]


__all__ = ["CaseZeroDataset"]
