"""Training determinism: same seed → same val loss across runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.imitation.case3.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case3.training.preprocess import NO_OP_LABEL
from pipeline.imitation.case3.training.train import train

pytestmark = pytest.mark.slow


def _make_mini_parquet(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        is_noop = bool(i % 4 == 0)
        from_label = NO_OP_LABEL if is_noop else int(rng.integers(0, 4))
        target_label = NO_OP_LABEL if is_noop else int(rng.integers(0, 4))
        ships_label = 0 if is_noop else int(rng.integers(0, 5))
        rows.append(
            {
                "planet_feats": rng.standard_normal(MAX_PLANETS * PLANET_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": [bool(j < 8) for j in range(MAX_PLANETS)],
                "my_planet_mask": [bool(j < 4) for j in range(MAX_PLANETS)],
                "target_mask": [bool(4 <= j < 8) for j in range(MAX_PLANETS)],
                "from_label": from_label,
                "target_label": target_label,
                "ships_label": ships_label,
                "is_noop": is_noop,
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _build_cfg(
    train_path: Path, val_path: Path, weights_path: Path
) -> dict[str, object]:
    return {
        "seed": 42,
        "data": {
            "out_train": str(train_path),
            "out_val": str(val_path),
        },
        "model": {
            "planet_in_dim": PLANET_FEAT_DIM,
            "global_in_dim": GLOBAL_FEAT_DIM,
            "hidden": 16,
            "ships_buckets": 5,
        },
        "train": {
            "batch_size": 64,
            "epochs": 2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "loss_weights": {"from": 1.0, "target": 1.0, "ships": 0.5},
            "weights_out": str(weights_path),
        },
    }


def test_same_seed_same_val_loss(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    _make_mini_parquet(train_path, n=800, seed=0)
    _make_mini_parquet(val_path, n=200, seed=1)

    cfg_a = _build_cfg(train_path, val_path, tmp_path / "w_a.pt")
    cfg_b = _build_cfg(train_path, val_path, tmp_path / "w_b.pt")

    rep_a = train(cfg_a)
    rep_b = train(cfg_b)
    assert rep_a.best_val_loss == pytest.approx(rep_b.best_val_loss, abs=1e-6)
