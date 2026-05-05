"""End-to-end smoke test for case8 train loop with scheduler + early stop.

Builds a tiny synthetic parquet dataset and runs `train()` for 2 epochs to
verify scheduler init / DataLoader / epoch loop / best.pt save / early stop
all work without hangs or exceptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.imitation.case8.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case8.training.preprocess import _arrow_schema
from pipeline.imitation.case8.training.train import train


def _build_synthetic_parquet(path: Path, n_rows: int = 32) -> None:
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        rows.append(
            {
                "planet_feats": rng.standard_normal(
                    MAX_PLANETS * PLANET_FEAT_DIM
                ).astype(np.float32).tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": ([True] * 4 + [False] * (MAX_PLANETS - 4)),
                "my_planet_mask": ([True] * 2 + [False] * (MAX_PLANETS - 2)),
                "target_mask": ([True] * 4 + [False] * (MAX_PLANETS - 4)),
                "candidate_feats": rng.standard_normal(
                    MAX_PLANETS * CAND_K * CAND_FEAT_DIM
                ).astype(np.float32).tolist(),
                "candidate_mask": ([True] * (MAX_PLANETS * CAND_K)),
                "candidate_pid": ([1] * (MAX_PLANETS * CAND_K)),
                "cand_slot_per_src": (
                    [int(i % CAND_K), 0]
                    + [-1] * (MAX_PLANETS - 2)
                ),
                "ship_label_per_src": (
                    [int(20 + (i % 30)), -1]
                    + [-1] * (MAX_PLANETS - 2)
                ),
                "is_noop": False,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_arrow_schema())
    pq.write_table(table, str(path), compression="zstd")


@pytest.mark.timeout(120)
def test_train_smoke_scheduler_and_early_stop(tmp_path: Path) -> None:
    """Run 2 epochs end-to-end: scheduler builds, dataloader yields, no hang."""
    train_pq = tmp_path / "train.parquet"
    val_pq = tmp_path / "val.parquet"
    _build_synthetic_parquet(train_pq, n_rows=32)
    _build_synthetic_parquet(val_pq, n_rows=16)

    weights_out = tmp_path / "weights.pt"
    cfg = {
        "seed": 0,
        "data": {
            "out_train": str(train_pq),
            "out_val": str(val_pq),
        },
        "model": {
            "planet_in_dim": PLANET_FEAT_DIM,
            "global_in_dim": GLOBAL_FEAT_DIM,
            "cand_in_dim": CAND_FEAT_DIM,
            "cand_k": CAND_K,
            "hidden": 16,
        },
        "train": {
            "case": "case8",
            "batch_size": 4,
            "epochs": 2,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "loss_weights": {
                "cand": 1.0,
                "use_class_weights": False,
                "ship": 1.0,
            },
            "scheduler": {
                "type": "cosine_warmup",
                "t_max": 2,
                "eta_min": 1.0e-5,
                "warmup_epochs": 1,
                "warmup_start_factor": 0.1,
            },
            "best_metric": "val_cand_fire_acc",
            "best_metric_mode": "max",
            "early_stop": {
                "metric": "val_cand_fire_acc",
                "patience": 5,
                "mode": "max",
            },
            "weights_out": str(weights_out),
            "run_dir": str(tmp_path / "run"),
        },
    }
    report = train(cfg)
    assert report.epochs_run >= 1
    assert report.epochs_run <= 2
    assert weights_out.exists() or (tmp_path / "run" / "best.pt").exists()
