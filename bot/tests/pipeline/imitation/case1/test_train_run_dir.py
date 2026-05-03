"""train.py の ORBIT_WARS_RUN_DIR override 経路を検証する。

- env なし: 従来通り params.yaml: train.weights_out にのみ書く (regression 確認)
- env あり: <run_dir>/{best.pt, metrics.json, run.json} を生成
- ORBIT_WARS_VAST_INSTANCE_ID あり + RUN_DIR なし: assertion error (Risk #4)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.imitation.case1.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case1.policy.templates import TEMPLATE_CTX_DIM
from pipeline.imitation.case1.training.train import _resolve_run_dir, train

pytestmark = pytest.mark.slow


def _make_mini_parquet(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        is_noop = bool(i % 4 == 0)
        from_multihot = [False] * MAX_PLANETS
        target_per_src = [-1] * MAX_PLANETS
        ships_per_src = [-1] * MAX_PLANETS
        if not is_noop:
            src = int(rng.integers(0, 4))
            from_multihot[src] = True
            target_per_src[src] = int(rng.integers(0, 8))
            ships_per_src[src] = int(rng.integers(0, 4))
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
                "template_ctx": rng.standard_normal(MAX_PLANETS * TEMPLATE_CTX_DIM)
                .astype(np.float32)
                .tolist(),
                "from_multihot": from_multihot,
                "target_per_src": target_per_src,
                "ships_per_src": ships_per_src,
                "is_noop": is_noop,
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _build_cfg(
    train_path: Path, val_path: Path, weights_path: Path
) -> dict[str, object]:
    return {
        "seed": 0,
        "data": {
            "out_train": str(train_path),
            "out_val": str(val_path),
        },
        "model": {
            "planet_in_dim": PLANET_FEAT_DIM,
            "global_in_dim": GLOBAL_FEAT_DIM,
            "hidden": 16,
            "ships_buckets": 4,
        },
        "train": {
            "batch_size": 64,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "loss_weights": {"from": 1.0, "target": 1.0, "ships": 0.5},
            "weights_out": str(weights_path),
        },
    }


@pytest.fixture
def mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    _make_mini_parquet(train_path, n=200, seed=0)
    _make_mini_parquet(val_path, n=80, seed=1)
    return train_path, val_path


def test_run_dir_override_writes_three_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: tuple[Path, Path],
) -> None:
    train_path, val_path = mini_dataset
    canonical = tmp_path / "canonical_should_not_be_written.pt"
    run_dir = tmp_path / "run_xxx"
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(run_dir))
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "test_run_xxx")
    monkeypatch.setenv("ORBIT_WARS_GIT_SHA", "abc1234deadbeef")
    monkeypatch.setenv("ORBIT_WARS_GIT_BRANCH", "feature/test")

    report = train(_build_cfg(train_path, val_path, canonical))

    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "run.json").is_file()
    # canonical path was not touched
    assert not canonical.exists()
    assert report.weights_path == run_dir / "best.pt"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_run"] == 1
    assert "train_loss_history" in metrics

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["run_id"] == "test_run_xxx"
    assert meta["git_sha"] == "abc1234deadbeef"
    assert meta["git_branch"] == "feature/test"
    assert meta["status"] == "pushed"
    assert meta["seed"] == 0
    assert len(meta["params_hash"]) == 12
    assert meta["weights_path"].endswith("best.pt")


def test_no_env_uses_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: tuple[Path, Path],
) -> None:
    train_path, val_path = mini_dataset
    canonical = tmp_path / "canonical.pt"
    monkeypatch.delenv("ORBIT_WARS_RUN_DIR", raising=False)
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)

    report = train(_build_cfg(train_path, val_path, canonical))
    assert canonical.is_file()
    assert report.weights_path == canonical
    # No metrics.json / run.json written next to it
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "run.json").exists()


def test_vast_id_without_run_dir_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_VAST_INSTANCE_ID", "12345")
    monkeypatch.delenv("ORBIT_WARS_RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ORBIT_WARS_RUN_DIR"):
        _resolve_run_dir()


def test_runpod_id_without_run_dir_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_RUNPOD_POD_ID", "abcd1234")
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ORBIT_WARS_RUN_DIR"):
        _resolve_run_dir()


def test_both_provider_ids_set_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_VAST_INSTANCE_ID", "12345")
    monkeypatch.setenv("ORBIT_WARS_RUNPOD_POD_ID", "abcd1234")
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", "/tmp/whatever")
    with pytest.raises(RuntimeError, match="Both"):
        _resolve_run_dir()


def test_run_dir_override_records_runpod_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: tuple[Path, Path],
) -> None:
    train_path, val_path = mini_dataset
    canonical = tmp_path / "canonical_should_not_be_written.pt"
    run_dir = tmp_path / "run_runpod"
    snapshot = {
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "cloud_type": "SECURE",
        "dph_total": 0.43,
    }
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(run_dir))
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "test_runpod_run")
    monkeypatch.setenv("ORBIT_WARS_GIT_SHA", "abc1234deadbeef")
    monkeypatch.setenv("ORBIT_WARS_GIT_BRANCH", "feature/test")
    monkeypatch.setenv("ORBIT_WARS_RUNPOD_POD_ID", "pod-abc123")
    monkeypatch.setenv("ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT", json.dumps(snapshot))
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)

    train(_build_cfg(train_path, val_path, canonical))

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["runpod_pod_id"] == "pod-abc123"
    assert meta["runpod_offer_snapshot"] == snapshot
    assert meta["vast_instance_id"] is None
    assert meta["vast_offer_snapshot"] is None
