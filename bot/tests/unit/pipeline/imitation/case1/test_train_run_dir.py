"""train.py の ORBIT_WARS_RUN_DIR override 経路を検証する。

- env なし: 従来通り params.yaml: train.weights_out にのみ書く (regression 確認)
- env あり: <run_dir>/{best.pt, metrics.json, run.json} を生成
- ORBIT_WARS_VAST_INSTANCE_ID あり + RUN_DIR なし: assertion error (Risk #4)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.imitation.case1.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case1.training.train import _resolve_run_dir, train

pytestmark = pytest.mark.slow


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
    monkeypatch.delenv("ORBIT_WARS_KAGGLE_KERNEL_SLUG", raising=False)
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", "/tmp/whatever")
    with pytest.raises(RuntimeError, match="Multiple provider"):
        _resolve_run_dir()


def test_kaggle_kernel_slug_without_run_dir_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORBIT_WARS_KAGGLE_KERNEL_SLUG", "user/orbit-wars-case1")
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ORBIT_WARS_RUN_DIR"):
        _resolve_run_dir()


def test_vast_and_kaggle_kernel_set_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_VAST_INSTANCE_ID", "12345")
    monkeypatch.setenv("ORBIT_WARS_KAGGLE_KERNEL_SLUG", "user/orbit-wars-case1")
    monkeypatch.delenv("ORBIT_WARS_RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", "/tmp/whatever")
    with pytest.raises(RuntimeError, match="Multiple provider"):
        _resolve_run_dir()


def test_runpod_and_kaggle_kernel_set_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WARS_RUNPOD_POD_ID", "pod_abc")
    monkeypatch.setenv("ORBIT_WARS_KAGGLE_KERNEL_SLUG", "user/orbit-wars-case1")
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", "/tmp/whatever")
    with pytest.raises(RuntimeError, match="Multiple provider"):
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
    monkeypatch.delenv("ORBIT_WARS_KAGGLE_KERNEL_SLUG", raising=False)

    train(_build_cfg(train_path, val_path, canonical))

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["runpod_pod_id"] == "pod-abc123"
    assert meta["runpod_offer_snapshot"] == snapshot
    assert meta["vast_instance_id"] is None
    assert meta["vast_offer_snapshot"] is None
    assert meta["kaggle_kernel_meta"] is None


def test_run_dir_override_records_kaggle_kernel_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: tuple[Path, Path],
) -> None:
    train_path, val_path = mini_dataset
    canonical = tmp_path / "canonical_should_not_be_written.pt"
    run_dir = tmp_path / "run_kaggle"
    kk_meta = {
        "kernel_slug": "yuri/orbit-wars-case1-20260520",
        "kernel_version": 3,
        "dataset_slug": "yuri/orbit-wars-bot",
        "dataset_version": "v17",
        "accelerator": "gpu-t4x2",
        "internet_enabled": True,
    }
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(run_dir))
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "test_kaggle_run")
    monkeypatch.setenv("ORBIT_WARS_GIT_SHA", "abc1234deadbeef")
    monkeypatch.setenv("ORBIT_WARS_GIT_BRANCH", "feature/test")
    monkeypatch.setenv(
        "ORBIT_WARS_KAGGLE_KERNEL_SLUG", "yuri/orbit-wars-case1-20260520"
    )
    monkeypatch.setenv("ORBIT_WARS_KAGGLE_KERNEL_META", json.dumps(kk_meta))
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUNPOD_POD_ID", raising=False)

    train(_build_cfg(train_path, val_path, canonical))

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["kaggle_kernel_meta"] == kk_meta
    assert meta["vast_instance_id"] is None
    assert meta["runpod_pod_id"] is None


def test_kaggle_kernel_meta_malformed_json_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mini_dataset: tuple[Path, Path],
) -> None:
    train_path, val_path = mini_dataset
    canonical = tmp_path / "canonical.pt"
    run_dir = tmp_path / "run_bad"
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(run_dir))
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "test_bad_kk_meta")
    monkeypatch.setenv("ORBIT_WARS_GIT_SHA", "abc1234deadbeef")
    monkeypatch.setenv(
        "ORBIT_WARS_KAGGLE_KERNEL_SLUG", "yuri/orbit-wars-case1-20260520"
    )
    monkeypatch.setenv("ORBIT_WARS_KAGGLE_KERNEL_META", "not-json{")
    monkeypatch.delenv("ORBIT_WARS_VAST_INSTANCE_ID", raising=False)
    monkeypatch.delenv("ORBIT_WARS_RUNPOD_POD_ID", raising=False)

    with pytest.raises(RuntimeError, match="ORBIT_WARS_KAGGLE_KERNEL_META"):
        train(_build_cfg(train_path, val_path, canonical))
