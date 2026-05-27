"""run_meta のユニットテスト。

- run_id 生成のフォーマット
- branch slug の `/` → `-` 変換
- params hash の dict 順序非依存
- write/read/update の round-trip
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu.vast.run_meta import (
    RUN_ID_PATTERN,
    SCHEMA_VERSION,
    RunMetadata,
    _branch_slug,
    generate_run_id,
    hash_params,
    read_run_json,
    update_run_json,
    write_run_json,
)


def test_generate_run_id_format() -> None:
    now = datetime(2026, 4, 25, 14, 30, 22, tzinfo=timezone.utc)
    rid = generate_run_id("feature/vast-ai-basis", "abc1234deadbeef", 0, now=now)
    assert rid == "20260425-143022__feature-vast-ai-basis__abc1234__seed0"
    assert RUN_ID_PATTERN.match(rid)


def test_generate_run_id_short_sha_rejected() -> None:
    now = datetime(2026, 4, 25, 14, 30, 22, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        generate_run_id("main", "abc12", 0, now=now)


def test_branch_slug_complex_chars() -> None:
    assert _branch_slug("feature/vast-ai-basis") == "feature-vast-ai-basis"
    assert _branch_slug("user/foo/bar.baz") == "user-foo-bar-baz"
    assert _branch_slug("---weird---") == "weird"
    assert _branch_slug("/") == "unknown"


def test_hash_params_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    # 同じ内容で順序違い
    a.write_text("seed: 0\ntrain:\n  lr: 0.001\n  epochs: 15\n", encoding="utf-8")
    b.write_text("train:\n  epochs: 15\n  lr: 0.001\nseed: 0\n", encoding="utf-8")
    assert hash_params(a) == hash_params(b)
    assert len(hash_params(a)) == 12


def test_hash_params_changes_on_value(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("seed: 0\n", encoding="utf-8")
    b.write_text("seed: 1\n", encoding="utf-8")
    assert hash_params(a) != hash_params(b)


def test_run_metadata_schema_version_default() -> None:
    meta = RunMetadata()
    assert meta.schema_version == SCHEMA_VERSION
    assert meta.status == "running"
    assert meta.local_eval_results is None


def test_write_read_run_json_roundtrip(tmp_path: Path) -> None:
    meta = RunMetadata(
        run_id="20260425-143022__main__abc1234__seed0",
        git_sha="abc1234deadbeef" * 2,
        git_branch="main",
        params_hash="0123456789ab",
        seed=0,
        command="dvc repro train_imitation_case1",
        weights_path="artifacts/.../best.pt",
        train_metrics={"epochs_run": 15, "best_val_loss": 3.84},
        status="pushed",
    )
    write_run_json(tmp_path, meta)
    restored = read_run_json(tmp_path)
    assert restored.run_id == meta.run_id
    assert restored.git_sha == meta.git_sha
    assert restored.train_metrics["epochs_run"] == 15
    assert restored.created_at  # filled
    assert restored.updated_at


def test_update_run_json_patches_and_bumps_updated_at(tmp_path: Path) -> None:
    meta = RunMetadata(run_id="r1", git_sha="abc1234x", seed=0, status="pushed")
    write_run_json(tmp_path, meta)
    first = read_run_json(tmp_path)
    updated = update_run_json(
        tmp_path, status="adopted", local_eval_results={"win_rate": 0.55}
    )
    assert updated.status == "adopted"
    assert updated.local_eval_results == {"win_rate": 0.55}
    assert updated.created_at == first.created_at
    # updated_at は >= first.updated_at
    assert updated.updated_at >= first.updated_at


def test_update_run_json_unknown_field_raises(tmp_path: Path) -> None:
    write_run_json(tmp_path, RunMetadata(run_id="r1", git_sha="abc1234x", seed=0))
    with pytest.raises(AttributeError):
        update_run_json(tmp_path, totally_not_a_field=42)


def test_run_json_is_human_readable(tmp_path: Path) -> None:
    meta = RunMetadata(run_id="r1", git_sha="abc1234x", seed=0)
    write_run_json(tmp_path, meta)
    raw = (tmp_path / "run.json").read_text(encoding="utf-8")
    data = json.loads(raw)  # parseable
    assert "\n" in raw  # indented
    assert data["schema_version"] == SCHEMA_VERSION


def test_runpod_fields_default_none() -> None:
    meta = RunMetadata()
    assert meta.runpod_pod_id is None
    assert meta.runpod_offer_snapshot is None


def test_runpod_fields_roundtrip(tmp_path: Path) -> None:
    snapshot = {
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "cloud_type": "SECURE",
        "dph_total": 0.43,
    }
    meta = RunMetadata(
        run_id="20260502-100000__main__deadbee__seed0",
        git_sha="deadbeefcafef00d",
        seed=0,
        runpod_pod_id="abcd1234ef",
        runpod_offer_snapshot=snapshot,
    )
    write_run_json(tmp_path, meta)
    restored = read_run_json(tmp_path)
    assert restored.runpod_pod_id == "abcd1234ef"
    assert restored.runpod_offer_snapshot == snapshot
    assert restored.vast_instance_id is None
    assert restored.vast_offer_snapshot is None


def test_legacy_run_json_without_runpod_fields_loads(tmp_path: Path) -> None:
    """Vast 既存 run.json (runpod_* 欠如) を読めて default で埋まること。"""
    legacy = {
        "schema_version": 1,
        "run_id": "legacy_run",
        "git_sha": "abc1234deadbe",
        "git_branch": "main",
        "params_hash": "0123456789ab",
        "seed": 0,
        "vast_instance_id": 99999,
        "gpu_name": "RTX_3090",
        "vast_offer_snapshot": {"dph_total": 0.13},
        "command": "x",
        "weights_path": "w",
        "train_metrics": {},
        "local_eval_results": None,
        "status": "pushed",
        "created_at": "2026-04-25T00:00:00Z",
        "updated_at": "2026-04-25T00:10:00Z",
    }
    (tmp_path / "run.json").write_text(json.dumps(legacy), encoding="utf-8")
    restored = read_run_json(tmp_path)
    assert restored.vast_instance_id == 99999
    assert restored.runpod_pod_id is None
    assert restored.runpod_offer_snapshot is None


def test_kaggle_kernel_meta_default_none() -> None:
    meta = RunMetadata()
    assert meta.kaggle_kernel_meta is None


def test_kaggle_kernel_meta_roundtrip(tmp_path: Path) -> None:
    kk_meta = {
        "kernel_slug": "yuri/orbit-wars-case1-20260520",
        "kernel_version": 3,
        "dataset_slug": "yuri/orbit-wars-bot",
        "dataset_version": "v17",
        "accelerator": "gpu-t4x2",
        "runtime_seconds": 1820,
        "internet_enabled": True,
    }
    meta = RunMetadata(
        run_id="20260520-103000__main__abc1234__seed0",
        git_sha="abc1234deadbeef",
        seed=0,
        kaggle_kernel_meta=kk_meta,
    )
    write_run_json(tmp_path, meta)
    restored = read_run_json(tmp_path)
    assert restored.kaggle_kernel_meta == kk_meta
    assert restored.vast_instance_id is None
    assert restored.runpod_pod_id is None


def test_legacy_run_json_without_kaggle_kernel_meta_loads(tmp_path: Path) -> None:
    """RunPod 既存 run.json (kaggle_kernel_meta 欠如) を読めて default で埋まること。"""
    legacy = {
        "schema_version": 1,
        "run_id": "legacy_runpod",
        "git_sha": "abc1234deadbe",
        "git_branch": "main",
        "params_hash": "0123456789ab",
        "seed": 0,
        "runpod_pod_id": "pod_abc",
        "runpod_offer_snapshot": {"dph_total": 0.5, "gpu_type_id": "X"},
        "command": "x",
        "weights_path": "w",
        "train_metrics": {},
        "local_eval_results": None,
        "status": "pushed",
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:10:00Z",
    }
    (tmp_path / "run.json").write_text(json.dumps(legacy), encoding="utf-8")
    restored = read_run_json(tmp_path)
    assert restored.runpod_pod_id == "pod_abc"
    assert restored.kaggle_kernel_meta is None
