"""gpu.runpod.execution.summary.build_summary のユニットテスト。"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpu.runpod.execution.progress import ProgressMarker
from gpu.runpod.execution.summary import build_summary


def _now_minus(seconds: int) -> str:
    ts = datetime.now(UTC) - timedelta(seconds=seconds)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_run_dir(
    repo_root: Path,
    run_id: str,
    case: str = "case1",
    *,
    write_launch: bool = True,
    write_run_json: dict[str, Any] | None = None,
    artifact_files: tuple[str, ...] = (),
    write_dvc_meta: bool = False,
) -> Path:
    run_dir = repo_root / f"data/output/models/imitation/{case}/runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_launch:
        (run_dir / "launch.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "pod_id": "pod-x",
                    "commit_sha": "abc",
                    "branch": "feature-x",
                    "case": case,
                    "cloud_type": "SECURE",
                    "gpu_type_id": "RTX A5000",
                    "dph_total": 0.27,
                    "data_center_id": "CA-MTL-3",
                    "launched_at": "2026-05-02T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
    for f in artifact_files:
        # run.json は後で write_run_json から書くのでここではスキップ
        if f == "run.json" and write_run_json is not None:
            continue
        (run_dir / f).write_bytes(b"x")
    if write_run_json is not None:
        (run_dir / "run.json").write_text(json.dumps(write_run_json), encoding="utf-8")
    if write_dvc_meta:
        (run_dir.parent / f"{run_id}.dvc").write_text("outs:\n", encoding="utf-8")
    return run_dir


class _StubProgress:
    def __init__(self, markers: list[ProgressMarker], artifacts: list[str]) -> None:
        self._markers = markers
        self._artifacts = artifacts

    def list_markers(
        self, run_id: str, *, profile: str | None = None
    ) -> list[ProgressMarker]:
        return list(self._markers)

    def list_artifacts(self, run_id: str, *, profile: str | None = None) -> list[str]:
        return list(self._artifacts)


def test_summary_succeeded_with_full_artifacts(tmp_path: Path) -> None:
    run_id = "run42"
    _make_run_dir(
        tmp_path,
        run_id,
        write_run_json={
            "schema_version": 1,
            "run_id": run_id,
            "train_metrics": {
                "epochs_run": 15,
                "train_loss_history": [4.0, 3.5, 3.0],
                "val_loss_history": [4.2, 3.8, 3.4],
            },
            "status": "pushed",
        },
        artifact_files=("best.pt", "metrics.json", "run.json", "onstart.log"),
        write_dvc_meta=True,
    )
    progress = _StubProgress(
        markers=[
            ProgressMarker(timestamp=_now_minus(900), step="00_container_started"),
            ProgressMarker(timestamp=_now_minus(60), step="99_done"),
        ],
        artifacts=["best.pt", "metrics.json", "run.json", "onstart.log"],
    )
    s = build_summary(
        run_id, repo_root=tmp_path, case="case1", progress_module=progress
    )
    assert s.status == "succeeded"
    assert s.last_step == "99_done"
    assert s.pod_id == "pod-x"
    assert s.epochs_completed == 15
    assert s.final_train_loss == 3.0
    assert s.final_val_loss == 3.4
    assert s.dvc_meta_committed is True
    # 全 artifact がローカル + S3 両方に存在
    by_name = {a.name: a for a in s.artifacts}
    assert by_name["best.pt"].in_run_dir is True
    assert by_name["best.pt"].in_s3_artifacts is True
    # 推定コスト > 0 (dph 0.27, elapsed ~840s)
    assert s.estimated_cost_usd is not None
    assert s.estimated_cost_usd > 0


def test_summary_failed_with_cleanup_exit_nonzero(tmp_path: Path) -> None:
    run_id = "run-fail"
    _make_run_dir(tmp_path, run_id, write_run_json=None)
    progress = _StubProgress(
        markers=[
            ProgressMarker(timestamp=_now_minus(600), step="00_container_started"),
            ProgressMarker(timestamp=_now_minus(60), step="90_cleanup_exit_1"),
        ],
        artifacts=[],
    )
    s = build_summary(
        run_id, repo_root=tmp_path, case="case1", progress_module=progress
    )
    assert s.status == "failed"
    assert s.last_step == "90_cleanup_exit_1"
    assert s.dvc_meta_committed is False
    # 全 artifact 不在
    assert all(not a.in_s3_artifacts for a in s.artifacts)
    assert all(not a.in_run_dir for a in s.artifacts)


def test_summary_running_no_99_no_cleanup(tmp_path: Path) -> None:
    run_id = "run-progress"
    _make_run_dir(tmp_path, run_id, write_run_json=None)
    progress = _StubProgress(
        markers=[
            ProgressMarker(timestamp=_now_minus(600), step="00_container_started"),
            ProgressMarker(timestamp=_now_minus(120), step="60_before_train"),
        ],
        artifacts=[],
    )
    s = build_summary(
        run_id, repo_root=tmp_path, case="case1", progress_module=progress
    )
    assert s.status == "running"
    assert s.last_step == "60_before_train"
    assert s.last_step_age_seconds is not None
    assert 100 <= s.last_step_age_seconds <= 200


def test_summary_no_markers_no_run_json(tmp_path: Path) -> None:
    run_id = "ghost"
    _make_run_dir(tmp_path, run_id, write_run_json=None)
    progress = _StubProgress(markers=[], artifacts=[])
    s = build_summary(
        run_id, repo_root=tmp_path, case="case1", progress_module=progress
    )
    # markers 0 件は status="running" (失敗 marker も成功 marker もないため)
    assert s.status == "running"
    assert s.last_step is None
    assert s.epochs_completed is None
    assert s.estimated_cost_usd is None  # elapsed_seconds が None


def test_summary_artifact_split_local_only(tmp_path: Path) -> None:
    """ローカルにあって S3 には無い artifact のケース。"""
    run_id = "run-split"
    _make_run_dir(
        tmp_path,
        run_id,
        write_run_json=None,
        artifact_files=("best.pt", "run.json"),
    )
    progress = _StubProgress(markers=[], artifacts=["onstart.log"])
    s = build_summary(
        run_id, repo_root=tmp_path, case="case1", progress_module=progress
    )
    by_name = {a.name: a for a in s.artifacts}
    assert by_name["best.pt"].in_run_dir is True
    assert by_name["best.pt"].in_s3_artifacts is False
    assert by_name["onstart.log"].in_run_dir is False
    assert by_name["onstart.log"].in_s3_artifacts is True
