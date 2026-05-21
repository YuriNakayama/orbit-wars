"""kaggle_kernel.artifacts.cost のユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_kernel.artifacts.cost import (
    aggregate_runs,
    default_report_path,
    render_markdown,
)


def _write_run_json(
    runs_root: Path,
    case: str,
    run_id: str,
    *,
    kaggle_kernel_meta: dict[str, object] | None,
    status: str = "pushed",
    created_at: str = "",
) -> Path:
    run_dir = runs_root / case / "runs" / run_id
    run_dir.mkdir(parents=True)
    data = {
        "schema_version": 1,
        "run_id": run_id,
        "git_sha": "abc1234deadbeef",
        "git_branch": "main",
        "params_hash": "0123456789ab",
        "seed": 0,
        "vast_instance_id": None,
        "runpod_pod_id": None,
        "gpu_name": None,
        "vast_offer_snapshot": None,
        "runpod_offer_snapshot": None,
        "kaggle_kernel_meta": kaggle_kernel_meta,
        "command": "x",
        "weights_path": "best.pt",
        "train_metrics": {},
        "local_eval_results": None,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def test_aggregate_runs_filters_by_month_and_kaggle_meta(tmp_path: Path) -> None:
    _write_run_json(
        tmp_path,
        "case1",
        "kk_may",
        kaggle_kernel_meta={
            "kernel_slug": "yuri/run-may",
            "accelerator": "gpu-t4x2",
            "runtime_seconds": 1800,
            "started_at": "2026-05-10T10:30:00Z",
        },
    )
    _write_run_json(
        tmp_path,
        "case1",
        "kk_april",
        kaggle_kernel_meta={
            "kernel_slug": "yuri/run-apr",
            "accelerator": "gpu-t4x2",
            "runtime_seconds": 3600,
            "started_at": "2026-04-25T00:00:00Z",
        },
    )
    _write_run_json(
        tmp_path,
        "case1",
        "vast_run",
        kaggle_kernel_meta=None,
        created_at="2026-05-15T00:00:00Z",
    )
    report = aggregate_runs(tmp_path, "2026-05")
    assert len(report.rows) == 1
    assert report.rows[0].run_id == "kk_may"
    assert report.total_runtime_seconds == 1800
    assert report.total_gpu_hours == pytest.approx(0.5)


def test_aggregate_runs_invalid_month_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid month"):
        aggregate_runs(tmp_path, "2026/05")


def test_render_markdown_contains_summary_and_rows(tmp_path: Path) -> None:
    _write_run_json(
        tmp_path,
        "case1",
        "r0",
        kaggle_kernel_meta={
            "kernel_slug": "yuri/r0",
            "accelerator": "gpu-t4x2",
            "runtime_seconds": 600,
            "started_at": "2026-05-01T00:00:00Z",
        },
    )
    report = aggregate_runs(tmp_path, "2026-05")
    md = render_markdown(report)
    assert "Total runs: **1**" in md
    assert "Total GPU hours used: **0.17**" in md
    assert "| r0 |" in md
    assert "| gpu-t4x2 |" in md


def test_default_report_path() -> None:
    p = default_report_path(Path("docs/experiment"), "2026-05")
    assert p == Path("docs/experiment/kaggle_kernel_cost_report_2026-05.md")


def test_aggregate_skips_unparseable_run_json(tmp_path: Path) -> None:
    case_dir = tmp_path / "case1" / "runs" / "bad"
    case_dir.mkdir(parents=True)
    (case_dir / "run.json").write_text("not-json{", encoding="utf-8")
    report = aggregate_runs(tmp_path, "2026-05")
    assert report.rows == []


def test_started_at_fallback_to_created_at(tmp_path: Path) -> None:
    _write_run_json(
        tmp_path,
        "case1",
        "fallback",
        kaggle_kernel_meta={
            "kernel_slug": "yuri/fb",
            "accelerator": "gpu-p100",
            "runtime_seconds": 100,
            # started_at 不在
        },
        created_at="2026-05-30T00:00:00Z",
    )
    report = aggregate_runs(tmp_path, "2026-05")
    assert len(report.rows) == 1
    assert report.rows[0].started_at == "2026-05-30T00:00:00Z"
