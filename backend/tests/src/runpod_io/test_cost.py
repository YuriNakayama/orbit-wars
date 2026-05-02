"""runpod_io.cost のユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runpod_io.cost import (
    CostReport,
    aggregate_runs,
    parse_month,
    render_markdown,
)


def _write_run(
    runs_root: Path,
    run_id: str,
    *,
    runpod: bool,
    dph: float = 0.5,
    runtime: float = 600.0,
    status: str = "pushed",
    created_at: str = "2026-05-02T10:00:00Z",
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "git_sha": "abc1234",
        "git_branch": "main",
        "params_hash": "0123456789ab",
        "seed": 0,
        "vast_instance_id": None if runpod else 12345,
        "runpod_pod_id": "pod-xxx" if runpod else None,
        "gpu_name": "RTX 3090",
        "vast_offer_snapshot": None
        if runpod
        else {"dph_total": 0.13, "gpu_name": "RTX_3090"},
        "runpod_offer_snapshot": {
            "gpu_type_id": "NVIDIA GeForce RTX 3090",
            "display_name": "RTX 3090",
            "memory_gb": 24,
            "cloud_type": "SECURE",
            "secure_cloud": True,
            "community_cloud": True,
            "dph_total": dph,
        }
        if runpod
        else None,
        "command": "x",
        "weights_path": "x",
        "train_metrics": {"runtime_seconds": runtime},
        "local_eval_results": None,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
    }
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_runs_filters_vast_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-runpod-1", runpod=True, dph=0.5, runtime=600)
    _write_run(tmp_path, "run-runpod-2", runpod=True, dph=0.3, runtime=300)
    _write_run(tmp_path, "run-vast-1", runpod=False)
    report = aggregate_runs(tmp_path)
    assert len(report.runs) == 2
    assert {r.run_id for r in report.runs} == {"run-runpod-1", "run-runpod-2"}
    # 0.5 * 600/3600 + 0.3 * 300/3600 = 0.0833... + 0.025 = 0.1083...
    assert report.total_cost_usd == pytest.approx(0.1083, abs=1e-3)


def test_aggregate_runs_month_filter(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-april", runpod=True, created_at="2026-04-30T23:59:59Z")
    _write_run(tmp_path, "run-may-1", runpod=True, created_at="2026-05-01T00:00:00Z")
    _write_run(tmp_path, "run-may-2", runpod=True, created_at="2026-05-15T12:00:00Z")
    report = aggregate_runs(tmp_path, month="2026-05")
    assert len(report.runs) == 2
    assert all(r.created_at.startswith("2026-05") for r in report.runs)


def test_aggregate_runs_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "missing"
    report = aggregate_runs(empty)
    assert report.runs == ()


def test_aggregate_runs_adopted_count(tmp_path: Path) -> None:
    _write_run(tmp_path, "r1", runpod=True, status="adopted")
    _write_run(tmp_path, "r2", runpod=True, status="pushed")
    _write_run(tmp_path, "r3", runpod=True, status="adopted")
    report = aggregate_runs(tmp_path)
    assert report.adopted_count == 2


def test_render_markdown_with_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "r1", runpod=True, dph=0.5, runtime=600)
    report = aggregate_runs(tmp_path)
    md = render_markdown(report)
    assert "RunPod Cost Report" in md
    assert "r1" in md
    assert "SECURE" in md
    assert "$0.500" in md or "$0.50" in md


def test_render_markdown_empty() -> None:
    md = render_markdown(CostReport(month="2026-05", runs=()))
    assert "(no runs)" in md
    assert "RunPod" in md


def test_parse_month_valid() -> None:
    assert parse_month("2026-05") == "2026-05"
    assert parse_month(None) is None


def test_parse_month_invalid() -> None:
    for bad in ["2026-5", "26-05", "2026/05", "2026-13"]:
        with pytest.raises(ValueError):
            parse_month(bad)


def test_aggregate_runs_corrupt_json_skipped(tmp_path: Path) -> None:
    _write_run(tmp_path, "good", runpod=True)
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    (bad_dir / "run.json").write_text("not json", encoding="utf-8")
    report = aggregate_runs(tmp_path)
    assert len(report.runs) == 1
    assert report.runs[0].run_id == "good"
