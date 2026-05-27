"""cost.py の集計と markdown レンダのユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpu.vast.cost import (
    aggregate_runs,
    list_running_instances,
    parse_month,
    render_markdown,
)


def _write_run(
    runs_root: Path,
    run_id: str,
    *,
    created_at: str,
    dph: float,
    runtime: float,
    status: str = "pushed",
    gpu: str = "RTX_3090",
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "git_sha": "abc1234",
        "git_branch": "main",
        "params_hash": "0123456789ab",
        "seed": 0,
        "vast_instance_id": 1,
        "gpu_name": gpu,
        "vast_offer_snapshot": {"dph_total": dph},
        "command": "dvc repro",
        "weights_path": f"artifacts/.../runs/{run_id}/best.pt",
        "train_metrics": {"runtime_seconds": runtime},
        "local_eval_results": None,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_aggregate_runs_filters_by_month(tmp_path: Path) -> None:
    _write_run(
        tmp_path, "r-april", created_at="2026-04-25T14:30:22Z", dph=0.13, runtime=600.0
    )
    _write_run(
        tmp_path, "r-march", created_at="2026-03-10T08:00:00Z", dph=0.20, runtime=300.0
    )
    _write_run(
        tmp_path,
        "r-april-adopted",
        created_at="2026-04-12T10:00:00Z",
        dph=0.29,
        runtime=900.0,
        status="adopted",
    )

    report = aggregate_runs(tmp_path, month="2026-04")
    assert {r.run_id for r in report.runs} == {"r-april", "r-april-adopted"}
    # 2 runs * (dph * runtime / 3600)
    expected_total = round(0.13 * 600 / 3600 + 0.29 * 900 / 3600, 4)
    assert pytest.approx(report.total_cost_usd, abs=1e-4) == expected_total
    assert report.adopted_count == 1
    assert report.average_cost_usd == pytest.approx(expected_total / 2)


def test_aggregate_runs_no_month_takes_all(tmp_path: Path) -> None:
    _write_run(
        tmp_path, "r-a", created_at="2026-04-01T00:00:00Z", dph=0.1, runtime=600.0
    )
    _write_run(
        tmp_path, "r-b", created_at="2026-05-01T00:00:00Z", dph=0.2, runtime=600.0
    )
    report = aggregate_runs(tmp_path)
    assert len(report.runs) == 2
    assert report.month == "all"


def test_aggregate_runs_skips_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad-run"
    bad.mkdir()
    (bad / "run.json").write_text("{not json", encoding="utf-8")
    _write_run(
        tmp_path, "good", created_at="2026-04-25T00:00:00Z", dph=0.1, runtime=60.0
    )
    report = aggregate_runs(tmp_path)
    assert len(report.runs) == 1
    assert report.runs[0].run_id == "good"


def test_aggregate_runs_missing_root_returns_empty(tmp_path: Path) -> None:
    report = aggregate_runs(tmp_path / "does_not_exist")
    assert report.runs == ()


def test_render_markdown_no_runs(tmp_path: Path) -> None:
    report = aggregate_runs(tmp_path / "missing", month="2026-04")
    md = render_markdown(report)
    assert "Vast.ai Cost Report — 2026-04" in md
    assert "(no runs)" in md


def test_render_markdown_includes_table(tmp_path: Path) -> None:
    _write_run(
        tmp_path, "r-1", created_at="2026-04-25T00:00:00Z", dph=0.13, runtime=600.0
    )
    report = aggregate_runs(tmp_path, month="2026-04")
    md = render_markdown(report)
    assert "| run_id |" in md
    assert "| r-1 |" in md
    assert "RTX_3090" in md


def test_parse_month_valid_and_invalid() -> None:
    assert parse_month("2026-04") == "2026-04"
    assert parse_month(None) is None
    with pytest.raises(ValueError):
        parse_month("2026-4")
    with pytest.raises(ValueError):
        parse_month("not-a-date")


def test_list_running_instances_handles_failure() -> None:
    sdk = MagicMock()
    sdk.show_instances.side_effect = RuntimeError("boom")
    assert list_running_instances(sdk) == []


def test_list_running_instances_success() -> None:
    sdk = MagicMock()
    sdk.show_instances.return_value = [{"id": 1}, {"id": 2}]
    assert list_running_instances(sdk) == [{"id": 1}, {"id": 2}]
