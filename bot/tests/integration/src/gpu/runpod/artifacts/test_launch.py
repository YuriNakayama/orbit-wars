"""gpu.runpod.artifacts.launch の write/read ラウンドトリップ。"""

from __future__ import annotations

import json
from pathlib import Path

from gpu.runpod.artifacts.launch import (
    LAUNCH_FILENAME,
    find_run_dir,
    read_launch_json,
    write_launch_json,
)


def test_write_creates_run_dir_and_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "test_run"
    path = write_launch_json(
        run_dir,
        run_id="test_run",
        pod_id="pod-abc",
        commit_sha="deadbeef",
        branch="feature/x",
        case="case1",
        cloud_type="SECURE",
        gpu_type_id="NVIDIA RTX 3090",
        dph_total=0.5,
        data_center_id="US-KS-2",
    )

    assert path == run_dir / LAUNCH_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"
    assert payload["pod_id"] == "pod-abc"
    assert payload["case"] == "case1"
    assert payload["dph_total"] == 0.5
    assert payload["data_center_id"] == "US-KS-2"
    assert "launched_at" in payload


def test_read_round_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "rt_run"
    write_launch_json(
        run_dir,
        run_id="rt_run",
        pod_id="pod-xyz",
        commit_sha="abc1234",
        branch="main",
        case="case3",
        cloud_type="COMMUNITY",
        gpu_type_id="RTX 4090",
        dph_total=0.7,
    )

    payload = read_launch_json(run_dir)
    assert payload["pod_id"] == "pod-xyz"
    assert payload["case"] == "case3"


def test_find_run_dir_builds_expected_path(tmp_path: Path) -> None:
    path = find_run_dir(tmp_path, "myrun", "case1")
    assert path == tmp_path / "data/output/models/imitation/case1/runs/myrun"
