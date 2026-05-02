"""runpod_io.cli の wiring + 各サブコマンドのフロー検証 (mock 中心)。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from runpod_io.cli import _verify_commit_pushed, app
from runpod_io.offers import Offer

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "train" in out
    assert "pull" in out
    assert "promote" in out
    assert "cost-report" in out
    assert "volume" in out
    assert "ps" in out
    assert "status" in out
    assert "logs" in out


def test_volume_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["volume", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "list" in out
    assert "search" in out
    assert "create" in out


def test_verify_commit_pushed_unpushed_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "cat-file"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        if cmd[:2] == ["git", "branch"] and "--contains" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    import typer

    with pytest.raises(typer.BadParameter, match="not pushed"):
        _verify_commit_pushed("abc1234deadbeef")


def test_verify_commit_pushed_remote_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "cat-file"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        if cmd[:2] == ["git", "branch"] and "--contains" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="  origin/main\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _verify_commit_pushed("abc1234deadbeef")


@pytest.fixture
def mock_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from runpod_io.auth import AwsCreds

    monkeypatch.setattr(
        "runpod_io.cli.load_aws_creds",
        lambda profile=None: AwsCreds(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="ap-northeast-1",
        ),
    )
    monkeypatch.setattr("runpod_io.cli.load_runpod_api_key", lambda: "RUNPOD_KEY_FAKE")


@pytest.fixture
def mock_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "runpod_io.cli._git_remote_url",
        lambda: "https://github.com/YuriNakayama/orbit-wars.git",
    )
    monkeypatch.setattr("runpod_io.cli._git_current_branch", lambda: "feature-test")
    monkeypatch.setattr("runpod_io.cli._verify_commit_pushed", lambda _sha: None)


@pytest.fixture
def mock_offer() -> Offer:
    return Offer(
        gpu_type_id="NVIDIA GeForce RTX 3090",
        display_name="RTX 3090",
        memory_gb=24,
        secure_cloud=True,
        community_cloud=True,
        secure_price=0.5,
        community_price=0.3,
        secure_spot_price=0.2,
        community_spot_price=0.1,
        cloud_type="SECURE",
        dph_total=0.5,
    )


def test_train_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
    mock_offer: Offer,
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr("runpod_io.cli._build_sdk", lambda _key: sdk)
    monkeypatch.setattr("runpod_io.cli._volume_sdk", lambda: MagicMock())
    monkeypatch.setattr("runpod_io.cli.list_volumes", lambda _sdk: [])
    monkeypatch.setattr(
        "runpod_io.cli.search_offers", lambda _sdk, **_kwargs: [mock_offer]
    )
    monkeypatch.setattr(
        "runpod_io.cli.pick_offer", lambda offers, console=None: offers[0]
    )
    monkeypatch.setattr(
        "runpod_io.cli.create_pod",
        lambda *_a, **_k: "pod-abc123",
    )
    result = runner.invoke(
        app,
        ["train", "abc1234deadbeef", "--seed", "0"],
    )
    assert result.exit_code == 0, result.output
    assert "Pod launched" in result.output
    assert "pod-abc123" in result.output


def test_train_no_offers_exits(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr("runpod_io.cli._build_sdk", lambda _key: sdk)
    monkeypatch.setattr("runpod_io.cli._volume_sdk", lambda: MagicMock())
    monkeypatch.setattr("runpod_io.cli.list_volumes", lambda _sdk: [])
    monkeypatch.setattr("runpod_io.cli.search_offers", lambda _sdk, **_kwargs: [])
    result = runner.invoke(app, ["train", "abc1234deadbeef"])
    assert result.exit_code == 1
    assert "No offers" in result.output


def test_train_cost_limit_aborts_when_declined(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
) -> None:
    expensive = Offer(
        gpu_type_id="NVIDIA A100 80GB PCIe",
        display_name="A100 80GB",
        memory_gb=80,
        secure_cloud=True,
        community_cloud=False,
        secure_price=5.0,
        community_price=None,
        secure_spot_price=None,
        community_spot_price=None,
        cloud_type="SECURE",
        dph_total=5.0,
    )
    monkeypatch.setattr("runpod_io.cli._build_sdk", lambda _key: MagicMock())
    monkeypatch.setattr("runpod_io.cli._volume_sdk", lambda: MagicMock())
    monkeypatch.setattr("runpod_io.cli.list_volumes", lambda _sdk: [])
    monkeypatch.setattr(
        "runpod_io.cli.search_offers", lambda _sdk, **_kwargs: [expensive]
    )
    monkeypatch.setattr(
        "runpod_io.cli.pick_offer", lambda offers, console=None: offers[0]
    )
    result = runner.invoke(
        app,
        ["train", "abc1234deadbeef", "--cost-limit", "0.5"],
        input="n\n",
    )
    assert result.exit_code == 1


def test_pull_runs_dvc_pull(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    runs_root = repo_root / "data/output/models/imitation/case1/runs"
    run_dir = runs_root / "test_run"
    run_dir.mkdir(parents=True)
    (runs_root / "test_run.dvc").write_text("outs:\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "test_run", "status": "pushed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(app, ["pull", "test_run"])
    assert result.exit_code == 0, result.output
    assert any("dvc" in c and "pull" in c for c in captured)


def test_pull_warns_on_failed_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    runs_root = repo_root / "data/output/models/imitation/case1/runs"
    run_dir = runs_root / "test_run"
    run_dir.mkdir(parents=True)
    (runs_root / "test_run.dvc").write_text("outs:\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "test_run", "status": "failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout=""),
    )
    result = runner.invoke(app, ["pull", "test_run"])
    assert result.exit_code == 0
    assert "warning" in result.output.lower()


def test_pull_missing_dvc_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    result = runner.invoke(app, ["pull", "missing_run"])
    assert result.exit_code == 1
    assert "missing" in result.output.lower()


def test_promote_copies_and_updates_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    runs_root_rel = Path("data/output/models/imitation/case1/runs")
    canonical_rel = Path("backend/pipeline/imitation/case1/policy/weights.pt")
    run_dir = repo_root / runs_root_rel / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"FAKEMODEL")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "test_run",
                "git_sha": "abc1234",
                "git_branch": "main",
                "params_hash": "0123456789ab",
                "seed": 0,
                "vast_instance_id": None,
                "runpod_pod_id": "pod-x",
                "gpu_name": None,
                "vast_offer_snapshot": None,
                "runpod_offer_snapshot": {"dph_total": 0.5},
                "command": "x",
                "weights_path": "x",
                "train_metrics": {},
                "local_eval_results": None,
                "status": "pushed",
                "created_at": "2026-05-02T00:00:00Z",
                "updated_at": "2026-05-02T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    result = runner.invoke(app, ["promote", "test_run"])
    assert result.exit_code == 0, result.output
    canonical_dst = repo_root / canonical_rel
    assert canonical_dst.is_file()
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "adopted"


def test_cost_report_writes_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    runs_root = repo_root / "data/output/models/imitation/case1/runs"
    runs_root.mkdir(parents=True)
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    result = runner.invoke(app, ["cost-report", "--month", "2026-05"])
    assert result.exit_code == 0
    out_path = repo_root / "docs/experiment/runpod_cost_report_2026-05.md"
    assert out_path.is_file()
    assert "RunPod Cost Report" in out_path.read_text(encoding="utf-8")


def test_volume_search_lists_dcs() -> None:
    result = runner.invoke(app, ["volume", "search"])
    assert result.exit_code == 0
    assert "US-KS-2" in result.output


def test_volume_search_no_match() -> None:
    result = runner.invoke(app, ["volume", "search", "--data-center-id", "MARS-1"])
    assert result.exit_code == 1
    assert "No matching" in result.output


def _write_launch(run_dir: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "run_id": "run42",
        "pod_id": "pod-abc",
        "commit_sha": "deadbeef",
        "branch": "feature-x",
        "case": "case1",
        "cloud_type": "SECURE",
        "gpu_type_id": "RTX 3090",
        "dph_total": 0.5,
        "data_center_id": None,
        "launched_at": "2026-05-02T00:00:00Z",
    }
    payload.update(overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launch.json").write_text(json.dumps(payload), encoding="utf-8")


def test_train_writes_launch_json(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
    mock_offer: Offer,
    tmp_path: Path,
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr("runpod_io.cli._build_sdk", lambda _key: sdk)
    monkeypatch.setattr("runpod_io.cli._volume_sdk", lambda: MagicMock())
    monkeypatch.setattr("runpod_io.cli.list_volumes", lambda _sdk: [])
    monkeypatch.setattr(
        "runpod_io.cli.search_offers", lambda _sdk, **_kwargs: [mock_offer]
    )
    monkeypatch.setattr(
        "runpod_io.cli.pick_offer", lambda offers, console=None: offers[0]
    )
    monkeypatch.setattr("runpod_io.cli.create_pod", lambda *_a, **_k: "pod-xyz")
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: tmp_path)

    result = runner.invoke(app, ["train", "abc1234deadbeef", "--seed", "0"])
    assert result.exit_code == 0, result.output

    runs = list((tmp_path / "data/output/models/imitation/case1/runs").iterdir())
    assert len(runs) == 1
    payload = json.loads((runs[0] / "launch.json").read_text(encoding="utf-8"))
    assert payload["pod_id"] == "pod-xyz"
    assert payload["case"] == "case1"


def test_ps_lists_active_pods(monkeypatch: pytest.MonkeyPatch) -> None:
    from runpod_io.pod_state import PodState

    monkeypatch.setattr("runpod_io.cli._ensure_runpod_key", lambda: None)
    monkeypatch.setattr(
        "runpod_io.pod_state.list_pods",
        lambda: [
            PodState(
                pod_id="pod-abc",
                name="my-run",
                desired_status="RUNNING",
                cost_per_hr=0.5,
                uptime_seconds=125,
                gpu_display_name="RTX 3090",
                image_name="x",
            ),
            PodState(
                pod_id="pod-old",
                name="old-run",
                desired_status="EXITED",
                cost_per_hr=0.5,
                uptime_seconds=10,
                gpu_display_name="RTX 3090",
                image_name="x",
            ),
        ],
    )

    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0, result.output
    assert "pod-abc" in result.output
    assert "pod-old" not in result.output  # active filter

    result_all = runner.invoke(app, ["ps", "--all"])
    assert "pod-old" in result_all.output


def test_ps_no_pods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runpod_io.cli._ensure_runpod_key", lambda: None)
    monkeypatch.setattr("runpod_io.pod_state.list_pods", lambda: [])
    result = runner.invoke(app, ["ps"])
    assert result.exit_code == 0
    assert "No active pods" in result.output


def test_status_full_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from runpod_io.pod_state import PodState
    from runpod_io.progress import ProgressMarker

    repo_root = tmp_path
    run_dir = repo_root / "data/output/models/imitation/case1/runs/run42"
    _write_launch(run_dir)
    (repo_root / "data/output/models/imitation/case1/runs/run42.dvc").write_text(
        "outs: []\n", encoding="utf-8"
    )
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr("runpod_io.cli._ensure_runpod_key", lambda: None)
    monkeypatch.setattr(
        "runpod_io.pod_state.get_pod",
        lambda _pid: PodState(
            pod_id="pod-abc",
            name="run42",
            desired_status="RUNNING",
            cost_per_hr=0.5,
            uptime_seconds=600,
            gpu_display_name="RTX 3090",
            image_name="x",
        ),
    )
    monkeypatch.setattr(
        "runpod_io.progress.list_markers",
        lambda _run, **_k: [
            ProgressMarker(
                timestamp="2026-05-02T00:00:00Z", step="00_container_started"
            ),
            ProgressMarker(timestamp="2026-05-02T00:10:00Z", step="60_before_train"),
        ],
    )

    result = runner.invoke(app, ["status", "run42"])
    assert result.exit_code == 0, result.output
    assert "pod-abc" in result.output
    assert "RUNNING" in result.output
    assert "60_before_train" in result.output
    assert "ready" in result.output  # dvc artifact present


def test_status_missing_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: tmp_path)
    result = runner.invoke(app, ["status", "missing_run"])
    assert result.exit_code == 1
    assert "run dir not found" in result.output


def test_status_progress_unavailable_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from runpod_io.pod_state import PodState

    repo_root = tmp_path
    run_dir = repo_root / "data/output/models/imitation/case1/runs/run42"
    _write_launch(run_dir)
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr("runpod_io.cli._ensure_runpod_key", lambda: None)
    monkeypatch.setattr(
        "runpod_io.pod_state.get_pod",
        lambda _pid: PodState(
            pod_id="pod-abc",
            name="run42",
            desired_status="RUNNING",
            cost_per_hr=0.5,
            uptime_seconds=10,
            gpu_display_name="RTX 3090",
            image_name="x",
        ),
    )

    def boom(*_a: object, **_k: object) -> list[object]:
        raise RuntimeError("no creds")

    monkeypatch.setattr("runpod_io.progress.list_markers", boom)
    result = runner.invoke(app, ["status", "run42"])
    assert result.exit_code == 0
    assert "progress markers unavailable" in result.output
    # dvc meta missing -> warning branch
    assert "not yet" in result.output


def test_logs_lists_s3_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """logs は S3 marker を timestamp 順に列挙する (runpodctl 不要)。"""
    from runpod_io.progress import ProgressMarker

    repo_root = tmp_path
    run_dir = repo_root / "data/output/models/imitation/case1/runs/run42"
    _write_launch(run_dir)
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)

    markers = [
        ProgressMarker(timestamp="2026-05-02T00:00:00Z", step="00_container_started"),
        ProgressMarker(timestamp="2026-05-02T00:01:00Z", step="10_before_clone"),
        ProgressMarker(timestamp="2026-05-02T00:05:00Z", step="40_uv_sync_done"),
    ]
    monkeypatch.setattr(
        "runpod_io.progress.list_markers",
        lambda run_id, profile=None: markers,
    )
    result = runner.invoke(app, ["logs", "run42"])
    assert result.exit_code == 0, result.output
    assert "00_container_started" in result.output
    assert "10_before_clone" in result.output
    assert "40_uv_sync_done" in result.output


def test_logs_grep_and_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from runpod_io.progress import ProgressMarker

    repo_root = tmp_path
    run_dir = repo_root / "data/output/models/imitation/case1/runs/run42"
    _write_launch(run_dir)
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)

    markers = [
        ProgressMarker(timestamp="2026-05-02T00:00:00Z", step="00_container_started"),
        ProgressMarker(timestamp="2026-05-02T00:01:00Z", step="10_before_clone"),
        ProgressMarker(timestamp="2026-05-02T00:02:00Z", step="20_clone_done"),
        ProgressMarker(timestamp="2026-05-02T00:05:00Z", step="40_uv_sync_done"),
    ]
    monkeypatch.setattr(
        "runpod_io.progress.list_markers",
        lambda run_id, profile=None: markers,
    )
    result = runner.invoke(app, ["logs", "run42", "--grep", "_done$", "--tail", "1"])
    assert result.exit_code == 0
    # tail=1 で `_done` を含む行のみ → 最後の 1 行
    assert "40_uv_sync_done" in result.output
    assert "00_container_started" not in result.output


def test_logs_no_markers_exits_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """marker が 0 件のときは exit 1 + actionable message。"""
    repo_root = tmp_path
    run_dir = repo_root / "data/output/models/imitation/case1/runs/run42"
    _write_launch(run_dir)
    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr(
        "runpod_io.progress.list_markers", lambda run_id, profile=None: []
    )
    result = runner.invoke(app, ["logs", "run42"])
    assert result.exit_code == 1
    assert "no S3 markers" in result.output


def test_logs_works_without_launch_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """launch.json 無しでも S3 marker は読める (旧 launched run 救済経路)。"""
    from runpod_io.progress import ProgressMarker

    monkeypatch.setattr("runpod_io.cli._repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "runpod_io.progress.list_markers",
        lambda run_id, profile=None: [
            ProgressMarker(
                timestamp="2026-05-02T00:00:00Z", step="00_container_started"
            )
        ],
    )
    result = runner.invoke(app, ["logs", "ghost_run"])
    assert result.exit_code == 0
    assert "00_container_started" in result.output
