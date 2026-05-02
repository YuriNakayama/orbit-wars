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
