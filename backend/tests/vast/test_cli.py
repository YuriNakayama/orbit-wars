"""cli.py の wiring + train/pull/promote/cost-report のフロー検証 (mock 中心)。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from vast.cli import _verify_commit_pushed, app
from vast.offers import Offer

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "train" in out
    assert "pull" in out
    assert "promote" in out
    assert "cost-report" in out


def test_verify_commit_pushed_unpushed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_verify_commit_pushed_remote_contains(monkeypatch: pytest.MonkeyPatch) -> None:
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
    from vast.auth import AwsCreds

    monkeypatch.setattr(
        "vast.cli.load_aws_creds",
        lambda profile=None: AwsCreds(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="ap-northeast-1",
        ),
    )
    monkeypatch.setattr("vast.cli.load_vast_api_key", lambda: "VAST_KEY_FAKE")


@pytest.fixture
def mock_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vast.cli._git_remote_url",
        lambda: "https://github.com/YuriNakayama/orbit-wars.git",
    )
    monkeypatch.setattr("vast.cli._git_current_branch", lambda: "feature-test")
    monkeypatch.setattr("vast.cli._verify_commit_pushed", lambda _sha: None)


@pytest.fixture
def mock_offer() -> Offer:
    return Offer(
        id=12345,
        gpu_name="RTX_3090",
        num_gpus=1,
        dph_total=0.13,
        reliability=0.995,
        geolocation="US",
        cuda_max_good=12.4,
        inet_down=500.0,
        verified=True,
    )


def test_train_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
    mock_offer: Offer,
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr("vast.cli._build_sdk", lambda _key: sdk)
    monkeypatch.setattr("vast.cli.search_offers", lambda _sdk: [mock_offer])
    monkeypatch.setattr("vast.cli.pick_offer", lambda offers, console=None: offers[0])
    monkeypatch.setattr(
        "vast.cli.create_instance",
        lambda *_a, **_k: 99999,
    )
    result = runner.invoke(
        app,
        ["train", "abc1234deadbeef", "--seed", "0"],
    )
    assert result.exit_code == 0, result.output
    assert "Instance launched" in result.output
    assert "99999" in result.output


def test_train_no_offers_exits(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr("vast.cli._build_sdk", lambda _key: sdk)
    monkeypatch.setattr("vast.cli.search_offers", lambda _sdk: [])
    result = runner.invoke(app, ["train", "abc1234deadbeef"])
    assert result.exit_code == 1
    assert "No offers" in result.output


def test_train_cost_limit_aborts_when_declined(
    monkeypatch: pytest.MonkeyPatch,
    mock_credentials: None,
    mock_git: None,
    mock_offer: Offer,
) -> None:
    expensive = Offer(
        id=12345,
        gpu_name="A100",
        num_gpus=1,
        dph_total=5.0,
        reliability=0.995,
        geolocation="US",
        cuda_max_good=12.4,
        inet_down=500.0,
        verified=True,
    )
    monkeypatch.setattr("vast.cli._build_sdk", lambda _key: MagicMock())
    monkeypatch.setattr("vast.cli.search_offers", lambda _sdk: [expensive])
    monkeypatch.setattr("vast.cli.pick_offer", lambda offers, console=None: offers[0])
    result = runner.invoke(
        app,
        ["train", "abc1234deadbeef", "--cost-limit", "0.5"],
        input="n\n",
    )
    assert result.exit_code == 1


def test_pull_runs_dvc_pull(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    runs_root = repo_root / "artifacts/models/imitation/case1/runs"
    run_dir = runs_root / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "test_run", "status": "pushed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("vast.cli._repo_root", lambda: repo_root)

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
    runs_root = repo_root / "artifacts/models/imitation/case1/runs"
    run_dir = runs_root / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "test_run", "status": "failed"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("vast.cli._repo_root", lambda: repo_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout=""),
    )
    result = runner.invoke(app, ["pull", "test_run"])
    assert result.exit_code == 0
    assert "warning" in result.output.lower()


def test_promote_copies_and_updates_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    runs_root_rel = Path("artifacts/models/imitation/case1/runs")
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
                "gpu_name": None,
                "vast_offer_snapshot": None,
                "command": "",
                "weights_path": "",
                "train_metrics": {},
                "local_eval_results": None,
                "status": "pushed",
                "created_at": "2026-04-25T00:00:00Z",
                "updated_at": "2026-04-25T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (repo_root / "backend/pipeline/imitation/case1/policy").mkdir(parents=True)

    monkeypatch.setattr("vast.cli._repo_root", lambda: repo_root)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout=""),
    )

    result = runner.invoke(
        app,
        [
            "promote",
            "test_run",
            "--runs-root",
            str(runs_root_rel),
            "--canonical",
            str(canonical_rel),
        ],
    )
    assert result.exit_code == 0, result.output

    canonical_out = repo_root / canonical_rel
    assert canonical_out.read_bytes() == b"FAKEMODEL"
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "adopted"


def test_promote_missing_best_pt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    monkeypatch.setattr("vast.cli._repo_root", lambda: repo_root)
    result = runner.invoke(app, ["promote", "missing_run"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cost_report_writes_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path
    runs_rel = Path("artifacts/models/imitation/case1/runs")
    run_dir = repo_root / runs_rel / "r-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "r-1",
                "git_sha": "abc",
                "gpu_name": "RTX_3090",
                "vast_offer_snapshot": {"dph_total": 0.13},
                "train_metrics": {"runtime_seconds": 600.0},
                "status": "pushed",
                "created_at": "2026-04-25T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("vast.cli._repo_root", lambda: repo_root)
    output_rel = Path("docs/experiment")

    result = runner.invoke(
        app,
        [
            "cost-report",
            "--month",
            "2026-04",
            "--runs-root",
            str(runs_rel),
            "--output-dir",
            str(output_rel),
        ],
    )
    assert result.exit_code == 0, result.output
    out_path = repo_root / output_rel / "vast_cost_report_2026-04.md"
    assert out_path.is_file()
    md = out_path.read_text(encoding="utf-8")
    assert "r-1" in md
    assert "RTX_3090" in md
