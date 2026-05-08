"""Integration tests for the kaggle sub-app in src/dataset/cli.py."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataset import cli
from dataset.kaggle import scraper


def _fake_result(**overrides: object) -> scraper.ScrapeResult:
    base = {
        "run_id": "rid",
        "teams_scanned": 0,
        "teams_without_submission": 0,
        "episodes_considered": 0,
        "episodes_skipped_existing": 0,
        "episodes_skipped_failed": 0,
        "episodes_skipped_mode": 0,
        "episodes_planned": 0,
        "episodes_fetched": 0,
        "episodes_failed": 0,
        "records_written": 0,
        "replays_written": 0,
        "dry_run": False,
    }
    base.update(overrides)
    return scraper.ScrapeResult(**base)  # type: ignore[arg-type]


def test_scrape_dry_run_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(spec: object) -> scraper.ScrapeResult:
        captured["spec"] = spec
        return _fake_result(dry_run=True, teams_scanned=1)

    monkeypatch.setattr(scraper, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "kaggle",
            "scrape",
            "--top",
            "1",
            "--dry-run",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Scrape summary" in result.stdout
    assert "dry_run" in result.stdout


def test_scrape_configures_logging_for_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`kaggle scrape` 起動時に root logger を INFO 以上に設定する。"""

    def fake_run(spec: object) -> scraper.ScrapeResult:
        return _fake_result(dry_run=True, teams_scanned=1)

    monkeypatch.setattr(scraper, "run", fake_run)
    import logging as _logging

    root = _logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", _logging.WARNING)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["kaggle", "scrape", "--top", "1", "--dry-run", "--data-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert root.level <= _logging.INFO


def test_scrape_rejects_unknown_mode(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "kaggle",
            "scrape",
            "--modes",
            "ffa8",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_list_empty_prints_no_matches(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["kaggle", "list", "--data-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "no matches found" in result.stdout


def test_inspect_reads_stored_replay(tmp_path: Path) -> None:
    payload = {
        "name": "orbit_wars",
        "steps": [
            [{"status": "ACTIVE", "reward": 0.0}, {"status": "ACTIVE", "reward": 0.0}],
            [{"status": "DONE", "reward": 10.0}, {"status": "DONE", "reward": -5.0}],
        ],
    }
    replays = tmp_path / "matches" / "replays"
    replays.mkdir(parents=True)
    (replays / "kaggle_ep_42.json.gz").write_bytes(
        gzip.compress(json.dumps(payload).encode("utf-8"))
    )

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["kaggle", "inspect", "42", "--data-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "episode_id: 42" in result.stdout
    assert "turns: 2" in result.stdout


def test_help_lists_three_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["kaggle", "--help"])
    assert result.exit_code == 0
    assert "scrape" in result.stdout
    assert "list" in result.stdout
    assert "inspect" in result.stdout


def test_scrape_dvc_add_runs_when_records_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dvc_root = tmp_path / "repo"
    (dvc_root / ".dvc").mkdir(parents=True)
    data_root = dvc_root / "data" / "kaggle"
    (data_root / "matches").mkdir(parents=True)
    monkeypatch.setattr(
        scraper, "run", lambda spec: _fake_result(records_written=3, replays_written=3)
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/dvc")
    captured: dict[str, object] = {}

    def fake_subprocess_run(
        cmd: list[str], check: bool, cwd: Path
    ) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["kaggle", "scrape", "--top", "1", "--dvc-add", "--data-root", str(data_root)],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["cmd"] == ["/usr/local/bin/dvc", "add", "data/kaggle/matches"]
    assert captured["cwd"] == dvc_root.resolve()


def test_scrape_dvc_add_fails_without_dvc_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "matches").mkdir()
    monkeypatch.setattr(
        scraper, "run", lambda spec: _fake_result(records_written=1, replays_written=1)
    )
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/dvc")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be invoked without .dvc/")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["kaggle", "scrape", "--top", "1", "--dvc-add", "--data-root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "no `.dvc/` directory found" in result.stdout


def test_scrape_dvc_add_skipped_when_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        scraper, "run", lambda spec: _fake_result(dry_run=True, records_written=0)
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be invoked in dry-run mode")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "kaggle",
            "scrape",
            "--top",
            "1",
            "--dry-run",
            "--dvc-add",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_scrape_fetch_runs_finalize_cmd_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """scrape-fetch が例外で落ちても --finalize-cmd は必ず実行される。"""

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "run_id": "rid",
                "spec": {
                    "top": 1,
                    "modes": ["1v1"],
                    "limit_per_team": None,
                    "data_root": str(tmp_path),
                    "include_failed": False,
                },
                "plan": [],
            }
        ),
        encoding="utf-8",
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(scraper, "fetch_with_plan", boom)
    captured: dict[str, object] = {}

    def fake_subprocess_run(
        cmd: object, *, check: bool, shell: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = cmd
        captured["outcome"] = env.get("FINALIZE_OUTCOME")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "kaggle",
            "scrape-fetch",
            "--plan-in",
            str(plan_file),
            "--data-root",
            str(tmp_path),
            "--finalize-cmd",
            "echo persist",
        ],
    )
    assert result.exit_code != 0  # propagated RuntimeError
    assert captured.get("cmd") == "echo persist"
    assert captured.get("outcome") == "failure"


def test_scrape_dvc_add_skipped_when_no_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scraper, "run", lambda spec: _fake_result(records_written=0))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "subprocess.run must not be invoked when no records written"
        )

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["kaggle", "scrape", "--top", "1", "--dvc-add", "--data-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
