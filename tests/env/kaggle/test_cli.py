"""Integration tests for src/env/kaggle/cli.py via typer CliRunner."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from env.kaggle import cli, scraper


def _fake_result(**overrides: object) -> scraper.ScrapeResult:
    base = {
        "run_id": "rid",
        "teams_scanned": 0,
        "teams_without_submission": 0,
        "episodes_considered": 0,
        "episodes_skipped_existing": 0,
        "episodes_skipped_failed": 0,
        "episodes_skipped_mode": 0,
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


def test_scrape_rejects_unknown_mode(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
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
    result = runner.invoke(cli.app, ["list", "--data-root", str(tmp_path)])
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
    result = runner.invoke(cli.app, ["inspect", "42", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "episode_id: 42" in result.stdout
    assert "turns: 2" in result.stdout


def test_help_lists_three_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "scrape" in result.stdout
    assert "list" in result.stdout
    assert "inspect" in result.stdout
