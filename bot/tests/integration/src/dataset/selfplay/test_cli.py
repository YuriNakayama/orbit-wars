"""Self-play CLI integration tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dataset.cli import app
from dataset.storage import loader


def test_run_cmd_with_random_agents_succeeds(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--agents",
            "random,random",
            "--mode",
            "1v1",
            "-n",
            "1",
            "--parallel",
            "1",
            "--no-save-replay",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert loader.list_matches(data_root=tmp_path, mode="1v1", limit=10).height == 1
