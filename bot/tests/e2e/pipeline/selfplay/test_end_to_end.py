"""End-to-end CLI → recorder → loader → replay reconstruction."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataset.cli import app
from dataset.storage import loader


@pytest.mark.slow
def test_run_then_list_then_replay(tmp_path: Path) -> None:
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
            "2",
            "--parallel",
            "2",
            "--save-replay",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    df = loader.list_matches(tmp_path, mode="1v1")
    assert df.height == 2

    match_id = df["match_id"][0]
    env = loader.load_replay(match_id, data_root=tmp_path)
    rendered = env.render(mode="json")
    assert isinstance(rendered, (dict, str))
