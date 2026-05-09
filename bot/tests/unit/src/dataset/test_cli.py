"""CLI tests using typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataset.cli import _parse_agents, app


def test_parse_agents_strips_whitespace() -> None:
    assert _parse_agents("a, b , c") == ("a", "b", "c")


def test_parse_agents_rejects_empty() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_agents("")


def test_list_cmd_empty_data_root_prints_no_matches(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--data-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "no matches" in result.stdout

