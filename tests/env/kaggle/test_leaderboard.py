"""Unit tests for src/env/kaggle/leaderboard.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from env.kaggle import leaderboard

_CSV = """teamId,teamName,score,submissionDate
101,AlphaSquad,880.12,2026-04-18 10:00:00
202,BravoBots,860.05,2026-04-18 11:30:00
303,Charlie,840.00,2026-04-18 12:00:00
404,Delta,820.50,2026-04-18 13:45:00
505,Echo,810.10,2026-04-18 14:10:00
"""


def _make_proc(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["kaggle"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_fetch_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _make_proc(_CSV)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ranks = leaderboard.fetch(top=3)

    assert len(ranks) == 3
    assert ranks[0].rank == 1
    assert ranks[0].team_id == 101
    assert ranks[0].team_name == "AlphaSquad"
    assert ranks[0].score == pytest.approx(880.12)
    assert ranks[2].team_id == 303


def test_fetch_top_limits_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _make_proc(_CSV),
    )
    assert len(leaderboard.fetch(top=2)) == 2


def test_fetch_rejects_non_positive_top() -> None:
    with pytest.raises(ValueError, match="top"):
        leaderboard.fetch(top=0)


def test_fetch_raw_snapshot_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_proc(_CSV))
    target = tmp_path / "snapshots" / "lb.csv"

    ranks = leaderboard.fetch(top=2, raw_snapshot=target)

    assert len(ranks) == 2
    assert target.exists()
    assert target.read_text(encoding="utf-8") == _CSV


def test_fetch_cli_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_fnf(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("kaggle not found")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    with pytest.raises(leaderboard.KaggleLeaderboardError, match="見つかりません"):
        leaderboard.fetch(top=1)


def test_fetch_cli_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _make_proc("", returncode=2, stderr="auth failed"),
    )

    with pytest.raises(leaderboard.KaggleLeaderboardError, match="auth failed"):
        leaderboard.fetch(top=1)
