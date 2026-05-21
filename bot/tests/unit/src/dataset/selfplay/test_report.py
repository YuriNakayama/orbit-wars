"""Unit tests for src/dataset/report.py."""

from __future__ import annotations

from rich.console import Console

from dataset.schema import AgentTiming, MatchRecord
from dataset.selfplay import report


def _record(
    match_id: str,
    agents: tuple[str, str],
    winner: int,
    draw: bool = False,
    timeouts: tuple[int, int] = (0, 0),
    p95: tuple[float, float] = (0.01, 0.01),
    turns: int = 100,
) -> MatchRecord:
    timings = tuple(
        AgentTiming(timeouts=timeouts[i], p50=0.0, p95=p95[i], max=0.0)
        for i in range(2)
    )
    return MatchRecord(
        match_id=match_id,
        run_id="run",
        mode="1v1",
        seed=0,
        started_at="2026-04-18T10:00:00+00:00",
        elapsed_sec=1.0,
        turns=turns,
        winner=winner,
        draw=draw,
        agent_names=agents,
        agent_versions=("v", "v"),
        agent_scores=(0, 0),
        agent_timings=timings,
        replay_uri="",
        git_sha="",
    )


def test_aggregate_computes_win_rate() -> None:
    records = [
        _record("m1", ("alpha", "beta"), winner=0),
        _record("m2", ("alpha", "beta"), winner=0),
        _record("m3", ("alpha", "beta"), winner=1),
        _record("m4", ("alpha", "beta"), winner=-1, draw=True),
    ]
    stats = {s.name: s for s in report.aggregate(records)}
    assert stats["alpha"].wins == 2
    assert stats["alpha"].games == 4
    assert stats["alpha"].draws == 1
    assert stats["alpha"].win_rate == 0.5
    assert stats["beta"].wins == 1


def test_aggregate_tracks_timeouts_and_p95() -> None:
    records = [
        _record("m1", ("alpha", "beta"), winner=0, timeouts=(1, 0), p95=(0.9, 0.1)),
        _record("m2", ("alpha", "beta"), winner=0, timeouts=(2, 1), p95=(0.5, 0.3)),
    ]
    stats = {s.name: s for s in report.aggregate(records)}
    assert stats["alpha"].timeouts == 3
    assert stats["alpha"].turn_p95_max == 0.9
    assert stats["beta"].timeouts == 1


def test_aggregate_empty_returns_empty_list() -> None:
    assert report.aggregate([]) == []


def test_build_table_has_expected_columns() -> None:
    records = [_record("m1", ("a", "b"), winner=0)]
    table = report.build_table(records, title="T")
    headers = [col.header for col in table.columns]
    assert headers == [
        "agent",
        "wins",
        "games",
        "win_rate",
        "draws",
        "avg_turns",
        "timeouts",
        "turn_p95",
    ]
    assert table.title == "T"


def test_summarize_prints_to_console() -> None:
    records = [_record("m1", ("a", "b"), winner=0)]
    console = Console(record=True, width=120)
    report.summarize(records, console=console)
    output = console.export_text()
    assert "Summary" in output
    assert "a" in output
