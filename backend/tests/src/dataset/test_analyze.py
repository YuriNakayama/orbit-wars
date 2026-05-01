"""Unit tests for src/dataset/analyze.py."""

from __future__ import annotations

from pathlib import Path

from dataset.schema import AgentTiming, MatchRecord
from dataset.storage import analyze, recorder


def _record(
    match_id: str,
    winner: int,
    agents: tuple[str, ...] = ("alpha", "beta"),
    mode: str = "1v1",
    draw: bool = False,
    p95s: tuple[float, ...] = (0.1, 0.1),
    timeouts: tuple[int, ...] = (0, 0),
) -> MatchRecord:
    n = len(agents)
    timings = tuple(
        AgentTiming(timeouts=timeouts[i], p50=0.05, p95=p95s[i], max=0.2)
        for i in range(n)
    )
    return MatchRecord(
        match_id=match_id,
        run_id="run",
        mode=mode,
        seed=0,
        started_at="2026-04-18T10:00:00+00:00",
        elapsed_sec=1.0,
        turns=100,
        winner=winner,
        draw=draw,
        agent_names=agents,
        agent_versions=tuple("v" for _ in range(n)),
        agent_scores=tuple(0 for _ in range(n)),
        agent_timings=timings,
        replay_path="",
        git_sha="",
    )


def test_scan_index_empty_returns_empty_lazyframe(tmp_path: Path) -> None:
    lf = analyze.scan_index(tmp_path)
    assert lf.collect().is_empty()


def test_agent_winrate_matches_hand_computed(tmp_path: Path) -> None:
    records = [
        _record("m1", winner=0),
        _record("m2", winner=0),
        _record("m3", winner=1),
        _record("m4", winner=-1, draw=True),
    ]
    recorder.write_records(records, tmp_path)

    lf = analyze.scan_index(tmp_path)
    df = analyze.agent_winrate(lf)
    rows = {(row["agent"], row["mode"]): row for row in df.to_dicts()}
    alpha = rows[("alpha", "1v1")]
    assert alpha["games"] == 4
    assert alpha["wins"] == 2
    assert alpha["win_rate"] == 0.5


def test_timing_distribution_aggregates_per_agent(tmp_path: Path) -> None:
    records = [
        _record("m1", winner=0, p95s=(0.3, 0.1), timeouts=(1, 0)),
        _record("m2", winner=0, p95s=(0.5, 0.2), timeouts=(2, 0)),
    ]
    recorder.write_records(records, tmp_path)

    lf = analyze.scan_index(tmp_path)
    df = analyze.timing_distribution(lf)
    rows = {row["agent"]: row for row in df.to_dicts()}
    assert rows["alpha"]["timeouts"] == 3
    assert rows["alpha"]["p95_mean"] > rows["beta"]["p95_mean"]


def test_mode_summary_counts_games(tmp_path: Path) -> None:
    records = [
        _record("m1", winner=0, mode="1v1"),
        _record("m2", winner=0, mode="1v1"),
        _record(
            "m3",
            winner=0,
            mode="ffa4",
            agents=("a", "b", "c", "d"),
            p95s=(0.1, 0.1, 0.1, 0.1),
            timeouts=(0, 0, 0, 0),
        ),
    ]
    recorder.write_records(records, tmp_path)

    lf = analyze.scan_index(tmp_path)
    df = analyze.mode_summary(lf)
    rows = {row["mode"]: row for row in df.to_dicts()}
    assert rows["1v1"]["games"] == 2
    assert rows["ffa4"]["games"] == 1
