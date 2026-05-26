"""Unit tests for src.evaluation.vs_baseline."""

from __future__ import annotations

from dataset.schema import MatchRecord
from evaluate.vs_baseline import summarize_records, wilson_ci


def _record(winner: int, *, draw: bool = False) -> MatchRecord:
    return MatchRecord(
        match_id="x",
        run_id="r",
        mode="1v1",
        seed=0,
        started_at="2026-04-29T00:00:00Z",
        elapsed_sec=1.0,
        turns=10,
        winner=winner,
        draw=draw,
        agent_names=("a", "b"),
        agent_versions=("v0", "v0"),
        agent_scores=(0, 0),
        agent_timings=tuple(),
        replay_path="",
        git_sha="abc",
    )


def test_wilson_ci_zero_total_returns_zero_pair() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_full_success_upper_bound_is_one() -> None:
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_ci_full_failure_lower_bound_is_zero() -> None:
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def test_wilson_ci_centred_for_equal_outcomes() -> None:
    lo, hi = wilson_ci(50, 100)
    midpoint = (lo + hi) / 2.0
    assert abs(midpoint - 0.5) < 0.05


def test_summarize_records_empty() -> None:
    out = summarize_records([], challenger_idx=0)
    assert out["episodes"] == 0.0
    assert out["wins"] == 0.0
    assert out["win_rate"] == 0.0


def test_summarize_records_clean_split() -> None:
    records = [_record(0), _record(0), _record(1), _record(-1, draw=True)]
    out = summarize_records(records, challenger_idx=0)
    assert out["episodes"] == 4.0
    assert out["wins"] == 2.0
    assert out["losses"] == 1.0
    assert out["draws"] == 1.0
    assert out["win_rate"] == 0.5
    assert out["non_draw_win_rate"] == 2.0 / 3.0


def test_summarize_records_all_wins() -> None:
    records = [_record(0) for _ in range(5)]
    out = summarize_records(records, challenger_idx=0)
    assert out["wins"] == 5.0
    assert out["win_rate"] == 1.0
    assert out["non_draw_win_rate"] == 1.0


def test_summarize_records_perspective_flip() -> None:
    records = [_record(0), _record(0), _record(1)]
    challenger_perspective = summarize_records(records, challenger_idx=0)
    opponent_perspective = summarize_records(records, challenger_idx=1)
    assert challenger_perspective["wins"] == 2.0
    assert opponent_perspective["wins"] == 1.0
