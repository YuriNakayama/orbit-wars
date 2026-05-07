"""Unit tests for src/dataset/kaggle/scraper.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from dataset.kaggle import scraper
from dataset.kaggle.rate_limit import TokenBucket
from dataset.kaggle.types import ScrapeSpec, TeamRank


def _team(team_id: int, *, rank: int = 1) -> TeamRank:
    return TeamRank(
        rank=rank,
        team_id=team_id,
        team_name=f"t{team_id}",
        score=900.0 - rank,
        submission_date="",
    )


def _agent(
    submission_id: int,
    *,
    index: int = 0,
    reward: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": submission_id * 10,
        "submissionId": submission_id,
        "index": index,
        "reward": reward,
        "updatedScore": 800.0,
    }


def _episode(
    episode_id: int,
    *,
    agent_count: int = 2,
    state: str = "COMPLETED",
) -> dict[str, Any]:
    agents = [
        _agent(submission_id=100 + i, index=i, reward=1.0 if i == 0 else -1.0)
        for i in range(agent_count)
    ]
    return {
        "id": episode_id,
        "createTime": "2026-04-18T10:00:00+00:00",
        "endTime": "2026-04-18T10:00:05+00:00",
        "state": state,
        "agents": agents,
    }


def _listing(
    episodes: list[dict[str, Any]],
    *,
    submissions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "submissions": submissions or [],
    }


def _team_response(sid: int = 500) -> dict[str, Any]:
    return {"id": 1, "publicLeaderboardSubmissionId": sid}


def _fast_bucket() -> TokenBucket:
    return TokenBucket(
        capacity=1000,
        window_sec=60.0,
        clock=lambda: 0.0,
        sleeper=lambda _s: None,
    )


def _make_spec(
    tmp_path: Path,
    *,
    modes: tuple[str, ...] = ("1v1", "ffa4"),
    top: int = 1,
    limit_per_team: int | None = None,
    dry_run: bool = False,
    include_failed: bool = False,
) -> ScrapeSpec:
    return ScrapeSpec(
        top=top,
        modes=modes,
        limit_per_team=limit_per_team,
        data_root=tmp_path,
        dry_run=dry_run,
        include_failed=include_failed,
    )


def _replay_response(episode_id: int) -> dict[str, Any]:
    return {"configuration": {}, "steps": [[{"reward": episode_id}]]}


def test_run_records_carry_team_rank(tmp_path: Path) -> None:
    """leaderboard_fetcher の rank が MatchRecord.agent_kaggle_meta.team_rank に伝わる。"""

    from dataset.storage import loader

    spec = _make_spec(tmp_path)
    submissions = [{"id": 100, "teamId": 999}, {"id": 101, "teamId": 999}]
    listing = _listing([_episode(1)], submissions=submissions)
    scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(999, rank=42)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: listing,
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
    )
    df = loader.list_matches(data_root=tmp_path, limit=10)
    assert df.height == 1
    assert int(df.row(0, named=True)["agent_0_team_rank"]) == 42


def test_run_writes_records_and_replays(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    episodes = [_episode(1), _episode(2, agent_count=4)]

    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
    )

    assert result.episodes_fetched == 2
    assert result.records_written == 2
    assert result.replays_written == 2
    replays_dir = tmp_path / "matches" / "replays"
    assert (replays_dir / "kaggle_ep_1.json.gz").exists()
    assert (replays_dir / "kaggle_ep_2.json.gz").exists()


def test_run_dry_run_skips_disk_writes(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, dry_run=True)
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
    )

    assert result.dry_run is True
    assert result.episodes_fetched == 1
    assert result.records_written == 0
    assert result.replays_written == 0
    assert not (tmp_path / "matches").exists()


def test_run_skips_failed_by_default(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    episodes = [_episode(1, state="ERRORED"), _episode(2)]
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_skipped_failed == 1
    assert result.episodes_fetched == 1


def test_run_include_failed_fetches_all(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, include_failed=True)
    episodes = [_episode(1, state="ERRORED"), _episode(2)]
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_skipped_failed == 0
    assert result.episodes_fetched == 2


def test_run_filters_by_mode(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, modes=("1v1",))
    episodes = [_episode(1), _episode(2, agent_count=4)]
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_fetched == 1
    assert result.episodes_skipped_mode == 1


def test_run_limits_per_team(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, limit_per_team=1)
    episodes = [_episode(1), _episode(2), _episode(3)]
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_fetched == 1


def test_run_skips_existing_episode_ids(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )

    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1), _episode(2)]),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_skipped_existing == 1
    assert result.episodes_fetched == 1


def test_run_counts_teams_without_submission(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: {"id": 1},  # no publicLeaderboardSubmissionId
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.teams_scanned == 1
    assert result.teams_without_submission == 1
    assert result.episodes_fetched == 0


def test_run_rejects_non_positive_top(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, top=0)
    with pytest.raises(ValueError, match="top"):
        scraper.run(spec)


def test_run_rejects_empty_modes(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path, modes=())
    with pytest.raises(ValueError, match="modes"):
        scraper.run(spec)


def test_run_reports_episodes_planned(tmp_path: Path) -> None:
    """Phase1 で確定した取得予定数が ScrapeResult.episodes_planned に出る。"""

    spec = _make_spec(tmp_path)
    episodes = [_episode(1), _episode(2, state="ERRORED"), _episode(3)]
    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
    )
    assert result.episodes_planned == 2
    assert result.episodes_fetched == 2
    assert result.episodes_skipped_failed == 1


def test_run_logs_phase_progress(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """plan/fetch phase の進捗ログが INFO で出る。"""

    spec = _make_spec(tmp_path)
    episodes = [_episode(eid) for eid in range(1, 4)]
    with caplog.at_level("INFO", logger="dataset.kaggle.scraper"):
        scraper.run(
            spec,
            session=MagicMock(),
            rate_limit=_fast_bucket(),
            leaderboard_fetcher=lambda _top: [_team(10)],
            team_fetcher=lambda _s, _t: _team_response(),
            episodes_lister=lambda _s, _sid: _listing(episodes),
            replay_fetcher=lambda _s, eid: _replay_response(eid),
            progress_every=1,
        )
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("phase=plan" in m and "planned=" in m for m in msgs)
    assert any("phase=fetch total=3" in m for m in msgs)
    assert any("phase=fetch [3/3]" in m for m in msgs)


def test_plan_only_emits_plan_without_fetching(tmp_path: Path) -> None:
    """`plan_only` は replay を呼ばず planned を ScrapeResult に反映する。"""

    spec = _make_spec(tmp_path)
    fetcher_calls: list[int] = []

    def _replay_fetcher(_s: Any, eid: int) -> dict[str, Any]:
        fetcher_calls.append(eid)
        return _replay_response(eid)

    result, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1), _episode(2)]),
    )
    assert len(plan) == 2
    assert result.episodes_planned == 2
    assert result.episodes_fetched == 0
    assert fetcher_calls == []


def test_serialize_round_trip_preserves_plan(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
    )
    payload = scraper.serialize_plan(plan)
    restored = scraper.deserialize_plan(payload)
    assert [p.episode_id for p in restored] == [p.episode_id for p in plan]


def test_fetch_with_plan_persists_records(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1), _episode(2)]),
    )
    result = scraper.fetch_with_plan(
        spec,
        plan,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
    )
    assert result.episodes_fetched == 2
    assert result.records_written == 2
    assert result.episodes_planned == 2


def test_fetch_with_plan_runs_checkpoint_callback(tmp_path: Path) -> None:
    """checkpoint_every 件取得ごとに on_checkpoint(idx, total) が呼ばれる。"""

    spec = _make_spec(tmp_path)
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(
            [_episode(eid) for eid in range(1, 11)]
        ),
    )
    captured: list[tuple[int, int]] = []

    def cb(idx: int, total: int) -> None:
        captured.append((idx, total))

    scraper.fetch_with_plan(
        spec,
        plan,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
        flush_every=2,
        checkpoint_every=4,
        on_checkpoint=cb,
    )
    # 10 episodes / checkpoint_every=4 → idx=4 と idx=8 の 2 回
    assert captured == [(4, 10), (8, 10)]


def test_fetch_with_plan_parallel_workers_preserve_results(tmp_path: Path) -> None:
    """workers=4 でも 1 episode 1 record 1 replay を取りこぼさず永続化する。"""

    spec = _make_spec(tmp_path)
    episodes = [_episode(eid) for eid in range(1, 21)]
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(episodes),
    )
    result = scraper.fetch_with_plan(
        spec,
        plan,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        replay_fetcher=lambda _s, eid: _replay_response(eid),
        run_id="rid",
        progress_every=10,
        flush_every=5,
        workers=4,
    )
    assert result.episodes_planned == 20
    assert result.episodes_fetched == 20
    assert result.records_written == 20
    assert result.replays_written == 20


def test_fetch_with_plan_periodic_flush_persists_partial(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """flush_every で永続化が小刻みに走り、cumulative ログが出る。"""

    spec = _make_spec(tmp_path)
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing(
            [_episode(eid) for eid in (1, 2, 3, 4, 5)]
        ),
    )
    with caplog.at_level("INFO", logger="dataset.kaggle.scraper"):
        result = scraper.fetch_with_plan(
            spec,
            plan,
            session=MagicMock(),
            rate_limit=_fast_bucket(),
            replay_fetcher=lambda _s, eid: _replay_response(eid),
            run_id="rid",
            progress_every=10,
            flush_every=2,
        )
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("phase=fetch flush" in m for m in msgs)
    assert result.records_written == 5
    assert result.replays_written == 5


def test_fetch_logs_eta_and_percent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec = _make_spec(tmp_path)
    _, plan = scraper.plan_only(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(eid) for eid in (1, 2)]),
    )
    with caplog.at_level("INFO", logger="dataset.kaggle.scraper"):
        scraper.fetch_with_plan(
            spec,
            plan,
            session=MagicMock(),
            rate_limit=_fast_bucket(),
            replay_fetcher=lambda _s, eid: _replay_response(eid),
            run_id="rid",
            progress_every=1,
        )
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("phase=fetch [1/2]" in m and "%" in m and "eta=" in m for m in msgs)


def test_run_no_plan_no_fetch(tmp_path: Path) -> None:
    """既取得 episode のみの場合 fetch が走らずも fetched=0。"""

    spec = _make_spec(tmp_path)
    fetcher_calls: list[int] = []

    def _replay_fetcher(_s: Any, eid: int) -> dict[str, Any]:
        fetcher_calls.append(eid)
        return _replay_response(eid)

    scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
        replay_fetcher=_replay_fetcher,
    )
    fetcher_calls.clear()

    result = scraper.run(
        spec,
        session=MagicMock(),
        rate_limit=_fast_bucket(),
        leaderboard_fetcher=lambda _top: [_team(10)],
        team_fetcher=lambda _s, _t: _team_response(),
        episodes_lister=lambda _s, _sid: _listing([_episode(1)]),
        replay_fetcher=_replay_fetcher,
    )
    assert result.episodes_planned == 0
    assert result.episodes_fetched == 0
    assert fetcher_calls == []
