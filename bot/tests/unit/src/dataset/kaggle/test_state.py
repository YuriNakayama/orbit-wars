"""Unit tests for src/dataset/kaggle/state.py."""

from __future__ import annotations

from pathlib import Path

from dataset.kaggle import state
from dataset.schema import AgentKaggleMeta, AgentTiming, MatchRecord
from dataset.storage import recorder


def _record(
    match_id: str,
    *,
    mode: str = "1v1",
    source: str = "kaggle",
    episode_id: int = -1,
) -> MatchRecord:
    metas: tuple[AgentKaggleMeta, ...] = (
        (
            AgentKaggleMeta(submission_id=1, team_id=10),
            AgentKaggleMeta(submission_id=2, team_id=20),
        )
        if source == "kaggle"
        else ()
    )
    return MatchRecord(
        match_id=match_id,
        run_id="r",
        mode=mode,
        seed=0,
        started_at="2026-04-18T10:00:00+00:00",
        elapsed_sec=1.0,
        turns=100,
        winner=0,
        draw=False,
        agent_names=("a", "b"),
        agent_versions=("v", "v"),
        agent_scores=(1, 0),
        agent_timings=(
            AgentTiming(0, 0.0, 0.0, 0.0),
            AgentTiming(0, 0.0, 0.0, 0.0),
        ),
        replay_uri=f"replays/{match_id}.json.gz",
        git_sha="",
        source=source,
        episode_id=episode_id,
        scraped_at="2026-04-18T11:00:00+00:00" if source == "kaggle" else "",
        agent_kaggle_meta=metas,
    )


def test_existing_ids_empty_directory(tmp_path: Path) -> None:
    assert state.existing_episode_ids(tmp_path) == set()


def test_existing_ids_collects_kaggle_only(tmp_path: Path) -> None:
    records = [
        _record("kaggle_ep_1", source="kaggle", episode_id=1),
        _record("kaggle_ep_2", source="kaggle", episode_id=2),
        _record("sp_1", source="selfplay", episode_id=-1),
    ]
    recorder.write_records(records, tmp_path)

    assert state.existing_episode_ids(tmp_path) == {1, 2}


def test_existing_ids_filters_by_mode(tmp_path: Path) -> None:
    records = [
        _record("kaggle_ep_1", mode="1v1", source="kaggle", episode_id=1),
        _record("kaggle_ep_2", mode="ffa4", source="kaggle", episode_id=2),
    ]
    recorder.write_records(records, tmp_path)

    assert state.existing_episode_ids(tmp_path, modes=("1v1",)) == {1}
    assert state.existing_episode_ids(tmp_path, modes=("ffa4",)) == {2}
    assert state.existing_episode_ids(tmp_path, modes=("1v1", "ffa4")) == {1, 2}
