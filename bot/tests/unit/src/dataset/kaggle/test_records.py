"""Unit tests for src/dataset/kaggle/records.py."""

from __future__ import annotations

from typing import Any

import pytest

from dataset.kaggle import records
from dataset.kaggle.types import EpisodeMeta


def _agent(
    *,
    submission_id: int,
    index: int = 0,
    reward: float | None = None,
    updated_score: float = 0.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": submission_id * 10,
        "submissionId": submission_id,
        "index": index,
        "updatedScore": updated_score,
    }
    if reward is not None:
        out["reward"] = reward
    return out


def test_infer_mode_1v1() -> None:
    assert records.infer_mode(2) == "1v1"


def test_infer_mode_ffa4() -> None:
    assert records.infer_mode(4) == "ffa4"


def test_infer_mode_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        records.infer_mode(3)


def test_is_failed_episode_detects_state() -> None:
    assert records.is_failed_episode({"state": "ERRORED"})
    assert records.is_failed_episode({"state": "INVALID"})
    assert not records.is_failed_episode({"state": "COMPLETED"})
    assert not records.is_failed_episode({})


def test_parse_episode_meta_from_response() -> None:
    raw = {
        "id": 42,
        "createTime": "2026-04-18T10:00:00+00:00",
        "endTime": "2026-04-18T10:00:05+00:00",
        "state": "COMPLETED",
        "agents": [
            _agent(submission_id=111, index=0),
            _agent(submission_id=222, index=1),
        ],
    }
    meta = records.parse_episode_meta(raw)
    assert meta.episode_id == 42
    assert meta.mode == "1v1"
    assert len(meta.agents) == 2


def test_build_match_record_1v1_winner_by_reward() -> None:
    meta = EpisodeMeta(
        episode_id=7,
        mode="1v1",
        create_time="2026-04-18T10:00:00+00:00",
        end_time="2026-04-18T10:00:10+00:00",
        agents=(
            _agent(submission_id=111, index=0, reward=1.0, updated_score=1490.0),
            _agent(submission_id=222, index=1, reward=-1.0, updated_score=1280.0),
        ),
    )
    record = records.build_match_record(
        meta,
        run_id="rid",
        scraped_at="2026-04-18T11:00:00+00:00",
        team_id_by_submission={111: 15637383, 222: 15634574},
    )

    assert record.match_id == "kaggle_ep_7"
    assert record.source == "kaggle"
    assert record.episode_id == 7
    assert record.mode == "1v1"
    assert record.winner == 0
    assert record.draw is False
    assert record.elapsed_sec == pytest.approx(10.0)
    assert record.agent_names == ("kaggle_sub_111", "kaggle_sub_222")
    assert record.agent_kaggle_meta[0].rating_mu == pytest.approx(1490.0)
    assert record.agent_kaggle_meta[0].team_id == 15637383
    assert (
        record.replay_uri
        == "s3://orbit-wars-dvc-286854171013/replays/kaggle/kaggle_ep_7.json.gz"
    )


def test_build_match_record_ffa4_draw_when_tied() -> None:
    meta = EpisodeMeta(
        episode_id=8,
        mode="ffa4",
        create_time="",
        end_time="",
        agents=tuple(
            _agent(submission_id=100 + i, index=i, reward=0.5) for i in range(4)
        ),
    )
    record = records.build_match_record(meta, run_id="rid", scraped_at="")
    assert record.winner == -1
    assert record.draw is True
    assert record.mode == "ffa4"
    assert len(record.agent_kaggle_meta) == 4


def test_build_match_record_respects_agent_index() -> None:
    meta = EpisodeMeta(
        episode_id=9,
        mode="1v1",
        create_time="",
        end_time="",
        agents=(
            _agent(submission_id=222, index=1, reward=1.0),
            _agent(submission_id=111, index=0, reward=-1.0),
        ),
    )
    record = records.build_match_record(meta, run_id="rid", scraped_at="")
    assert record.agent_names[0] == "kaggle_sub_111"
    assert record.winner == 1


def test_extract_submission_team_map() -> None:
    listing = {
        "submissions": [
            {"id": 1, "teamId": 100},
            {"id": 2, "teamId": 200},
        ]
    }
    assert records.extract_submission_team_map(listing) == {1: 100, 2: 200}


def test_extract_submission_team_map_handles_missing() -> None:
    assert records.extract_submission_team_map({}) == {}
    assert records.extract_submission_team_map({"submissions": None}) == {}
