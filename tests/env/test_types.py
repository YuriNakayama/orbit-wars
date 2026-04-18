"""Unit tests for src/env/types.py."""

from __future__ import annotations

import pytest

from env.types import AgentTiming, MatchRecord


def _sample_record(num_agents: int = 2) -> MatchRecord:
    timings = tuple(
        AgentTiming(timeouts=i, p50=0.01 * (i + 1), p95=0.05 * (i + 1), max=0.1)
        for i in range(num_agents)
    )
    return MatchRecord(
        match_id="m1",
        run_id="r1",
        mode="1v1" if num_agents == 2 else "ffa4",
        seed=42,
        started_at="2026-04-18T10:00:00+00:00",
        elapsed_sec=1.234,
        turns=120,
        winner=0,
        draw=False,
        agent_names=tuple(f"agent_{i}" for i in range(num_agents)),
        agent_versions=tuple("abc1234" for _ in range(num_agents)),
        agent_scores=tuple(10 + i for i in range(num_agents)),
        agent_timings=timings,
        replay_path="replays/m1.json.gz",
        git_sha="abc1234",
    )


def test_match_record_is_frozen() -> None:
    record = _sample_record()
    with pytest.raises(AttributeError):
        record.winner = 1  # type: ignore[misc]


def test_to_row_has_fixed_max_player_columns() -> None:
    record = _sample_record(num_agents=2)
    row = record.to_row()
    for idx in range(4):
        assert f"agent_{idx}_name" in row
        assert f"agent_{idx}_score" in row
        assert f"agent_{idx}_timeouts" in row
    assert row["agent_0_name"] == "agent_0"
    assert row["agent_1_name"] == "agent_1"
    assert row["agent_2_name"] == ""
    assert row["agent_3_name"] == ""
    assert row["agent_2_score"] == 0
    assert row["agent_3_timeouts"] == 0


def test_to_row_roundtrip_1v1() -> None:
    original = _sample_record(num_agents=2)
    row = original.to_row()
    restored = MatchRecord.from_row(row)
    assert restored == original


def test_to_row_roundtrip_ffa4() -> None:
    original = _sample_record(num_agents=4)
    row = original.to_row()
    restored = MatchRecord.from_row(row)
    assert restored == original


def test_to_row_contains_schema_version() -> None:
    record = _sample_record()
    row = record.to_row()
    assert row["schema_version"] == 2


def test_to_row_defaults_kaggle_fields_for_selfplay() -> None:
    row = _sample_record().to_row()
    assert row["source"] == "selfplay"
    assert row["episode_id"] == -1
    assert row["scraped_at"] == ""
    for idx in range(4):
        assert row[f"agent_{idx}_submission_id"] == 0
        assert row[f"agent_{idx}_team_id"] == 0
        assert row[f"agent_{idx}_rating_mu"] == 0.0
        assert row[f"agent_{idx}_state"] == ""


def test_to_row_roundtrip_with_kaggle_meta() -> None:
    from env.types import AgentKaggleMeta, MatchRecord

    metas = (
        AgentKaggleMeta(
            submission_id=111,
            team_id=999,
            rating_mu=820.5,
            rating_sigma=40.1,
            state="ACTIVE",
        ),
        AgentKaggleMeta(
            submission_id=222,
            team_id=888,
            rating_mu=700.0,
            rating_sigma=55.0,
            state="INACTIVE",
        ),
    )
    original = _sample_record(num_agents=2)
    kaggle_record = MatchRecord(
        **{
            **original.__dict__,
            "source": "kaggle",
            "episode_id": 42,
            "scraped_at": "2026-04-18T11:00:00+00:00",
            "agent_kaggle_meta": metas,
        }
    )
    row = kaggle_record.to_row()
    restored = MatchRecord.from_row(row)
    assert restored.source == "kaggle"
    assert restored.episode_id == 42
    assert restored.agent_kaggle_meta == metas
