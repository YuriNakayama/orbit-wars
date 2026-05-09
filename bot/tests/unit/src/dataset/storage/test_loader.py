"""Unit + integration tests for src/dataset/loader.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from dataset.schema import AgentTiming, MatchRecord
from dataset.storage import loader, recorder


def _record(match_id: str, mode: str = "1v1") -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        run_id="run",
        mode=mode,
        seed=0,
        started_at=f"2026-04-18T10:{match_id[-2:]}:00+00:00",
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
        replay_path=f"replays/{match_id}.json.gz",
        git_sha="",
    )


def test_list_matches_empty_returns_empty_dataframe(tmp_path: Path) -> None:
    df = loader.list_matches(tmp_path)
    assert df.is_empty()


def test_list_matches_orders_newest_first(tmp_path: Path) -> None:
    records = [_record("m01"), _record("m02"), _record("m03")]
    recorder.write_records(records, tmp_path)

    df = loader.list_matches(tmp_path, limit=2)
    assert df.height == 2
    assert df["match_id"].to_list()[0] == "m03"


def test_list_matches_filters_by_mode(tmp_path: Path) -> None:
    records = [_record("m01", mode="1v1"), _record("m02", mode="ffa4")]
    recorder.write_records(records, tmp_path)

    df = loader.list_matches(tmp_path, mode="1v1")
    assert df.height == 1
    assert df["mode"].to_list() == ["1v1"]


def test_load_replay_payload_decodes_gzip(tmp_path: Path) -> None:
    payload = {"steps": [[{"status": "DONE"}]], "name": "orbit_wars"}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    recorder.write_replay("mXYZ", raw, tmp_path)

    decoded = loader.load_replay_payload("mXYZ", data_root=tmp_path)
    assert decoded == payload


def test_load_replay_payload_rejects_non_dict(tmp_path: Path) -> None:
    raw = gzip.compress(b"[]")
    recorder.write_replay("m1", raw, tmp_path)

    with pytest.raises(ValueError, match="not a dict"):
        loader.load_replay_payload("m1", data_root=tmp_path)


def test_recorder_works_with_alternate_data_root(tmp_path: Path) -> None:
    kaggle_root = tmp_path / "kaggle_episodes"
    records = [_record("kaggle_ep_1"), _record("kaggle_ep_2")]
    written = recorder.write_records(records, kaggle_root)

    assert all(str(p).startswith(str(kaggle_root)) for p in written)
    df = loader.list_matches(kaggle_root)
    assert set(df["match_id"].to_list()) == {"kaggle_ep_1", "kaggle_ep_2"}


def test_load_kaggle_replay_roundtrip(tmp_path: Path) -> None:
    payload = {"name": "orbit_wars", "configuration": {}, "steps": []}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    kaggle_root = tmp_path / "kaggle_episodes"
    recorder.write_replay("kaggle_ep_42", raw, kaggle_root)

    decoded = loader.load_replay_payload("kaggle_ep_42", data_root=kaggle_root)
    assert decoded == payload
