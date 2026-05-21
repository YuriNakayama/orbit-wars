"""Unit + integration tests for src/dataset/loader.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from dataset.schema import AgentTiming, MatchRecord
from dataset.storage import loader, recorder
from dataset.storage.paths import REPLAY_S3_BUCKET, REPLAY_S3_PREFIX, replay_uri


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
        replay_uri=replay_uri(match_id, "selfplay"),
        git_sha="",
    )


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Stub `recorder._s3()` and `loader._s3()` with a shared in-memory dict."""

    store: dict[str, bytes] = {}

    class _FakeFile:
        def __init__(self, key: str, mode: str, backing: dict[str, bytes]) -> None:
            self._key = key
            self._mode = mode
            self._backing = backing
            self._buf = bytearray()

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_exc: object) -> None:
            if "w" in self._mode:
                self._backing[self._key] = bytes(self._buf)

        def write(self, data: bytes) -> int:
            self._buf.extend(data)
            return len(data)

        def read(self) -> bytes:
            return self._backing[self._key]

    class _FakeFS:
        def open(self, path: str, mode: str = "rb"):  # type: ignore[no-untyped-def]
            key = path.removeprefix("s3://")
            return _FakeFile(key, mode, store)

    monkeypatch.setattr(recorder, "_s3", lambda: _FakeFS())
    monkeypatch.setattr(loader, "_s3", lambda: _FakeFS())
    return store


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


def test_load_replay_payload_decodes_gzip(fake_s3: dict[str, bytes]) -> None:
    payload = {"steps": [[{"status": "DONE"}]], "name": "orbit_wars"}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    recorder.write_replay("mXYZ", raw, source="selfplay")

    decoded = loader.load_replay_payload("mXYZ", source="selfplay")
    assert decoded == payload


def test_load_replay_payload_rejects_non_dict(fake_s3: dict[str, bytes]) -> None:
    raw = gzip.compress(b"[]")
    recorder.write_replay("m1", raw, source="selfplay")

    with pytest.raises(ValueError, match="not a dict"):
        loader.load_replay_payload("m1", source="selfplay")


def test_recorder_works_with_alternate_data_root(tmp_path: Path) -> None:
    kaggle_root = tmp_path / "kaggle_episodes"
    records = [_record("kaggle_ep_1"), _record("kaggle_ep_2")]
    written = recorder.write_records(records, kaggle_root)

    assert all(str(p).startswith(str(kaggle_root)) for p in written)
    df = loader.list_matches(kaggle_root)
    assert set(df["match_id"].to_list()) == {"kaggle_ep_1", "kaggle_ep_2"}


def test_load_kaggle_replay_roundtrip(fake_s3: dict[str, bytes]) -> None:
    payload = {"name": "orbit_wars", "configuration": {}, "steps": []}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    recorder.write_replay("kaggle_ep_42", raw, source="kaggle")

    decoded = loader.load_replay_payload("kaggle_ep_42", source="kaggle")
    assert decoded == payload
    # Ensure object is in the kaggle path, not selfplay.
    assert (
        f"{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/kaggle/kaggle_ep_42.json.gz" in fake_s3
    )
