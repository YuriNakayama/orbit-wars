"""Unit tests for src/dataset/recorder.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from dataset.schema import AgentTiming, MatchRecord
from dataset.storage import recorder
from dataset.storage.paths import REPLAY_S3_BUCKET, REPLAY_S3_PREFIX


def _record(
    match_id: str,
    run_id: str = "run_a",
    mode: str = "1v1",
    num_agents: int = 2,
) -> MatchRecord:
    timings = tuple(AgentTiming(0, 0.01, 0.05, 0.1) for _ in range(num_agents))
    return MatchRecord(
        match_id=match_id,
        run_id=run_id,
        mode=mode,
        seed=0,
        started_at="2026-04-18T10:00:00+00:00",
        elapsed_sec=1.0,
        turns=100,
        winner=0,
        draw=False,
        agent_names=tuple(f"a{i}" for i in range(num_agents)),
        agent_versions=tuple("sha" for _ in range(num_agents)),
        agent_scores=tuple(10 + i for i in range(num_agents)),
        agent_timings=timings,
        replay_uri=f"replays/{match_id}.json.gz",
        git_sha="sha",
    )


def test_write_records_creates_hive_partition_dirs(tmp_path: Path) -> None:
    records = [
        _record("m1", mode="1v1", num_agents=2),
        _record("m2", mode="ffa4", num_agents=4),
    ]
    written = recorder.write_records(records, tmp_path)
    assert len(written) == 2

    assert (tmp_path / "matches/index.parquet/mode=1v1").is_dir()
    assert (tmp_path / "matches/index.parquet/mode=ffa4").is_dir()


def test_write_records_readable_with_hive_partitioning(tmp_path: Path) -> None:
    records = [
        _record("m1", mode="1v1"),
        _record("m2", mode="1v1"),
        _record("m3", mode="ffa4", num_agents=4),
    ]
    recorder.write_records(records, tmp_path)

    lf = pl.scan_parquet(
        tmp_path / "matches/index.parquet/**/*.parquet",
        hive_partitioning=True,
    )
    df = lf.collect()
    assert df.height == 3
    assert set(df["mode"].to_list()) == {"1v1", "ffa4"}


def test_write_records_suffixes_duplicate_run_id(tmp_path: Path) -> None:
    first = [_record("m1", run_id="run_x")]
    second = [_record("m2", run_id="run_x")]
    path_a = recorder.write_records(first, tmp_path)[0]
    path_b = recorder.write_records(second, tmp_path)[0]
    assert path_a != path_b
    assert path_a.name == "run_run_x.parquet"
    assert path_b.name == "run_run_x_1.parquet"


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Replace `recorder._s3()` with an in-memory dict-backed S3 stub."""

    store: dict[str, bytes] = {}

    class _FakeFS:
        def open(self, path: str, mode: str = "rb"):  # type: ignore[no-untyped-def]
            key = path.removeprefix("s3://")
            return _FakeFile(key, mode, store)

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

    monkeypatch.setattr(recorder, "_s3", lambda: _FakeFS())
    return store


def test_write_replay_uploads_to_s3(fake_s3: dict[str, bytes]) -> None:
    payload: dict[str, list[list[dict[str, dict[str, object]]]]] = {
        "steps": [[{"observation": {}}]]
    }
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    uri = recorder.write_replay("m42", raw, source="selfplay")

    expected_uri = f"s3://{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/selfplay/m42.json.gz"
    assert uri == expected_uri
    stored = fake_s3[f"{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/selfplay/m42.json.gz"]
    decoded = json.loads(gzip.decompress(stored).decode("utf-8"))
    assert decoded == payload


def test_write_run_writes_parquet_and_uploads_replays(
    tmp_path: Path, fake_s3: dict[str, bytes]
) -> None:
    records = [_record("m1"), _record("m2")]
    raw = gzip.compress(b'{"steps": []}')
    index_paths, replay_uris = recorder.write_run(
        records, {"m1": raw, "m2": raw}, tmp_path, source="selfplay"
    )
    assert len(index_paths) == 1
    assert len(replay_uris) == 2
    assert f"{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/selfplay/m1.json.gz" in fake_s3
    assert f"{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/selfplay/m2.json.gz" in fake_s3


def test_write_records_empty_returns_empty_list(tmp_path: Path) -> None:
    assert recorder.write_records([], tmp_path) == []
