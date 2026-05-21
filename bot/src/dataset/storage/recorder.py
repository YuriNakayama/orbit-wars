"""Persist match records (index) locally and replay bytes to S3."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import s3fs

from dataset.schema import MatchRecord
from dataset.storage.paths import index_root, replay_uri

logger = logging.getLogger(__name__)


def _partition_dir(data_root: Path, mode: str) -> Path:
    return index_root(data_root) / f"mode={mode}"


def _unique_parquet_path(partition_dir: Path, run_id: str) -> Path:
    candidate = partition_dir / f"run_{run_id}.parquet"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alt = partition_dir / f"run_{run_id}_{suffix}.parquet"
        if not alt.exists():
            return alt
        suffix += 1


def write_records(records: list[MatchRecord], data_root: Path) -> list[Path]:
    """Write records as one parquet per (run_id, mode) partition."""
    if not records:
        return []

    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (record.run_id, record.mode)
        rows_by_key.setdefault(key, []).append(record.to_row())

    written: list[Path] = []
    for (run_id, mode), rows in rows_by_key.items():
        partition_dir = _partition_dir(data_root, mode)
        partition_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_parquet_path(partition_dir, run_id)
        pl.DataFrame(rows).write_parquet(target)
        written.append(target)
    return written


def _s3() -> s3fs.S3FileSystem:
    # Anonymous=False uses AWS_* env vars / shared credentials / IAM role.
    return s3fs.S3FileSystem()


def write_replay(match_id: str, replay_bytes: bytes, *, source: str) -> str:
    """Write a replay payload to S3 and return its s3:// URI.

    `source` must be `"kaggle"` or `"selfplay"`. The URI returned matches
    `paths.replay_uri(match_id, source)` and is what consumers should store
    in `MatchRecord.replay_uri`.
    """

    uri = replay_uri(match_id, source)
    fs = _s3()
    # s3fs writes via `with fs.open(...) as f: f.write(bytes)` — atomic per object.
    with fs.open(uri, "wb") as f:
        f.write(replay_bytes)
    logger.debug("wrote replay %s (%d bytes)", uri, len(replay_bytes))
    return uri


def write_run(
    records: list[MatchRecord],
    replay_bytes: dict[str, bytes],
    data_root: Path,
    *,
    source: str,
) -> tuple[list[Path], list[str]]:
    """Write all records and replays for a completed run.

    Returns (written_index_files, written_replay_uris).
    """

    written_index = write_records(records, data_root)
    written_uris: list[str] = [
        write_replay(match_id, payload, source=source)
        for match_id, payload in replay_bytes.items()
    ]
    return written_index, written_uris
