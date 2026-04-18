"""Persist match records and replay bytes to local disk.

Layout:
  data_root/matches/index.parquet/mode={mode}/run_{run_id}[_N].parquet
  data_root/matches/replays/{match_id}.json.gz
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from env.types import MatchRecord

INDEX_DIRNAME = "index.parquet"
REPLAYS_DIRNAME = "replays"


def _partition_dir(data_root: Path, mode: str) -> Path:
    return data_root / "matches" / INDEX_DIRNAME / f"mode={mode}"


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


def write_replay(match_id: str, replay_bytes: bytes, data_root: Path) -> Path:
    replays_dir = data_root / "matches" / REPLAYS_DIRNAME
    replays_dir.mkdir(parents=True, exist_ok=True)
    target = replays_dir / f"{match_id}.json.gz"
    target.write_bytes(replay_bytes)
    return target


def write_run(
    records: list[MatchRecord],
    replay_bytes: dict[str, bytes],
    data_root: Path,
) -> list[Path]:
    """Write all records and replays for a completed run."""
    written = write_records(records, data_root)
    for match_id, payload in replay_bytes.items():
        written.append(write_replay(match_id, payload, data_root))
    return written
