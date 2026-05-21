"""Read match index and replay payloads from S3."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import s3fs

from dataset.storage.paths import index_root, replay_uri

logger = logging.getLogger(__name__)


def list_matches(
    data_root: Path,
    mode: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Return recent match rows from the parquet index, newest first."""
    root = index_root(data_root)
    files = list(root.glob("**/*.parquet")) if root.exists() else []
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
    if mode is not None:
        lf = lf.filter(pl.col("mode") == mode)
    lf = lf.sort("started_at", descending=True)
    if limit is not None:
        lf = lf.limit(limit)
    return lf.collect()


def _s3() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem()


def load_replay_payload_from_uri(uri: str) -> dict[str, Any]:
    """Read and decode a replay payload directly from an s3:// URI."""

    fs = _s3()
    with fs.open(uri, "rb") as f:
        raw_bytes = f.read()
    raw = gzip.decompress(raw_bytes).decode("utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"replay payload is not a dict: {type(parsed).__name__}")
    return parsed


def load_replay_payload(match_id: str, *, source: str) -> dict[str, Any]:
    """Convenience wrapper that builds the URI from (match_id, source)."""
    return load_replay_payload_from_uri(replay_uri(match_id, source))


def load_replay(match_id: str, *, source: str) -> Any:
    """Reconstruct an Orbit Wars Environment from a stored replay."""
    from env.orbit_wars import make_orbit_wars_env

    payload = load_replay_payload(match_id, source=source)
    if payload.get("name", "orbit_wars") != "orbit_wars":
        raise ValueError("only orbit_wars replays are supported")
    return make_orbit_wars_env(
        configuration=payload.get("configuration", {}),
        steps=payload.get("steps", []),
    )


def load_kaggle_replay(episode_id: int) -> Any:
    """Load a Kaggle-sourced episode replay as an Environment."""
    return load_replay(f"kaggle_ep_{episode_id}", source="kaggle")
