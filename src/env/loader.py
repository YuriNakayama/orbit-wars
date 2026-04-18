"""Read match index and replay payloads."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl

INDEX_GLOB = "matches/index.parquet/**/*.parquet"
REPLAYS_DIR = "matches/replays"


def _index_glob(data_root: Path) -> Path:
    return data_root / INDEX_GLOB


def list_matches(
    data_root: Path,
    mode: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Return recent match rows from the parquet index, newest first."""
    index_root = data_root / "matches" / "index.parquet"
    files = list(index_root.glob("**/*.parquet")) if index_root.exists() else []
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(str(index_root / "**/*.parquet"), hive_partitioning=True)
    if mode is not None:
        lf = lf.filter(pl.col("mode") == mode)
    lf = lf.sort("started_at", descending=True)
    if limit is not None:
        lf = lf.limit(limit)
    return lf.collect()


def load_replay_payload(match_id: str, data_root: Path) -> dict[str, Any]:
    path = data_root / REPLAYS_DIR / f"{match_id}.json.gz"
    raw = gzip.decompress(path.read_bytes()).decode("utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"replay payload is not a dict: {type(parsed).__name__}")
    return parsed


def load_replay(match_id: str, data_root: Path) -> Any:
    """Reconstruct a kaggle_environments Environment from a stored replay."""
    from kaggle_environments import make

    payload = load_replay_payload(match_id, data_root=data_root)
    return make(
        payload.get("name", "orbit_wars"),
        configuration=payload.get("configuration", {}),
        steps=payload.get("steps", []),
    )
