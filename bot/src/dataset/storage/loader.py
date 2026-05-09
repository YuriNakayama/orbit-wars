"""Read match index and replay payloads."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl

from dataset.storage.paths import index_root, replays_root


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


def load_replay_payload(match_id: str, data_root: Path) -> dict[str, Any]:
    path = replays_root(data_root) / f"{match_id}.json.gz"
    raw = gzip.decompress(path.read_bytes()).decode("utf-8")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"replay payload is not a dict: {type(parsed).__name__}")
    return parsed


def load_replay(match_id: str, data_root: Path) -> Any:
    """Reconstruct an Orbit Wars Environment from a stored replay."""
    from env.orbit_wars import make_orbit_wars_env

    payload = load_replay_payload(match_id, data_root=data_root)
    if payload.get("name", "orbit_wars") != "orbit_wars":
        raise ValueError("only orbit_wars replays are supported")
    return make_orbit_wars_env(
        configuration=payload.get("configuration", {}),
        steps=payload.get("steps", []),
    )


def load_kaggle_replay(episode_id: int, data_root: Path) -> Any:
    """Load a Kaggle-sourced episode replay as an Environment."""
    return load_replay(f"kaggle_ep_{episode_id}", data_root=data_root)
