"""Shared on-disk layout constants for the match dataset.

Layout:
  data_root/matches/index.parquet/mode={mode}/run_{run_id}[_N].parquet
  data_root/matches/replays/{match_id}.json.gz
"""

from __future__ import annotations

from pathlib import Path

INDEX_DIRNAME = "index.parquet"
REPLAYS_DIRNAME = "replays"


def index_root(data_root: Path) -> Path:
    return data_root / "matches" / INDEX_DIRNAME


def replays_root(data_root: Path) -> Path:
    return data_root / "matches" / REPLAYS_DIRNAME
