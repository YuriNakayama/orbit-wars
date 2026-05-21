"""Shared on-disk layout constants for the match dataset.

Layout:
  data_root/matches/index.parquet/mode={mode}/run_{run_id}[_N].parquet

Replay payloads (.json.gz) live in S3 directly, NOT under data_root:
  s3://{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/{source}/{match_id}.json.gz

`source` is `"kaggle"` for scraped Kaggle episodes and `"selfplay"` for
locally-generated self-play matches. The MatchRecord.replay_uri column
stores the full URI per row so consumers can read replays from S3 directly
without any path convention coupling.
"""

from __future__ import annotations

from pathlib import Path

INDEX_DIRNAME = "index.parquet"

REPLAY_S3_BUCKET = "orbit-wars-dvc-286854171013"
REPLAY_S3_PREFIX = "replays"


def index_root(data_root: Path) -> Path:
    return data_root / "matches" / INDEX_DIRNAME


def replay_uri(match_id: str, source: str) -> str:
    """Build the canonical S3 URI for a replay payload."""
    return f"s3://{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/{source}/{match_id}.json.gz"
