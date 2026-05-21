"""Persistence and read-side for match records.

Writers (selfplay, kaggle) feed `recorder` with `MatchRecord` + replay bytes.
The replay bytes are persisted to S3 directly (not under data_root); the
parquet index keeps a `replay_uri` column pointing back to S3.
Readers use `loader` and `analyze` for parquet index access and aggregation.
"""

from __future__ import annotations

from dataset.storage.analyze import (
    agent_winrate,
    mode_summary,
    scan_index,
    timing_distribution,
)
from dataset.storage.loader import (
    list_matches,
    load_kaggle_replay,
    load_replay,
    load_replay_payload,
    load_replay_payload_from_uri,
)
from dataset.storage.paths import (
    INDEX_DIRNAME,
    REPLAY_S3_BUCKET,
    REPLAY_S3_PREFIX,
    index_root,
    replay_uri,
)
from dataset.storage.recorder import write_records, write_replay, write_run

__all__ = [
    "INDEX_DIRNAME",
    "REPLAY_S3_BUCKET",
    "REPLAY_S3_PREFIX",
    "agent_winrate",
    "index_root",
    "list_matches",
    "load_kaggle_replay",
    "load_replay",
    "load_replay_payload",
    "load_replay_payload_from_uri",
    "mode_summary",
    "replay_uri",
    "scan_index",
    "timing_distribution",
    "write_records",
    "write_replay",
    "write_run",
]
