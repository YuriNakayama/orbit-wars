"""Persistence and read-side for match records.

Writers (selfplay, kaggle) feed `recorder` with `MatchRecord` + replay bytes.
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
)
from dataset.storage.paths import INDEX_DIRNAME, REPLAYS_DIRNAME, index_root
from dataset.storage.recorder import write_records, write_replay, write_run

__all__ = [
    "INDEX_DIRNAME",
    "REPLAYS_DIRNAME",
    "agent_winrate",
    "index_root",
    "list_matches",
    "load_kaggle_replay",
    "load_replay",
    "load_replay_payload",
    "mode_summary",
    "scan_index",
    "timing_distribution",
    "write_records",
    "write_replay",
    "write_run",
]
