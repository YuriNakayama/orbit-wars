"""Kaggle episode ログ収集モジュール。

Kaggle Orbit Wars の上位エージェントの対戦リプレイを
`kaggle_environments.api` 相当の EpisodeService 経由で取得する。
"""

from __future__ import annotations

from env.kaggle.client import (
    BASE_URL,
    ClientConfig,
    KaggleEpisodeError,
    build_session,
    extract_replay_json,
    get_episode_replay,
    get_team,
    list_episodes_by_ids,
    list_episodes_for_submission,
)
from env.kaggle.rate_limit import TokenBucket
from env.kaggle.scraper import ScrapeResult, run
from env.kaggle.types import EpisodeMeta, ScrapeSpec, TeamRank

__all__ = [
    "BASE_URL",
    "ClientConfig",
    "EpisodeMeta",
    "KaggleEpisodeError",
    "ScrapeResult",
    "ScrapeSpec",
    "TeamRank",
    "TokenBucket",
    "build_session",
    "extract_replay_json",
    "get_episode_replay",
    "get_team",
    "list_episodes_by_ids",
    "list_episodes_for_submission",
    "run",
]
