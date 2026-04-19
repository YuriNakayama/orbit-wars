"""Match dataset: selfplay + Kaggle scraping → unified parquet index.

Public API:
  run_episodes, load_replay, list_matches, AGENT_REGISTRY, scrape_kaggle
"""

from __future__ import annotations

from dataset.kaggle import ScrapeSpec
from dataset.kaggle import run as scrape_kaggle
from dataset.schema import (
    AgentKaggleMeta,
    AgentSpec,
    AgentTiming,
    MatchRecord,
    MatchSpec,
)
from dataset.selfplay import AGENT_REGISTRY, RunSpec, list_agents, resolve, run_episodes
from dataset.storage import (
    list_matches,
    load_kaggle_replay,
    load_replay,
    load_replay_payload,
)

__all__ = [
    "AGENT_REGISTRY",
    "AgentKaggleMeta",
    "AgentSpec",
    "AgentTiming",
    "MatchRecord",
    "MatchSpec",
    "RunSpec",
    "ScrapeSpec",
    "list_agents",
    "list_matches",
    "load_kaggle_replay",
    "load_replay",
    "load_replay_payload",
    "resolve",
    "run_episodes",
    "scrape_kaggle",
]
