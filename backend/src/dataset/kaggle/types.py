"""Kaggle scraping 用の frozen dataclass 群。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeamRank:
    rank: int
    team_id: int
    team_name: str
    score: float
    submission_date: str


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: int
    mode: str
    create_time: str
    end_time: str
    agents: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ScrapeSpec:
    top: int
    modes: tuple[str, ...]
    limit_per_team: int | None
    data_root: Path
    dry_run: bool
    include_failed: bool
