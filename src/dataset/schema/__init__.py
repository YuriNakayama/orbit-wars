"""Shared data contracts for the dataset package.

Only frozen dataclasses and constants — no dependencies on other subpackages.
All persisted rows (selfplay + kaggle) conform to `MatchRecord`.
"""

from __future__ import annotations

from dataset.schema.types import (
    MAX_PLAYERS,
    SCHEMA_VERSION,
    SOURCE_KAGGLE,
    SOURCE_SELFPLAY,
    AgentKaggleMeta,
    AgentSpec,
    AgentTiming,
    MatchRecord,
    MatchSpec,
)

__all__ = [
    "MAX_PLAYERS",
    "SCHEMA_VERSION",
    "SOURCE_KAGGLE",
    "SOURCE_SELFPLAY",
    "AgentKaggleMeta",
    "AgentSpec",
    "AgentTiming",
    "MatchRecord",
    "MatchSpec",
]
