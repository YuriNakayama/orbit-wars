"""Local self-play: run agents in `the local simulator` and emit MatchRecords."""

from __future__ import annotations

from dataset.selfplay.agents import (
    AGENT_REGISTRY,
    agent_version,
    list_agents,
    resolve,
)
from dataset.selfplay.report import aggregate, build_table, summarize
from dataset.selfplay.runner import RunSpec, run_episodes

__all__ = [
    "AGENT_REGISTRY",
    "RunSpec",
    "agent_version",
    "aggregate",
    "build_table",
    "list_agents",
    "resolve",
    "run_episodes",
    "summarize",
]
