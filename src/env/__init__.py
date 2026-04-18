"""Evaluation framework for Orbit Wars bot agents.

Public API:
  run_episodes, load_replay, list_matches, AGENT_REGISTRY
"""

from __future__ import annotations

from env.agents import AGENT_REGISTRY, list_agents, resolve
from env.loader import list_matches, load_replay, load_replay_payload
from env.runner import RunSpec, run_episodes
from env.types import AgentSpec, AgentTiming, MatchRecord, MatchSpec

__all__ = [
    "AGENT_REGISTRY",
    "AgentSpec",
    "AgentTiming",
    "MatchRecord",
    "MatchSpec",
    "RunSpec",
    "list_agents",
    "list_matches",
    "load_replay",
    "load_replay_payload",
    "resolve",
    "run_episodes",
]
