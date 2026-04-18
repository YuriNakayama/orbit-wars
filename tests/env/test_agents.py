"""Unit tests for src/env/agents.py."""

from __future__ import annotations

import re

import pytest

from env import agents


def test_list_agents_returns_registered_names() -> None:
    names = agents.list_agents()
    assert "baseline_v1" in names
    assert "case0" in names
    assert "random" in names


def test_resolve_case0_returns_callable() -> None:
    resolved = agents.resolve("case0")
    assert callable(resolved)


def test_resolve_baseline_returns_callable() -> None:
    resolved = agents.resolve("baseline_v1")
    assert callable(resolved)


def test_resolve_random_returns_string() -> None:
    assert agents.resolve("random") == "random"


def test_resolve_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        agents.resolve("does_not_exist")


def test_agent_version_is_hex_or_empty() -> None:
    version = agents.agent_version()
    assert version == "" or re.fullmatch(r"[0-9a-f]{7,}", version)
