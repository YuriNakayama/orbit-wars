"""Snapshot test: trained agent produces deterministic output for a fixed obs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.imitation.case1.policy.agent import agent

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def test_agent_snapshot_obs_001() -> None:
    obs = json.loads((SNAPSHOT_DIR / "obs_001.json").read_text())
    expected = json.loads((SNAPSHOT_DIR / "action_001.json").read_text())
    actual = agent(obs)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert int(a[0]) == int(e[0])
        assert float(a[1]) == pytest.approx(float(e[1]), abs=1e-6)
        assert int(a[2]) == int(e[2])


def test_agent_snapshot_is_deterministic() -> None:
    obs = json.loads((SNAPSHOT_DIR / "obs_001.json").read_text())
    a = agent(obs)
    b = agent(obs)
    assert a == b
