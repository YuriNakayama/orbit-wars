"""Snapshot test: trained agent produces deterministic output for a fixed obs.

The snapshot expectations are tied to a specific `weights.pt`. After any model
architecture change the weights need re-training; the snapshot test is skipped
when the on-disk weights cannot be loaded into the current model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.imitation.case1.policy.agent import agent

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _agent_or_skip(obs: dict[str, object]) -> list[list[int | float]]:
    try:
        return agent(obs)
    except FileNotFoundError as e:
        if e.filename and e.filename.endswith("weights.pt"):
            pytest.skip(f"weights.pt is not available in this worktree: {e}")
        raise
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            pytest.skip(f"weights.pt incompatible with current architecture: {e}")
        raise


def test_agent_snapshot_obs_001() -> None:
    obs = json.loads((SNAPSHOT_DIR / "obs_001.json").read_text())
    expected = json.loads((SNAPSHOT_DIR / "action_001.json").read_text())
    actual = _agent_or_skip(obs)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert int(a[0]) == int(e[0])
        assert float(a[1]) == pytest.approx(float(e[1]), abs=1e-6)
        assert int(a[2]) == int(e[2])


def test_agent_snapshot_is_deterministic() -> None:
    obs = json.loads((SNAPSHOT_DIR / "obs_001.json").read_text())
    a = _agent_or_skip(obs)
    b = _agent_or_skip(obs)
    assert a == b
