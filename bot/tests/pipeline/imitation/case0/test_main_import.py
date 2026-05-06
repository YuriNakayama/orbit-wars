"""Verify case0/main.py exposes ``agent`` under the expected import patterns.

Locally we import via the package path; the Kaggle-side ``Path.cwd()``
pattern is exercised by ``submit submit ... --dry-run`` in CI.
"""

from __future__ import annotations


def test_local_package_import_exposes_agent() -> None:
    from pipeline.imitation.case0.policy.agent import agent

    assert callable(agent)
    assert agent({}) == []


def test_agent_registry_contains_il_v0() -> None:
    from dataset.selfplay.agents import AGENT_REGISTRY

    assert "il_v0" in AGENT_REGISTRY
    assert AGENT_REGISTRY["il_v0"] == "pipeline.imitation.case0.policy.agent:agent"
