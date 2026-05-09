"""E2E tests for the case9 baseline agent."""

from __future__ import annotations

from pipeline.rulebase.case9.baseline import agent
from tests.e2e.pipeline.util import assert_agent_runs_1v1_to_done


def test_agent_runs_1v1_to_done() -> None:
    assert_agent_runs_1v1_to_done(agent)
