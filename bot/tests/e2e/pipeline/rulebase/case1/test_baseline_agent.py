"""E2E tests for the case1 baseline agent."""

from __future__ import annotations

from pipeline.rulebase.case1.baseline import agent
from tests.e2e.pipeline.util import assert_agent_runs_against_random_to_done


def test_agent_runs_against_random_to_done() -> None:
    assert_agent_runs_against_random_to_done(agent)
