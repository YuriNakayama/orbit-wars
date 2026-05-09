"""case8 baseline_v8 simulator smoke test."""

from __future__ import annotations

from pipeline.rulebase.case8.baseline.agent import agent
from tests.e2e.pipeline.util import make_orbit_env, run_orbit_wars_episode


def test_baseline_v8_runs_against_random() -> None:
    env = make_orbit_env(seed=42, debug=True)
    run_orbit_wars_episode(env, [agent, "random"])
    assert env.state[0]["status"] in {"DONE", "ACTIVE"}
