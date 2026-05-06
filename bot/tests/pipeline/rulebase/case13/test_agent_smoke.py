"""case13 baseline_v13 が kaggle_environments で正常起動する smoke test。"""

from __future__ import annotations

from kaggle_environments import make

from pipeline.rulebase.case13.baseline.agent import agent


def test_baseline_v13_runs_against_random() -> None:
    env = make("orbit_wars", configuration={"agents": 2, "seed": 42}, debug=True)
    env.run([agent, "random"])
    assert env.state[0]["status"] in {"DONE", "ACTIVE"}
