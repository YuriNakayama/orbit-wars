"""case8 baseline_v8 が kaggle_environments で正常起動する smoke test。"""

from __future__ import annotations

from kaggle_environments import make

from pipeline.rulebase.case8.baseline.agent import agent


def test_baseline_v8_runs_against_random() -> None:
    env = make("orbit_wars", configuration={"agents": 2, "seed": 42}, debug=True)
    env.run([agent, "random"])
    assert env.state[0]["status"] in {"DONE", "ACTIVE"}
