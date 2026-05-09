"""E2E smoke test: rulebase/case1 must not send any fleet into the sun."""

from __future__ import annotations

from pipeline.rulebase.case1.baseline import agent
from pipeline.rulebase.case1.baseline.core.config import CENTER_X, CENTER_Y, SUN_R
from tests.e2e.pipeline.util import (
    assert_no_fleet_enters_sun_zone,
    make_orbit_env,
    noop_agent,
    run_orbit_wars_episode,
)


def test_no_fleet_enters_sun_zone_under_baseline_v1() -> None:
    env = make_orbit_env(seed=0, agents=2, episode_steps=12)
    run_orbit_wars_episode(env, [agent, noop_agent])
    assert len(env.steps) > 0
    assert_no_fleet_enters_sun_zone(
        env,
        center_x=CENTER_X,
        center_y=CENTER_Y,
        sun_r=SUN_R,
    )
