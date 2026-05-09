"""E2E smoke test: imitation/case5 must not send any fleet into the sun."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.imitation.case5.policy.geometry import CENTER_X, CENTER_Y, SUN_R
from tests.e2e.pipeline.util import (
    assert_no_fleet_enters_sun_zone,
    make_orbit_env,
    noop_agent,
    run_orbit_wars_episode,
)

_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[5]
    / "pipeline"
    / "imitation"
    / "case5"
    / "policy"
    / "weights.pt"
)


def test_no_fleet_enters_sun_zone_under_case5_il() -> None:
    if not _WEIGHTS_PATH.exists():
        pytest.skip(f"weights.pt missing at {_WEIGHTS_PATH} — run `dev/dvc pull`")

    from pipeline.imitation.case5.policy.agent import agent

    env = make_orbit_env(seed=0, agents=2, episode_steps=12)
    run_orbit_wars_episode(env, [agent, noop_agent])
    assert len(env.steps) > 0
    assert_no_fleet_enters_sun_zone(
        env,
        center_x=CENTER_X,
        center_y=CENTER_Y,
        sun_r=SUN_R,
    )
