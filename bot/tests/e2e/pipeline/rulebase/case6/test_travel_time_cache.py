"""Parity test for ``WorldModel.cached_travel_time``.

The per-turn cache must return *exactly* the same integer as a fresh
``physics.travel_time`` call for every (src, target, ships) probed by
``stay.py``. Drift would silently change STAY decisions and invalidate the
case6 evaluation results.
"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.rulebase.case6.baseline.agent import build_world
from pipeline.rulebase.case6.baseline.core.physics import travel_time
from tests.e2e.pipeline.util import make_orbit_env, noop_agent


def _seeded_obs(seed: int, step: int = 30) -> dict[str, Any]:
    env = make_orbit_env(seed=seed, agents=2)

    # Run a deterministic noop episode for `step` turns to get a non-trivial
    # mid-game obs (planets + a couple of fleets in flight).
    env.reset(num_agents=2)
    for _ in range(step):
        if env.done:
            break
        env.step([noop_agent(s["observation"]) for s in env.steps[-1]])
    obs: dict[str, Any] = env.steps[-1][0]["observation"]
    return obs


def test_cached_travel_time_matches_direct_for_seed_zero() -> None:
    """For every (src, target, ships) used by stay.py, cached == direct."""
    obs = _seeded_obs(seed=0, step=40)
    world = build_world(obs)

    probes: list[tuple[int, int, int]] = []
    # Sample the full (planet, planet, ships) grid that stay.py would walk.
    for src in world.planets:
        for tgt in world.planets:
            if src.id == tgt.id:
                continue
            for ships in (1, 5, max(1, int(src.ships)), int(src.ships) + 3):
                probes.append((src.id, tgt.id, max(1, ships)))

    assert probes, "fixture should produce a non-empty probe set"

    for src_id, tgt_id, ships in probes:
        src = world.planet_by_id[src_id]
        tgt = world.planet_by_id[tgt_id]
        direct = travel_time(src.x, src.y, src.radius, tgt.x, tgt.y, tgt.radius, ships)
        cached = world.cached_travel_time(src_id, tgt_id, ships)
        assert cached == direct, (
            f"cache drift at src={src_id} tgt={tgt_id} ships={ships}: "
            f"cached={cached} direct={direct}"
        )


def test_cached_travel_time_is_idempotent() -> None:
    """A second lookup must return the same value (cache must not mutate)."""
    obs = _seeded_obs(seed=1, step=20)
    world = build_world(obs)
    if len(world.planets) < 2:
        pytest.skip("not enough planets to probe")
    a, b = world.planets[0].id, world.planets[1].id
    first = world.cached_travel_time(a, b, 7)
    second = world.cached_travel_time(a, b, 7)
    assert first == second
