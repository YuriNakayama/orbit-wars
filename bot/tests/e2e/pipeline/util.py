"""Small shared helpers for pipeline e2e tests."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from env.orbit_wars import make_orbit_wars_env, run_orbit_wars_episode

__all__ = [
    "assert_agent_runs_against_random_to_done",
    "assert_done_or_inactive",
    "assert_no_fleet_enters_sun_zone",
    "avg_peak_over_seeds",
    "fire_one_fleet",
    "fleet_by_id",
    "make_initialized_orbit_env",
    "make_orbit_env",
    "max_observed_ships_per_fleet",
    "noop_agent",
    "planet_by_id",
    "run_orbit_wars_episode",
    "walk_fleets",
]

AgentFn = Callable[[dict[str, Any]], list[list[int | float]]]
NoopFn = Callable[[object], list[list[int | float]]]

DONE_STATUSES = frozenset({"DONE", "INACTIVE"})


def noop_agent(_obs: object) -> list[list[int | float]]:
    """Return no Orbit Wars actions."""
    return []


def make_orbit_env(
    *,
    seed: int = 0,
    agents: int = 2,
    episode_steps: int | None = None,
    debug: bool = False,
) -> Any:
    """Create a configured Orbit Wars environment."""
    configuration: dict[str, int] = {"agents": agents, "seed": seed}
    if episode_steps is not None:
        configuration["episodeSteps"] = episode_steps
    return make_orbit_wars_env(configuration=configuration, debug=debug)


def make_initialized_orbit_env(*, seed: int = 0, agents: int = 2) -> Any:
    """Create an env and advance once so planets/fleets are initialized."""
    random.seed(seed)
    env = make_orbit_env(seed=seed, agents=agents, debug=True)
    env.reset(agents)
    env.step([[], []])
    return env


def assert_done_or_inactive(env: Any) -> None:
    """Assert all final seat statuses are terminal/inactive."""
    assert len(env.steps) > 0
    final = env.steps[-1]
    statuses = [s.get("status") for s in final]
    assert all(status in DONE_STATUSES for status in statuses)


def assert_agent_runs_against_random_to_done(agent: AgentFn, *, seed: int = 0) -> None:
    """Run an agent against the fixed random opponent and assert terminal statuses."""
    env = make_orbit_env(seed=seed, agents=2)
    run_orbit_wars_episode(env, [agent, "random"])
    assert_done_or_inactive(env)


def walk_fleets(env: Any) -> list[tuple[int, int, float, float]]:
    """Return [(step_idx, fleet_id, x, y), ...] across all env steps."""
    out: list[tuple[int, int, float, float]] = []
    for step_idx, step in enumerate(env.steps):
        for entry in step:
            obs = entry.get("observation") or {}
            for fleet in obs.get("fleets") or []:
                if len(fleet) < 4:
                    continue
                out.append((step_idx, int(fleet[0]), float(fleet[2]), float(fleet[3])))
    return out


def assert_no_fleet_enters_sun_zone(
    env: Any,
    *,
    center_x: float,
    center_y: float,
    sun_r: float,
    safety: float = 0.5,
) -> None:
    """Assert no observed fleet position enters the sun safety radius."""
    threshold = sun_r + safety
    violations: list[tuple[int, int, float, float, float]] = []
    for step_idx, fid, fx, fy in walk_fleets(env):
        distance = math.hypot(fx - center_x, fy - center_y)
        if distance < threshold:
            violations.append((step_idx, fid, fx, fy, distance))
    assert not violations, (
        f"Fleet inside sun zone (threshold={threshold}): "
        f"{violations[:5]}{'...' if len(violations) > 5 else ''}"
    )


def planet_by_id(obs: dict[str, Any], pid: int) -> list[Any] | None:
    """Find a planet row by id in an observation."""
    for planet in obs.get("planets", []) or []:
        if int(planet[0]) == pid:
            return list(planet)
    return None


def fleet_by_id(obs: dict[str, Any], fid: int) -> list[Any] | None:
    """Find a fleet row by id in an observation."""
    for fleet in obs.get("fleets", []) or []:
        if int(fleet[0]) == fid:
            return list(fleet)
    return None


def fire_one_fleet(
    env: Any, player_seat: int, from_pid: int, angle: float, ships: int
) -> tuple[dict[str, Any], int]:
    """Submit one shot and return (obs after launch, newest fleet id or -1)."""
    actions: list[list[Any]] = [[], []]
    actions[player_seat] = [[from_pid, angle, ships]]
    env.step(actions)
    obs_after: dict[str, Any] = env.steps[-1][player_seat]["observation"]
    fleets = obs_after.get("fleets", []) or []
    if not fleets:
        return obs_after, -1
    return obs_after, max(int(fleet[0]) for fleet in fleets)


def _random_action(env: Any, seat: int) -> Any:
    random_agent = env.agents.get("random")
    if random_agent is None:
        return "random"
    obs = env.steps[-1][seat]["observation"]
    try:
        return random_agent(obs, env.configuration)
    except TypeError:
        return random_agent(obs)


def max_observed_ships_per_fleet(
    agent: AgentFn, *, seed: int, max_turns: int = 80
) -> dict[int, int]:
    """Run the agent against a fixed random opponent and record friendly peaks."""
    env = make_orbit_env(seed=seed, agents=2)
    peaks: dict[int, int] = {}
    for _ in range(max_turns):
        if env.done:
            break
        env.step([agent(env.steps[-1][0]["observation"]), _random_action(env, 1)])
        fleets = env.steps[-1][0]["observation"].get("fleets", [])
        for fleet in fleets:
            fid, owner, _x, _y, _angle, _from_pid, ships = fleet
            if owner != 0:
                continue
            ships_int = int(ships)
            peaks[int(fid)] = max(peaks.get(int(fid), 0), ships_int)
    return peaks


def avg_peak_over_seeds(agent: AgentFn, seeds: list[int]) -> float:
    """Average peak friendly fleet size over several seeds."""
    all_peaks: list[int] = []
    for seed in seeds:
        all_peaks.extend(max_observed_ships_per_fleet(agent, seed=seed).values())
    if not all_peaks:
        return 0.0
    return sum(all_peaks) / len(all_peaks)
