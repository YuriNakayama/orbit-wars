"""Single-match worker: runs one orbit_wars match and returns a record.

Top-level functions only (pickle-friendly for multiprocessing).
The in-repo simulator is imported lazily inside the worker so that each
spawned process owns its own engine state.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from statistics import median
from typing import Any

from dataset.schema import AgentTiming, MatchSpec
from dataset.selfplay.agents import resolve

TIMEOUT_THRESHOLD_SEC = 1.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summarize_timings(timings: list[float]) -> AgentTiming:
    if not timings:
        return AgentTiming(timeouts=0, p50=0.0, p95=0.0, max=0.0)
    timeouts = sum(1 for t in timings if t > TIMEOUT_THRESHOLD_SEC)
    return AgentTiming(
        timeouts=timeouts,
        p50=float(median(timings)),
        p95=_percentile(timings, 0.95),
        max=max(timings),
    )


def _episode_winner(final_state: list[dict[str, Any]]) -> int:
    if not final_state:
        return -1
    active = [i for i, s in enumerate(final_state) if s.get("status") == "ACTIVE"]
    if len(active) == 1:
        return active[0]
    rewards = [float(s.get("reward") or 0.0) for s in final_state]
    top = max(rewards)
    leaders = [i for i, r in enumerate(rewards) if r == top]
    if len(leaders) == 1:
        return leaders[0]
    return -1


def _player_scores(env: Any, num_agents: int) -> list[int]:
    steps = getattr(env, "steps", None) or []
    if not steps:
        return [0] * num_agents
    last = steps[-1]
    scores: list[int] = []
    for agent_state in last:
        obs = agent_state.get("observation") or {}
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        player = obs.get("player", 0)
        ships = sum(int(p[5]) for p in planets if p[1] == player)
        ships += sum(int(f[6]) for f in fleets if f[1] == player)
        scores.append(ships)
    while len(scores) < num_agents:
        scores.append(0)
    return scores[:num_agents]


def _wrap_with_timing(agent: Any, timings: list[float]) -> Any:
    if isinstance(agent, str):
        return agent

    def _wrapped(obs: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return agent(obs, *args, **kwargs)
        finally:
            timings.append(time.perf_counter() - start)

    return _wrapped


def run_one_match(spec: MatchSpec) -> tuple[dict[str, Any], bytes | None]:
    """Run a single match and return (record_dict, gzipped_replay_or_none)."""
    from env.orbit_wars import make_orbit_wars_env, run_orbit_wars_episode

    num_agents = len(spec.agents)
    config: dict[str, Any] = {"agents": num_agents, "seed": spec.seed}
    env = make_orbit_wars_env(configuration=config)

    timings_per_agent: list[list[float]] = [[] for _ in range(num_agents)]
    resolved_agents: list[Any] = [resolve(a.name) for a in spec.agents]
    wrapped = [
        _wrap_with_timing(resolved_agents[i], timings_per_agent[i])
        for i in range(num_agents)
    ]

    started_at = dt.datetime.now(dt.UTC).isoformat()
    t0 = time.perf_counter()
    run_orbit_wars_episode(env, wrapped)
    elapsed_sec = time.perf_counter() - t0

    final_state = env.steps[-1] if env.steps else []
    winner = _episode_winner(final_state)
    draw = winner < 0
    turns = len(env.steps)
    scores = _player_scores(env, num_agents)
    agent_timings = tuple(_summarize_timings(t) for t in timings_per_agent)

    replay_bytes: bytes | None = None
    replay_path = ""
    if spec.save_replay:
        payload = env.toJSON() if hasattr(env, "toJSON") else {"steps": env.steps}
        encoded = json.dumps(payload, default=str).encode("utf-8")
        replay_bytes = gzip.compress(encoded)
        replay_path = f"replays/{spec.match_id}.json.gz"

    record = {
        "match_id": spec.match_id,
        "run_id": spec.run_id,
        "mode": spec.mode,
        "seed": spec.seed,
        "started_at": started_at,
        "elapsed_sec": elapsed_sec,
        "turns": turns,
        "winner": winner,
        "draw": draw,
        "agent_names": tuple(a.name for a in spec.agents),
        "agent_versions": tuple(a.version for a in spec.agents),
        "agent_scores": tuple(scores),
        "agent_timings": agent_timings,
        "replay_path": replay_path,
        "git_sha": spec.agents[0].version if spec.agents else "",
    }
    return record, replay_bytes
