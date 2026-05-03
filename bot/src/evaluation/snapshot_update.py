"""Snapshot capture/write helpers shared by per-case `evaluation/snapshot_update.py`.

The kaggle_environments `orbit_wars` engine is only partially deterministic
(episode length varies across identical-seed runs), so snapshots cover a single
representative observation. Each case calls `capture_observation` to grab the
obs at (seed, turn), runs its own `agent(obs)` to compute the action, then
calls `write_snapshot` to persist both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def capture_observation(seed: int, turn: int, *, agents: int = 2) -> dict[str, Any]:
    """Reset orbit_wars and step `turn + 1` no-op turns; return the latest obs."""
    import kaggle_environments

    env = kaggle_environments.make(
        "orbit_wars", configuration={"agents": agents, "seed": seed}
    )
    no_op_actions: list[list[Any]] = [[] for _ in range(agents)]
    for _ in range(turn + 1):
        env.step(no_op_actions)
    obs = env.steps[-1][0]["observation"]
    if not isinstance(obs, dict):
        raise TypeError(f"expected dict observation, got {type(obs).__name__}")
    return obs


def write_snapshot(
    obs: dict[str, Any],
    action: list[Any],
    obs_out: Path,
    action_out: Path,
) -> None:
    """Write `obs` and `action` JSON files atomically (mkdir parents first)."""
    obs_out.parent.mkdir(parents=True, exist_ok=True)
    obs_out.write_text(json.dumps(obs), encoding="utf-8")
    action_out.parent.mkdir(parents=True, exist_ok=True)
    action_out.write_text(json.dumps(action), encoding="utf-8")
