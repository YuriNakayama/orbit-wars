"""Regenerate the action snapshot used by tests/pipeline/case2/.

The kaggle_environments `orbit_wars` engine is only partially deterministic
(episode length varies across identical-seed runs), so we do NOT snapshot the
full episode. Instead we capture a single representative observation from a
fresh env reset and snapshot `agent(obs)` — this is deterministic given
the baseline's pure-Python logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from pipeline.case2.baseline import agent as baseline_agent

app = typer.Typer(add_completion=False, help="Snapshot updater for case2 agent.")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "tests" / "pipeline" / "case2" / "snapshots"
DEFAULT_OBS = DEFAULT_SNAPSHOT_DIR / "obs_seed0_turn10.json"
DEFAULT_ACTION = DEFAULT_SNAPSHOT_DIR / "action_seed0_turn10.json"


def _captured_observation(seed: int, turn: int) -> dict[str, Any]:
    import kaggle_environments

    env = kaggle_environments.make(
        "orbit_wars", configuration={"agents": 2, "seed": seed}
    )
    for _ in range(turn + 1):
        env.step([[], []])
    obs = env.steps[-1][0]["observation"]
    assert isinstance(obs, dict)
    return obs


@app.command()
def update(
    seed: int = typer.Option(0, "--seed"),
    turn: int = typer.Option(10, "--turn"),
    obs_out: Path = typer.Option(DEFAULT_OBS, "--obs-out"),
    action_out: Path = typer.Option(DEFAULT_ACTION, "--action-out"),
) -> None:
    """Capture (obs, agent(obs)) at (seed, turn) and write both to disk."""
    obs = _captured_observation(seed, turn)
    action = baseline_agent(obs)

    obs_out.parent.mkdir(parents=True, exist_ok=True)
    obs_out.write_text(json.dumps(obs), encoding="utf-8")
    action_out.write_text(json.dumps(action), encoding="utf-8")
    typer.echo(f"obs written: {obs_out}")
    typer.echo(f"action written: {action_out} ({len(action)} moves)")


if __name__ == "__main__":
    app()
