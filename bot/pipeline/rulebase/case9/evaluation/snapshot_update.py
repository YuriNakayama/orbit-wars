"""Regenerate the action snapshot used by tests/pipeline/rulebase/case9/.

Capture/write helpers live in `src/evaluation/snapshot_update.py`; this wrapper
binds them to the case9 baseline agent and the case9 snapshot paths.
"""

from __future__ import annotations

from pathlib import Path

import typer

from evaluation.snapshot_update import capture_observation, write_snapshot
from pipeline.rulebase.case9.baseline import agent as baseline_agent
from utils.repo_root import find_repo_root

app = typer.Typer(add_completion=False, help="Snapshot updater for case9 baseline.")

REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_SNAPSHOT_DIR = (
    REPO_ROOT / "tests" / "pipeline" / "rulebase" / "case9" / "snapshots"
)
DEFAULT_OBS = DEFAULT_SNAPSHOT_DIR / "obs_seed0_turn10.json"
DEFAULT_ACTION = DEFAULT_SNAPSHOT_DIR / "action_seed0_turn10.json"


@app.command()
def update(
    seed: int = typer.Option(0, "--seed"),
    turn: int = typer.Option(10, "--turn"),
    obs_out: Path = typer.Option(DEFAULT_OBS, "--obs-out"),
    action_out: Path = typer.Option(DEFAULT_ACTION, "--action-out"),
) -> None:
    """Capture (obs, agent(obs)) at (seed, turn) and write both to disk."""
    obs = capture_observation(seed, turn)
    action = baseline_agent(obs)
    write_snapshot(obs, action, obs_out, action_out)
    typer.echo(f"obs written: {obs_out}")
    typer.echo(f"action written: {action_out} ({len(action)} moves)")


if __name__ == "__main__":
    app()
