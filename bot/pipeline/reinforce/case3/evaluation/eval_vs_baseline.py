"""Evaluate rl_v1 (reinforce/case3) vs a chosen baseline over N episodes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from dataset.selfplay.runner import RunSpec, run_episodes
from evaluate.vs_baseline import summarize_records
from utils.repo_root import absolute_under_repo

logger = logging.getLogger(__name__)

CHALLENGER = "rl_v1"

app = typer.Typer(
    add_completion=False,
    help="Evaluate reinforce/case3 (PPO + BC warm-start) vs a chosen baseline.",
)


def _abspath(rel: str | Path) -> Path:
    return absolute_under_repo(rel, start=Path(__file__))


_EPISODES_OPT = typer.Option(100, "--episodes", "-n")
_BASELINE_OPT = typer.Option("baseline_v1", "--baseline")
_MODE_OPT = typer.Option("1v1", "--mode")
_SEED_OPT = typer.Option(0, "--seed")
_PARALLEL_OPT = typer.Option(1, "--parallel", "-p")
_DATA_ROOT_OPT = typer.Option(Path("data"), "--data-root")
_OUT_OPT = typer.Option(None, "--out")
_LABEL_OPT = typer.Option("", "--label")


@app.command()
def main(
    episodes: int = _EPISODES_OPT,
    baseline: str = _BASELINE_OPT,
    mode: str = _MODE_OPT,
    seed: int = _SEED_OPT,
    parallel: int = _PARALLEL_OPT,
    data_root: Path = _DATA_ROOT_OPT,
    out: Path | None = _OUT_OPT,
    label: str = _LABEL_OPT,
) -> None:
    if mode != "1v1":
        raise typer.BadParameter("only 1v1 supported in this entry point")
    out_path = (
        _abspath(out)
        if out is not None
        else _abspath("data/mart/reinforce/case3/eval_metrics.json")
    )
    spec = RunSpec(
        agents=(CHALLENGER, baseline),
        mode=mode,
        episodes=episodes,
        seed=seed,
        parallel=parallel,
        save_replay=False,
        data_root=data_root,
    )
    records = run_episodes(spec)
    summary: dict[str, Any] = dict(summarize_records(records, challenger_idx=0))
    summary["challenger"] = CHALLENGER
    summary["baseline"] = baseline
    summary["mode"] = mode
    summary["seed_start"] = float(seed)
    summary["seed_end_exclusive"] = float(seed + episodes)
    if label:
        summary["label"] = label
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("vs-baseline result: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    app()
