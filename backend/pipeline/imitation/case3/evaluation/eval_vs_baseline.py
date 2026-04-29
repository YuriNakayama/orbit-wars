"""Evaluate il_v3 (imitation/case3) vs baseline_v1 (rulebase/case1) over N episodes.

Wilson CI summarization lives in `src/evaluation/vs_baseline.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from dataset.selfplay.runner import RunSpec, run_episodes
from evaluation.vs_baseline import summarize_records

logger = logging.getLogger(__name__)

CHALLENGER = "il_v3"
BASELINE = "baseline_v1"

app = typer.Typer(
    add_completion=False,
    help="Evaluate imitation/case3 IL vs rulebase/case1 baseline.",
)


@app.command()
def main(
    episodes: int = typer.Option(100, "--episodes", "-n"),
    mode: str = typer.Option("1v1", "--mode"),
    seed: int = typer.Option(0, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),  # noqa: B008
    out: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case3/evaluation/results.json"), "--out"
    ),
    label: str = typer.Option(
        "", "--label", help="Free-form tag (e.g. iter10) recorded in the JSON"
    ),
) -> None:
    if mode != "1v1":
        raise typer.BadParameter("only 1v1 supported in this entry point")
    agents = (CHALLENGER, BASELINE)
    spec = RunSpec(
        agents=agents,
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
    summary["baseline"] = BASELINE
    summary["mode"] = mode
    summary["seed_start"] = float(seed)
    summary["seed_end_exclusive"] = float(seed + episodes)
    if label:
        summary["label"] = label
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("vs-baseline result: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    app()
