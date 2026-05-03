"""Evaluate il_v1 (imitation/case1) vs baseline_v1 (rulebase/case1) over N episodes.

Params are read from repo-root `params.yaml` (`evaluation.*`).
CLI flags (`--episodes`, `--seed`, `--out`, `--label`) override params for ad-hoc runs.

Wilson CI summarization lives in `src/evaluation/vs_baseline.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
import yaml

from dataset.selfplay.runner import RunSpec, run_episodes
from evaluation.vs_baseline import summarize_records
from utils.repo_root import absolute_under_repo, find_repo_root

logger = logging.getLogger(__name__)

CHALLENGER = "il_v1"
BASELINE = "baseline_v1"

app = typer.Typer(
    add_completion=False,
    help="Evaluate imitation/case1 IL vs rulebase/case1 baseline.",
)


def _abspath(rel: str | Path) -> Path:
    return absolute_under_repo(rel, start=Path(__file__))


def _load_params() -> dict[str, Any]:
    with (find_repo_root(Path(__file__)) / "params.yaml").open() as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict), "params.yaml must be a mapping"
    return loaded


@app.command()
def main(
    episodes: int | None = typer.Option(None, "--episodes", "-n"),
    mode: str = typer.Option("1v1", "--mode"),
    seed: int | None = typer.Option(None, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    out: Path | None = typer.Option(None, "--out"),
    label: str = typer.Option(
        "", "--label", help="Free-form tag (e.g. iter10) recorded in the JSON"
    ),
) -> None:
    if mode != "1v1":
        raise typer.BadParameter("only 1v1 supported in this entry point")
    cfg = _load_params()
    eval_cfg = cfg.get("evaluation", {})
    episodes = int(episodes if episodes is not None else eval_cfg.get("episodes", 100))
    seed = int(seed if seed is not None else eval_cfg.get("seed", 0))
    out_path = (
        _abspath(out)
        if out is not None
        else _abspath(
            eval_cfg.get("metrics_out", "data/mart/imitation/case1/eval_metrics.json")
        )
    )

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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("vs-baseline result: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    app()
