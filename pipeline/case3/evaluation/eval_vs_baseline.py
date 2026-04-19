"""Evaluate case3_il_v1 vs case1_baseline_v1 over N self-play episodes.

Usage:
    uv run python -m pipeline.case3.evaluation.eval_vs_baseline \
        --episodes 100 --mode 1v1 --parallel 4
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env.runner import RunSpec, run_episodes  # noqa: E402
from env.types import MatchRecord  # noqa: E402

logger = logging.getLogger(__name__)

CHALLENGER = "case3_il_v1"
BASELINE = "baseline_v1"

app = typer.Typer(add_completion=False, help="Evaluate case3 IL vs case1 baseline.")


def _summarize(records: list[MatchRecord], challenger_idx: int) -> dict[str, float]:
    n = len(records)
    wins = sum(1 for r in records if r.winner == challenger_idx and not r.draw)
    losses = sum(
        1
        for r in records
        if r.winner != challenger_idx and r.winner >= 0 and not r.draw
    )
    draws = sum(1 for r in records if r.draw)
    return {
        "episodes": float(n),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / n if n else 0.0,
        "non_draw_win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
    }


@app.command()
def main(
    episodes: int = typer.Option(100, "--episodes", "-n"),
    mode: str = typer.Option("1v1", "--mode"),
    seed: int = typer.Option(0, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    out: Path = typer.Option(Path("pipeline/case3/evaluation/results.json"), "--out"),
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
    summary = _summarize(records, challenger_idx=0)
    summary["challenger"] = CHALLENGER  # type: ignore[assignment]
    summary["baseline"] = BASELINE  # type: ignore[assignment]
    summary["mode"] = mode  # type: ignore[assignment]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
