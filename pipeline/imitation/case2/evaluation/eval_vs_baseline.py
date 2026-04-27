"""Evaluate il_v2 (imitation/case2) vs baseline_v1 (rulebase/case1) over N episodes.

Usage:
    uv run python -m pipeline.imitation.case2.evaluation.eval_vs_baseline \
        --episodes 300 --mode 1v1 --parallel 4

Reports wins/losses/draws plus a Wilson 95% CI on win rate so iter-to-iter
comparisons can tell signal from kaggle_environments non-determinism noise.
Episodes use deterministic seeds [seed, seed+episodes) — keep the same seed
range across iterations to make comparisons reproducible.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.schema import MatchRecord  # noqa: E402
from dataset.selfplay.runner import RunSpec, run_episodes  # noqa: E402

logger = logging.getLogger(__name__)

CHALLENGER = "il_v2"
BASELINE = "baseline_v1"

app = typer.Typer(
    add_completion=False,
    help="Evaluate imitation/case2 IL vs rulebase/case1 baseline.",
)


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — stable for small successes / small n."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarize(records: list[MatchRecord], challenger_idx: int) -> dict[str, float]:
    n = len(records)
    wins = sum(1 for r in records if r.winner == challenger_idx and not r.draw)
    losses = sum(
        1
        for r in records
        if r.winner != challenger_idx and r.winner >= 0 and not r.draw
    )
    draws = sum(1 for r in records if r.draw)
    ci_lo, ci_hi = _wilson_ci(wins, n)
    return {
        "episodes": float(n),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / n if n else 0.0,
        "non_draw_win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
        "win_rate_ci95_lo": ci_lo,
        "win_rate_ci95_hi": ci_hi,
    }


@app.command()
def main(
    episodes: int = typer.Option(100, "--episodes", "-n"),
    mode: str = typer.Option("1v1", "--mode"),
    seed: int = typer.Option(0, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    out: Path = typer.Option(
        Path("pipeline/imitation/case2/evaluation/results.json"), "--out"
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
    summary = _summarize(records, challenger_idx=0)
    summary["challenger"] = CHALLENGER  # type: ignore[assignment]
    summary["baseline"] = BASELINE  # type: ignore[assignment]
    summary["mode"] = mode  # type: ignore[assignment]
    summary["seed_start"] = float(seed)
    summary["seed_end_exclusive"] = float(seed + episodes)
    if label:
        summary["label"] = label  # type: ignore[assignment]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
