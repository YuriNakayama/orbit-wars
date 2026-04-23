"""Evaluate il_v1 (imitation/case1) vs baseline_v1 (rulebase/case1) over N episodes.

Params are read from repo-root `params.yaml` (`evaluation.*`).
CLI flags (`--episodes`, `--seed`, `--out`) override params for ad-hoc runs.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.schema import MatchRecord  # noqa: E402
from dataset.selfplay.runner import RunSpec, run_episodes  # noqa: E402

logger = logging.getLogger(__name__)

CHALLENGER = "il_v1"
BASELINE = "baseline_v1"

app = typer.Typer(
    add_completion=False,
    help="Evaluate imitation/case1 IL vs rulebase/case1 baseline.",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "backend").is_dir() and (p / ".git").exists():
            return p
    raise RuntimeError("repo root not found from " + str(here))


def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def _load_params() -> dict[str, Any]:
    with (_repo_root() / "params.yaml").open() as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict), "params.yaml must be a mapping"
    return loaded


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
    episodes: int | None = typer.Option(None, "--episodes", "-n"),
    mode: str = typer.Option("1v1", "--mode"),
    seed: int | None = typer.Option(None, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    out: Path | None = typer.Option(None, "--out"),
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
    summary = _summarize(records, challenger_idx=0)
    summary["challenger"] = CHALLENGER  # type: ignore[assignment]
    summary["baseline"] = BASELINE  # type: ignore[assignment]
    summary["mode"] = mode  # type: ignore[assignment]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
