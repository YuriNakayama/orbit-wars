"""Evaluate a case10 weights file vs rulebase/case1 over 20×P1/P2 = 40 episodes.

Used by the data_volume_sweep driver: each sweep point produces a .pt file
under pipeline/imitation/case10/policy/data_volume_sweep/, and this script
loads it (by copying to the canonical agent path) and runs the head-to-head
evaluation against baseline_v1.

Output JSON shape (per tag):

    {
      "tag": "top80",
      "weights": "...",
      "win_rate": 0.40,
      "draw_rate": 0.05,
      "challenger_wins": 16,
      "baseline_wins": 22,
      "draws": 2,
      "episodes": 40,
      "mean_turns": 175.3,
      "side_breakdown": {
        "challenger_p1": {"wins": 8, "losses": 11, "draws": 1, "n": 20},
        "challenger_p2": {"wins": 8, "losses": 11, "draws": 1, "n": 20}
      }
    }

skip list adherence:
- always 1v1; FFA never run
- 20 episodes per side × 2 sides = 40 total (NOT 100, NOT 300)
- baseline opponent is always rulebase/case1 (= baseline_v1)
- no Kaggle publicScore lookup
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from dataset.selfplay.runner import RunSpec, run_episodes

logger = logging.getLogger(__name__)

CHALLENGER_HEAD_KEYS = {
    "candidate_ships": "il_v10_candidate",
    "template_ships": "il_v10_template",
}
BASELINE = "baseline_v1"
DEFAULT_OUT_DIR = Path("data/output/experiment/case10_data_volume_sweep")

# Canonical agent weights path used by pipeline/imitation/case10/policy/agent.py
CANONICAL_WEIGHTS_PATH = Path("bot/pipeline/imitation/case10/policy/weights.pt")


app = typer.Typer(
    add_completion=False,
    help="case10 sweep evaluator: weights -> 40 episodes vs baseline_v1.",
)


def _install_weights(src: Path) -> Path:
    """Copy ``src`` to the canonical agent path so the registered agent loads it.

    Returns the canonical path. Raises if ``src`` does not exist. The file
    overwrite is intentional and matches dev/runpod promote semantics.
    """
    if not src.exists():
        raise FileNotFoundError(f"weights not found: {src}")
    CANONICAL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, CANONICAL_WEIGHTS_PATH)
    return CANONICAL_WEIGHTS_PATH


def _run_one_side(
    challenger_key: str,
    n: int,
    seed: int,
    parallel: int,
    data_root: Path,
    *,
    challenger_first: bool,
) -> list[dict[str, Any]]:
    agents = (
        (challenger_key, BASELINE) if challenger_first else (BASELINE, challenger_key)
    )
    spec = RunSpec(
        agents=agents,
        mode="1v1",
        episodes=n,
        seed=seed,
        parallel=parallel,
        save_replay=False,
        data_root=data_root,
    )
    records = run_episodes(spec)
    challenger_idx = 0 if challenger_first else 1
    out: list[dict[str, Any]] = []
    for rec in records:
        winner = rec.get("winner")
        if winner is None or winner < 0:
            outcome = "draw"
        elif int(winner) == challenger_idx:
            outcome = "win"
        else:
            outcome = "loss"
        out.append(
            {
                "outcome": outcome,
                "turns": int(rec.get("turns", 0) or 0),
                "challenger_first": challenger_first,
                "seed": rec.get("seed"),
            }
        )
    return out


def _summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rs = list(records)
    n = len(rs)
    wins = sum(1 for r in rs if r["outcome"] == "win")
    losses = sum(1 for r in rs if r["outcome"] == "loss")
    draws = sum(1 for r in rs if r["outcome"] == "draw")
    turns = [r["turns"] for r in rs if r["turns"] > 0]
    side = {"p1": [], "p2": []}
    for r in rs:
        bucket = "p1" if r["challenger_first"] else "p2"
        side[bucket].append(r)
    side_summary: dict[str, Any] = {}
    for tag, items in side.items():
        side_summary[f"challenger_{tag}"] = {
            "wins": sum(1 for r in items if r["outcome"] == "win"),
            "losses": sum(1 for r in items if r["outcome"] == "loss"),
            "draws": sum(1 for r in items if r["outcome"] == "draw"),
            "n": len(items),
        }
    return {
        "episodes": n,
        "challenger_wins": wins,
        "baseline_wins": losses,
        "draws": draws,
        "win_rate": wins / max(1, n),
        "draw_rate": draws / max(1, n),
        "mean_turns": (sum(turns) / len(turns)) if turns else 0.0,
        "side_breakdown": side_summary,
    }


@app.command()
def main(
    weights: Path = typer.Option(  # noqa: B008
        ...,
        "--weights",
        "-w",
        help="path to .pt produced by training",
    ),
    tag: str = typer.Option(..., "--tag", help="sweep tag, e.g. top80 / swap_off"),
    head_mode: str = typer.Option(
        "template_ships",
        "--head-mode",
        help="case10 head_mode the weights were trained with",
    ),
    episodes_per_side: int = typer.Option(
        20,
        "--episodes-per-side",
        "-n",
        help="seed 0..N-1 per side; default 20 -> 40 total",
    ),
    seed: int = typer.Option(0, "--seed"),
    parallel: int = typer.Option(1, "--parallel", "-p"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),  # noqa: B008
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),  # noqa: B008
) -> None:
    if head_mode not in CHALLENGER_HEAD_KEYS:
        valid = sorted(CHALLENGER_HEAD_KEYS)
        raise typer.BadParameter(
            f"unknown head_mode {head_mode!r}; expected one of {valid}"
        )
    challenger_key = CHALLENGER_HEAD_KEYS[head_mode]
    weights = weights.resolve()
    canonical = _install_weights(weights)
    logger.info("installed weights %s -> %s", weights, canonical)

    started_at = time.monotonic()
    p1_records = _run_one_side(
        challenger_key,
        episodes_per_side,
        seed,
        parallel,
        data_root,
        challenger_first=True,
    )
    p2_records = _run_one_side(
        challenger_key,
        episodes_per_side,
        seed,
        parallel,
        data_root,
        challenger_first=False,
    )
    elapsed = time.monotonic() - started_at
    summary = _summarize(p1_records + p2_records)
    summary.update(
        {
            "tag": tag,
            "weights": str(weights),
            "head_mode": head_mode,
            "challenger_key": challenger_key,
            "baseline": BASELINE,
            "episodes_per_side": episodes_per_side,
            "seed_start": seed,
            "wall_clock_sec": round(elapsed, 2),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("eval done: %s", out_path)
    typer.echo(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    app()
