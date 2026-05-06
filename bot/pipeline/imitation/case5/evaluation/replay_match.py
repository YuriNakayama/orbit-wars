"""Replay a single il_v5 vs baseline_v1 match for a given seed with save_replay=True.

Uses the canonical selfplay runner (same code path as eval_vs_baseline) so the
outcome matches the original 100-game batch. Writes a JSON trace summary.

Usage:
    uv run python -m pipeline.imitation.case5.evaluation.replay_match \\
        --seed 19 --label win
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import typer

from dataset.selfplay.runner import RunSpec, run_episodes

DEFAULT_OUT_DIR = Path("data/output/experiment/imitation_case5_replays")

ReplayLabel = Literal["win", "loss"]


@dataclass(frozen=True)
class ReplayConfig:
    seed: int
    label: ReplayLabel
    data_root: Path
    out_dir: Path = DEFAULT_OUT_DIR


def _summarize_replay(replay_path: Path) -> dict[str, Any]:
    data = json.loads(replay_path.read_text())
    steps = data.get("steps") or []
    if not steps:
        return {"empty": True}

    il_act: Counter[int] = Counter()
    base_act: Counter[int] = Counter()
    rows: list[tuple[int, int, int, int, int]] = []
    for t, step in enumerate(steps):
        if t == 0:
            continue
        il_entry, base_entry = step[0], step[1]
        il_a = il_entry.get("action") or []
        base_a = base_entry.get("action") or []
        il_act[len(il_a)] += 1
        base_act[len(base_a)] += 1
        if t % 20 == 0 or t == len(steps) - 1:
            obs = il_entry.get("observation") or {}
            planets = obs.get("planets") or []
            il_ships = sum(int(p[5]) for p in planets if int(p[1]) == 0)
            base_ships = sum(int(p[5]) for p in planets if int(p[1]) == 1)
            il_p = sum(1 for p in planets if int(p[1]) == 0)
            base_p = sum(1 for p in planets if int(p[1]) == 1)
            rows.append((t, il_ships, base_ships, il_p, base_p))

    return {
        "actions_per_turn_il": dict(sorted(il_act.items())),
        "actions_per_turn_base": dict(sorted(base_act.items())),
        "trace": [
            {
                "turn": t,
                "il_ships": a,
                "base_ships": b,
                "il_planets": c,
                "base_planets": d,
            }
            for t, a, b, c, d in rows
        ],
    }


def run(cfg: ReplayConfig) -> dict[str, Any]:
    spec = RunSpec(
        agents=("il_v1", "baseline_v1"),
        mode="1v1",
        episodes=1,
        seed=cfg.seed,
        parallel=1,
        save_replay=True,
        data_root=cfg.data_root,
    )
    records = run_episodes(spec)
    rec = records[0]

    summary: dict[str, Any] = {
        "seed": cfg.seed,
        "label": cfg.label,
        "match_id": rec.match_id,
        "winner_idx": rec.winner,
        "draw": rec.draw,
        "turns": rec.turns,
        "scores": list(rec.agent_scores),
        "timings_p50": [
            rec.agent_timings[i].p50 for i in range(len(rec.agent_timings))
        ],
    }

    replay_path = Path(rec.replay_path) if rec.replay_path else None
    if replay_path and replay_path.exists():
        summary["replay_path"] = str(replay_path)
        summary.update(_summarize_replay(replay_path))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / f"iter9_{cfg.label}_seed{cfg.seed}.json"
    out.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(out)
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    seed: int = typer.Option(..., "--seed", "-s"),
    label: str = typer.Option(..., "--label", "-l", help="win or loss"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),  # noqa: B008
) -> None:
    if label not in ("win", "loss"):
        raise typer.BadParameter("label must be 'win' or 'loss'")
    cfg = ReplayConfig(seed=seed, label=label, data_root=data_root)  # type: ignore[arg-type]
    summary = run(cfg)
    typer.echo(f"=== {label.upper()} (seed={seed}) ===")
    typer.echo(f"match_id   : {summary['match_id']}")
    typer.echo(
        f"winner idx : {summary['winner_idx']}  draw={summary['draw']}  "
        f"turns={summary['turns']}"
    )
    typer.echo(f"scores     : {summary['scores']}")
    typer.echo(f"saved      : {summary['summary_path']}")


if __name__ == "__main__":
    app()
