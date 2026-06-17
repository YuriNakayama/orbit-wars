"""Evaluate any per-iter JAX ckpt vs a rulebase locally (requirement 1).

Takes a single `ckpt_i*.pt` (JAX npz of Equinox leaves, as saved every iter by
train_jax) or a `best.pt`, converts it to a PyTorch state_dict, points the case7
agent at it via `ORBIT_WARS_CASE7_WEIGHTS`, and runs N episodes vs a chosen
rulebase agent. Lets a mid-training checkpoint be battle-tested without touching
the canonical submit `weights.pt`.

Usage (from bot/):
    uv run python -m pipeline.reinforce.case7.evaluation.eval_ckpt_vs_rulebase \
        --ckpt data/output/models/reinforce/case7_train_jax/runs/<run>/ckpt_i010.pt \
        --baseline baseline_v8 --episodes 30 --seed 0
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import typer

from dataset.selfplay.runner import RunSpec, run_episodes
from evaluate.vs_baseline import summarize_records

from ..training.jax_to_torch import _load_jax_model_from_npz, _state_dict

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Evaluate a case7 JAX ckpt vs a rulebase.")

CHALLENGER = "rl_v7"

_CKPT_OPT = typer.Option(..., "--ckpt", help="JAX ckpt_i*.pt / best.pt (npz) path")
_BASELINE_OPT = typer.Option("baseline_v8", "--baseline", help="rulebase agent name")
_EPISODES_OPT = typer.Option(30, "--episodes", "-n")
_SEED_OPT = typer.Option(0, "--seed")
_PARALLEL_OPT = typer.Option(1, "--parallel", "-p")
_OUT_OPT = typer.Option(None, "--out")


def _convert_to_torch(ckpt: Path, out: Path) -> None:
    """Convert a JAX npz ckpt into a PyTorch state_dict at `out`."""
    model = _load_jax_model_from_npz(ckpt)
    import torch  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(model), out)


@app.command()
def main(
    ckpt: Path = _CKPT_OPT,
    baseline: str = _BASELINE_OPT,
    episodes: int = _EPISODES_OPT,
    seed: int = _SEED_OPT,
    parallel: int = _PARALLEL_OPT,
    out: Path | None = _OUT_OPT,
) -> None:
    if not ckpt.exists():
        raise typer.BadParameter(f"ckpt not found: {ckpt}")
    with tempfile.TemporaryDirectory() as tmp:
        torch_weights = Path(tmp) / "ckpt_weights.pt"
        _convert_to_torch(ckpt, torch_weights)
        os.environ["ORBIT_WARS_CASE7_WEIGHTS"] = str(torch_weights)
        spec = RunSpec(
            agents=(CHALLENGER, baseline),
            mode="1v1",
            episodes=episodes,
            seed=seed,
            parallel=parallel,
            save_replay=False,
            data_root=Path("data"),
        )
        records = run_episodes(spec)
    summary: dict[str, Any] = dict(summarize_records(records, challenger_idx=0))
    summary["ckpt"] = str(ckpt)
    summary["baseline"] = baseline
    summary["episodes"] = episodes
    summary["seed_start"] = seed
    summary["seed_end_exclusive"] = seed + episodes
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("ckpt-vs-rulebase result: %s", json.dumps(summary, default=str))
    typer.echo(json.dumps(summary, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
