"""Play ONE game of a case7 JAX ckpt vs a Python rulebase and render the replay.

Converts the JAX ckpt to torch, points the case7 agent at it via
`ORBIT_WARS_CASE7_WEIGHTS`, runs a single kaggle_environments orbit_wars match
against a chosen rulebase agent, and writes the kaggle HTML player to `--out`
(openable in a browser).

Usage (from bot/):
    uv run python -m pipeline.reinforce.case7.evaluation.replay_ckpt_vs_rulebase \
        --ckpt /tmp/h4_ckpts/ckpt_i137.pt --baseline baseline_v1 --seed 0 \
        --out /tmp/replay.html
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import typer

from dataset.selfplay.agents import resolve
from orbit_wars_sim import make_orbit_wars_env

from ..training.jax_to_torch import _load_jax_model_from_npz, _state_dict

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Render a case7 ckpt vs rulebase replay.")

_CKPT_OPT = typer.Option(..., "--ckpt", help="JAX ckpt_i*.pt / best.pt (npz)")
_BASELINE_OPT = typer.Option("baseline_v1", "--baseline", help="rulebase agent name")
_SEED_OPT = typer.Option(0, "--seed")
_OUT_OPT = typer.Option(Path("/tmp/case7_replay.html"), "--out")


def _convert_to_torch(ckpt: Path, out: Path) -> None:
    import torch  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(_load_jax_model_from_npz(ckpt)), out)


@app.command()
def main(
    ckpt: Path = _CKPT_OPT,
    baseline: str = _BASELINE_OPT,
    seed: int = _SEED_OPT,
    out: Path = _OUT_OPT,
) -> None:
    if not ckpt.exists():
        raise typer.BadParameter(f"ckpt not found: {ckpt}")
    with tempfile.TemporaryDirectory() as tmp:
        torch_weights = Path(tmp) / "ckpt_weights.pt"
        _convert_to_torch(ckpt, torch_weights)
        os.environ["ORBIT_WARS_CASE7_WEIGHTS"] = str(torch_weights)
        # rl_v7 (seat 0) = our ckpt; baseline (seat 1) = the Python rulebase.
        challenger = resolve("rl_v7")
        opponent = resolve(baseline)
        env = make_orbit_wars_env(agents=2, seed=seed)
        env.run([challenger, opponent])
        rewards = [s.get("reward") for s in env.steps[-1]]
        outcome = (
            "win" if (rewards[0] or 0) > (rewards[1] or 0)
            else "loss" if (rewards[0] or 0) < (rewards[1] or 0)
            else "draw"
        )
        html = env.render(mode="html", width=900, height=700)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
    typer.echo(f"rl_v7(ckpt) vs {baseline} (seed={seed}): {outcome} → {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
