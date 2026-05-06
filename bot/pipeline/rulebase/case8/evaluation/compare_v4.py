"""Head-to-head evaluation: baseline_v8 (case8 beam) vs baseline_v4 (production).

Wraps `python -m dataset run` so the canonical 300-game (seat 入替 150x2) protocol
from `docs/experiment/rulebase/20260504_case8_multistep_beam/plan.md` lives next
to the case it evaluates. For ad-hoc smoke runs, calling `python -m dataset run`
directly is equivalent — this script just pins the seed/parallel/episode defaults.

Usage (from `bot/`):

    uv run python -m pipeline.rulebase.case8.evaluation.compare_v4
    uv run python -m pipeline.rulebase.case8.evaluation.compare_v4 --episodes 50

Each invocation runs both seats sequentially.
"""

from __future__ import annotations

import subprocess
import sys

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _run_seat(agents: str, episodes: int, seed: int, parallel: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "dataset",
        "run",
        "--agents",
        agents,
        "--mode",
        "1v1",
        "-n",
        str(episodes),
        "--seed",
        str(seed),
        "--parallel",
        str(parallel),
        "--no-save-replay",
    ]
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


@app.callback()
def main(
    episodes: int = typer.Option(150, "--episodes", "-n", help="Episodes per seat."),
    seed_a: int = typer.Option(50000, "--seed-a", help="Seat A base seed (v4 first)."),
    seed_b: int = typer.Option(50500, "--seed-b", help="Seat B base seed (v8 first)."),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Worker processes."),
) -> None:
    """Run case8 vs case4 in both seats and print summaries."""
    rc_a = _run_seat("baseline_v4,baseline_v8", episodes, seed_a, parallel)
    rc_b = _run_seat("baseline_v8,baseline_v4", episodes, seed_b, parallel)
    if rc_a or rc_b:
        sys.exit(rc_a or rc_b)


if __name__ == "__main__":
    app()
