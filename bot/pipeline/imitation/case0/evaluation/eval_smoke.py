"""Single-match smoke evaluator for imitation/case0.

Runs one ``env.run`` between the case0 NO_OP agent and a built-in random
opponent. The result content is irrelevant — we only care that the agent
loads, packages, and runs without raising. Used as a final RunPod stage
before declaring a green E2E run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import typer
from kaggle_environments import make

from pipeline.imitation.case0.policy.agent import agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmokeResult:
    completed: bool
    num_steps: int


def run_smoke() -> SmokeResult:
    env = make("orbit_wars", debug=False)
    env.run([agent, "random"])
    return SmokeResult(completed=True, num_steps=len(env.steps))


app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_smoke()
    typer.echo(json.dumps(asdict(result)))


if __name__ == "__main__":
    app()
