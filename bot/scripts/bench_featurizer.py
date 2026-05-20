"""Microbenchmark for the reinforce/case1 featurizer.

Loads the parity fixture obs (5 turns, seed 0), runs featurize() N times,
reports mean / median wall-clock per call. Used to verify the NumPy refactor
delivers a real speedup without round-tripping through full rollouts.

Usage::

    cd bot && uv run python scripts/bench_featurizer.py --iters 200
"""

from __future__ import annotations

import pickle
import statistics
import time
from pathlib import Path

import typer

from pipeline.reinforce.case1.policy.featurizer import HistoryState, featurize

app = typer.Typer(add_completion=False)

_FIXTURE_OBS = (
    Path(__file__).resolve().parents[1]
    / "tests/unit/pipeline/reinforce/case1/fixtures/featurize_parity_obs.pkl"
)
_DEFAULT_ITERS = typer.Option(200, help="Iterations per turn for averaging.")


@app.command()
def main(iters: int = _DEFAULT_ITERS) -> None:
    if not _FIXTURE_OBS.exists():
        raise SystemExit(
            f"missing fixture {_FIXTURE_OBS}; "
            f"run pytest tests/unit/pipeline/reinforce/case1/"
            f"test_featurizer_parity.py first"
        )
    with _FIXTURE_OBS.open("rb") as f:
        obs_log = pickle.load(f)

    # warmup
    for obs in obs_log:
        featurize(obs, HistoryState())

    per_turn_times: list[list[float]] = [[] for _ in obs_log]
    for _ in range(iters):
        history = HistoryState()
        for idx, obs in enumerate(obs_log):
            t0 = time.perf_counter()
            featurize(obs, history)
            per_turn_times[idx].append(time.perf_counter() - t0)

    print(f"=== featurizer bench ({iters} iters per turn) ===")
    total_means = []
    for idx, samples in enumerate(per_turn_times):
        mean_ms = 1000 * statistics.mean(samples)
        median_ms = 1000 * statistics.median(samples)
        total_means.append(mean_ms)
        print(
            f"  turn {idx}: mean={mean_ms:7.3f}ms  "
            f"median={median_ms:7.3f}ms  n={len(samples)}"
        )
    print(f"  total mean per call: {sum(total_means) / len(total_means):.3f}ms")


if __name__ == "__main__":
    app()
