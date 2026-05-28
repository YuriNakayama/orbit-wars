# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""GPU bench + win-rate sanity for the case_jax vmapped self-play harness.

Mirrors `pipeline/reinforce/_bench/rollout_gpu/run_bench.py` so the RunPod
onstart artifact uploader picks it up unchanged. This is the production harness
for the loop's two goals:

  1. **Strength sanity** — jax_v4-vs-jax_v4 self-play seat0 win-rate over a batch
     (should be ~50% by symmetry; a lopsided number means the harness or agent is
     biased). The real "互角 vs case8" 300-game eval uses the same
     `run_selfplay_batch` with `agent1_fn` swapped to the Python case8 opponent
     (a follow-up; needs a host-callback or JAX case8 — out of scope here).
  2. **Speed** — wall-clock per parallel batch at batch sizes {1, 16, 64} and
     horizon 500, reporting per-game throughput and vmap batch-scaling. The whole
     point of the JAX port: `compute_actions` is ~24s/call on CPU, but vmapped
     across the batch on GPU the per-game cost collapses.

GPU-only at scale: `compute_actions`'s graph is huge, so CPU runs are confined to
a tiny smoke (batch=2, horizon=8) for wiring validation. Real numbers come from
RunPod (RUNPOD_POD_ID set + jax[cuda12] installed by onstart).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import typer

from utils.gpu_bench import diagnostic_log, run_dir

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@dataclass
class BenchResult:
    variant: str
    batch: int
    horizon: int
    wall_seconds: float
    per_game_seconds: float
    games_per_second: float
    seat0_win_rate: float
    draw_rate: float
    device: str


def _bench_batch(batch: int, horizon: int) -> BenchResult:
    import jax

    from pipeline.rulebase.case_jax.baseline_jax.selfplay_jax import (
        OUTCOME_DRAW,
        OUTCOME_SEAT0_WIN,
        run_selfplay_batch,
    )

    seeds = list(range(batch))

    # Warmup (compile cost paid here).
    out = run_selfplay_batch(seeds, horizon=horizon)
    out.outcome.block_until_ready()

    t0 = time.perf_counter()
    out = run_selfplay_batch(seeds, horizon=horizon)
    out.outcome.block_until_ready()
    wall = time.perf_counter() - t0

    outcome = out.outcome
    seat0_wins = int((outcome == OUTCOME_SEAT0_WIN).sum())
    draws = int((outcome == OUTCOME_DRAW).sum())
    device = str(jax.devices()[0].platform)
    return BenchResult(
        variant="jax",
        batch=batch,
        horizon=horizon,
        wall_seconds=wall,
        per_game_seconds=wall / max(1, batch),
        games_per_second=batch / max(1e-9, wall),
        seat0_win_rate=seat0_wins / max(1, batch),
        draw_rate=draws / max(1, batch),
        device=device,
    )


def _bench_full(batches: list[int], horizon: int) -> list[BenchResult]:
    results: list[BenchResult] = []

    if not os.environ.get("RUNPOD_POD_ID"):
        # CPU smoke only — a single tiny batch to validate wiring without
        # paying the full compile/run cost (compute_actions is ~24s/call).
        logger.info("not on RunPod; CPU smoke (batch=2 horizon=8) only")
        results.append(_bench_batch(batch=2, horizon=8))
        return results

    import jax

    devices = jax.devices()
    logger.info("jax devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.warning("no GPU device present — skipping GPU bench")
        return results

    for batch in batches:
        logger.info("=== jax gpu batch=%d horizon=%d ===", batch, horizon)
        r = _bench_batch(batch=batch, horizon=horizon)
        results.append(r)
        logger.info("result: %s", asdict(r))

    return results


@app.command()
def main(
    horizon: int = typer.Option(500, help="game horizon (turns)"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    diagnostic_log()
    run_path = run_dir()
    logger.info("run_dir=%s", run_path)

    started = time.perf_counter()
    results = _bench_full(batches=[1, 16, 64], horizon=horizon)
    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "rulebase/case_jax/_bench/selfplay_gpu",
        "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "results": [asdict(r) for r in results],
        "runtime_seconds": round(runtime, 3),
    }
    out_path = run_path / "bench_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)

    (run_path / "best.pt").write_bytes(b"")
    (run_path / "metrics.json").write_text(
        json.dumps(
            {
                "iterations_run": len(results),
                "best_win_rate": (results[-1].seat0_win_rate if results else 0.0),
                "runtime_seconds": runtime,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    sys.stdout.flush()


if __name__ == "__main__":
    app()
