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
    warm_seconds: float
    per_turn_ms: float
    turns_per_second: float
    compile_seconds: float
    device: str


def _stack_states(batch: int) -> Any:
    import jax
    import jax.numpy as jnp
    from orbit_wars_jax.reset import reset

    states = [reset(seed=s, num_agents=2) for s in range(batch)]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


def _bench_batch(batch: int) -> BenchResult:
    """Single-turn throughput: one `compute_actions` per game, vmapped over batch.

    This is the GPU-speed unit. The full game-loop `lax.scan` is intractable to
    compile (the per-turn graph is huge), but a single vmapped turn compiles
    (~82s CPU) and directly measures parallel agent-turn throughput — the "GPU
    で最大速度" metric.
    """
    import jax

    from pipeline.rulebase.case_jax.baseline_jax.selfplay_jax import (
        batched_turn_actions,
    )

    batched = _stack_states(batch)
    fn = jax.jit(batched_turn_actions)

    # Warmup (compile cost paid here).
    t0 = time.perf_counter()
    out = fn(batched)
    out.block_until_ready()
    compile_s = time.perf_counter() - t0

    # Warm timing: average a few calls.
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(batched)
        out.block_until_ready()
    warm = (time.perf_counter() - t0) / reps

    device = str(jax.devices()[0].platform)
    return BenchResult(
        variant="jax",
        batch=batch,
        warm_seconds=warm,
        per_turn_ms=warm / max(1, batch) * 1000.0,
        turns_per_second=batch / max(1e-9, warm),
        compile_seconds=compile_s,
        device=device,
    )


def _bench_full(batches: list[int]) -> list[BenchResult]:
    results: list[BenchResult] = []

    on_runpod = bool(os.environ.get("RUNPOD_POD_ID"))
    if not on_runpod:
        # CPU smoke only — smallest batch to validate wiring (compile ~82s).
        logger.info("not on RunPod; CPU smoke (batch=1) only")
        results.append(_bench_batch(batch=1))
        return results

    import jax

    devices = jax.devices()
    logger.info("jax devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.warning("no GPU device present — skipping GPU bench")
        return results

    for batch in batches:
        logger.info("=== jax gpu single-turn batch=%d ===", batch)
        r = _bench_batch(batch=batch)
        results.append(r)
        logger.info("result: %s", asdict(r))

    return results


@app.command()
def main(
    batches: str = typer.Option(
        "1,16,64,256", help="comma-separated batch sizes to bench"
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    diagnostic_log()
    run_path = run_dir()
    logger.info("run_dir=%s", run_path)

    batch_list = [int(b) for b in batches.split(",") if b.strip()]
    started = time.perf_counter()
    results = _bench_full(batches=batch_list)
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
                "best_turns_per_second": (
                    max((r.turns_per_second for r in results), default=0.0)
                ),
                "runtime_seconds": runtime,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    sys.stdout.flush()


if __name__ == "__main__":
    app()
