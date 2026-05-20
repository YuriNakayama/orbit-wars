"""GPU bench for the W4 JAX rollout (end-to-end featurize + forward + env).

Mirrors `pipeline/_bench/featurizer_gpu/run_bench.py` so the RunPod
onstart artifact uploader picks it up unchanged. Measures wall-clock
for full rollouts at horizons {50, 200, 500} and episode counts {1, 16}
on:

  - PyTorch baseline (CPU) — `collect_rollout` from `rollout.py`
  - JAX (cpu wheel) — `collect_rollout_jax`
  - JAX (GPU after `uv pip install jax[cuda12]`)

The PyTorch baseline is only run with episodes_per_iter=1 to keep the
bench under the cost cap; the comparison number that matters is "JAX GPU
per-episode wall-clock at horizon=500" vs the 4.5-min/iter serial
baseline.
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
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@dataclass
class BenchResult:
    variant: str
    episodes: int
    horizon: int
    wall_seconds: float
    per_step_seconds: float
    per_ep_seconds: float
    device: str


def _run_dir() -> Path:
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _bench_jax(episodes: int, horizon: int, seed: int) -> BenchResult:
    import jax

    from pipeline.reinforce.case1.policy.model_jax import ActorCriticJax
    from pipeline.reinforce.case1.training.rollout_jax import (
        collect_rollout_jax,
    )

    model = ActorCriticJax.from_init(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(seed)

    # Warmup (compile cost paid here).
    out = collect_rollout_jax(
        model, key, episodes_per_iter=episodes, horizon=horizon, seed=seed
    )
    out.planet_feats.block_until_ready()

    t0 = time.perf_counter()
    out = collect_rollout_jax(
        model, key, episodes_per_iter=episodes, horizon=horizon, seed=seed
    )
    out.planet_feats.block_until_ready()
    wall = time.perf_counter() - t0

    device = str(jax.devices()[0].platform)
    total_steps = episodes * horizon
    return BenchResult(
        variant="jax",
        episodes=episodes,
        horizon=horizon,
        wall_seconds=wall,
        per_step_seconds=wall / max(1, total_steps),
        per_ep_seconds=wall / max(1, episodes),
        device=device,
    )


def _bench_full(horizons: list[int], seed: int) -> list[BenchResult]:
    results: list[BenchResult] = []

    # CPU JAX baseline at all (episodes, horizon) combos.
    for episodes in (1, 16):
        for h in horizons:
            logger.info("=== jax cpu episodes=%d horizon=%d ===", episodes, h)
            r = _bench_jax(episodes=episodes, horizon=h, seed=seed)
            results.append(r)
            logger.info("result: %s", asdict(r))

    if not os.environ.get("RUNPOD_POD_ID"):
        logger.info("not running on RunPod; skipping GPU bench")
        return results

    # The cuda12 plugin (jax-cuda12-plugin + jax-cuda12-pjrt) is now
    # installed by onstart via `uv sync --group cuda`, BEFORE this
    # bench process starts and BEFORE jax is imported. We can detect
    # it directly via `jax.devices()`. The earlier in-process
    # `uv pip install --reinstall jax[cuda12]` triggered SIGABRT (the
    # already-loaded jax process can't survive its jaxlib shared
    # library being replaced).
    import jax

    devices = jax.devices()
    logger.info("jax devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.warning("no GPU device present — skipping GPU bench")
        return results

    for episodes in (1, 16):
        for h in horizons:
            logger.info("=== jax gpu episodes=%d horizon=%d ===", episodes, h)
            r = _bench_jax(episodes=episodes, horizon=h, seed=seed)
            results.append(r)
            logger.info("result: %s", asdict(r))

    return results


@app.command()
def main(
    seed: int = typer.Option(0, help="seed"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_dir = _run_dir()
    logger.info("run_dir=%s", run_dir)

    started = time.perf_counter()
    results = _bench_full(horizons=[50, 200, 500], seed=seed)
    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "_bench/rollout_gpu",
        "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "results": [asdict(r) for r in results],
        "runtime_seconds": round(runtime, 3),
    }
    out_path = run_dir / "bench_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_path)

    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "iterations_run": len(results),
                "best_win_rate": 0.0,
                "runtime_seconds": runtime,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    sys.stdout.flush()


if __name__ == "__main__":
    app()
