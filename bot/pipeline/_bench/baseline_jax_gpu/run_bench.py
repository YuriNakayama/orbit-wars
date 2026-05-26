"""GPU bench for the rulebase/case2 JAX core port (geometry / physics).

Mirrors `pipeline/_bench/rollout_gpu/run_bench.py` so the RunPod onstart
artifact uploader picks it up unchanged (anything under `_bench/<name>_gpu/`).

What it measures
----------------
The hot scalar helpers in `baseline/core/physics.py` are called once per
(source planet, target planet, ship-count) triple inside the rule-based
agent's candidate search (`search_safe_intercept` / `aim_with_prediction`),
which on a busy board is thousands of calls per turn. We benchmark the
representative `estimate_arrival` primitive (it pulls in `safe_angle_and_distance`
-> `point_to_segment_distance` + `fleet_speed`, i.e. the whole geometry+physics
numeric chain) on a batch of `N` triples:

  - Python baseline (scalar loop) — `estimate_arrival` from `physics.py`
  - JAX (vmapped + jit) — `estimate_arrival_jax`, swept over batch size N

The comparison number that matters is **JAX-GPU per-batch wall-clock vs the
Python scalar-loop per-batch wall-clock at a realistic N**, and the vmap
scaling curve across N in {256, 1024, 4096, 16384}.

GPU only: per the skill's benchmark rule we never report a CPU substitute. On a
non-GPU host this still runs (CPU JAX) for a smoke check, but the headline
figure is taken from the RunPod GPU run (detected via `RUNPOD_POD_ID` +
`jax.devices()`).
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

import jax
import jax.numpy as jnp
import numpy as np
import typer

from pipeline.rulebase.case2.baseline.core.physics import estimate_arrival
from pipeline.rulebase.case2.baseline_jax.physics_jax import estimate_arrival_jax

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

BATCH_SIZES: tuple[int, ...] = (256, 1024, 4096, 16384)


@dataclass
class BenchResult:
    variant: str
    batch: int
    wall_seconds: float
    per_call_seconds: float
    device: str


def _run_dir() -> Path:
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _make_inputs(batch: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "sx": rng.uniform(0.0, 100.0, size=batch),
        "sy": rng.uniform(0.0, 100.0, size=batch),
        "sr": rng.uniform(1.0, 5.0, size=batch),
        "tx": rng.uniform(0.0, 100.0, size=batch),
        "ty": rng.uniform(0.0, 100.0, size=batch),
        "tr": rng.uniform(1.0, 5.0, size=batch),
        "ships": rng.integers(1, 500, size=batch),
    }


def _bench_python(batch: int, seed: int) -> BenchResult:
    data = _make_inputs(batch, seed)
    t0 = time.perf_counter()
    for i in range(batch):
        estimate_arrival(
            data["sx"][i],
            data["sy"][i],
            data["sr"][i],
            data["tx"][i],
            data["ty"][i],
            data["tr"][i],
            int(data["ships"][i]),
        )
    wall = time.perf_counter() - t0
    return BenchResult(
        variant="python",
        batch=batch,
        wall_seconds=wall,
        per_call_seconds=wall / max(1, batch),
        device="cpu",
    )


def _bench_jax(batch: int, seed: int) -> BenchResult:
    data = _make_inputs(batch, seed)
    args = (
        jnp.asarray(data["sx"], dtype=jnp.float32),
        jnp.asarray(data["sy"], dtype=jnp.float32),
        jnp.asarray(data["sr"], dtype=jnp.float32),
        jnp.asarray(data["tx"], dtype=jnp.float32),
        jnp.asarray(data["ty"], dtype=jnp.float32),
        jnp.asarray(data["tr"], dtype=jnp.float32),
        jnp.asarray(data["ships"], dtype=jnp.int32),
    )
    runner = jax.jit(jax.vmap(estimate_arrival_jax, in_axes=(0, 0, 0, 0, 0, 0, 0)))

    # Warm-up (compile cost paid here, NOT timed). Blocking on `turns` forces
    # the whole fused kernel — angle/valid are computed in the same trace.
    _, turns, _ = runner(*args)
    turns.block_until_ready()

    t0 = time.perf_counter()
    _, turns, _ = runner(*args)
    turns.block_until_ready()
    wall = time.perf_counter() - t0

    device = str(jax.devices()[0].platform)
    return BenchResult(
        variant="jax",
        batch=batch,
        wall_seconds=wall,
        per_call_seconds=wall / max(1, batch),
        device=device,
    )


def _bench_full(seed: int) -> list[BenchResult]:
    results: list[BenchResult] = []

    # Python scalar baseline + CPU-JAX smoke at every batch size.
    for n in BATCH_SIZES:
        logger.info("=== python batch=%d ===", n)
        r = _bench_python(batch=n, seed=seed)
        results.append(r)
        logger.info("result: %s", asdict(r))

        logger.info("=== jax (default backend) batch=%d ===", n)
        r = _bench_jax(batch=n, seed=seed)
        results.append(r)
        logger.info("result: %s", asdict(r))

    if not os.environ.get("RUNPOD_POD_ID"):
        logger.info("not running on RunPod; GPU figure pending a RunPod run")
        return results

    devices = jax.devices()
    logger.info("jax devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.warning("no GPU device present — skipping GPU bench")
        return results

    # On a real GPU the jax rows above already used the GPU backend, but re-run
    # explicitly tagged so the JSON makes the device unambiguous.
    for n in BATCH_SIZES:
        logger.info("=== jax gpu batch=%d ===", n)
        r = _bench_jax(batch=n, seed=seed)
        results.append(r)
        logger.info("result: %s", asdict(r))

    return results


@app.command()
def main(seed: int = typer.Option(0, help="seed")) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("jax version=%s devices=%s", jax.__version__, jax.devices())
    run_dir = _run_dir()
    logger.info("run_dir=%s", run_dir)

    started = time.perf_counter()
    results = _bench_full(seed=seed)
    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "_bench/baseline_jax_gpu",
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
