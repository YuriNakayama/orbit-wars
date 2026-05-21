"""GPU benchmark for the JAX featurizer (`featurizer_jax_w1`).

Runs as a RunPod onstart "train" stage. Measures wall-clock for:
  - PyTorch baseline `featurize` (per-env CPU, vendor obs source)
  - JAX `featurize_jax_w1` (CPU wheel, vmap=1/16)
  - JAX `featurize_jax_w1` (GPU after `uv pip install jax[cuda12]`,
    vmap=1/16/32/64)

Writes `bench_results.json` to the run dir. Mirrors the structure of
`pipeline/_bench/jax_env_gpu/run_bench.py` so the onstart artifacts
uploader and `dev/runpod pull` flow already work.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import socket
import subprocess
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
    calls: int
    vmap: int
    wall_seconds: float
    per_call_seconds: float
    per_env_seconds: float
    device: str


def _run_dir() -> Path:
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _install_cuda_jax() -> None:
    logger.info("installing jax[cuda12] via uv pip…")
    proc = subprocess.run(
        ["uv", "pip", "install", "--reinstall", "jax[cuda12]"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error(
            "uv pip install jax[cuda12] failed: %s", (proc.stderr or "")[-500:]
        )
        proc.check_returncode()
    logger.info("jax[cuda12] install OK")


def _reload_jax() -> Any:
    """Reload jax so cuda plugin is picked up. Also reload jax_env / our
    featurizer_jax module since they import jax at module-load time."""
    for mod in list(sys.modules):
        if (
            mod == "jax"
            or mod.startswith(("jax.", "jaxlib", "jax_env"))
            or mod.startswith("pipeline.reinforce.case1.policy.featurizer_jax")
        ):
            sys.modules.pop(mod, None)
    return importlib.import_module("jax")


def _bench_pytorch(calls: int, seed: int) -> BenchResult:
    """PyTorch baseline: warm up + measure mean per-call wall-clock."""
    # Local imports so this module stays import-light.
    from jax_env.observation import state_to_obs
    from jax_env.reset import reset
    from pipeline.reinforce.case1.policy.featurizer import (
        HistoryState,
        featurize,
    )

    state = reset(seed=seed, num_agents=2)
    obs = state_to_obs(state, player=0)
    # warmup
    for _ in range(3):
        featurize(obs, history=HistoryState())
    t0 = time.perf_counter()
    for _ in range(calls):
        featurize(obs, history=HistoryState())
    wall = time.perf_counter() - t0
    return BenchResult(
        variant="pytorch",
        calls=calls,
        vmap=1,
        wall_seconds=wall,
        per_call_seconds=wall / max(1, calls),
        per_env_seconds=wall / max(1, calls),
        device="cpu",
    )


def _bench_jax(calls: int, vmap: int, seed: int) -> BenchResult:
    """JAX featurizer bench at the given vmap batch size."""
    import jax
    import jax.numpy as jnp

    # Reload so we pick up cuda jax if it was just installed.
    from jax_env.reset import reset
    from pipeline.reinforce.case1.policy.featurizer_jax import (
        featurize_jax_w1,
        init_history_jax,
    )

    state = reset(seed=seed, num_agents=2)
    hist = init_history_jax()
    if vmap == 1:
        jit_fn = jax.jit(featurize_jax_w1, static_argnums=(1,))
        call_state = state
        call_hist = hist
    else:
        jit_fn = jax.jit(
            jax.vmap(featurize_jax_w1, in_axes=(0, None, 0)),
            static_argnums=(1,),
        )
        call_state = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (vmap,) + x.shape), state
        )
        call_hist = jax.tree.map(lambda x: jnp.broadcast_to(x, (vmap,) + x.shape), hist)

    # Warmup (compile + a few runs).
    for _ in range(3):
        out = jit_fn(call_state, 0, call_hist)
        out.planet_feats.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(calls):
        out = jit_fn(call_state, 0, call_hist)
        out.planet_feats.block_until_ready()
    wall = time.perf_counter() - t0
    device = str(jax.devices()[0].platform)
    return BenchResult(
        variant=f"jax-vmap{vmap}",
        calls=calls,
        vmap=vmap,
        wall_seconds=wall,
        per_call_seconds=wall / max(1, calls),
        per_env_seconds=wall / max(1, calls * vmap),
        device=device,
    )


def _bench_full(calls: int, seed: int) -> list[BenchResult]:
    results: list[BenchResult] = []
    logger.info("=== pytorch baseline ===")
    results.append(_bench_pytorch(calls=calls, seed=seed))
    logger.info("pytorch: %s", asdict(results[-1]))

    for vmap in (1, 16):
        logger.info("=== jax (cpu wheel) vmap=%d ===", vmap)
        results.append(_bench_jax(calls=calls, vmap=vmap, seed=seed))
        logger.info("jax cpu vmap=%d: %s", vmap, asdict(results[-1]))

    if not os.environ.get("RUNPOD_POD_ID"):
        logger.info("not running on RunPod; skipping jax[cuda12] install + GPU bench")
        return results

    try:
        _install_cuda_jax()
    except subprocess.CalledProcessError as exc:
        logger.warning("jax[cuda12] install failed, skipping GPU bench: %s", exc)
        return results

    jax = _reload_jax()
    devices = jax.devices()
    logger.info("jax devices after cuda install: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.warning("no GPU device after install — skipping GPU bench")
        return results

    for vmap in (1, 16, 32, 64):
        logger.info("=== jax (gpu) vmap=%d ===", vmap)
        r = _bench_jax(calls=calls, vmap=vmap, seed=seed)
        results.append(r)
        logger.info("jax gpu vmap=%d: %s", vmap, asdict(r))

    return results


@app.command()
def main(
    calls: int = typer.Option(100, help="calls per variant"),
    seed: int = typer.Option(0, help="initial seed"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_dir = _run_dir()
    logger.info("run_dir=%s", run_dir)

    started = time.perf_counter()
    results = _bench_full(calls=calls, seed=seed)
    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "_bench/featurizer_gpu",
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

    # Onstart artifacts uploader expects best.pt + metrics.json.
    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "iterations_run": len(results),
                "total_calls": sum(r.calls for r in results),
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
