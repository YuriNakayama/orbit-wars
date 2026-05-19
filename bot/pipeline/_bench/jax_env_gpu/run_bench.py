"""GPU benchmark for JAX env, runs as a RunPod onstart "train" stage.

Compares vendor sim vs JAX env (CPU and GPU, vmap=1/16/32). Writes a single
`bench_results.json` to the run dir which `dev/runpod pull` then fetches.

The container starts with `jax` (cpu wheel) already installed via uv. This
script installs `jax[cuda12]` on demand so the same Phase B test suite can
run against the GPU device without changing the project's CPU pyproject.

Designed to exit 0 cleanly so the onstart pipeline marks 99_done and the
artifacts uploader picks up the JSON.
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
from types import SimpleNamespace
from typing import Any

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@dataclass
class BenchResult:
    variant: str
    episodes: int
    vmap: int
    wall_seconds: float
    per_ep_seconds: float
    device: str


def _vendor_root() -> Path:
    """Path to simulator/python (added to sys.path lazily)."""
    return Path(__file__).resolve().parents[4] / "simulator" / "python"


def _ensure_vendor_on_path() -> None:
    p = str(_vendor_root())
    if p not in sys.path:
        sys.path.insert(0, p)


def _run_vendor_episode(seed: int, max_steps: int = 500) -> int:
    _ensure_vendor_on_path()
    from orbit_wars_vendor.orbit_wars import interpreter

    state = [
        SimpleNamespace(
            observation=SimpleNamespace(), action=None, status="ACTIVE", reward=0
        )
        for _ in range(2)
    ]
    env = SimpleNamespace(
        configuration=SimpleNamespace(episodeSteps=500, shipSpeed=6.0, cometSpeed=4.0),
        info={"seed": seed},
        done=False,
    )
    interpreter(state, env)
    for _ in range(max_steps):
        if state[0].status == "DONE":
            break
        for s in state:
            s.action = []
        interpreter(state, env)
    return int(getattr(state[0].observation, "step", 0))


def _bench_vendor(episodes: int, seed: int = 0) -> BenchResult:
    t0 = time.perf_counter()
    for i in range(episodes):
        _run_vendor_episode(seed + i)
    wall = time.perf_counter() - t0
    return BenchResult(
        variant="vendor",
        episodes=episodes,
        vmap=1,
        wall_seconds=wall,
        per_ep_seconds=wall / max(1, episodes),
        device="cpu",
    )


def _bench_jax(episodes: int, vmap: int, seed: int = 0) -> BenchResult:
    """JAX env with vmap=N. Uses the current default backend (cpu / gpu)."""
    import jax
    import jax.numpy as jnp

    from jax_env.reset import reset
    from jax_env.step import empty_actions, step

    device = jax.default_backend()
    actions = empty_actions()

    def body(state: Any, _: Any) -> tuple[Any, Any]:
        new_state, _r, term = step(state, actions)
        return new_state, term

    def run(state: Any) -> tuple[Any, Any]:
        final, terms = jax.lax.scan(body, state, jnp.arange(500))
        return final, terms

    runner = jax.jit(run)

    states = [reset(seed + i, num_agents=2) for i in range(max(episodes, vmap))]

    # Warmup compile.
    final, _ = runner(states[0])
    jax.block_until_ready(final.step)  # type: ignore[no-untyped-call]

    if vmap <= 1:
        t0 = time.perf_counter()
        for s in states[:episodes]:
            final, _ = runner(s)
            jax.block_until_ready(final.step)  # type: ignore[no-untyped-call]
        wall = time.perf_counter() - t0
        return BenchResult(
            variant=f"jax-vmap{vmap}",
            episodes=episodes,
            vmap=vmap,
            wall_seconds=wall,
            per_ep_seconds=wall / max(1, episodes),
            device=device,
        )

    batched_runner = jax.jit(jax.vmap(lambda s: runner(s)))
    # Trim episodes to multiple of vmap.
    eps = (episodes // vmap) * vmap
    if eps < vmap:
        eps = vmap
    batches = [
        jax.tree.map(lambda *xs: jnp.stack(xs), *states[b : b + vmap])
        for b in range(0, eps, vmap)
    ]
    # Warmup vmap.
    f0, _ = batched_runner(batches[0])
    jax.block_until_ready(f0.step)  # type: ignore[no-untyped-call]

    t0 = time.perf_counter()
    for b in batches:
        final, _ = batched_runner(b)
        jax.block_until_ready(final.step)  # type: ignore[no-untyped-call]
    wall = time.perf_counter() - t0
    return BenchResult(
        variant=f"jax-vmap{vmap}",
        episodes=eps,
        vmap=vmap,
        wall_seconds=wall,
        per_ep_seconds=wall / max(1, eps),
        device=device,
    )


def _install_cuda_jax() -> None:
    """Reinstall jax + jaxlib with cuda12 plugin in the active venv.

    Pinning is intentionally loose (`jax[cuda12]`): jax/jaxlib has minor-version
    drift on the cuda plugin and we just need *any* GPU-capable build at bench
    time. The cpu wheel already in `.venv` is replaced.
    """
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
    """Reload jax + dependent modules so cuda plugin is picked up."""
    for mod in list(sys.modules):
        if mod == "jax" or mod.startswith(("jax.", "jaxlib", "jax_env.")):
            sys.modules.pop(mod, None)
    return importlib.import_module("jax")


def _run_dir() -> Path:
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _bench_full(episodes: int, seed: int) -> list[BenchResult]:
    results: list[BenchResult] = []
    logger.info("=== vendor ===")
    results.append(_bench_vendor(episodes=episodes, seed=seed))
    logger.info("vendor: %s", asdict(results[-1]))

    logger.info("=== jax (cpu wheel) vmap=1 ===")
    results.append(_bench_jax(episodes=episodes, vmap=1, seed=seed))
    logger.info("jax cpu vmap=1: %s", asdict(results[-1]))

    logger.info("=== jax (cpu wheel) vmap=16 ===")
    results.append(_bench_jax(episodes=episodes, vmap=min(16, episodes), seed=seed))
    logger.info("jax cpu vmap=16: %s", asdict(results[-1]))

    # GPU bench is pod-only (macOS jax wheels don't have a cuda plugin and a
    # `pip --reinstall` can break the cpu install on host). Gate on RUNPOD_POD_ID
    # so local smoke runs skip the GPU section entirely.
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
        if vmap > episodes:
            continue
        logger.info("=== jax (gpu) vmap=%d ===", vmap)
        r = _bench_jax(episodes=episodes, vmap=vmap, seed=seed)
        results.append(r)
        logger.info("jax gpu vmap=%d: %s", vmap, asdict(r))

    return results


@app.command()
def main(
    episodes: int = typer.Option(64, help="episodes per variant"),
    seed: int = typer.Option(0, help="initial seed"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_dir = _run_dir()
    logger.info("run_dir=%s", run_dir)

    started = time.perf_counter()
    results = _bench_full(episodes=episodes, seed=seed)
    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "_bench/jax_env_gpu",
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

    # Onstart artifacts uploader expects best.pt + metrics.json (cf. case0 smoke).
    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "iterations_run": len(results),
                "total_episodes": sum(r.episodes for r in results),
                "best_win_rate": 0.0,
                "runtime_seconds": runtime,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    app()
