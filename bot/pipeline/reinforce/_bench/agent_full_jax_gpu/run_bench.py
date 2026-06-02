"""GPU bench: vmapped self-play throughput of the faithful JAX v1 agent.

This is the headline number that justifies the rulebase→JAX port: how many
env-steps/sec a batch of JAX-vs-JAX self-play games sustain when the opponent is
`core_jax.agent_full_jax` (vmap-friendly, no host roundtrip), vs the rate a
single Python `baseline_v1` game sustains on CPU.

The whole point of the port (see docs/plans/rulebase-to-jax): JAX wins THROUGHPUT
by running B independent games in parallel on the GPU. We sweep B and report
env-steps/sec; the headline is the RunPod-GPU figure (detected via RUNPOD_POD_ID
+ jax.devices()). On a non-GPU host this still runs (CPU JAX) as a smoke check.

Run (RunPod onstart auto-uploads anything under _bench/<name>_gpu/):
    uv run python -m pipeline.reinforce._bench.agent_full_jax_gpu.run_bench
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import typer
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import empty_actions
from orbit_wars_jax.step import step as jax_env_step

from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline.core.types import Planet  # noqa: F401
from pipeline.rulebase.case1.baseline_jax.core_jax.agent_full_jax import (
    compute_actions_jax,
)

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

BATCH_SIZES: tuple[int, ...] = (1, 8, 64, 256)
STEPS: int = 30  # env steps to time (after warmup)


@dataclass
class BenchResult:
    variant: str
    batch: int
    wall_seconds: float
    env_steps_per_sec: float
    device: str


def _run_dir() -> Path:
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _vstep_fn() -> Callable[[Any], Any]:
    def one(state: Any) -> Any:
        a0 = compute_actions_jax(state, seat=0)
        a1 = compute_actions_jax(state, seat=1)
        acts = empty_actions().at[0].set(a0).at[1].set(a1)
        ns, _r, _t = jax_env_step(state, acts)
        return ns

    return jax.jit(jax.vmap(one))


def _bench_jax(batch: int) -> BenchResult:
    states = [reset(seed=s, num_agents=2) for s in range(batch)]
    batched = jax.tree.map(lambda *xs: jnp.stack(xs, 0), *states)
    vstep = _vstep_fn()
    ns = vstep(batched)  # warmup / compile
    ns.step.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ns = vstep(ns)
    ns.step.block_until_ready()
    wall = time.perf_counter() - t0
    return BenchResult(
        variant="jax_vmap_selfplay",
        batch=batch,
        wall_seconds=wall,
        env_steps_per_sec=batch * STEPS / wall,
        device=str(jax.devices()[0]),
    )


def _bench_python_single() -> BenchResult:
    """Python baseline_v1 single-game per-turn rate (reference, CPU)."""
    from orbit_wars_jax.observation import state_to_obs
    from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT

    def pyrow(m: list[Any]) -> jax.Array:
        r = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32)
        for i, mv in enumerate(m[:MAX_LAUNCHES_PER_AGENT]):
            r = r.at[i].set(jnp.asarray([mv[0], mv[1], mv[2]], dtype=jnp.float32))
        return r

    state = reset(seed=0, num_agents=2)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        a0 = pyrow(v1_py(state_to_obs(state, player=0)))
        a1 = pyrow(v1_py(state_to_obs(state, player=1)))
        state, _r, _t = jax_env_step(state, empty_actions().at[0].set(a0).at[1].set(a1))
    wall = time.perf_counter() - t0
    return BenchResult(
        variant="python_v1_single",
        batch=1,
        wall_seconds=wall,
        env_steps_per_sec=STEPS / wall,
        device="cpu",
    )


@app.command()
def main() -> None:
    _run()


def _run() -> None:
    logging.basicConfig(level=logging.INFO)
    results: list[BenchResult] = [_bench_python_single()]
    for b in BATCH_SIZES:
        results.append(_bench_jax(b))
        logger.info("jax B=%d: %.0f env-steps/s", b, results[-1].env_steps_per_sec)

    run_dir = _run_dir()
    out = {
        "host": platform.node(),
        "jax_devices": [str(d) for d in jax.devices()],
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "steps": STEPS,
        "results": [asdict(r) for r in results],
    }
    path = run_dir / "agent_full_jax_bench.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    app()
