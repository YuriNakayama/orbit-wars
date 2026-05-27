# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""GPU bench for the case2 intercept hot path (the real act() bottleneck).

Compares, on the SAME (src x target) grid:
  - Python: the nested `for src: for target:` loop calling aim_with_prediction
  - JAX:    one vmap over all pairs calling aim_with_prediction_jax

`aim_with_prediction` is the per-pair solver every mission generator invokes, so
this grid is exactly what `act()` does each turn. Per-pair Python wall-clock vs
JAX-vmap per-pair wall-clock (warm-up excluded, block_until_ready) is the number
that says whether porting the bottleneck actually speeds the agent up.

Runs grids sized {8x8, 16x16, 24x24} (planets x targets) — realistic Orbit Wars
boards have up to ~48 planets, so 24x24=576 pairs/turn is a plausible upper end.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import typer

from pipeline.rulebase.case2.baseline.core.config import HORIZON
from pipeline.rulebase.case2.baseline.core.physics import aim_with_prediction
from pipeline.rulebase.case2.baseline.core.types import Planet
from pipeline.rulebase.case2.baseline_jax.aim_jax import (
    MAX_COMET_PATH_LEN,
    aim_with_prediction_jax,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

GRIDS = [8, 16, 24]
ANG_VEL = 0.02


def _make_planets(n: int, seed: int) -> list[Planet]:
    import random

    rng = random.Random(seed)
    return [
        Planet(i, 1, rng.uniform(15, 85), rng.uniform(15, 85),
               rng.uniform(1.0, 3.0), rng.randint(1, 400), 1)
        for i in range(n)
    ]


def _python_grid(srcs: list[Planet], tgts: list[Planet]) -> int:
    hits = 0
    for s in srcs:
        ibid = {t.id: t for t in tgts}
        ibid[s.id] = s
        for t in tgts:
            r = aim_with_prediction(s, t, s.ships, ibid, ANG_VEL, [], set())
            if r is not None:
                hits += 1
    return hits


def _build_jax_arrays(
    srcs: list[Planet], tgts: list[Planet]
) -> tuple[jax.Array, ...]:
    sx = jnp.array([[s.x] for s in srcs], dtype=jnp.float32)
    sy = jnp.array([[s.y] for s in srcs], dtype=jnp.float32)
    sr = jnp.array([[s.radius] for s in srcs], dtype=jnp.float32)
    ships = jnp.array([[s.ships] for s in srcs], dtype=jnp.int32)
    tx = jnp.array([[t.x for t in tgts]], dtype=jnp.float32)
    ty = jnp.array([[t.y for t in tgts]], dtype=jnp.float32)
    trr = jnp.array([[t.radius for t in tgts]], dtype=jnp.float32)
    ns, nt = len(srcs), len(tgts)
    sx, sy, sr, ships = (jnp.broadcast_to(a, (ns, nt)) for a in (sx, sy, sr, ships))
    tx, ty, trr = (jnp.broadcast_to(a, (ns, nt)) for a in (tx, ty, trr))
    return sx, sy, sr, tx, ty, trr, ships


def _jax_runner() -> jax.stages.Wrapped:
    # Bench grid is all orbital targets -> path_len=0 (JAX orbital branch).
    no_comet = jnp.zeros((MAX_COMET_PATH_LEN, 2), dtype=jnp.float32)
    grid_fn = jax.vmap(
        jax.vmap(
            lambda sx, sy, sr, tx, ty, tr, sh: aim_with_prediction_jax(
                sx, sy, sr, tx, ty, tx, ty, tr, tr, sh,
                jnp.float32(ANG_VEL), jnp.int32(HORIZON),
                no_comet, jnp.int32(0), jnp.int32(0),
            )
        )
    )
    return jax.jit(grid_fn)


@app.command()
def main(seed: int = typer.Option(0)) -> None:
    logger.info("jax version=%s devices=%s", jax.__version__, jax.devices())
    runner = _jax_runner()
    results = []
    for n in GRIDS:
        srcs = _make_planets(n, seed)
        tgts = _make_planets(n, seed + 1000)
        pairs = n * n

        # Python: time the whole grid, report per-pair.
        t0 = time.perf_counter()
        _python_grid(srcs, tgts)
        py_wall = time.perf_counter() - t0

        arrs = _build_jax_arrays(srcs, tgts)
        # Warm-up (compile) — not timed.
        out = runner(*arrs)
        out[0].block_until_ready()  # out = (angle, turns, ix, iy, valid) arrays
        t0 = time.perf_counter()
        out = runner(*arrs)
        out[0].block_until_ready()  # out = (angle, turns, ix, iy, valid) arrays
        jax_wall = time.perf_counter() - t0

        dev = jax.devices()[0].platform
        rec = {
            "grid": f"{n}x{n}",
            "pairs": pairs,
            "python_wall_s": py_wall,
            "python_per_pair_s": py_wall / pairs,
            "jax_wall_s": jax_wall,
            "jax_per_pair_s": jax_wall / pairs,
            "speedup": py_wall / jax_wall,
            "device": dev,
        }
        results.append(rec)
        logger.info("result: %s", rec)

    # --- act()-level bench: real agent turn time, Python vs JAX backend ---
    # This is the agent-level number (not the isolated grid): how long one
    # `agent(obs)` call takes when the hot path runs Python vs JAX, on obs drawn
    # from the JAX env. The grid bench above is the pure solver speedup; this
    # shows what the agent actually gains per turn.
    act_results = _bench_act(seed)
    results.append({"act_level": act_results})
    for r in act_results:
        logger.info("act result: %s", r)

    run_dir = Path(
        "data/output/models/reinforce/aim_jax_gpu/runs"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "bench_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_path)
    if jax.devices()[0].platform != "gpu":
        logger.info("not on GPU; the headline GPU number is pending a RunPod run")


def _bench_act(seed: int) -> list[dict[str, object]]:
    """Time agent(obs) per turn under Python vs JAX aim backend, on jax_env obs."""
    import importlib
    import os
    import statistics

    from orbit_wars_jax.observation import state_to_obs
    from orbit_wars_jax.reset import reset
    from orbit_wars_jax.step import empty_actions, step

    agent_mod = importlib.import_module("pipeline.rulebase.case2.baseline.agent")

    state = reset(seed=seed, num_agents=2)
    obs_snapshots = []
    for _ in range(10):
        obs_snapshots.append(state_to_obs(state, player=0))
        state, _r, term = step(state, empty_actions())
        if bool(term):
            break

    def time_backend(backend: str) -> float:
        os.environ["ORBIT_WARS_AIM_BACKEND"] = backend
        # warm-up (jit compile for jax) — not timed
        setattr(agent_mod, "_OM_STATE", agent_mod.om.OMState())
        agent_mod.agent(obs_snapshots[0])
        per_turn = []
        for obs in obs_snapshots:
            setattr(agent_mod, "_OM_STATE", agent_mod.om.OMState())
            t0 = time.perf_counter()
            agent_mod.agent(obs)
            per_turn.append(time.perf_counter() - t0)
        return statistics.median(per_turn)

    py_t = time_backend("python")
    jax_t = time_backend("jax")
    os.environ["ORBIT_WARS_AIM_BACKEND"] = "python"
    return [
        {
            "metric": "act_median_turn_s",
            "python_s": py_t,
            "jax_s": jax_t,
            "speedup": py_t / jax_t if jax_t > 0 else 0.0,
            "device": jax.devices()[0].platform,
        }
    ]


if __name__ == "__main__":
    app()
