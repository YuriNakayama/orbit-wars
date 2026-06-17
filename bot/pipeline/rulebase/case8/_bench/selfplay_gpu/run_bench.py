# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""GPU bench + win-rate sanity for the case8 vmapped self-play harness.

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

    from pipeline.rulebase.case8.baseline_jax.selfplay_jax import (
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


def _vs_v8_winrate(games: int, horizon: int) -> dict[str, Any]:
    """jax_v4 (seat0) vs Python case8.baseline.agent (seat1) via host callback.

    The gold-standard "is jax_v4 losing to the original v8" check. Uses
    `jax.pure_callback` to invoke the Python v8 agent for seat 1 each turn;
    this forces a host roundtrip per game per turn so vmap parallelism is
    largely lost, but per-game throughput stays acceptable for win-rate eval.
    """
    from pipeline.rulebase.case8.baseline_jax.selfplay_jax import (
        OUTCOME_DRAW,
        OUTCOME_SEAT0_WIN,
        OUTCOME_SEAT1_WIN,
        jax_v4_action_fn,
        python_v8_action_fn,
        run_selfplay_batch,
    )

    seeds = list(range(games))
    a0 = jax_v4_action_fn(0)
    a1 = python_v8_action_fn(1)
    t0 = time.perf_counter()
    out = run_selfplay_batch(seeds, horizon=horizon, agent0_fn=a0, agent1_fn=a1)
    out.outcome.block_until_ready()
    wall = time.perf_counter() - t0
    oc = [int(x) for x in out.outcome]
    s0p = [int(x) for x in out.seat0_planets]
    s1p = [int(x) for x in out.seat1_planets]
    s0s = [int(x) for x in out.seat0_ships]
    s1s = [int(x) for x in out.seat1_ships]
    planets_jax_ahead = sum(1 for a, b in zip(s0p, s1p, strict=True) if a > b)
    planets_v8_ahead = sum(1 for a, b in zip(s0p, s1p, strict=True) if a < b)
    ships_jax_ahead = sum(1 for a, b in zip(s0s, s1s, strict=True) if a > b)
    ships_v8_ahead = sum(1 for a, b in zip(s0s, s1s, strict=True) if a < b)
    return {
        "match": "jax_v4 (seat0) vs python_v8 (seat1)",
        "games": games,
        "horizon": horizon,
        "jax_win": oc.count(OUTCOME_SEAT0_WIN),
        "v8_win": oc.count(OUTCOME_SEAT1_WIN),
        "draw": oc.count(OUTCOME_DRAW),
        "terminated": int(sum(bool(x) for x in out.terminated)),
        "wall_seconds": round(wall, 1),
        "outcomes": oc,
        "planets_jax_ahead": planets_jax_ahead,
        "planets_v8_ahead": planets_v8_ahead,
        "ships_jax_ahead": ships_jax_ahead,
        "ships_v8_ahead": ships_v8_ahead,
        "jax_planets": s0p,
        "v8_planets": s1p,
        "jax_ships": s0s,
        "v8_ships": s1s,
    }


def _selfplay_winrate(games: int, horizon: int) -> dict[str, Any]:
    """jax_v4 self-play win-rate over `games` (short horizon) — non-degenerate check.

    Confirms the agent plays to a spread of outcomes (not all-draw / degenerate)
    and is ~50% by symmetry. NOT a vs-Python-v8 eval (that needs a hybrid harness
    with v8 in the opponent seat — a follow-up). The game-loop compile is heavy
    but tractable on GPU at modest horizon.

    Also reports per-game final_rewards so that even when horizon is too short to
    reach termination (all DRAW outcomes), the reward sign indicates which seat
    was advantaged at the cutoff — the "全敗してないか" check.
    """
    from pipeline.rulebase.case8.baseline_jax.selfplay_jax import (
        OUTCOME_DRAW,
        OUTCOME_SEAT0_WIN,
        OUTCOME_SEAT1_WIN,
        run_selfplay_batch,
    )

    seeds = list(range(games))
    t0 = time.perf_counter()
    out = run_selfplay_batch(seeds, horizon=horizon)
    out.outcome.block_until_ready()
    wall = time.perf_counter() - t0
    oc = [int(x) for x in out.outcome]
    # Per-game board summary at end of scan — useful when horizon cuts off
    # before termination (env reward is 0 unless terminated).
    s0p = [int(x) for x in out.seat0_planets]
    s1p = [int(x) for x in out.seat1_planets]
    s0s = [int(x) for x in out.seat0_ships]
    s1s = [int(x) for x in out.seat1_ships]
    s0r = [int(x) for x in out.seat0_prod]
    s1r = [int(x) for x in out.seat1_prod]
    # "Advantage" = more planets at the cutoff; tie-break on ships.
    planets_seat0_ahead = sum(1 for a, b in zip(s0p, s1p, strict=True) if a > b)
    planets_seat1_ahead = sum(1 for a, b in zip(s0p, s1p, strict=True) if a < b)
    ships_seat0_ahead = sum(1 for a, b in zip(s0s, s1s, strict=True) if a > b)
    ships_seat1_ahead = sum(1 for a, b in zip(s0s, s1s, strict=True) if a < b)
    return {
        "games": games,
        "horizon": horizon,
        "seat0_win": oc.count(OUTCOME_SEAT0_WIN),
        "seat1_win": oc.count(OUTCOME_SEAT1_WIN),
        "draw": oc.count(OUTCOME_DRAW),
        "terminated": int(sum(bool(x) for x in out.terminated)),
        "wall_seconds": round(wall, 1),
        "outcomes": oc,
        # Board-state advantage at cutoff (= the meaningful "is jax_v4 losing
        # everywhere" check when reward is 0).
        "planets_seat0_ahead": planets_seat0_ahead,
        "planets_seat1_ahead": planets_seat1_ahead,
        "ships_seat0_ahead": ships_seat0_ahead,
        "ships_seat1_ahead": ships_seat1_ahead,
        "seat0_planets": s0p,
        "seat1_planets": s1p,
        "seat0_ships": s0s,
        "seat1_ships": s1s,
        "seat0_prod": s0r,
        "seat1_prod": s1r,
    }


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
        "",
        help=(
            "comma-separated batch sizes for the throughput stage; empty (default) "
            "skips it. Throughput numbers are already on file (RTX 4090: batch=256 "
            "→ 120ms/turn, 8.3 turns/s, 267x scaling). Override to re-measure."
        ),
    ),
    winrate_games: int = typer.Option(
        16, help="jax_v4 self-play games for the non-degenerate win-rate check"
    ),
    winrate_horizon: int = typer.Option(
        50,
        help=(
            "horizon for the self-play win-rate games. 50 compiles in ~50min on "
            "RTX 4090; 200+ hangs in XLA compile. The bench reports per-game "
            "final_rewards so the seat0/seat1 advantage is observable even when "
            "horizon cuts off before termination."
        ),
    ),
    skip_winrate: bool = typer.Option(
        True,
        help=(
            "skip the jax_v4 self-play win-rate stage (already captured: "
            "non-degenerate, planets/ships ~50/50, see prior run 20260601-075234)"
        ),
    ),
    vs_v8: bool = typer.Option(
        True,
        help=(
            "run jax_v4 vs Python case8 baseline (hybrid host callback for "
            "seat1). The gold-standard 'is jax_v4 losing to the original' check"
        ),
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
    results = _bench_full(batches=batch_list) if batch_list else []

    # Non-degenerate self-play win-rate check (RunPod GPU only; game-loop compile
    # is intractable on CPU). Confirms jax_v4 plays to a spread of outcomes
    # (not all-draw / degenerate) and is ~50% by symmetry.
    winrate: dict[str, Any] | None = None
    vs_v8_result: dict[str, Any] | None = None
    if os.environ.get("RUNPOD_POD_ID") and not skip_winrate:
        logger.info(
            "=== self-play win-rate games=%d horizon=%d ===",
            winrate_games,
            winrate_horizon,
        )
        winrate = _selfplay_winrate(winrate_games, winrate_horizon)
        logger.info("winrate: %s", winrate)

    if os.environ.get("RUNPOD_POD_ID") and vs_v8:
        logger.info(
            "=== jax_v4 vs python_v8 games=%d horizon=%d ===",
            winrate_games,
            winrate_horizon,
        )
        vs_v8_result = _vs_v8_winrate(winrate_games, winrate_horizon)
        logger.info("vs_v8: %s", vs_v8_result)

    runtime = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": 0,
        "case": "rulebase/case8/_bench/selfplay_gpu",
        "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "results": [asdict(r) for r in results],
        "selfplay_winrate": winrate,
        "vs_v8_winrate": vs_v8_result,
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
