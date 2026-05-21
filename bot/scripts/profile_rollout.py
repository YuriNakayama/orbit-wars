"""Profile a single iter1 rollout to identify the wall-clock bottleneck.

Vendor sim sustains per_ep ~0.04s on RTX 4090 hosts (~0.05s on M-series).
With episodes_per_iter=16 the env should take ~0.6-0.8s — yet iter1 took
~270s/iter. The gap (267s) is presumably agent forward + PPO update, but
we want to verify with cProfile before committing to JAX env integration.

Usage::

    cd bot && uv run python scripts/profile_rollout.py --episodes 4

Outputs:
- cProfile pstats dump to stdout (top 30 by cumulative time)
- per-section wall-clock breakdown: rollout (env + forward) vs ppo update
"""

from __future__ import annotations

import cProfile
import logging
import pstats
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from torch import optim

# Mirror train.py imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.reinforce.case1.policy.model import (  # noqa: E402
    ActorCritic,
    ModelConfig,
)
from pipeline.reinforce.case1.training.env import (  # noqa: E402
    random_noop_agent,
)
from pipeline.reinforce.case1.training.ppo import (  # noqa: E402
    PPOConfig,
    ppo_update,
)
from pipeline.reinforce.case1.training.rollout import (  # noqa: E402
    RolloutWorkerPool,
    collect_rollout,
    resolve_num_workers,
)

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = typer.Option(
    Path("pipeline/reinforce/case1/configs/cpu_stable_v1.yaml"),
    "--config",
    help="Path to PPO YAML config (relative to bot/).",
)
_DEFAULT_EPISODES = typer.Option(4, help="Episodes to roll out (smaller = faster)")
_DEFAULT_NO_BC = typer.Option(
    True, help="Skip BC warm-start to isolate env/forward costs"
)
_DEFAULT_SKIP_UPDATE = typer.Option(False, help="Skip PPO update; measure rollout only")
_DEFAULT_NUM_WORKERS = typer.Option(1, help="Rollout worker count (1=serial, 0=auto)")


def _build_model(cfg: dict[str, Any]) -> ActorCritic:
    m = cfg["model"]
    model_cfg = ModelConfig(
        hidden=int(m.get("hidden", 192)),
        attn_heads=int(m.get("attn_heads", 8)),
        inducing_points=int(m.get("inducing_points", 24)),
        encoder_layers=int(m.get("encoder_layers", 4)),
        head_dropout=float(m.get("head_dropout", 0.0)),
        ship_log_std_init=float(m.get("ship_log_std_init", 0.5)),
        no_op_bias=float(m.get("no_op_bias", 8.0)),
    )
    return ActorCritic(model_cfg)


@app.command()
def main(
    config: Path = _DEFAULT_CONFIG,
    episodes: int = _DEFAULT_EPISODES,
    no_bc: bool = _DEFAULT_NO_BC,
    skip_update: bool = _DEFAULT_SKIP_UPDATE,
    num_workers: int = _DEFAULT_NUM_WORKERS,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg_path = Path(__file__).resolve().parents[1] / config
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]
    if no_bc:
        cfg["bc_warmstart"]["enabled"] = False

    logger.info(
        "config=%s episodes=%d no_bc=%s skip_update=%s num_workers=%d",
        config,
        episodes,
        no_bc,
        skip_update,
        num_workers,
    )

    torch.manual_seed(0)
    model = _build_model(cfg)
    opponent = random_noop_agent

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    ppo_cfg = PPOConfig(
        clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        value_coef=float(train_cfg.get("value_coef", 0.5)),
        entropy_coef=float(train_cfg.get("entropy_coef", 0.003)),
        kl_beta=float(train_cfg.get("kl_beta", 0.0)),  # no BC anchor
        epochs=int(train_cfg.get("ppo_epochs", 4)),
        minibatch_size=int(train_cfg.get("minibatch_size", 64)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        target_kl=None,
    )

    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    shaping_coef = float(train_cfg.get("shaping_coef", 0.001))

    # Warmup (compile-ish; PyTorch eager but caches autograd graphs). Always
    # serial — warmup is cheap and avoids pool startup overhead skewing later
    # numbers.
    _ = collect_rollout(
        model,
        opponent,
        1,
        seed=999,
        gamma=gamma,
        gae_lambda=gae_lambda,
        shaping_coef=shaping_coef,
        worker_pool=None,
    )
    logger.info("warmup done; starting profiled rollout")

    effective_workers = resolve_num_workers(num_workers, episodes)
    worker_pool: RolloutWorkerPool | None = (
        RolloutWorkerPool(model, effective_workers) if effective_workers > 1 else None
    )
    if worker_pool is not None:
        worker_pool.update_model(model)
    try:
        profiler = cProfile.Profile()
        profiler.enable()
        t0 = time.perf_counter()

        rollout_t0 = time.perf_counter()
        rollout = collect_rollout(
            model,
            opponent,
            episodes,
            seed=0,
            gamma=gamma,
            gae_lambda=gae_lambda,
            shaping_coef=shaping_coef,
            worker_pool=worker_pool,
        )
        rollout_wall = time.perf_counter() - rollout_t0
    finally:
        if worker_pool is not None:
            worker_pool.close()

    update_wall = 0.0
    if not skip_update:
        update_t0 = time.perf_counter()
        stats = ppo_update(model, None, optimizer, rollout, ppo_cfg)
        update_wall = time.perf_counter() - update_t0
        logger.info(
            "ppo_update: policy_loss=%.4f value_loss=%.4f approx_kl=%.4f",
            stats.policy_loss,
            stats.value_loss,
            stats.approx_kl,
        )

    total = time.perf_counter() - t0
    profiler.disable()

    transitions = rollout.planet_feats.shape[0]
    logger.info("=== wall-clock breakdown ===")
    logger.info(
        "rollout:    %6.2fs (%5.1f%%, %d transitions, %.3fs/ep)",
        rollout_wall,
        100 * rollout_wall / total,
        transitions,
        rollout_wall / episodes,
    )
    if not skip_update:
        logger.info(
            "ppo update: %6.2fs (%5.1f%%)", update_wall, 100 * update_wall / total
        )
    logger.info("total:      %6.2fs", total)

    # Top 30 by cumulative time
    buf = StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(30)
    print("\n=== cProfile top 30 by cumulative time ===")
    print(buf.getvalue())

    # Also top 20 by tottime to highlight hot leaf functions
    buf2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=buf2).sort_stats("tottime")
    ps2.print_stats(20)
    print("\n=== cProfile top 20 by tottime ===")
    print(buf2.getvalue())


if __name__ == "__main__":
    app()
