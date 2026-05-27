"""PPO training loop for reinforce/case0.

Drives `collect_rollout` → `ppo_update` for `iterations` rounds against a
fixed opponent (configurable; default = built-in `random_agent` that fires
no fleets). Emits `best.pt`, `metrics.json`, and `run.json` under
`run_dir` (env var `ORBIT_WARS_RUN_DIR` honored for RunPod compatibility).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch import optim

from pipeline.reinforce.case0.policy.model import (
    ActorCritic,
    ModelConfig,
    count_parameters,
)
from pipeline.reinforce.case0.training.env import OpponentAgent, random_agent
from pipeline.reinforce.case0.training.ppo import PPOConfig, ppo_update
from pipeline.reinforce.case0.training.rollout import collect_rollout
from utils.repo_root import absolute_under_repo, find_repo_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainReport:
    iterations_run: int
    total_episodes: int
    final_policy_loss: float
    final_value_loss: float
    win_rate: float
    weights_path: Path
    runtime_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_run_dir(cfg: dict[str, Any]) -> Path:
    env_run_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    base = Path(absolute_under_repo(cfg["output"]["run_dir"], start=Path(__file__)))
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"local_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _mark_progress(step: str) -> None:
    logger.info(json.dumps({"event": "progress", "step": step}))
    if not os.environ.get("RUNPOD_POD_ID"):
        return
    try:
        from gpu.runpod.execution.progress import write_progress_marker

        write_progress_marker(step)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mark_progress failed: %s", exc)


def _resolve_opponent(name: str) -> OpponentAgent:
    if name == "random_noop":
        return random_agent
    # Optional: rule-based opponent by registry name. Kept lazy so this module
    # is importable without selfplay deps when only PPO internals are needed.
    from dataset.selfplay.agents import resolve

    agent = resolve(name)
    if callable(agent):
        return agent  # type: ignore[return-value]
    raise ValueError(f"opponent {name!r} is not a callable agent")


def _win_rate(outcomes: list[float]) -> float:
    if not outcomes:
        return 0.0
    wins = sum(1 for o in outcomes if o > 0.5)
    return wins / len(outcomes)


def train(cfg: dict[str, Any]) -> TrainReport:
    train_cfg = cfg["training"]
    seed = int(train_cfg.get("seed", 0))
    _seed_all(seed)

    run_dir = _resolve_run_dir(cfg)
    _mark_progress("10_model_init")

    model_cfg = ModelConfig(
        hidden=int(cfg["model"].get("hidden", 64)),
        ships_buckets=int(cfg["model"].get("ships_buckets", 4)),
    )
    model = ActorCritic(model_cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    ppo_cfg = PPOConfig(
        clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        value_coef=float(train_cfg.get("value_coef", 0.5)),
        entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
        epochs=int(train_cfg.get("ppo_epochs", 4)),
        minibatch_size=int(train_cfg.get("minibatch_size", 64)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
    )

    iterations = int(train_cfg["iterations"])
    episodes_per_iter = int(train_cfg["episodes_per_iter"])
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    shaping_coef = float(train_cfg.get("shaping_coef", 0.001))
    opponent_name = str(train_cfg.get("opponent", "random_noop"))
    opponent = _resolve_opponent(opponent_name)

    logger.info(
        json.dumps(
            {
                "event": "init",
                "params": count_parameters(model),
                "iterations": iterations,
                "episodes_per_iter": episodes_per_iter,
                "opponent": opponent_name,
            }
        )
    )

    _mark_progress("20_train_start")
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    total_episodes = 0
    best_win_rate = -1.0
    weights_path = run_dir / cfg["output"].get("weights_filename", "best.pt")

    for it in range(iterations):
        rollout = collect_rollout(
            model,
            opponent,
            episodes_per_iter,
            seed=seed + it * 10_000,
            gamma=gamma,
            gae_lambda=gae_lambda,
            shaping_coef=shaping_coef,
        )
        stats = ppo_update(model, optimizer, rollout, ppo_cfg)
        wr = _win_rate(rollout.episode_outcomes)
        total_episodes += episodes_per_iter
        record = {
            "iter": float(it),
            "transitions": float(rollout.planet_feats.shape[0]),
            "win_rate": float(wr),
            "policy_loss": float(stats.policy_loss),
            "value_loss": float(stats.value_loss),
            "entropy": float(stats.entropy),
            "approx_kl": float(stats.approx_kl),
            "clip_fraction": float(stats.clip_fraction),
        }
        history.append(record)
        logger.info(json.dumps({"event": "iter", **record}))
        _mark_progress(f"30_train_iter_{it:04d}")

        if wr >= best_win_rate:
            best_win_rate = wr
            torch.save(model.state_dict(), weights_path)

    runtime = time.perf_counter() - started
    _mark_progress("50_save")
    if not weights_path.exists():
        torch.save(model.state_dict(), weights_path)

    final = history[-1] if history else {"policy_loss": 0.0, "value_loss": 0.0}
    metrics = {
        "iterations_run": iterations,
        "total_episodes": total_episodes,
        "best_win_rate": best_win_rate,
        "final_win_rate": history[-1]["win_rate"] if history else 0.0,
        "final_policy_loss": final["policy_loss"],
        "final_value_loss": final["value_loss"],
        "history": history,
        "runtime_seconds": round(runtime, 3),
        "opponent": opponent_name,
    }
    metrics_path = run_dir / cfg["output"].get("metrics_filename", "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    run_meta = {
        "schema_version": 0,
        "case": "reinforce/case0",
        "purpose": "ppo_baseline",
        "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "config": cfg,
    }
    run_meta_path = run_dir / cfg["output"].get("run_meta_filename", "run.json")
    run_meta_path.write_text(json.dumps(run_meta, indent=2, default=str))

    _mark_progress("99_done")
    report = TrainReport(
        iterations_run=iterations,
        total_episodes=total_episodes,
        final_policy_loss=final["policy_loss"],
        final_value_loss=final["value_loss"],
        win_rate=best_win_rate,
        weights_path=weights_path,
        runtime_seconds=runtime,
        history=history,
    )
    logger.info(
        json.dumps(
            {
                "event": "done",
                **{k: v for k, v in asdict(report).items() if k != "history"},
            },
            default=str,
        )
    )
    return report


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        loaded = yaml.safe_load(fp)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def _resolve_config_path(config: Path) -> Path:
    if config.is_absolute():
        return config
    repo_root = find_repo_root(Path(__file__))
    bot_relative = repo_root / "bot" / config
    if bot_relative.is_file():
        return bot_relative
    return Path(absolute_under_repo(config, start=Path(__file__)))


app = typer.Typer(add_completion=False)

_CONFIG_OPTION = typer.Option(
    Path("pipeline/reinforce/case0/configs/smoke.yaml"),
    "--config",
    help="Path to PPO YAML config (relative to repo root or absolute).",
)


@app.command()
def main(config: Path = _CONFIG_OPTION) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_path = _resolve_config_path(config)
    cfg = _load_config(config_path)
    report = train(cfg)
    typer.echo(
        f"iters={report.iterations_run} win_rate={report.win_rate:.3f} "
        f"runtime={report.runtime_seconds:.1f}s weights={report.weights_path}"
    )


if __name__ == "__main__":
    app()
