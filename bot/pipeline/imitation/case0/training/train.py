"""Minimal training loop for imitation/case0 — RunPod E2E smoke only.

Pipeline:
  1) load smoke YAML config
  2) seed torch / numpy / random
  3) generate dummy parquet (no DVC dependency)
  4) build TinyMLP + AdamW
  5) train 1 epoch with `max_steps` cap; emit JSON-line metrics per N steps
  6) save best.pt / metrics.json / run.json under run_dir

The structure mirrors `pipeline.imitation.case1.training.train` so the
RunPod onstart template can drive both cases through the same harness.
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
from torch import nn, optim
from torch.utils.data import DataLoader

from pipeline.imitation.case0.policy.model import ModelConfig, TinyMLP
from pipeline.imitation.case0.training.dataset import CaseZeroDataset
from pipeline.imitation.case0.training.dummy_data import (
    DummyDatasetSpec,
    sha256_of_file,
    write,
)
from utils.repo_root import absolute_under_repo, find_repo_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainReport:
    epochs_run: int
    steps_run: int
    final_loss: float
    weights_path: Path
    device: str
    runtime_seconds: float
    max_steps_hit: bool
    loss_history: list[float] = field(default_factory=list)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but cuda is not available")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device: {name!r}")


def _resolve_run_dir(cfg: dict[str, Any]) -> Path:
    """Honor ORBIT_WARS_RUN_DIR (set by RunPod onstart) or fall back to a
    timestamped local dir under ``output.run_dir``.
    """
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
    """Best-effort progress marker.

    Calls runpod_io.execution.progress.write_progress_marker when running on a pod
    (RUNPOD_POD_ID env set). Locally we just log.
    """
    logger.info(json.dumps({"event": "progress", "step": step}))
    if not os.environ.get("RUNPOD_POD_ID"):
        return
    try:
        from runpod_io.execution.progress import write_progress_marker

        write_progress_marker(step)
    except Exception as exc:  # noqa: BLE001 — never let logging break training
        logger.warning("mark_progress failed: %s", exc)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    max_steps: int,
    log_every_n_steps: int,
    epoch_index: int,
) -> tuple[int, list[float], bool]:
    model.train()
    losses: list[float] = []
    step = 0
    max_hit = False
    start = time.perf_counter()
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        losses.append(loss_value)
        step += 1
        if step % log_every_n_steps == 0 or step == max_steps:
            elapsed = time.perf_counter() - start
            logger.info(
                json.dumps(
                    {
                        "event": "step",
                        "epoch": epoch_index,
                        "step": step,
                        "max_steps": max_steps,
                        "loss": round(loss_value, 6),
                        "elapsed_s": round(elapsed, 3),
                    }
                )
            )
            _mark_progress(f"30_train_step_{step:04d}")
        if step >= max_steps:
            max_hit = True
            break
    return step, losses, max_hit


def train(cfg: dict[str, Any]) -> TrainReport:
    train_cfg = cfg["training"]
    seed = int(train_cfg.get("seed", 0))
    _seed_all(seed)

    _mark_progress("00_data_load")
    data_cfg = cfg["dataset"]
    spec = DummyDatasetSpec(
        num_samples=int(data_cfg["num_samples"]),
        in_dim=int(data_cfg["in_dim"]),
        num_classes=int(data_cfg["num_classes"]),
        seed=int(data_cfg.get("seed", seed)),
    )
    run_dir = _resolve_run_dir(cfg)
    parquet_path = run_dir / "dummy_train.parquet"
    write(spec, parquet_path)
    parquet_sha = sha256_of_file(parquet_path)
    logger.info(
        json.dumps(
            {
                "event": "data_ready",
                "rows": spec.num_samples,
                "sha256": parquet_sha[:12],
                "path": str(parquet_path.relative_to(run_dir)),
            }
        )
    )

    _mark_progress("10_model_init")
    device = _resolve_device(str(train_cfg.get("device", "auto")))
    logger.info(json.dumps({"event": "device", "device": device.type}))

    dataset = CaseZeroDataset(parquet_path, in_dim=spec.in_dim)
    loader_cfg = cfg["dataloader"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        generator=generator,
    )

    model_cfg_raw = cfg["model"]
    model = TinyMLP(
        ModelConfig(
            in_dim=int(model_cfg_raw["in_dim"]),
            hidden_dim=int(model_cfg_raw["hidden_dim"]),
            out_dim=int(model_cfg_raw["out_dim"]),
        )
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(train_cfg["epochs"])
    max_steps = int(train_cfg["max_steps"])
    log_every_n_steps = int(train_cfg.get("log_every_n_steps", 1))

    _mark_progress("20_train_start")
    started = time.perf_counter()
    total_steps = 0
    all_losses: list[float] = []
    max_hit_any = False
    for epoch in range(epochs):
        steps, losses, max_hit = _train_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device,
            max_steps=max_steps,
            log_every_n_steps=log_every_n_steps,
            epoch_index=epoch,
        )
        total_steps += steps
        all_losses.extend(losses)
        max_hit_any = max_hit_any or max_hit
        if max_hit:
            break
    runtime = time.perf_counter() - started

    _mark_progress("40_eval_start")
    model.eval()
    with torch.no_grad():
        features = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
        labels = torch.stack([dataset[i][1] for i in range(len(dataset))]).to(device)
        logits = model(features)
        eval_loss = float(loss_fn(logits, labels).item())
        accuracy = float((logits.argmax(dim=-1) == labels).float().mean().item())
    logger.info(
        json.dumps(
            {
                "event": "eval",
                "loss": round(eval_loss, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    )

    _mark_progress("50_save")
    weights_path = run_dir / cfg["output"].get("weights_filename", "best.pt")
    torch.save(model.state_dict(), weights_path)

    metrics_path = run_dir / cfg["output"].get("metrics_filename", "metrics.json")
    metrics = {
        "epochs_run": min(epochs, epoch + 1),
        "steps_run": total_steps,
        "max_steps_hit": max_hit_any,
        "final_loss": all_losses[-1] if all_losses else float("nan"),
        "eval_loss": eval_loss,
        "eval_accuracy": accuracy,
        "loss_history": [round(loss, 6) for loss in all_losses],
        "runtime_seconds": round(runtime, 3),
        "device": device.type,
        "parquet_sha256": parquet_sha,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    final_loss = all_losses[-1] if all_losses else float("nan")
    epochs_run = min(epochs, epoch + 1)

    run_meta_path = run_dir / cfg["output"].get("run_meta_filename", "run.json")
    run_meta = {
        "schema_version": 0,
        "case": "imitation/case0",
        "purpose": "runpod_e2e_smoke",
        "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "device": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "metrics_path": str(metrics_path),
        "config": cfg,
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, default=str))

    _mark_progress("99_done")
    report = TrainReport(
        epochs_run=epochs_run,
        steps_run=total_steps,
        final_loss=final_loss,
        weights_path=weights_path,
        device=device.type,
        runtime_seconds=runtime,
        max_steps_hit=max_hit_any,
        loss_history=all_losses,
    )
    logger.info(
        json.dumps(
            {
                "event": "done",
                **{k: v for k, v in asdict(report).items() if k != "loss_history"},
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
    """Locate the YAML config without depending on cwd.

    Tries (in order):
      1. ``config`` if absolute
      2. ``<repo>/bot/<config>`` (typical local + RunPod path)
      3. ``absolute_under_repo(config)`` (last-resort repo-rooted resolve)
    """
    if config.is_absolute():
        return config
    repo_root: Path = find_repo_root(Path(__file__))
    bot_relative: Path = repo_root / "bot" / config
    if bot_relative.is_file():
        return bot_relative
    return Path(absolute_under_repo(config, start=Path(__file__)))


app = typer.Typer(add_completion=False)

_CONFIG_OPTION = typer.Option(
    Path("pipeline/imitation/case0/configs/smoke.yaml"),
    "--config",
    help="Path to smoke YAML config (relative to repo root or absolute).",
)
_DEVICE_OPTION = typer.Option(
    None,
    "--device",
    help="Override training.device (cpu | cuda | auto).",
)


@app.command()
def main(
    config: Path = _CONFIG_OPTION,
    device: str | None = _DEVICE_OPTION,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_path = _resolve_config_path(config)
    cfg = _load_config(config_path)
    if device is not None:
        cfg["training"]["device"] = device
    report = train(cfg)
    typer.echo(
        f"steps={report.steps_run} loss={report.final_loss:.4f} "
        f"runtime={report.runtime_seconds:.2f}s weights={report.weights_path}"
    )


if __name__ == "__main__":
    app()
