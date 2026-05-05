"""BC training loop for imitation/case8 (cand + ship-count regression heads)."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch import optim
from torch.utils.data import DataLoader

from pipeline.imitation.case8.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case8.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case8.policy.model import CandidatePolicy, ModelConfig
from pipeline.imitation.case8.policy.types import BatchFeatures
from pipeline.imitation.case8.training.dataset import (
    BatchedSample,
    CaseFourDataset,
    collate,
)
from pipeline.imitation.case8.training.losses import LossWeights, compute_loss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainReport:
    epochs_run: int
    best_val_loss: float
    best_epoch: int
    weights_path: Path
    run_dir: Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bot").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def _git_branch_slug() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        )
        branch = out.decode().strip()
    except Exception:
        branch = "nobranch"
    return branch.replace("/", "-").replace("_", "-")


def _build_run_dir(case: str, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}__{_git_branch_slug()}__{_git_sha()}__seed{seed}"
    return Path("data/output/models/imitation") / case / "runs" / run_id


def _resolve_run_dir(cfg_run_dir: str | None, case: str, seed: int) -> Path:
    env_run_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
    runpod_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
    if vast_id and runpod_id:
        raise RuntimeError(
            "Both ORBIT_WARS_VAST_INSTANCE_ID and ORBIT_WARS_RUNPOD_POD_ID are set. "
            "Only one provider should be active per run."
        )
    if (vast_id or runpod_id) and not env_run_dir:
        raise RuntimeError(
            "ORBIT_WARS_RUN_DIR is required when a provider id "
            "(ORBIT_WARS_VAST_INSTANCE_ID / ORBIT_WARS_RUNPOD_POD_ID) is set. "
            "Refusing to overwrite canonical weights from a cloud instance."
        )
    if env_run_dir:
        return _abspath(env_run_dir)
    if cfg_run_dir:
        return _abspath(cfg_run_dir)
    return _abspath(_build_run_dir(case=case, seed=seed))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _to_batch_features(sample: BatchedSample, device: torch.device) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats.to(device, non_blocking=True),
        planet_mask=sample.planet_mask.to(device, non_blocking=True),
        my_planet_mask=sample.my_planet_mask.to(device, non_blocking=True),
        target_mask=sample.target_mask.to(device, non_blocking=True),
        global_feats=sample.global_feats.to(device, non_blocking=True),
        candidate_feats=sample.candidate_feats.to(device, non_blocking=True),
        candidate_mask=sample.candidate_mask.to(device, non_blocking=True),
        candidate_pid=sample.candidate_pid.to(device, non_blocking=True),
    )


BATCH_LOG_EVERY = 100


_METRIC_KEY_MAP: dict[str, str] = {
    "val_total": "total",
    "val_cand_loss": "cand",
    "val_cand_acc": "cand_acc",
    "val_cand_noop_acc": "cand_noop_acc",
    "val_cand_fire_acc": "cand_fire_acc",
    "val_ship_loss": "ship",
    "val_ship_mae": "ship_mae",
}


def _select_metric_value(val_metrics: dict[str, float], name: str) -> float:
    key = _METRIC_KEY_MAP.get(name)
    if key is None:
        raise KeyError(
            f"unknown metric name {name!r}; expected one of {sorted(_METRIC_KEY_MAP)}"
        )
    return float(val_metrics[key])


def _build_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict[str, Any] | None,
    *,
    epochs: int,
) -> optim.lr_scheduler.LRScheduler | None:
    """Build an LR scheduler from config. Supports `cosine_warmup` only.

    Returns None when cfg is empty/None — caller treats it as "fixed lr".
    """
    if not cfg:
        return None
    sched_type = str(cfg.get("type", "")).lower()
    if sched_type != "cosine_warmup":
        raise ValueError(
            f"unsupported scheduler type {sched_type!r}"
            " (only 'cosine_warmup' supported)"
        )
    t_max = int(cfg.get("t_max", epochs))
    eta_min = float(cfg.get("eta_min", 0.0))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(cfg.get("warmup_start_factor", 0.1))

    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, t_max - warmup_epochs), eta_min=eta_min
    )
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return cosine


def _run_epoch(
    model: CandidatePolicy,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_weights: LossWeights,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    *,
    epoch: int = -1,
    phase: str = "epoch",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "total": 0.0,
        "cand": 0.0,
        "cand_acc": 0.0,
        "cand_noop_acc": 0.0,
        "cand_fire_acc": 0.0,
        "ship": 0.0,
        "ship_mae": 0.0,
    }
    n_batches = 0
    started = time.monotonic()
    n_total = len(loader)
    grad_ctx = torch.enable_grad if is_train else torch.no_grad
    with grad_ctx():
        for batch in loader:
            features = _to_batch_features(batch, device)
            output = model(features)
            report = compute_loss(
                output,
                cand_slot_per_src=batch.cand_slot_per_src.to(device, non_blocking=True),
                my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
                weights=loss_weights,
                ship_label_per_src=batch.ship_label_per_src.to(
                    device, non_blocking=True
                ),
            )
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                report.total.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            totals["total"] += float(report.total.detach().item())
            totals["cand"] += float(report.cand_loss.item())
            totals["cand_acc"] += report.cand_acc
            totals["cand_noop_acc"] += report.cand_noop_acc
            totals["cand_fire_acc"] += report.cand_fire_acc
            totals["ship"] += float(report.ship_loss.item())
            totals["ship_mae"] += report.ship_mae
            n_batches += 1

            if n_batches % BATCH_LOG_EVERY == 0:
                elapsed = time.monotonic() - started
                rate = n_batches / elapsed if elapsed > 0 else 0.0
                eta = (n_total - n_batches) / rate if rate > 0 else float("inf")
                logger.info(
                    json.dumps(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "batch": n_batches,
                            "of": n_total,
                            "running_loss": round(
                                totals["total"] / max(1, n_batches), 4
                            ),
                            "running_acc": round(
                                totals["cand_acc"] / max(1, n_batches), 4
                            ),
                            "elapsed_s": round(elapsed, 1),
                            "rate_b_per_s": round(rate, 2),
                            "eta_s": round(eta, 1),
                        }
                    )
                )

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


def _stamp(msg: str) -> None:
    """logger.info + handler flush で各 step の進捗を確実にディスクに残す。

    iter2 で `60_before_train` から marker が出ないまま 30 分 hang した事象を
    debug するため、train.py の主要 checkpoint で stamp を書き出す。logger
    自体のバッファは Python の logging が flush しないことがあるので、
    handler 全部を明示的に flush して S3 へ push される前に確実に残す。
    """
    logger.info(f"[case8/train] {msg}")
    for handler in logger.handlers + logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def train(cfg: dict[str, Any]) -> TrainReport:
    _stamp("ENTER train()")
    seed = int(cfg.get("seed", 0))
    _seed_all(seed)
    _stamp(f"seed_all done seed={seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(json.dumps({"device": device.type}))
    _stamp(f"device={device.type} cuda_available={torch.cuda.is_available()}")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    ablation_cfg = data_cfg.get("ablation", {}) or {}
    mask_planet_cols = list(ablation_cfg.get("planet_cols", []) or [])
    mask_global_cols = list(ablation_cfg.get("global_cols", []) or [])
    _stamp(f"loading train_ds from {data_cfg['out_train']}")
    train_ds = CaseFourDataset(
        _abspath(data_cfg["out_train"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )
    _stamp(f"train_ds loaded len={len(train_ds)}")
    _stamp(f"loading val_ds from {data_cfg['out_val']}")
    val_ds = CaseFourDataset(
        _abspath(data_cfg["out_val"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )
    _stamp(f"val_ds loaded len={len(val_ds)}")

    g = torch.Generator()
    g.manual_seed(seed)
    pin_memory = device.type == "cuda"
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
        generator=g,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    model_config = ModelConfig(
        planet_in_dim=int(model_cfg.get("planet_in_dim", PLANET_FEAT_DIM)),
        global_in_dim=int(model_cfg.get("global_in_dim", GLOBAL_FEAT_DIM)),
        cand_in_dim=int(model_cfg.get("cand_in_dim", CAND_FEAT_DIM)),
        cand_k=int(model_cfg.get("cand_k", CAND_K)),
        hidden=int(model_cfg.get("hidden", 128)),
    )
    model = CandidatePolicy(model_config).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg["epochs"])
    _stamp(f"building scheduler epochs={epochs} cfg={train_cfg.get('scheduler')!r}")
    scheduler = _build_scheduler(optimizer, train_cfg.get("scheduler"), epochs=epochs)
    _stamp(f"scheduler built type={type(scheduler).__name__ if scheduler else 'None'}")

    early_stop_cfg = train_cfg.get("early_stop") or {}
    early_stop_metric = str(early_stop_cfg.get("metric", "")) or None
    early_stop_patience = int(early_stop_cfg.get("patience", 0)) or 0
    early_stop_mode = str(early_stop_cfg.get("mode", "max")).lower()
    if early_stop_mode not in {"min", "max"}:
        raise ValueError(
            f"early_stop.mode must be 'min' or 'max', got {early_stop_mode}"
        )

    lw_cfg = train_cfg.get("loss_weights", {}) or {}
    cand_cw: torch.Tensor | None = None
    if bool(lw_cfg.get("use_class_weights", False)):
        cand_cw = train_ds.class_weight_on_slots(
            num_classes=model_config.cand_k,
            beta=float(lw_cfg.get("class_weight_beta", 0.999)),
            ignore_index=-1,
        )
        logger.info(
            json.dumps(
                {"cand_class_weights": [round(float(x), 4) for x in cand_cw.tolist()]}
            )
        )

    weights = LossWeights(
        cand_w=float(lw_cfg.get("cand", 1.0)),
        cand_class_weights=cand_cw,
        label_smoothing=float(lw_cfg.get("label_smoothing", 0.0)),
        ship_w=float(lw_cfg.get("ship", 1.0)),
    )

    case_name = str(train_cfg.get("case", "case8"))
    run_dir = _resolve_run_dir(
        cfg_run_dir=train_cfg.get("run_dir"),
        case=case_name,
        seed=int(cfg.get("seed", 0)),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    use_run_dir_as_canonical = bool(os.environ.get("ORBIT_WARS_RUN_DIR"))
    if use_run_dir_as_canonical:
        weights_out = run_dir / "best.pt"
    else:
        weights_out = _abspath(train_cfg["weights_out"])
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.jsonl"
    history_path.write_text("")
    run_weights_path = run_dir / "best.pt"
    config_snapshot = run_dir / "config.yaml"
    with config_snapshot.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(json.dumps({"run_dir": str(run_dir), "weights_out": str(weights_out)}))

    best_metric_name = str(train_cfg.get("best_metric", "val_total")).lower()
    best_metric_mode = str(train_cfg.get("best_metric_mode", "min")).lower()
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(
            f"best_metric_mode must be 'min' or 'max', got {best_metric_mode}"
        )
    best_metric_value = float("inf") if best_metric_mode == "min" else float("-inf")
    best_val = float("inf")  # legacy field for summary.json compat
    best_epoch = -1
    early_stop_counter = 0
    early_stop_best = float("-inf") if early_stop_mode == "max" else float("inf")
    epochs_run = epochs
    train_started = time.monotonic()
    _stamp(f"entering epoch loop epochs={epochs}")
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        _stamp(f"epoch={epoch} starting (lr={optimizer.param_groups[0]['lr']:.6f})")
        logger.info(
            json.dumps(
                {
                    "event": "epoch_start",
                    "epoch": epoch,
                    "of": epochs,
                    "elapsed_total_s": round(time.monotonic() - train_started, 1),
                }
            )
        )
        train_metrics = _run_epoch(
            model,
            train_loader,
            weights,
            optimizer,
            device,
            epoch=epoch,
            phase="train",
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            weights,
            optimizer=None,
            device=device,
            epoch=epoch,
            phase="val",
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        log_row = {
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "train_total": round(train_metrics["total"], 4),
            "train_cand_loss": round(train_metrics["cand"], 4),
            "train_ship_loss": round(train_metrics["ship"], 4),
            "train_ship_mae": round(train_metrics["ship_mae"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_cand_loss": round(val_metrics["cand"], 4),
            "val_cand_acc": round(val_metrics["cand_acc"], 4),
            "val_cand_noop_acc": round(val_metrics["cand_noop_acc"], 4),
            "val_cand_fire_acc": round(val_metrics["cand_fire_acc"], 4),
            "val_ship_loss": round(val_metrics["ship"], 4),
            "val_ship_mae": round(val_metrics["ship_mae"], 4),
            "epoch_elapsed_s": round(time.monotonic() - epoch_started, 1),
        }

        candidate_value = _select_metric_value(val_metrics, best_metric_name)
        improved = (
            candidate_value < best_metric_value
            if best_metric_mode == "min"
            else candidate_value > best_metric_value
        )
        if improved:
            best_metric_value = candidate_value
            best_val = val_metrics["total"]
            best_epoch = epoch
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, weights_out)
            if weights_out.resolve() != run_weights_path.resolve():
                shutil.copyfile(weights_out, run_weights_path)
            logger.info(
                json.dumps(
                    {
                        "event": "best_updated",
                        "epoch": epoch,
                        "best_metric": best_metric_name,
                        "best_metric_value": round(best_metric_value, 6),
                        "best_val_total": round(best_val, 4),
                    }
                )
            )

        if early_stop_metric and early_stop_patience > 0:
            es_value = _select_metric_value(val_metrics, early_stop_metric)
            es_improved = (
                es_value > early_stop_best
                if early_stop_mode == "max"
                else es_value < early_stop_best
            )
            if es_improved:
                early_stop_best = es_value
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            log_row["early_stop_counter"] = early_stop_counter
            log_row["early_stop_best"] = round(early_stop_best, 6)
            if early_stop_counter >= early_stop_patience:
                logger.info(json.dumps(log_row))
                with history_path.open("a") as f:
                    f.write(json.dumps(log_row) + "\n")
                logger.info(
                    json.dumps(
                        {
                            "event": "early_stop",
                            "epoch": epoch,
                            "metric": early_stop_metric,
                            "best": round(early_stop_best, 6),
                            "patience": early_stop_patience,
                        }
                    )
                )
                epochs_run = epoch + 1
                break

        logger.info(json.dumps(log_row))
        with history_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")

        if scheduler is not None:
            scheduler.step()
            _stamp(
                f"epoch={epoch} done train_total={train_metrics['total']:.2f} "
                f"val_total={val_metrics['total']:.2f} "
                f"val_cand_fire_acc={val_metrics['cand_fire_acc']:.4f} "
                f"next_lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    summary = {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6),
        "best_metric": best_metric_name,
        "best_metric_value": round(best_metric_value, 6),
        "weights_out": str(weights_out),
        "run_weights": str(run_weights_path),
        "git_sha": _git_sha(),
        "git_branch": _git_branch_slug(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _write_run_json(run_dir, summary, seed=int(cfg.get("seed", 0)))

    return TrainReport(
        epochs_run=epochs_run,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        weights_path=weights_out,
        run_dir=run_dir,
    )


def _write_run_json(run_dir: Path, summary: dict[str, Any], *, seed: int) -> None:
    """`<run_dir>/run.json` を case1/3 互換 schema で生成する。"""
    from vast.run_meta import RunMetadata, write_run_json

    git_sha = os.environ.get("ORBIT_WARS_GIT_SHA") or _git_sha()
    git_branch = os.environ.get("ORBIT_WARS_GIT_BRANCH") or _git_branch_slug()
    run_id = os.environ.get("ORBIT_WARS_RUN_ID") or run_dir.name
    vast_id_raw = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
    vast_id: int | None = None
    if vast_id_raw:
        try:
            vast_id = int(vast_id_raw)
        except ValueError:
            vast_id = None
    runpod_pod_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID") or None
    runpod_offer_snapshot: dict[str, Any] | None = None
    snapshot_raw = os.environ.get("ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT")
    if snapshot_raw:
        try:
            parsed = json.loads(snapshot_raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            runpod_offer_snapshot = parsed
    gpu_name: str | None = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    command = os.environ.get(
        "ORBIT_WARS_COMMAND",
        "uv run --directory bot python -m pipeline.imitation.case8.training.train",
    )
    weights_path_rel = str(run_dir / "best.pt")
    meta = RunMetadata(
        run_id=run_id,
        git_sha=git_sha,
        git_branch=git_branch,
        params_hash="",
        seed=seed,
        vast_instance_id=vast_id,
        runpod_pod_id=runpod_pod_id,
        gpu_name=gpu_name,
        vast_offer_snapshot=None,
        runpod_offer_snapshot=runpod_offer_snapshot,
        command=command,
        weights_path=weights_path_rel,
        train_metrics=summary,
        local_eval_results=None,
        status="pushed",
    )
    write_run_json(run_dir, meta)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case8/configs/il_case8.yaml"),
        "--config",
        "-c",
        help="YAML config path",
    ),
) -> None:
    """CLI: run training with the given config."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with config.open() as f:
        cfg = yaml.safe_load(f)
    report = train(cfg)
    typer.echo(f"run_dir={report.run_dir}")
    typer.echo(
        f"best_val_loss={report.best_val_loss:.4f} "
        f"best_epoch={report.best_epoch} "
        f"weights={report.weights_path}"
    )


if __name__ == "__main__":
    app()
