"""BC training loop for imitation/case8 (cand + ship-count regression heads)."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch import nn, optim
from torch.optim import swa_utils
from torch.utils.data import DataLoader

from pipeline.imitation.case9.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case9.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case9.policy.model import Case9Policy, ModelConfig
from pipeline.imitation.case9.policy.types import BatchFeatures
from pipeline.imitation.case9.training.dataset import (
    BatchedSample,
    CaseFourDataset,
    collate,
)
from pipeline.imitation.case9.training.losses import (
    LossWeights,
    compute_candidate_ships_loss,
    compute_dual_head_loss,
    compute_loss,
    compute_template_ships_loss,
    compute_three_head_loss,
)

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
        template_ctx=sample.template_ctx.to(device, non_blocking=True),
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
    # three_head metrics (head_mode=three_head)
    "val_from_loss": "from_loss",
    "val_target_loss": "target_loss",
    "val_ships_loss": "ships_loss",
    "val_from_acc": "from_acc",
    "val_target_acc": "target_acc",
    "val_ships_acc": "ships_acc",
    # template_ships metrics
    "val_template_loss": "template_loss",
    "val_template_acc": "template_acc",
    "val_template_noop_acc": "template_noop_acc",
    "val_template_fire_acc": "template_fire_acc",
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


def _make_ema_step(
    ema_model: swa_utils.AveragedModel, model: nn.Module
) -> Callable[[], None]:
    """Build a closure-free callback for EMA update (avoids B023 loop binding)."""

    def step() -> None:
        ema_model.update_parameters(model)

    return step


def _compute_loss_dispatch(
    head_mode: str,
    output: object,
    batch: BatchedSample,
    loss_weights: LossWeights,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run the appropriate loss for head_mode and return (total, metrics_dict)."""
    if head_mode == "three_head":
        rep3 = compute_three_head_loss(
            output,  # type: ignore[arg-type]
            from_multihot=batch.from_multihot.to(device, non_blocking=True),
            target_per_src=batch.target_per_src.to(device, non_blocking=True),
            ships_per_src=batch.ships_per_src.to(device, non_blocking=True),
            my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
        )
        return rep3.total, {
            "total": float(rep3.total.detach().item()),
            "from_loss": float(rep3.from_loss.item()),
            "target_loss": float(rep3.target_loss.item()),
            "ships_loss": float(rep3.ships_loss.item()),
            "from_acc": rep3.from_acc,
            "target_acc": rep3.target_acc,
            "ships_acc": rep3.ships_acc,
        }
    if head_mode == "candidate_ships":
        rep_cs = compute_candidate_ships_loss(
            output,  # type: ignore[arg-type]
            cand_slot_per_src=batch.cand_slot_per_src.to(device, non_blocking=True),
            ships_bucket_per_src=batch.ships_bucket_per_src.to(
                device, non_blocking=True
            ),
            my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
            weights=loss_weights,
        )
        return rep_cs.total, {
            "total": float(rep_cs.total.detach().item()),
            "cand": float(rep_cs.cand_loss.item()),
            "cand_acc": rep_cs.cand_acc,
            "cand_noop_acc": rep_cs.cand_noop_acc,
            "cand_fire_acc": rep_cs.cand_fire_acc,
            "ships_loss": float(rep_cs.ships_loss.item()),
            "ships_acc": rep_cs.ships_acc,
        }
    if head_mode == "template_ships":
        rep_ts = compute_template_ships_loss(
            output,  # type: ignore[arg-type]
            target_per_src=batch.target_per_src.to(device, non_blocking=True),
            ships_per_src=batch.ships_per_src.to(device, non_blocking=True),
            my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
        )
        return rep_ts.total, {
            "total": float(rep_ts.total.detach().item()),
            "template_loss": float(rep_ts.template_loss.item()),
            "template_acc": rep_ts.template_acc,
            "template_noop_acc": rep_ts.template_noop_acc,
            "template_fire_acc": rep_ts.template_fire_acc,
            "ships_loss": float(rep_ts.ships_loss.item()),
            "ships_acc": rep_ts.ships_acc,
        }
    if head_mode == "dual":
        rep_d = compute_dual_head_loss(
            output,  # type: ignore[arg-type]
            from_multihot=batch.from_multihot.to(device, non_blocking=True),
            target_per_src=batch.target_per_src.to(device, non_blocking=True),
            ships_per_src=batch.ships_per_src.to(device, non_blocking=True),
            cand_slot_per_src=batch.cand_slot_per_src.to(device, non_blocking=True),
            ship_label_per_src=batch.ship_label_per_src.to(device, non_blocking=True),
            my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
            weights=loss_weights,
        )
        return rep_d.total, {
            "total": float(rep_d.total.detach().item()),
            "three_total": float(rep_d.three.total.detach().item()),
            "cand_total": float(rep_d.candidate.total.detach().item()),
            "from_loss": float(rep_d.three.from_loss.item()),
            "target_loss": float(rep_d.three.target_loss.item()),
            "ships_loss": float(rep_d.three.ships_loss.item()),
            "from_acc": rep_d.three.from_acc,
            "target_acc": rep_d.three.target_acc,
            "ships_acc": rep_d.three.ships_acc,
            "cand": float(rep_d.candidate.cand_loss.item()),
            "cand_acc": rep_d.candidate.cand_acc,
            "cand_noop_acc": rep_d.candidate.cand_noop_acc,
            "cand_fire_acc": rep_d.candidate.cand_fire_acc,
            "ship": float(rep_d.candidate.ship_loss.item()),
            "ship_mae": rep_d.candidate.ship_mae,
            "dual_alpha": rep_d.alpha,
        }
    # default: candidate (case8 style)
    rep_c = compute_loss(
        output,  # type: ignore[arg-type]
        cand_slot_per_src=batch.cand_slot_per_src.to(device, non_blocking=True),
        my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
        weights=loss_weights,
        ship_label_per_src=batch.ship_label_per_src.to(device, non_blocking=True),
    )
    return rep_c.total, {
        "total": float(rep_c.total.detach().item()),
        "cand": float(rep_c.cand_loss.item()),
        "cand_acc": rep_c.cand_acc,
        "cand_noop_acc": rep_c.cand_noop_acc,
        "cand_fire_acc": rep_c.cand_fire_acc,
        "ship": float(rep_c.ship_loss.item()),
        "ship_mae": rep_c.ship_mae,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_weights: LossWeights,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    *,
    epoch: int = -1,
    phase: str = "epoch",
    grad_clip_max_norm: float = 0.0,
    on_step: Callable[[], None] | None = None,
    head_mode: str = "candidate",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    n_batches = 0
    started = time.monotonic()
    n_total = len(loader)
    grad_ctx = torch.enable_grad if is_train else torch.no_grad
    with grad_ctx():
        for batch in loader:
            features = _to_batch_features(batch, device)
            output = model(features)
            total_loss, metrics = _compute_loss_dispatch(
                head_mode, output, batch, loss_weights, device
            )
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()  # type: ignore[no-untyped-call]
                if grad_clip_max_norm > 0.0:
                    pre_clip_norm = float(
                        nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip_max_norm
                        )
                    )
                    totals.setdefault("grad_norm", 0.0)
                    totals["grad_norm"] += pre_clip_norm
                optimizer.step()
                if on_step is not None:
                    on_step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

            if n_batches % BATCH_LOG_EVERY == 0:
                elapsed = time.monotonic() - started
                rate = n_batches / elapsed if elapsed > 0 else 0.0
                eta = (n_total - n_batches) / rate if rate > 0 else float("inf")
                if head_mode == "three_head":
                    acc_key = "from_acc"
                elif head_mode == "template_ships":
                    acc_key = "template_acc"
                else:
                    acc_key = "cand_acc"
                running_acc_val = totals.get(acc_key, 0.0) / max(1, n_batches)
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
                            "running_acc": round(running_acc_val, 4),
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
    # iter5: yaml override (default: cuda 上では pin_memory=True、host RAM が
    # 厳しいときは false 推奨)
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
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

    head_mode = str(model_cfg.get("head_mode", "candidate"))
    model_config = ModelConfig(
        planet_in_dim=int(model_cfg.get("planet_in_dim", PLANET_FEAT_DIM)),
        global_in_dim=int(model_cfg.get("global_in_dim", GLOBAL_FEAT_DIM)),
        cand_in_dim=int(model_cfg.get("cand_in_dim", CAND_FEAT_DIM)),
        cand_k=int(model_cfg.get("cand_k", CAND_K)),
        hidden=int(model_cfg.get("hidden", 128)),
        head_dropout=float(train_cfg.get("head_dropout", 0.0)),
        head_mode=head_mode,
    )
    model = Case9Policy(model_config).to(device)
    _stamp(
        f"model built head_mode={head_mode} head_dropout={model_config.head_dropout}"
    )

    # iter4: EMA wrapper. eval / best.pt 選定は EMA weights を使う。
    ema_cfg = train_cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_model: swa_utils.AveragedModel | None = None
    if ema_enabled:
        ema_model = swa_utils.AveragedModel(
            model,
            multi_avg_fn=swa_utils.get_ema_multi_avg_fn(ema_decay),  # type: ignore[no-untyped-call]
        )
        _stamp(f"EMA enabled decay={ema_decay}")
    else:
        _stamp("EMA disabled")

    grad_clip_max_norm = float(train_cfg.get("grad_clip_max_norm", 0.0))
    _stamp(f"grad_clip_max_norm={grad_clip_max_norm}")

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
        cand_loss_type=str(lw_cfg.get("cand_loss_type", "ce")),
        focal_alpha=float(lw_cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(lw_cfg.get("focal_gamma", 2.0)),
        dual_alpha=float(lw_cfg.get("dual_alpha", 0.5)),
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
        ema_step = _make_ema_step(ema_model, model) if ema_model is not None else None
        train_metrics = _run_epoch(
            model,
            train_loader,
            weights,
            optimizer,
            device,
            epoch=epoch,
            phase="train",
            grad_clip_max_norm=grad_clip_max_norm,
            on_step=ema_step,
            head_mode=head_mode,
        )
        # iter4: val は EMA weights で評価 (live model は dropout が train-time
        # 振る舞いになっていなくても、AveragedModel(model) は同じ structure を
        # 共有し eval mode で dropout を無効化)
        eval_model: nn.Module = ema_model if ema_model is not None else model
        val_metrics = _run_epoch(
            eval_model,
            val_loader,
            weights,
            optimizer=None,
            device=device,
            epoch=epoch,
            phase="val",
            head_mode=head_mode,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        avg_grad_norm = (
            train_metrics["grad_norm"] if "grad_norm" in train_metrics else 0.0
        )
        log_row = {
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "avg_grad_norm_pre_clip": round(avg_grad_norm, 4),
            "train_total": round(train_metrics["total"], 4),
            "val_total": round(val_metrics["total"], 4),
            "ema_eval": ema_model is not None,
            "epoch_elapsed_s": round(time.monotonic() - epoch_started, 1),
        }
        if head_mode == "three_head":
            for k in (
                "from_loss",
                "target_loss",
                "ships_loss",
                "from_acc",
                "target_acc",
                "ships_acc",
            ):
                if k in train_metrics:
                    log_row[f"train_{k}"] = round(train_metrics[k], 4)
                if k in val_metrics:
                    log_row[f"val_{k}"] = round(val_metrics[k], 4)
        elif head_mode == "dual":
            for k in (
                "three_total",
                "cand_total",
                "from_loss",
                "target_loss",
                "ships_loss",
                "from_acc",
                "target_acc",
                "ships_acc",
                "cand",
                "cand_acc",
                "cand_noop_acc",
                "cand_fire_acc",
                "ship",
                "ship_mae",
                "dual_alpha",
            ):
                if k in train_metrics:
                    log_row[f"train_{k}"] = round(train_metrics[k], 4)
                if k in val_metrics:
                    log_row[f"val_{k}"] = round(val_metrics[k], 4)
        elif head_mode == "candidate_ships":
            for k in (
                "cand",
                "cand_acc",
                "cand_noop_acc",
                "cand_fire_acc",
                "ships_loss",
                "ships_acc",
            ):
                if k in train_metrics:
                    log_row[f"train_{k}"] = round(train_metrics[k], 4)
                if k in val_metrics:
                    log_row[f"val_{k}"] = round(val_metrics[k], 4)
        elif head_mode == "template_ships":
            for k in (
                "template_loss",
                "template_acc",
                "template_noop_acc",
                "template_fire_acc",
                "ships_loss",
                "ships_acc",
            ):
                if k in train_metrics:
                    log_row[f"train_{k}"] = round(train_metrics[k], 4)
                if k in val_metrics:
                    log_row[f"val_{k}"] = round(val_metrics[k], 4)
        else:  # candidate
            for k in (
                "cand",
                "cand_acc",
                "cand_noop_acc",
                "cand_fire_acc",
                "ship",
                "ship_mae",
            ):
                if k in train_metrics:
                    log_row[f"train_{k}"] = round(train_metrics[k], 4)
                if k in val_metrics:
                    log_row[f"val_{k}"] = round(val_metrics[k], 4)

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
            # iter4: best.pt は EMA weights を保存 (eval も EMA で実施済み)。
            # AveragedModel.module.state_dict() は live model と同じ key 構造。
            save_source: nn.Module = (
                ema_model.module if ema_model is not None else model
            )
            cpu_state = {
                k: v.detach().cpu() for k, v in save_source.state_dict().items()
            }
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
            if head_mode == "three_head":
                primary_acc_key = "from_acc"
            elif head_mode == "template_ships":
                primary_acc_key = "template_fire_acc"
            else:
                primary_acc_key = "cand_fire_acc"
            primary_acc_val = val_metrics.get(primary_acc_key, 0.0)
            _stamp(
                f"epoch={epoch} done train_total={train_metrics['total']:.2f} "
                f"val_total={val_metrics['total']:.2f} "
                f"val_{primary_acc_key}={primary_acc_val:.4f} "
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
        "uv run --directory bot python -m pipeline.imitation.case9.training.train",
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
