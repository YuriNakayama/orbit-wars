"""BC training loop for imitation/case5 IL baseline.

Pipeline:
  1) load configs from YAML
  2) seed torch/numpy/random
  3) load train/val parquet via CaseThreeDataset
  4) build DeepSetsPolicy + AdamW
  5) for each epoch: train + val; emit JSON-line metrics; keep best val total loss
  6) save best state_dict to weights_out
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from pipeline.imitation.case7.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case7.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case7.policy.types import BatchFeatures
from pipeline.imitation.case7.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)
from pipeline.imitation.case7.training.losses import (
    LossWeights,
    compute_class_weights,
    compute_loss,
)
from runpod_io.execution.progress import mark_progress
from utils.repo_root import absolute_under_repo, find_repo_root

logger = logging.getLogger(__name__)

STEP_LOG_INTERVAL = 100


@dataclass(frozen=True)
class TrainReport:
    epochs_run: int
    best_val_loss: float
    best_epoch: int
    weights_path: Path
    device: str = "cpu"
    runtime_seconds: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # CPU MLP — already deterministic


def _to_batch_features(sample: BatchedSample, device: torch.device) -> BatchFeatures:
    """Build BatchFeatures, moving tensors to the target device."""
    return BatchFeatures(
        planet_feats=sample.planet_feats.to(device, non_blocking=True),
        planet_mask=sample.planet_mask.to(device, non_blocking=True),
        my_planet_mask=sample.my_planet_mask.to(device, non_blocking=True),
        target_mask=sample.target_mask.to(device, non_blocking=True),
        global_feats=sample.global_feats.to(device, non_blocking=True),
        template_ctx=sample.template_ctx.to(device, non_blocking=True),
    )


def _run_epoch(
    model: DeepSetsPolicy,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_weights: LossWeights,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    *,
    run_id: str | None = None,
    epoch_idx: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "total": 0.0,
        "from": 0.0,
        "target": 0.0,
        "ships": 0.0,
        "from_acc": 0.0,
        "target_acc": 0.0,
        "ships_acc": 0.0,
    }
    n_batches = 0
    grad_ctx = torch.enable_grad if is_train else torch.no_grad
    with grad_ctx():
        for batch in loader:
            features = _to_batch_features(batch, device)
            output = model(features)
            report = compute_loss(
                output,
                from_multihot=batch.from_multihot.to(device, non_blocking=True),
                target_per_src=batch.target_per_src.to(device, non_blocking=True),
                ships_per_src=batch.ships_per_src.to(device, non_blocking=True),
                my_planet_mask=batch.my_planet_mask.to(device, non_blocking=True),
                weights=loss_weights,
            )
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                report.total.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            batch_loss = float(report.total.detach().item())
            totals["total"] += batch_loss
            totals["from"] += float(report.from_loss.item())
            totals["target"] += float(report.target_loss.item())
            totals["ships"] += float(report.ships_loss.item())
            totals["from_acc"] += report.from_acc
            totals["target_acc"] += report.target_acc
            totals["ships_acc"] += report.ships_acc
            n_batches += 1
            if (
                is_train
                and run_id is not None
                and epoch_idx is not None
                and n_batches % STEP_LOG_INTERVAL == 0
            ):
                mark_progress(
                    run_id,
                    "train.step",
                    {
                        "epoch": epoch_idx,
                        "batch": n_batches,
                        "loss": round(batch_loss, 4),
                    },
                )

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


def _resolve_run_dir() -> Path | None:
    """`ORBIT_WARS_RUN_DIR` env が設定されていれば run dir 絶対パスを返す。

    Vast / RunPod 上では指定必須、ローカルでは未指定で従来挙動を維持する。
    どちらかの provider env (`ORBIT_WARS_VAST_INSTANCE_ID` /
    `ORBIT_WARS_RUNPOD_POD_ID`) があるのに `ORBIT_WARS_RUN_DIR` が無い場合は
    canonical weights.pt を誤上書きするリスクがあるため assertion で停止する
    (Risk #4 防御弾)。両 provider env が同時 set されているのも誤設定なので拒否。
    """
    run_dir_env = os.environ.get("ORBIT_WARS_RUN_DIR")
    vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
    runpod_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
    if vast_id and runpod_id:
        raise RuntimeError(
            "Both ORBIT_WARS_VAST_INSTANCE_ID and ORBIT_WARS_RUNPOD_POD_ID are set. "
            "Only one provider should be active per run."
        )
    if (vast_id or runpod_id) and not run_dir_env:
        raise RuntimeError(
            "ORBIT_WARS_RUN_DIR is required when a provider id "
            "(ORBIT_WARS_VAST_INSTANCE_ID / ORBIT_WARS_RUNPOD_POD_ID) is set. "
            "Refusing to overwrite canonical weights.pt from a cloud instance."
        )
    if not run_dir_env:
        return None
    return Path(run_dir_env).resolve()


def train(cfg: dict[str, Any]) -> TrainReport:
    seed = int(cfg.get("seed", 0))
    _seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(json.dumps({"device": device.type}))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    train_ds = CaseThreeDataset(_abspath(data_cfg["out_train"]))
    val_ds = CaseThreeDataset(_abspath(data_cfg["out_val"]))

    g = torch.Generator()
    g.manual_seed(seed)
    # Optional minority-target oversampling (Phase 3-A1). When enabled we
    # replace shuffling with WeightedRandomSampler so rare target templates
    # (NEAREST_NEUTRAL, HIGHEST_PROD_NEUTRAL, ...) appear more often per epoch.
    oversample_cfg = train_cfg.get("target_oversample", {}) or {}
    use_target_oversample = bool(oversample_cfg.get("enabled", False))
    target_sampler: WeightedRandomSampler | None = None
    if use_target_oversample:
        sample_w = train_ds.sample_weights_from_target(
            num_classes=NUM_TEMPLATES,
            power=float(oversample_cfg.get("power", 0.5)),
            ignore_index=-1,
            aggregate=str(oversample_cfg.get("aggregate", "mean")),
        )
        target_sampler = WeightedRandomSampler(
            weights=sample_w.tolist(),
            num_samples=len(train_ds),
            replacement=True,
            generator=g,
        )
        logger.info(
            json.dumps(
                {
                    "target_oversample": {
                        "power": float(oversample_cfg.get("power", 0.5)),
                        "min_weight": round(float(sample_w.min()), 4),
                        "max_weight": round(float(sample_w.max()), 4),
                        "ratio_max_min": round(
                            float(sample_w.max() / max(sample_w.min(), 1e-12)), 2
                        ),
                    }
                }
            )
        )
    pin_memory = device.type == "cuda"
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=target_sampler is None,
        sampler=target_sampler,
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
        planet_in_dim=int(model_cfg.get("planet_in_dim", 11)),
        global_in_dim=int(model_cfg.get("global_in_dim", 6)),
        hidden=int(model_cfg.get("hidden", 128)),
        ships_buckets=int(model_cfg.get("ships_buckets", 4)),
        attn_heads=int(model_cfg.get("attn_heads", 4)),
        inducing_points=int(model_cfg.get("inducing_points", 16)),
        encoder_layers=int(model_cfg.get("encoder_layers", 3)),
        model_type=str(model_cfg.get("model_type", "set_transformer")),
    )
    # build_model は model_type で SetTransformer/Pointer を分岐
    from pipeline.imitation.case7.policy.model import build_model

    model = build_model(model_config).to(device)
    model.train()  # build_model は eval mode で返す

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    lw_cfg = train_cfg.get("loss_weights", {})

    # Phase 2: compute per-class weights from train labels. Only "fired" rows
    # (target/ships_per_src != -1) contribute. This is an O(N*P) scan done once
    # at startup — cheaper than re-counting inside each epoch.
    target_cw: torch.Tensor | None = None
    ships_cw: torch.Tensor | None = None
    if bool(lw_cfg.get("use_target_class_weights", False)):
        tgt_all = torch.from_numpy(train_ds._target_per_src).flatten()
        target_cw = compute_class_weights(
            tgt_all,
            num_classes=NUM_TEMPLATES,
            beta=float(lw_cfg.get("target_class_weight_beta", 0.9999)),
            ignore_index=-1,
        ).to(device)
        logger.info(
            json.dumps(
                {
                    "target_class_weights": [
                        round(float(x), 4) for x in target_cw.tolist()
                    ]
                }
            )
        )
    if bool(lw_cfg.get("use_ships_class_weights", False)):
        ships_all = torch.from_numpy(train_ds._ships_per_src).flatten()
        num_ships_buckets = int(model_cfg.get("ships_buckets", 4))
        ships_cw = compute_class_weights(
            ships_all,
            num_classes=num_ships_buckets,
            beta=float(lw_cfg.get("ships_class_weight_beta", 0.9999)),
            ignore_index=-1,
        ).to(device)
        logger.info(
            json.dumps(
                {"ships_class_weights": [round(float(x), 4) for x in ships_cw.tolist()]}
            )
        )

    ships_focal_alpha_cfg = lw_cfg.get("ships_focal_alpha")
    ships_focal_alpha: torch.Tensor | None = None
    if ships_focal_alpha_cfg is not None:
        ships_focal_alpha = torch.tensor(
            [float(v) for v in ships_focal_alpha_cfg], dtype=torch.float32
        ).to(device)

    weights = LossWeights(
        from_w=float(lw_cfg.get("from", 1.0)),
        target_w=float(lw_cfg.get("target", 1.0)),
        ships_w=float(lw_cfg.get("ships", 0.5)),
        from_pos_weight=float(lw_cfg.get("from_pos_weight", 8.5)),
        from_focal_gamma=float(lw_cfg.get("from_focal_gamma", 2.0)),
        from_focal_alpha=float(lw_cfg.get("from_focal_alpha", 0.75)),
        target_label_smoothing=float(lw_cfg.get("target_label_smoothing", 0.1)),
        target_entropy_bonus=float(lw_cfg.get("target_entropy_bonus", 0.05)),
        target_class_weights=target_cw,
        ships_class_weights=ships_cw,
        ships_ordinal_smoothing=float(lw_cfg.get("ships_ordinal_smoothing", 0.0)),
        ships_focal_gamma=float(lw_cfg.get("ships_focal_gamma", 0.0)),
        ships_focal_alpha=ships_focal_alpha,
    )

    run_dir = _resolve_run_dir()
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_out = run_dir / "best.pt"
    else:
        weights_out = _abspath(train_cfg["weights_out"])
        weights_out.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    epochs = int(train_cfg["epochs"])
    started = time.monotonic()
    run_id = os.environ.get("ORBIT_WARS_RUN_ID", "local")
    for epoch in range(epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            weights,
            optimizer,
            device,
            run_id=run_id,
            epoch_idx=epoch,
        )
        val_metrics = _run_epoch(
            model, val_loader, weights, optimizer=None, device=device
        )
        train_loss_history.append(round(train_metrics["total"], 4))
        val_loss_history.append(round(val_metrics["total"], 4))
        log_row = {
            "epoch": epoch,
            "train_total": round(train_metrics["total"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_from_acc": round(val_metrics["from_acc"], 4),
            "val_target_acc": round(val_metrics["target_acc"], 4),
            "val_ships_acc": round(val_metrics["ships_acc"], 4),
        }
        logger.info(json.dumps(log_row))
        mark_progress(run_id, "train.epoch", log_row)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            # CPU tensor で保存して再読込時のデバイス互換性を確保
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, weights_out)
    runtime_seconds = round(time.monotonic() - started, 2)

    report = TrainReport(
        epochs_run=epochs,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        weights_path=weights_out,
        device=device.type,
        runtime_seconds=runtime_seconds,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
    )

    if run_dir is not None:
        _write_run_artifacts(run_dir, report, seed)

    return report


def _write_run_artifacts(run_dir: Path, report: TrainReport, seed: int) -> None:
    """metrics.json と run.json を run_dir に書き出す。"""
    # 遅延 import: 学習スクリプト本体の import グラフを汚染しないため
    from vast.run_meta import RunMetadata, hash_params, write_run_json

    metrics = {
        "epochs_run": report.epochs_run,
        "best_epoch": report.best_epoch,
        "best_val_loss": report.best_val_loss,
        "train_loss_history": report.train_loss_history,
        "val_loss_history": report.val_loss_history,
        "device": report.device,
        "runtime_seconds": report.runtime_seconds,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    git_sha = os.environ.get("ORBIT_WARS_GIT_SHA") or _git_rev_parse("HEAD")
    git_branch = os.environ.get("ORBIT_WARS_GIT_BRANCH") or _git_branch_name()
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
        except Exception:  # pragma: no cover - defensive
            gpu_name = None
    weights_path_rel = _safe_relative_to_repo(run_dir / "best.pt")
    command = os.environ.get(
        "ORBIT_WARS_COMMAND",
        "uv run --directory bot python -m pipeline.imitation.case7.training.train",
    )
    meta = RunMetadata(
        run_id=run_id,
        git_sha=git_sha,
        git_branch=git_branch,
        params_hash=hash_params(_repo_root() / "params.yaml"),
        seed=seed,
        vast_instance_id=vast_id,
        runpod_pod_id=runpod_pod_id,
        gpu_name=gpu_name,
        vast_offer_snapshot=None,
        runpod_offer_snapshot=runpod_offer_snapshot,
        command=command,
        weights_path=weights_path_rel,
        train_metrics=metrics,
        local_eval_results=None,
        status="pushed",
    )
    write_run_json(run_dir, meta)


def _git_rev_parse(rev: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(_repo_root()),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch_name() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(_repo_root()),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _safe_relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path)


app = typer.Typer(add_completion=False)


def _repo_root() -> Path:
    return find_repo_root(Path(__file__))


def _abspath(rel: str | Path) -> Path:
    return absolute_under_repo(rel, start=Path(__file__))


def _load_params(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        path = _repo_root() / "params.yaml"
    else:
        path = config_path
        if not path.is_absolute():
            # 相対 path は cwd → repo_root/bot → repo_root の順で探す
            # (`uv run --directory bot` cwd 配慮)。
            candidates = [
                Path.cwd() / path,
                _repo_root() / "bot" / path,
                _repo_root() / path,
            ]
            for c in candidates:
                if c.is_file():
                    path = c
                    break
            else:
                path = candidates[0]
    with path.open() as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict), f"{path} must be a mapping"
    return loaded


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML config path (default=params.yaml, case5 は configs/il_case5.yaml)",
    ),
) -> None:
    """CLI: run training using repo-root params.yaml or --config override."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = _load_params(config)
    report = train(cfg)
    typer.echo(
        f"best_val_loss={report.best_val_loss:.4f} "
        f"best_epoch={report.best_epoch} "
        f"weights={report.weights_path}"
    )


if __name__ == "__main__":
    app()
