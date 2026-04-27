"""BC training loop for imitation/case1 IL baseline.

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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from pipeline.imitation.case1.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case1.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case1.policy.types import BatchFeatures
from pipeline.imitation.case1.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)
from pipeline.imitation.case1.training.losses import (
    LossWeights,
    compute_class_weights,
    compute_loss,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainReport:
    epochs_run: int
    best_val_loss: float
    best_epoch: int
    weights_path: Path


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU MLP — already deterministic


def _to_batch_features(sample: BatchedSample) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats,
        planet_mask=sample.planet_mask,
        my_planet_mask=sample.my_planet_mask,
        target_mask=sample.target_mask,
        global_feats=sample.global_feats,
        template_ctx=sample.template_ctx,
    )


def _run_epoch(
    model: DeepSetsPolicy,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_weights: LossWeights,
    optimizer: optim.Optimizer | None,
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
            features = _to_batch_features(batch)
            output = model(features)
            report = compute_loss(
                output,
                from_multihot=batch.from_multihot,
                target_per_src=batch.target_per_src,
                ships_per_src=batch.ships_per_src,
                my_planet_mask=batch.my_planet_mask,
                weights=loss_weights,
            )
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                report.total.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            totals["total"] += float(report.total.detach().item())
            totals["from"] += float(report.from_loss.item())
            totals["target"] += float(report.target_loss.item())
            totals["ships"] += float(report.ships_loss.item())
            totals["from_acc"] += report.from_acc
            totals["target_acc"] += report.target_acc
            totals["ships_acc"] += report.ships_acc
            n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


def train(cfg: dict[str, Any]) -> TrainReport:
    seed = int(cfg.get("seed", 0))
    _seed_all(seed)

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
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=target_sampler is None,
        sampler=target_sampler,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
        generator=g,
    )
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model_config = ModelConfig(
        planet_in_dim=int(model_cfg.get("planet_in_dim", 11)),
        global_in_dim=int(model_cfg.get("global_in_dim", 6)),
        hidden=int(model_cfg.get("hidden", 64)),
        ships_buckets=int(model_cfg.get("ships_buckets", 4)),
    )
    model = DeepSetsPolicy(model_config)

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
        )
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
        )
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
        )

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

    weights_out = _abspath(train_cfg["weights_out"])
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    epochs = int(train_cfg["epochs"])
    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, weights, optimizer)
        val_metrics = _run_epoch(model, val_loader, weights, optimizer=None)
        log_row = {
            "epoch": epoch,
            "train_total": round(train_metrics["total"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_from_acc": round(val_metrics["from_acc"], 4),
            "val_target_acc": round(val_metrics["target_acc"], 4),
            "val_ships_acc": round(val_metrics["ships_acc"], 4),
        }
        logger.info(json.dumps(log_row))
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), weights_out)

    return TrainReport(
        epochs_run=epochs,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        weights_path=weights_out,
    )


app = typer.Typer(add_completion=False)


def _repo_root() -> Path:
    """Locate the repository root (parent of `backend/`)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "backend").is_dir() and (p / ".git").exists():
            return p
    raise RuntimeError("repo root not found from " + str(here))


def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (_repo_root() / p).resolve()


def _load_params() -> dict[str, Any]:
    path = _repo_root() / "params.yaml"
    with path.open() as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict), "params.yaml must be a mapping"
    return loaded


@app.command()
def main() -> None:
    """CLI: run training using repo-root params.yaml."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = _load_params()
    report = train(cfg)
    typer.echo(
        f"best_val_loss={report.best_val_loss:.4f} "
        f"best_epoch={report.best_epoch} "
        f"weights={report.weights_path}"
    )


if __name__ == "__main__":
    app()
