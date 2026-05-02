"""BC training loop for imitation/case4 (single candidate head)."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
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

from pipeline.imitation.case4.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case4.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case4.policy.model import CandidatePolicy, ModelConfig
from pipeline.imitation.case4.policy.types import BatchFeatures
from pipeline.imitation.case4.training.dataset import (
    BatchedSample,
    CaseFourDataset,
    collate,
)
from pipeline.imitation.case4.training.losses import LossWeights, compute_loss

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
        if (parent / "backend").is_dir() and (parent / ".git").exists():
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


def _run_epoch(
    model: CandidatePolicy,
    loader: DataLoader,  # type: ignore[type-arg]
    loss_weights: LossWeights,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "total": 0.0,
        "cand": 0.0,
        "cand_acc": 0.0,
        "cand_noop_acc": 0.0,
        "cand_fire_acc": 0.0,
    }
    n_batches = 0
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
            n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in totals.items()}


def train(cfg: dict[str, Any]) -> TrainReport:
    seed = int(cfg.get("seed", 0))
    _seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(json.dumps({"device": device.type}))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    ablation_cfg = data_cfg.get("ablation", {}) or {}
    mask_planet_cols = list(ablation_cfg.get("planet_cols", []) or [])
    mask_global_cols = list(ablation_cfg.get("global_cols", []) or [])
    train_ds = CaseFourDataset(
        _abspath(data_cfg["out_train"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )
    val_ds = CaseFourDataset(
        _abspath(data_cfg["out_val"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )

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
    )

    case_name = str(train_cfg.get("case", "case4"))
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

    best_val = float("inf")
    best_epoch = -1
    epochs = int(train_cfg["epochs"])
    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, weights, optimizer, device)
        val_metrics = _run_epoch(
            model, val_loader, weights, optimizer=None, device=device
        )
        log_row = {
            "epoch": epoch,
            "train_total": round(train_metrics["total"], 4),
            "train_cand_loss": round(train_metrics["cand"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_cand_loss": round(val_metrics["cand"], 4),
            "val_cand_acc": round(val_metrics["cand_acc"], 4),
            "val_cand_noop_acc": round(val_metrics["cand_noop_acc"], 4),
            "val_cand_fire_acc": round(val_metrics["cand_fire_acc"], 4),
        }
        logger.info(json.dumps(log_row))
        with history_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, weights_out)
            if weights_out.resolve() != run_weights_path.resolve():
                shutil.copyfile(weights_out, run_weights_path)

    summary = {
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6),
        "weights_out": str(weights_out),
        "run_weights": str(run_weights_path),
        "git_sha": _git_sha(),
        "git_branch": _git_branch_slug(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _write_run_json(run_dir, summary, seed=int(cfg.get("seed", 0)))

    return TrainReport(
        epochs_run=epochs,
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
        "uv run --directory backend python -m pipeline.imitation.case4.training.train",
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
        Path("pipeline/imitation/case4/configs/il_case4.yaml"),
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
