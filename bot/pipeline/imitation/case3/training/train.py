"""BC training loop for imitation/case3 IL baseline.

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from pipeline.imitation.case3.evaluation.eval_metrics import (
    collect_arrays,
    from_metrics,
    ships_metrics,
    target_metrics,
)
from pipeline.imitation.case3.policy.featurizer_phase2 import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case3.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case3.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case3.policy.types import BatchFeatures
from pipeline.imitation.case3.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)
from pipeline.imitation.case3.training.losses import (
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
    run_dir: Path


def _repo_root() -> Path:
    """`bot/` の親ディレクトリ (リポジトリ root) を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bot").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _abspath(rel: str | Path) -> Path:
    """相対パスを repo root 起点の絶対パスに解決する.

    Vast.ai では `uv run --directory bot ...` で CWD が bot/ になるため、
    config の `data/...` 相対パスをそのまま開くと `bot/data/...` になる.
    """
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


def _round_or_none(value: Any, ndigits: int) -> float | None:
    """Round if value is a real number; pass through None / n=0 sentinels."""
    if value is None:
        return None
    return round(float(value), ndigits)


def _build_run_dir(case: str, seed: int) -> Path:
    """data/output/models/imitation/<case>/runs/<run_id>/ を生成する.

    run_id 形式: <YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>
    (data/output/README.md の規約に準拠)
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}__{_git_branch_slug()}__{_git_sha()}__seed{seed}"
    return Path("data/output/models/imitation") / case / "runs" / run_id


def _resolve_run_dir(cfg_run_dir: str | None, case: str, seed: int) -> Path:
    """run_dir を解決する.

    優先順: ORBIT_WARS_RUN_DIR env (Vast / RunPod 注入) > cfg run_dir > 自動生成.
    どちらかの provider env (`ORBIT_WARS_VAST_INSTANCE_ID` /
    `ORBIT_WARS_RUNPOD_POD_ID`) があるのに `ORBIT_WARS_RUN_DIR` が無い場合は
    canonical weights を誤上書きするリスクがあるため停止する (Risk #4 防御弾)。
    両 provider env が同時 set されているのも誤設定なので拒否。
    """
    env_run_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
    runpod_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
    kk_slug = os.environ.get("ORBIT_WARS_KAGGLE_KERNEL_SLUG")
    active = [bool(vast_id), bool(runpod_id), bool(kk_slug)]
    if sum(active) > 1:
        raise RuntimeError(
            "Multiple provider env vars are set simultaneously. "
            "Set only one of ORBIT_WARS_VAST_INSTANCE_ID / "
            "ORBIT_WARS_RUNPOD_POD_ID / ORBIT_WARS_KAGGLE_KERNEL_SLUG per run."
        )
    if any(active) and not env_run_dir:
        raise RuntimeError(
            "ORBIT_WARS_RUN_DIR is required when a provider id "
            "(ORBIT_WARS_VAST_INSTANCE_ID / ORBIT_WARS_RUNPOD_POD_ID / "
            "ORBIT_WARS_KAGGLE_KERNEL_SLUG) is set. "
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(json.dumps({"device": device.type}))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {})

    ablation_cfg = data_cfg.get("ablation", {}) or {}
    mask_planet_cols = list(ablation_cfg.get("planet_cols", []) or [])
    mask_global_cols = list(ablation_cfg.get("global_cols", []) or [])
    train_ds = CaseThreeDataset(
        _abspath(data_cfg["out_train"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )
    val_ds = CaseThreeDataset(
        _abspath(data_cfg["out_val"]),
        mask_planet_cols=mask_planet_cols,
        mask_global_cols=mask_global_cols,
    )

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
        planet_in_dim=int(model_cfg.get("planet_in_dim", PLANET_FEAT_DIM)),
        global_in_dim=int(model_cfg.get("global_in_dim", GLOBAL_FEAT_DIM)),
        hidden=int(model_cfg.get("hidden", 64)),
        ships_buckets=int(model_cfg.get("ships_buckets", 4)),
    )
    model = DeepSetsPolicy(model_config).to(device)

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
        tgt_all = torch.from_numpy(train_ds._target_per_src).flatten()  # type: ignore[attr-defined]
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
        ships_all = torch.from_numpy(train_ds._ships_per_src).flatten()  # type: ignore[attr-defined]
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

    case_name = str(train_cfg.get("case", "case3"))
    run_dir = _resolve_run_dir(
        cfg_run_dir=train_cfg.get("run_dir"),
        case=case_name,
        seed=int(cfg.get("seed", 0)),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Vast モード (ORBIT_WARS_RUN_DIR 注入時) では weights_out も run_dir/best.pt に
    # 揃え、canonical weights.pt を誤上書きしない。ローカル時は cfg の weights_out
    # を使い、加えて run_dir にもコピーする (履歴保持)。
    use_run_dir_as_canonical = bool(os.environ.get("ORBIT_WARS_RUN_DIR"))
    if use_run_dir_as_canonical:
        weights_out = run_dir / "best.pt"
    else:
        weights_out = _abspath(train_cfg["weights_out"])
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.jsonl"
    history_path.write_text("")  # truncate any prior content for this run_dir
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
        # Per-epoch full-validation AUC / F1: re-runs the model on val_loader in
        # eval mode to collect probabilities, then computes the same multi-faceted
        # metrics as evaluation/eval_metrics.py.
        from_arr, target_arr, ships_arr = collect_arrays(model, val_loader)
        from_m = from_metrics(from_arr)
        target_m = target_metrics(target_arr)
        ships_m = ships_metrics(ships_arr)
        log_row = {
            "epoch": epoch,
            "train_total": round(train_metrics["total"], 4),
            "train_from_loss": round(train_metrics["from"], 4),
            "train_target_loss": round(train_metrics["target"], 4),
            "train_ships_loss": round(train_metrics["ships"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_from_loss": round(val_metrics["from"], 4),
            "val_target_loss": round(val_metrics["target"], 4),
            "val_ships_loss": round(val_metrics["ships"], 4),
            "val_from_acc": round(val_metrics["from_acc"], 4),
            "val_target_acc": round(val_metrics["target_acc"], 4),
            "val_ships_acc": round(val_metrics["ships_acc"], 4),
            # from head: ROC-AUC / PR-AUC / best-F1
            "val_from_roc_auc": _round_or_none(from_m.get("roc_auc"), 4),
            "val_from_pr_auc": _round_or_none(from_m.get("pr_auc"), 4),
            "val_from_f1_best": _round_or_none(from_m.get("f1_best"), 4),
            "val_from_f1_threshold": _round_or_none(from_m.get("f1_threshold"), 3),
            # target head: top-1 / top-2 acc + macro-F1
            "val_target_top1_acc": _round_or_none(target_m.get("top1_acc"), 4),
            "val_target_top2_acc": _round_or_none(target_m.get("top2_acc"), 4),
            "val_target_macro_f1": _round_or_none(target_m.get("macro_f1"), 4),
            # ships head: macro-F1 + ordinal MAE
            "val_ships_macro_f1": _round_or_none(ships_m.get("macro_f1"), 4),
            "val_ships_ordinal_mae": _round_or_none(ships_m.get("ordinal_mae"), 4),
        }
        logger.info(json.dumps(log_row))
        with history_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            # Kaggle Sandbox は CPU 推論なので state_dict を CPU に揃えて保存.
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, weights_out)
            # Vast モードでは weights_out == run_weights_path なので copy 不要.
            # ローカルモードでのみ run_dir にも履歴コピーする.
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

    # case1 と同じ schema の run.json を生成 (vast pull / promote / cost-report が
    # 共通コードで動くようにするため)。
    _write_run_json(run_dir, summary, seed=int(cfg.get("seed", 0)))

    return TrainReport(
        epochs_run=epochs,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        weights_path=weights_out,
        run_dir=run_dir,
    )


def _write_run_json(run_dir: Path, summary: dict[str, Any], *, seed: int) -> None:
    """`<run_dir>/run.json` を case1 互換 schema で生成する。"""
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
    kaggle_kernel_meta: dict[str, Any] | None = None
    kk_meta_raw = os.environ.get("ORBIT_WARS_KAGGLE_KERNEL_META")
    if kk_meta_raw:
        try:
            kk_parsed = json.loads(kk_meta_raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"ORBIT_WARS_KAGGLE_KERNEL_META is set but not valid JSON: {e}"
            ) from e
        if isinstance(kk_parsed, dict):
            kaggle_kernel_meta = kk_parsed
    gpu_name: str | None = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - defensive
            gpu_name = None
    command = os.environ.get(
        "ORBIT_WARS_COMMAND",
        "uv run --directory bot python -m pipeline.imitation.case3.training.train",
    )
    weights_path_rel = str(run_dir / "best.pt")
    meta = RunMetadata(
        run_id=run_id,
        git_sha=git_sha,
        git_branch=git_branch,
        params_hash="",  # case3 は configs/*.yaml 直読 → params.yaml hash 対象外
        seed=seed,
        vast_instance_id=vast_id,
        runpod_pod_id=runpod_pod_id,
        gpu_name=gpu_name,
        vast_offer_snapshot=None,
        runpod_offer_snapshot=runpod_offer_snapshot,
        kaggle_kernel_meta=kaggle_kernel_meta,
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
        Path("pipeline/imitation/case3/configs/il_baseline.yaml"),
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
