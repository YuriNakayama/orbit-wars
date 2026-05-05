"""Compute F1 / PR-AUC / ROC-AUC for imitation/case5 saved weights on val set.

Runs inference once over val.parquet with the current weights.pt and emits a
rich metric report alongside the training-time accuracies. case1-specific
diagnostic: pulls in case1's DeepSetsPolicy + CaseThreeDataset directly.

Usage:
    uv run python -m pipeline.imitation.case10.evaluation.diagnose_weights \\
        --weights pipeline/imitation/case5/policy/weights.pt \\
        --report data/output/experiment/imitation_case1_val_metrics.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from pipeline.imitation.case10.policy.model import DeepSetsPolicy, ModelConfig
from pipeline.imitation.case10.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case10.policy.types import BatchFeatures
from pipeline.imitation.case10.training.dataset import (
    BatchedSample,
    CaseThreeDataset,
    collate,
)

DEFAULT_CONFIG = Path("pipeline/imitation/case5/configs/il_baseline.yaml")
DEFAULT_WEIGHTS = Path("pipeline/imitation/case5/policy/weights.pt")
DEFAULT_REPORT = Path("data/output/experiment/imitation_case1_val_metrics.json")
SHIPS_BUCKETS = 4


@dataclass(frozen=True)
class EvalConfig:
    weights: Path
    report: Path
    config: Path


def _to_batch_features(sample: BatchedSample) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats,
        planet_mask=sample.planet_mask,
        my_planet_mask=sample.my_planet_mask,
        target_mask=sample.target_mask,
        global_feats=sample.global_feats,
        template_ctx=sample.template_ctx,
    )


def _load_model(config: Path, weights: Path) -> DeepSetsPolicy:
    cfg_yaml = yaml.safe_load(config.read_text())
    mcfg = cfg_yaml.get("model", {})
    model_cfg = ModelConfig(
        planet_in_dim=int(mcfg.get("planet_in_dim", 11)),
        global_in_dim=int(mcfg.get("global_in_dim", 6)),
        hidden=int(mcfg.get("hidden", 128)),
        ships_buckets=int(mcfg.get("ships_buckets", SHIPS_BUCKETS)),
    )
    model = DeepSetsPolicy(model_cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _gather_preds(
    model: DeepSetsPolicy, val_ds: CaseThreeDataset
) -> dict[str, np.ndarray]:
    loader: DataLoader[BatchedSample] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=256,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    from_logits_all: list[np.ndarray] = []
    from_gt_all: list[np.ndarray] = []
    my_mask_all: list[np.ndarray] = []
    target_logits_all: list[np.ndarray] = []
    target_gt_all: list[np.ndarray] = []
    ships_logits_all: list[np.ndarray] = []
    ships_gt_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features = _to_batch_features(batch)
            output = model(features)

            from_logits_all.append(output.from_logits.cpu().numpy())
            from_gt_all.append(batch.from_multihot.cpu().numpy())
            my_mask_all.append(batch.my_planet_mask.cpu().numpy())

            fired = batch.from_multihot & batch.my_planet_mask
            if fired.any():
                b_idx, s_idx = fired.nonzero(as_tuple=True)
                target_logits_all.append(
                    output.target_logits[b_idx, s_idx].cpu().numpy()
                )
                target_gt_all.append(batch.target_per_src[b_idx, s_idx].cpu().numpy())
                ships_logits_all.append(output.ships_logits[b_idx, s_idx].cpu().numpy())
                ships_gt_all.append(batch.ships_per_src[b_idx, s_idx].cpu().numpy())

    return {
        "from_logits": np.concatenate(from_logits_all),
        "from_gt": np.concatenate(from_gt_all),
        "my_mask": np.concatenate(my_mask_all),
        "target_logits": np.concatenate(target_logits_all),
        "target_gt": np.concatenate(target_gt_all),
        "ships_logits": np.concatenate(ships_logits_all),
        "ships_gt": np.concatenate(ships_gt_all),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    result: np.ndarray = e / e.sum(axis=-1, keepdims=True)
    return result


def _top_k_accuracy(logits: np.ndarray, gt: np.ndarray, k: int) -> float:
    topk = np.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    match = (topk == gt[:, None]).any(axis=-1)
    return float(match.mean())


def _macro_ovr_pr_auc(gt: np.ndarray, probs: np.ndarray, labels: list[int]) -> float:
    if len(labels) < 2:
        return float("nan")
    aps = []
    for c in labels:
        y = (gt == c).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aps.append(average_precision_score(y, probs[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def _from_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = data["from_logits"]
    gt = data["from_gt"].astype(bool)
    valid = data["my_mask"].astype(bool)

    logits_v = logits[valid]
    gt_v = gt[valid].astype(int)
    probs_v = 1.0 / (1.0 + np.exp(-logits_v))
    pred_v = (probs_v > 0.5).astype(int)

    acc = float((pred_v == gt_v).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        gt_v, pred_v, average="binary", zero_division=0
    )
    prauc = (
        float(average_precision_score(gt_v, probs_v))
        if gt_v.sum() > 0
        else float("nan")
    )
    rocauc = (
        float(roc_auc_score(gt_v, probs_v))
        if 0 < gt_v.sum() < len(gt_v)
        else float("nan")
    )

    per_threshold = []
    for th in (0.05, 0.1, 0.3, 0.5):
        pred_th = (probs_v > th).astype(int)
        p_th, r_th, f1_th, _ = precision_recall_fscore_support(
            gt_v, pred_th, average="binary", zero_division=0
        )
        per_threshold.append(
            {
                "threshold": th,
                "precision": float(p_th),
                "recall": float(r_th),
                "f1": float(f1_th),
                "pred_pos_rate": float(pred_th.mean()),
            }
        )

    return {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": prauc,
        "roc_auc": rocauc,
        "positive_rate_gt": float(gt_v.mean()),
        "positive_rate_pred@0.5": float(pred_v.mean()),
        "prob_mean": float(probs_v.mean()),
        "prob_median": float(np.median(probs_v)),
        "per_threshold": per_threshold,
        "n_valid_slots": int(valid.sum()),
        "n_positive": int(gt_v.sum()),
    }


def _target_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = data["target_logits"]
    gt = data["target_gt"]
    n = len(gt)
    if n == 0:
        return {"n_fired": 0}

    probs = _softmax(logits)
    pred = logits.argmax(axis=-1)
    acc = float((pred == gt).mean())

    macro_f1 = float(f1_score(gt, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gt, pred, average="weighted", zero_division=0))
    per_class_f1 = f1_score(
        gt, pred, labels=list(range(NUM_TEMPLATES)), average=None, zero_division=0
    )

    labels_present = sorted({int(x) for x in gt.tolist()})
    try:
        rocauc_ovr = float(
            roc_auc_score(
                gt, probs, multi_class="ovr", average="macro", labels=labels_present
            )
        )
    except ValueError:
        rocauc_ovr = float("nan")

    prauc_macro = _macro_ovr_pr_auc(gt, probs, labels_present)
    top2 = _top_k_accuracy(logits, gt, k=2)
    cm = confusion_matrix(gt, pred, labels=list(range(NUM_TEMPLATES))).tolist()

    gt_dist = np.bincount(gt, minlength=NUM_TEMPLATES) / n
    pred_dist = np.bincount(pred, minlength=NUM_TEMPLATES) / n

    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    entropy = float(-(probs * log_p).sum(axis=-1).mean())

    return {
        "n_fired": int(n),
        "accuracy": acc,
        "top2_accuracy": top2,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": [float(x) for x in per_class_f1],
        "roc_auc_ovr_macro": rocauc_ovr,
        "pr_auc_ovr_macro": prauc_macro,
        "mean_prediction_entropy": entropy,
        "gt_class_distribution": [float(x) for x in gt_dist],
        "pred_class_distribution": [float(x) for x in pred_dist],
        "confusion_matrix": cm,
    }


def _ships_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = data["ships_logits"]
    gt = data["ships_gt"]
    n = len(gt)
    if n == 0:
        return {"n_fired": 0}

    probs = _softmax(logits)
    pred = logits.argmax(axis=-1)
    acc = float((pred == gt).mean())
    macro_f1 = float(f1_score(gt, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(gt, pred, average="weighted", zero_division=0))
    per_class_f1 = f1_score(
        gt, pred, labels=list(range(SHIPS_BUCKETS)), average=None, zero_division=0
    )
    mae_bucket = float(np.abs(pred.astype(int) - gt.astype(int)).mean())

    labels_present = sorted({int(x) for x in gt.tolist()})
    try:
        rocauc_ovr = float(
            roc_auc_score(
                gt, probs, multi_class="ovr", average="macro", labels=labels_present
            )
        )
    except ValueError:
        rocauc_ovr = float("nan")
    prauc_macro = _macro_ovr_pr_auc(gt, probs, labels_present)

    cm = confusion_matrix(gt, pred, labels=list(range(SHIPS_BUCKETS))).tolist()
    gt_dist = np.bincount(gt, minlength=SHIPS_BUCKETS) / n
    pred_dist = np.bincount(pred, minlength=SHIPS_BUCKETS) / n

    return {
        "n_fired": int(n),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": [float(x) for x in per_class_f1],
        "roc_auc_ovr_macro": rocauc_ovr,
        "pr_auc_ovr_macro": prauc_macro,
        "mae_bucket": mae_bucket,
        "gt_class_distribution": [float(x) for x in gt_dist],
        "pred_class_distribution": [float(x) for x in pred_dist],
        "confusion_matrix": cm,
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    f = report["from_head"]
    t = report["target_head"]
    s = report["ships_head"]
    return {
        "from": {
            "acc": round(f["accuracy"], 4),
            "f1": round(f["f1"], 4),
            "pr_auc": round(f["pr_auc"], 4),
            "roc_auc": round(f["roc_auc"], 4),
            "positive_rate_gt": round(f["positive_rate_gt"], 4),
        },
        "target": {
            "acc": round(t["accuracy"], 4),
            "top2_acc": round(t["top2_accuracy"], 4),
            "macro_f1": round(t["macro_f1"], 4),
            "weighted_f1": round(t["weighted_f1"], 4),
            "pr_auc_macro": round(t["pr_auc_ovr_macro"], 4),
            "roc_auc_macro": round(t["roc_auc_ovr_macro"], 4),
        },
        "ships": {
            "acc": round(s["accuracy"], 4),
            "macro_f1": round(s["macro_f1"], 4),
            "mae_bucket": round(s["mae_bucket"], 4),
            "pr_auc_macro": round(s["pr_auc_ovr_macro"], 4),
            "roc_auc_macro": round(s["roc_auc_ovr_macro"], 4),
        },
    }


def run(cfg: EvalConfig) -> dict[str, Any]:
    cfg_yaml = yaml.safe_load(cfg.config.read_text())
    val_parquet = Path(cfg_yaml["data"]["out_val"])
    val_ds = CaseThreeDataset(val_parquet)
    model = _load_model(cfg.config, cfg.weights)
    preds = _gather_preds(model, val_ds)

    report: dict[str, Any] = {
        "weights": str(cfg.weights),
        "val_parquet": str(val_parquet),
        "n_frames": int(len(val_ds)),
        "from_head": _from_metrics(preds),
        "target_head": _target_metrics(preds),
        "ships_head": _ships_metrics(preds),
    }

    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(json.dumps(report, indent=2))
    return report


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    weights: Path = typer.Option(DEFAULT_WEIGHTS, "--weights", "-w"),  # noqa: B008
    report: Path = typer.Option(DEFAULT_REPORT, "--report", "-r"),  # noqa: B008
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),  # noqa: B008
) -> None:
    cfg = EvalConfig(weights=weights, report=report, config=config)
    report_data = run(cfg)
    typer.echo(f"saved: {cfg.report}")
    typer.echo(json.dumps(_summary(report_data), indent=2))


if __name__ == "__main__":
    app()
