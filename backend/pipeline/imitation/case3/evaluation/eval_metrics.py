"""Multi-faceted validation metrics for imitation/case3 IL heads.

Loads val.parquet via CaseThreeDataset, runs the model in eval mode, and
collects per-head predictions/labels for the full set. Computes:

- from head (per-planet multi-hot, masked to my_planet_mask):
    ROC-AUC, PR-AUC, best-F1 + threshold, log-loss, ECE (10 bins),
    n_pos / n_neg, support count.

- target head (per-fired-source 8-class):
    top-1 / top-2 accuracy, macro-F1, per-class precision/recall/F1,
    confusion matrix, NLL, ECE.

- ships head (per-fired-source 4-class, ordinal):
    accuracy, macro-F1, ordinal MAE (|pred - true|), per-class recall,
    NLL, ECE.

Output: JSON for iter-to-iter comparison. Self-play is intentionally not run
here — feature-engineering effects are evaluated on validation metrics only.
"""

from __future__ import annotations

import json
import logging
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
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FromArrays:
    probs: np.ndarray  # (N,) float in [0, 1] over my-planet positives
    labels: np.ndarray  # (N,) bool


@dataclass(frozen=True)
class TargetArrays:
    probs: np.ndarray  # (M, NUM_TEMPLATES) softmax
    labels: np.ndarray  # (M,) int64 in [0, NUM_TEMPLATES)


@dataclass(frozen=True)
class ShipsArrays:
    probs: np.ndarray  # (M, K) softmax
    labels: np.ndarray  # (M,) int64 in [0, K)


def _to_batch_features(sample: BatchedSample) -> BatchFeatures:
    return BatchFeatures(
        planet_feats=sample.planet_feats,
        planet_mask=sample.planet_mask,
        my_planet_mask=sample.my_planet_mask,
        target_mask=sample.target_mask,
        global_feats=sample.global_feats,
        template_ctx=sample.template_ctx,
    )


def collect_arrays(
    model: DeepSetsPolicy, loader: DataLoader[BatchedSample]
) -> tuple[FromArrays, TargetArrays, ShipsArrays]:
    """Run model in eval mode over loader, return per-head arrays."""
    model.eval()
    from_probs_list: list[np.ndarray] = []
    from_labels_list: list[np.ndarray] = []
    target_probs_list: list[np.ndarray] = []
    target_labels_list: list[np.ndarray] = []
    ships_probs_list: list[np.ndarray] = []
    ships_labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features = _to_batch_features(batch)
            output = model(features)
            from_logits = output.from_logits  # (B, P)
            target_logits = output.target_logits  # (B, P, NUM_TEMPLATES)
            ships_logits = output.ships_logits  # (B, P, K)

            my_mask = batch.my_planet_mask  # (B, P)
            from_multihot = batch.from_multihot  # (B, P)

            # from head: collect probs/labels at every my-planet slot.
            valid = my_mask
            from_probs_full = torch.sigmoid(from_logits)
            from_probs_list.append(
                from_probs_full[valid].detach().cpu().numpy().astype(np.float32)
            )
            from_labels_list.append(
                from_multihot[valid].detach().cpu().numpy().astype(np.bool_)
            )

            # target / ships heads: collect at fired sources only (label != -1
            # equivalent to fired_mask = from_multihot & my_mask).
            fired = from_multihot & valid
            if fired.any():
                b_idx, p_idx = fired.nonzero(as_tuple=True)
                tp = (
                    torch.softmax(target_logits[b_idx, p_idx], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                tl = (
                    batch.target_per_src[b_idx, p_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                sp = (
                    torch.softmax(ships_logits[b_idx, p_idx], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                sl = (
                    batch.ships_per_src[b_idx, p_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                target_probs_list.append(tp)
                target_labels_list.append(tl)
                ships_probs_list.append(sp)
                ships_labels_list.append(sl)

    from_arr = FromArrays(
        probs=(
            np.concatenate(from_probs_list)
            if from_probs_list
            else np.zeros(0, dtype=np.float32)
        ),
        labels=(
            np.concatenate(from_labels_list)
            if from_labels_list
            else np.zeros(0, dtype=np.bool_)
        ),
    )
    if target_probs_list:
        target_arr = TargetArrays(
            probs=np.concatenate(target_probs_list, axis=0),
            labels=np.concatenate(target_labels_list, axis=0),
        )
        ships_arr = ShipsArrays(
            probs=np.concatenate(ships_probs_list, axis=0),
            labels=np.concatenate(ships_labels_list, axis=0),
        )
    else:
        target_arr = TargetArrays(
            probs=np.zeros((0, NUM_TEMPLATES), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )
        ships_arr = ShipsArrays(
            probs=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )
    return from_arr, target_arr, ships_arr


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE for binary predictions (probs of positive class) vs bool labels."""
    if probs.size == 0:
        return 0.0
    probs = probs.astype(np.float64)
    labels = labels.astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (
            (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(probs[in_bin].mean())
        avg_acc = float(labels[in_bin].mean())
        ece += abs(avg_conf - avg_acc) * (count / n)
    return float(ece)


def expected_calibration_error_multiclass(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE on top-1 confidence vs correctness for multiclass softmax."""
    if probs.size == 0 or labels.size == 0:
        return 0.0
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1).astype(np.float64)
    correct = (pred == labels).astype(np.float64)
    return expected_calibration_error(conf, correct.astype(np.bool_), n_bins=n_bins)


def best_f1_threshold(
    probs: np.ndarray, labels: np.ndarray, num_thresholds: int = 101
) -> tuple[float, float]:
    """Sweep thresholds in [0,1] and return (best_f1, best_threshold)."""
    if probs.size == 0:
        return 0.0, 0.5
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.0, 1.0, num_thresholds):
        preds = probs >= t
        if preds.sum() == 0 and labels.sum() == 0:
            f1 = 1.0
        elif preds.sum() == 0 or labels.sum() == 0:
            f1 = 0.0
        else:
            f1 = float(f1_score(labels.astype(np.int32), preds.astype(np.int32)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_f1, best_t


def from_metrics(arr: FromArrays) -> dict[str, Any]:
    n = int(arr.probs.size)
    n_pos = int(arr.labels.sum())
    n_neg = n - n_pos
    if n == 0:
        return {
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "roc_auc": None,
            "pr_auc": None,
            "f1_best": None,
            "f1_threshold": None,
            "log_loss": None,
            "ece": None,
        }
    out: dict[str, Any] = {"n": n, "n_pos": n_pos, "n_neg": n_neg}
    if 0 < n_pos < n:
        out["roc_auc"] = float(roc_auc_score(arr.labels.astype(np.int32), arr.probs))
        out["pr_auc"] = float(
            average_precision_score(arr.labels.astype(np.int32), arr.probs)
        )
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    out["log_loss"] = float(
        log_loss(
            arr.labels.astype(np.int32), arr.probs.clip(1e-7, 1 - 1e-7), labels=[0, 1]
        )
    )
    f1_best, t_best = best_f1_threshold(arr.probs, arr.labels)
    out["f1_best"] = f1_best
    out["f1_threshold"] = t_best
    out["ece"] = expected_calibration_error(arr.probs, arr.labels, n_bins=10)
    return out


def target_metrics(arr: TargetArrays) -> dict[str, Any]:
    n = int(arr.labels.size)
    if n == 0:
        return {"n": 0}
    pred1 = arr.probs.argmax(axis=1)
    top2 = np.argsort(-arr.probs, axis=1)[:, :2]
    top1_acc = float((pred1 == arr.labels).mean())
    top2_acc = float((top2 == arr.labels[:, None]).any(axis=1).mean())
    macro_f1 = float(
        f1_score(arr.labels, pred1, average="macro", labels=list(range(NUM_TEMPLATES)))
    )
    prec, rec, f1c, sup = precision_recall_fscore_support(
        arr.labels,
        pred1,
        labels=list(range(NUM_TEMPLATES)),
        zero_division=0,
    )
    cm = confusion_matrix(arr.labels, pred1, labels=list(range(NUM_TEMPLATES)))
    nll = float(
        log_loss(
            arr.labels,
            arr.probs.clip(1e-7, 1 - 1e-7),
            labels=list(range(NUM_TEMPLATES)),
        )
    )
    ece = expected_calibration_error_multiclass(arr.probs, arr.labels, n_bins=10)
    return {
        "n": n,
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "macro_f1": macro_f1,
        "per_class_precision": [float(x) for x in prec],
        "per_class_recall": [float(x) for x in rec],
        "per_class_f1": [float(x) for x in f1c],
        "per_class_support": [int(x) for x in sup],
        "confusion_matrix": cm.tolist(),
        "nll": nll,
        "ece": ece,
    }


def ships_metrics(arr: ShipsArrays) -> dict[str, Any]:
    n = int(arr.labels.size)
    if n == 0:
        return {"n": 0}
    k = int(arr.probs.shape[1])
    pred = arr.probs.argmax(axis=1)
    acc = float((pred == arr.labels).mean())
    macro_f1 = float(f1_score(arr.labels, pred, average="macro", labels=list(range(k))))
    ordinal_mae = float(np.abs(pred.astype(np.int64) - arr.labels).mean())
    _, rec, _, sup = precision_recall_fscore_support(
        arr.labels, pred, labels=list(range(k)), zero_division=0
    )
    nll = float(
        log_loss(arr.labels, arr.probs.clip(1e-7, 1 - 1e-7), labels=list(range(k)))
    )
    ece = expected_calibration_error_multiclass(arr.probs, arr.labels, n_bins=10)
    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ordinal_mae": ordinal_mae,
        "per_class_recall": [float(x) for x in rec],
        "per_class_support": [int(x) for x in sup],
        "nll": nll,
        "ece": ece,
    }


def compute_all_metrics(
    model: DeepSetsPolicy, loader: DataLoader[BatchedSample]
) -> dict[str, Any]:
    from_arr, target_arr, ships_arr = collect_arrays(model, loader)
    return {
        "from": from_metrics(from_arr),
        "target": target_metrics(target_arr),
        "ships": ships_metrics(ships_arr),
    }


def _build_model(weights_path: Path, model_cfg: dict[str, Any]) -> DeepSetsPolicy:
    cfg = ModelConfig(
        planet_in_dim=int(model_cfg.get("planet_in_dim", PLANET_FEAT_DIM)),
        global_in_dim=int(model_cfg.get("global_in_dim", GLOBAL_FEAT_DIM)),
        hidden=int(model_cfg.get("hidden", 128)),
        ships_buckets=int(model_cfg.get("ships_buckets", 4)),
    )
    model = DeepSetsPolicy(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    weights: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case3/policy/weights.pt"),
        "--weights",
        "-w",
        help="Trained weights path",
    ),
    val: Path = typer.Option(  # noqa: B008
        Path("data/mart/imitation/case3/val.parquet"),
        "--val",
        "-v",
        help="Validation parquet path",
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case3/configs/il_baseline.yaml"),
        "--config",
        "-c",
        help="Config YAML (read for model dims)",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("pipeline/imitation/case3/evaluation/results_metrics.json"),
        "--out",
        "-o",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    label: str = typer.Option("", "--label", help="Free-form tag stored in JSON"),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with config.open() as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    model = _build_model(weights, model_cfg)
    ablation_cfg = (cfg.get("data", {}) or {}).get("ablation", {}) or {}
    ds = CaseThreeDataset(
        val,
        mask_planet_cols=list(ablation_cfg.get("planet_cols", []) or []),
        mask_global_cols=list(ablation_cfg.get("global_cols", []) or []),
    )
    loader: DataLoader[BatchedSample] = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    metrics = compute_all_metrics(model, loader)
    metrics["weights"] = str(weights)
    metrics["val"] = str(val)
    if label:
        metrics["label"] = label
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
