"""Pure-computation metric helpers shared across imitation cases.

These functions take pre-collected NumPy arrays and never touch the model,
the dataset, or the filesystem. The case-specific `eval_metrics.py` wrappers
build the arrays (calling `collect_arrays` with case-local model/loader types)
and then delegate the metric math to functions here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class FromArrays:
    probs: np.ndarray  # (N,) float in [0, 1]
    labels: np.ndarray  # (N,) bool


@dataclass(frozen=True)
class TargetArrays:
    probs: np.ndarray  # (M, NUM_TEMPLATES) softmax
    labels: np.ndarray  # (M,) int64


@dataclass(frozen=True)
class ShipsArrays:
    probs: np.ndarray  # (M, K) softmax
    labels: np.ndarray  # (M,) int64


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
            arr.labels.astype(np.int32),
            arr.probs.clip(1e-7, 1 - 1e-7),
            labels=[0, 1],
        )
    )
    f1_best, t_best = best_f1_threshold(arr.probs, arr.labels)
    out["f1_best"] = f1_best
    out["f1_threshold"] = t_best
    out["ece"] = expected_calibration_error(arr.probs, arr.labels, n_bins=10)
    return out


def target_metrics(arr: TargetArrays, num_classes: int) -> dict[str, Any]:
    """Per-class metrics for the target head with `num_classes` templates."""
    n = int(arr.labels.size)
    if n == 0:
        return {"n": 0}
    pred1 = arr.probs.argmax(axis=1)
    top2 = np.argsort(-arr.probs, axis=1)[:, :2]
    top1_acc = float((pred1 == arr.labels).mean())
    top2_acc = float((top2 == arr.labels[:, None]).any(axis=1).mean())
    macro_f1 = float(
        f1_score(arr.labels, pred1, average="macro", labels=list(range(num_classes)))
    )
    prec, rec, f1c, sup = precision_recall_fscore_support(
        arr.labels,
        pred1,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(arr.labels, pred1, labels=list(range(num_classes)))
    nll = float(
        log_loss(
            arr.labels,
            arr.probs.clip(1e-7, 1 - 1e-7),
            labels=list(range(num_classes)),
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
