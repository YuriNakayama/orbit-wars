"""Epoch-level diagnostic metrics for case9 imitation training."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

from pipeline.imitation.case9.policy.templates import NUM_TEMPLATES
from pipeline.imitation.case9.policy.types import PolicyOutput
from pipeline.imitation.case9.training.dataset import BatchedSample


@dataclass
class _BinaryStore:
    y_true: list[np.ndarray] = field(default_factory=list)
    y_score: list[np.ndarray] = field(default_factory=list)


@dataclass
class _MultiStore:
    y_true: list[np.ndarray] = field(default_factory=list)
    y_prob: list[np.ndarray] = field(default_factory=list)


class EpochMetricAccumulator:
    """Collect logits/labels and compute non-accuracy diagnostics per epoch.

    Stores at most `max_samples` examples per metric stream. This keeps train
    epochs bounded while still making validation AUC/F1/MAE comparable.
    """

    def __init__(self, max_samples: int = 500_000) -> None:
        self.max_samples = max(0, int(max_samples))
        self.fire = _BinaryStore()
        self.decision = _MultiStore()
        self.ships = _MultiStore()
        self._counts = {"fire": 0, "decision": 0, "ships": 0}

    def update(
        self,
        head_mode: str,
        output: PolicyOutput,
        batch: BatchedSample,
    ) -> None:
        my = batch.my_planet_mask.bool()
        if head_mode == "three_head":
            self._update_three_head(output, batch, my)
        elif head_mode == "candidate_ships":
            self._update_candidate_ships(output, batch, my)
        elif head_mode == "template_ships":
            self._update_template_ships(output, batch, my)
        elif head_mode == "per_planet":
            self._update_per_planet(output, batch, my)
        elif head_mode == "dual":
            self._update_three_head(output, batch, my)
        else:
            self._update_candidate_decision(output, batch, my)

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(_prefix("fire", _binary_metrics(self.fire)))
        out.update(_prefix("decision", _multi_metrics(self.decision)))
        out.update(_prefix("ships", _multi_metrics(self.ships)))
        return out

    def _room(self, name: str, n: int) -> int:
        if self.max_samples <= 0:
            return n
        room = self.max_samples - self._counts[name]
        return max(0, min(room, n))

    def _add_binary(self, name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
        n = self._room(name, int(y_true.shape[0]))
        if n <= 0:
            return
        store = self.fire
        store.y_true.append(y_true[:n].astype(np.int64, copy=False))
        store.y_score.append(y_score[:n].astype(np.float64, copy=False))
        self._counts[name] += n

    def _add_multi(self, name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        n = self._room(name, int(y_true.shape[0]))
        if n <= 0:
            return
        store = self.decision if name == "decision" else self.ships
        store.y_true.append(y_true[:n].astype(np.int64, copy=False))
        store.y_prob.append(y_prob[:n].astype(np.float64, copy=False))
        self._counts[name] += n

    def _update_three_head(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
    ) -> None:
        if output.from_logits is None or output.target_logits is None:
            return
        y_fire = _to_numpy(batch.from_multihot[my]).astype(np.int64)
        s_fire = _to_numpy(torch.sigmoid(output.from_logits[my]))
        self._add_binary("fire", y_fire, s_fire)

        target_valid = my & (batch.target_per_src != -1)
        if target_valid.any():
            self._add_multi(
                "decision",
                _to_numpy(batch.target_per_src[target_valid]).astype(np.int64),
                _to_numpy(torch.softmax(output.target_logits[target_valid], dim=-1)),
            )
        self._update_ships(output, batch, my, use_bucket=False)

    def _update_candidate_ships(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
    ) -> None:
        self._update_candidate_decision(output, batch, my)
        self._update_ships(output, batch, my, use_bucket=True)

    def _update_template_ships(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
    ) -> None:
        if output.target_logits is None:
            return
        labels_full = torch.where(
            batch.target_per_src >= 0,
            batch.target_per_src,
            torch.full_like(batch.target_per_src, NUM_TEMPLATES - 1),
        )
        probs = _to_numpy(torch.softmax(output.target_logits[my], dim=-1))
        labels = _to_numpy(labels_full[my]).astype(np.int64)
        self._add_binary(
            "fire",
            (labels != NUM_TEMPLATES - 1).astype(np.int64),
            1.0 - probs[:, NUM_TEMPLATES - 1],
        )
        self._add_multi("decision", labels, probs)
        self._update_ships(output, batch, my, use_bucket=False)

    def _update_per_planet(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
    ) -> None:
        if output.per_planet_logits is None:
            return
        # last index of (P+1) is the no-op sentinel.
        no_op_idx = int(output.per_planet_logits.shape[-1]) - 1
        if not my.any():
            return
        probs = _to_numpy(torch.softmax(output.per_planet_logits[my], dim=-1))
        labels = _to_numpy(batch.target_pid_per_src[my]).astype(np.int64)
        self._add_binary(
            "fire",
            (labels != no_op_idx).astype(np.int64),
            1.0 - probs[:, no_op_idx],
        )
        self._add_multi("decision", labels, probs)

    def _update_candidate_decision(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
    ) -> None:
        if output.candidate_logits is None:
            return
        valid = my & (batch.cand_slot_per_src != -1)
        if not valid.any():
            return
        probs = _to_numpy(torch.softmax(output.candidate_logits[valid], dim=-1))
        labels = _to_numpy(batch.cand_slot_per_src[valid]).astype(np.int64)
        self._add_binary("fire", (labels != 0).astype(np.int64), 1.0 - probs[:, 0])
        self._add_multi("decision", labels, probs)

    def _update_ships(
        self,
        output: PolicyOutput,
        batch: BatchedSample,
        my: torch.Tensor,
        *,
        use_bucket: bool,
    ) -> None:
        if output.ships_logits is None:
            return
        labels_tensor = (
            batch.ships_bucket_per_src if use_bucket else batch.ships_per_src
        )
        valid = my & (labels_tensor != -1)
        if not valid.any():
            return
        self._add_multi(
            "ships",
            _to_numpy(labels_tensor[valid]).astype(np.int64),
            _to_numpy(torch.softmax(output.ships_logits[valid], dim=-1)),
        )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _join(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def _safe_float(fn: Callable[[], float]) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(fn())
    except ValueError:
        return float("nan")
    return value


def _binary_metrics(store: _BinaryStore) -> dict[str, float]:
    y_true = _join(store.y_true).astype(np.int64)
    y_score = _join(store.y_score).astype(np.float64)
    if y_true.size == 0:
        return {"count": 0.0}
    y_pred = (y_score >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "count": float(y_true.size),
        "positive_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
        "roc_auc": _safe_float(lambda: roc_auc_score(y_true, y_score)),
        "pr_auc": _safe_float(lambda: average_precision_score(y_true, y_score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": _safe_float(lambda: balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def _multi_metrics(store: _MultiStore) -> dict[str, float]:
    y_true = _join(store.y_true).astype(np.int64)
    y_prob = _join(store.y_prob).astype(np.float64)
    if y_true.size == 0:
        return {"count": 0.0}
    y_pred = y_prob.argmax(axis=1).astype(np.int64)
    return {
        "count": float(y_true.size),
        "macro_roc_auc": _multiclass_auc(y_true, y_prob),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_acc": _safe_float(lambda: balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    labels = sorted(np.unique(y_true).astype(int).tolist())
    if len(labels) < 2:
        return float("nan")
    if len(labels) == 2:
        pos = labels[1]
        return _safe_float(
            lambda: roc_auc_score((y_true == pos).astype(np.int64), y_prob[:, pos])
        )
    return _safe_float(lambda: _ovr_auc_present_labels(y_true, y_prob, labels))


def _ovr_auc_present_labels(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[int]
) -> float:
    probs = y_prob[:, labels]
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    remapped = np.array([labels.index(int(v)) for v in y_true], dtype=np.int64)
    return float(roc_auc_score(remapped, probs, multi_class="ovr", average="macro"))


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
