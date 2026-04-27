"""Unit tests for pipeline.imitation.case2.evaluation.eval_metrics."""

from __future__ import annotations

import numpy as np

from pipeline.imitation.case2.evaluation.eval_metrics import (
    FromArrays,
    ShipsArrays,
    TargetArrays,
    best_f1_threshold,
    expected_calibration_error,
    expected_calibration_error_multiclass,
    from_metrics,
    ships_metrics,
    target_metrics,
)
from pipeline.imitation.case2.policy.templates import NUM_TEMPLATES


def test_ece_perfect_calibration_is_zero() -> None:
    probs = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    labels = np.array([False, False, True, True], dtype=np.bool_)
    assert expected_calibration_error(probs, labels) == 0.0


def test_ece_full_miscalibration() -> None:
    probs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    labels = np.array([False, False, True, True], dtype=np.bool_)
    # Confidence 1.0 / accuracy 0.0 -> ECE = 1.0
    assert expected_calibration_error(probs, labels) == 1.0


def test_best_f1_perfect_separation() -> None:
    probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    labels = np.array([False, False, True, True], dtype=np.bool_)
    f1, t = best_f1_threshold(probs, labels)
    assert f1 == 1.0
    assert 0.2 < t <= 0.8


def test_from_metrics_keys_and_ranges() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.bool_)
    probs = np.where(labels, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)).astype(
        np.float32
    )
    arr = FromArrays(probs=probs, labels=labels)
    m = from_metrics(arr)
    assert m["n"] == n
    assert m["n_pos"] + m["n_neg"] == n
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["f1_best"] <= 1.0
    assert 0.0 <= m["f1_threshold"] <= 1.0
    assert m["log_loss"] >= 0.0
    assert 0.0 <= m["ece"] <= 1.0


def test_from_metrics_handles_all_negative() -> None:
    probs = np.full(50, 0.1, dtype=np.float32)
    labels = np.zeros(50, dtype=np.bool_)
    m = from_metrics(FromArrays(probs=probs, labels=labels))
    # AUC undefined when only one class present.
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None
    assert m["log_loss"] >= 0.0


def test_target_metrics_perfect_predictions() -> None:
    n = 80
    labels = np.array([i % NUM_TEMPLATES for i in range(n)], dtype=np.int64)
    probs = np.zeros((n, NUM_TEMPLATES), dtype=np.float32)
    for i, c in enumerate(labels):
        probs[i, c] = 1.0
    m = target_metrics(TargetArrays(probs=probs, labels=labels))
    assert m["top1_acc"] == 1.0
    assert m["top2_acc"] == 1.0
    assert m["macro_f1"] == 1.0
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (NUM_TEMPLATES, NUM_TEMPLATES)
    # Diagonal should be the only non-zero entries.
    assert int(cm.sum() - np.trace(cm)) == 0


def test_target_metrics_random_predictions_macro_f1_in_range() -> None:
    rng = np.random.default_rng(1)
    n = 400
    labels = rng.integers(0, NUM_TEMPLATES, size=n).astype(np.int64)
    logits = rng.normal(size=(n, NUM_TEMPLATES)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    m = target_metrics(TargetArrays(probs=probs.astype(np.float32), labels=labels))
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert len(m["per_class_f1"]) == NUM_TEMPLATES
    assert len(m["per_class_recall"]) == NUM_TEMPLATES
    assert sum(m["per_class_support"]) == n


def test_ships_metrics_ordinal_mae() -> None:
    # Predictions off-by-one from labels → MAE should be 1.0.
    n = 40
    k = 4
    labels = np.tile(np.array([0, 1, 2, 3]), n // 4).astype(np.int64)
    probs = np.zeros((n, k), dtype=np.float32)
    for i, c in enumerate(labels):
        pred = (c + 1) % k
        probs[i, pred] = 1.0
    m = ships_metrics(ShipsArrays(probs=probs, labels=labels))
    assert m["accuracy"] == 0.0
    # off-by-one within ring; c=3 -> pred=0 distance 3, others distance 1.
    assert m["ordinal_mae"] > 0.0
    assert len(m["per_class_recall"]) == k


def test_ece_multiclass_perfect_is_zero() -> None:
    n = 30
    labels = np.array([i % 4 for i in range(n)], dtype=np.int64)
    probs = np.zeros((n, 4), dtype=np.float32)
    for i, c in enumerate(labels):
        probs[i, c] = 1.0
    assert expected_calibration_error_multiclass(probs, labels) == 0.0


def test_empty_metrics_safe() -> None:
    empty_from = FromArrays(
        probs=np.zeros(0, dtype=np.float32), labels=np.zeros(0, dtype=np.bool_)
    )
    m = from_metrics(empty_from)
    assert m["n"] == 0
    assert m["roc_auc"] is None

    empty_target = TargetArrays(
        probs=np.zeros((0, NUM_TEMPLATES), dtype=np.float32),
        labels=np.zeros(0, dtype=np.int64),
    )
    assert target_metrics(empty_target) == {"n": 0}

    empty_ships = ShipsArrays(
        probs=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros(0, dtype=np.int64),
    )
    assert ships_metrics(empty_ships) == {"n": 0}
