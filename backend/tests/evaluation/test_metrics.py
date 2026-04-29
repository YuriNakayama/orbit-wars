"""Unit tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import (
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


def test_ece_perfect_calibration_returns_zero() -> None:
    # All predictions exactly match labels (probs in {0, 1}).
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([False, False, True, True])
    assert expected_calibration_error(probs, labels, n_bins=2) == 0.0


def test_ece_complete_miscalibration() -> None:
    # All probabilities 1.0 but all labels False → ECE = 1.0.
    probs = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([False, False, False, False])
    assert expected_calibration_error(probs, labels, n_bins=10) == 1.0


def test_ece_empty_input() -> None:
    assert expected_calibration_error(np.zeros(0), np.zeros(0, dtype=np.bool_)) == 0.0


def test_ece_multiclass_perfect() -> None:
    probs = np.eye(3, dtype=np.float32)
    labels = np.array([0, 1, 2])
    assert expected_calibration_error_multiclass(probs, labels, n_bins=2) == 0.0


def test_ece_multiclass_empty() -> None:
    assert expected_calibration_error_multiclass(np.zeros((0, 3)), np.zeros(0)) == 0.0


def test_best_f1_threshold_perfect_separation() -> None:
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    f1, threshold = best_f1_threshold(probs, labels, num_thresholds=11)
    assert f1 == 1.0
    assert 0.0 < threshold <= 0.8


def test_best_f1_threshold_empty_input() -> None:
    f1, threshold = best_f1_threshold(np.zeros(0), np.zeros(0, dtype=np.bool_))
    assert f1 == 0.0
    assert threshold == 0.5


def test_from_metrics_zero_samples() -> None:
    arr = FromArrays(probs=np.zeros(0), labels=np.zeros(0, dtype=np.bool_))
    out = from_metrics(arr)
    assert out["n"] == 0
    assert out["roc_auc"] is None


def test_from_metrics_balanced_separable() -> None:
    arr = FromArrays(
        probs=np.array([0.1, 0.2, 0.8, 0.9]),
        labels=np.array([False, False, True, True]),
    )
    out = from_metrics(arr)
    assert out["n"] == 4
    assert out["n_pos"] == 2
    assert out["n_neg"] == 2
    assert out["roc_auc"] == 1.0
    assert out["f1_best"] == 1.0


def test_from_metrics_all_positive_class() -> None:
    arr = FromArrays(probs=np.array([0.5, 0.6]), labels=np.array([True, True]))
    out = from_metrics(arr)
    # roc_auc undefined when only one class present.
    assert out["roc_auc"] is None
    assert out["pr_auc"] is None


def test_target_metrics_perfect() -> None:
    probs = np.array(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], dtype=np.float32
    )
    labels = np.array([0, 1, 2])
    out = target_metrics(TargetArrays(probs=probs, labels=labels), num_classes=3)
    assert out["n"] == 3
    assert out["top1_acc"] == 1.0
    assert out["macro_f1"] == 1.0


def test_target_metrics_empty() -> None:
    out = target_metrics(
        TargetArrays(probs=np.zeros((0, 4), dtype=np.float32), labels=np.zeros(0)),
        num_classes=4,
    )
    assert out == {"n": 0}


def test_ships_metrics_perfect() -> None:
    probs = np.array([[0.9, 0.05, 0.05, 0.0], [0.0, 0.9, 0.05, 0.05]], dtype=np.float32)
    labels = np.array([0, 1])
    out = ships_metrics(ShipsArrays(probs=probs, labels=labels))
    assert out["n"] == 2
    assert out["accuracy"] == 1.0
    assert out["ordinal_mae"] == 0.0


def test_ships_metrics_off_by_one() -> None:
    probs = np.array([[0.05, 0.9, 0.05, 0.0], [0.0, 0.05, 0.9, 0.05]], dtype=np.float32)
    labels = np.array([0, 1])
    out = ships_metrics(ShipsArrays(probs=probs, labels=labels))
    # Pred is always one bucket above the label.
    assert out["accuracy"] == 0.0
    assert out["ordinal_mae"] == 1.0


def test_ships_metrics_empty() -> None:
    out = ships_metrics(ShipsArrays(probs=np.zeros((0, 4)), labels=np.zeros(0)))
    assert out == {"n": 0}
