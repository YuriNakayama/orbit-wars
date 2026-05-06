"""Tests for runpod_io.retry decision logic."""

from __future__ import annotations

from runpod_io import retry
from runpod_io.retry import FailureReason


def test_is_retryable_subset() -> None:
    assert retry.is_retryable(FailureReason.SCHEDULING_FAILED)
    assert retry.is_retryable(FailureReason.IMAGE_PULL_STUCK)
    assert retry.is_retryable(FailureReason.DRIVER_MISMATCH)
    assert not retry.is_retryable(FailureReason.OOM)
    assert not retry.is_retryable(FailureReason.NON_ZERO_EXIT)
    assert not retry.is_retryable(FailureReason.COST_LIMIT_HIT)


def test_decide_first_retryable_failure_retries() -> None:
    d = retry.decide(
        failure_reason=FailureReason.SCHEDULING_FAILED,
        attempts_so_far=1,
        max_retries=2,
    )
    assert d.should_retry is True
    assert d.attempt_after_this == 2


def test_decide_caps_at_max_retries() -> None:
    """attempts_so_far=3 with max_retries=2 means we've already used the
    initial attempt + 2 retries — no more retries."""
    d = retry.decide(
        failure_reason=FailureReason.SCHEDULING_FAILED,
        attempts_so_far=3,
        max_retries=2,
    )
    assert d.should_retry is False
    assert "max_retries" in d.explanation


def test_decide_non_retryable_does_not_retry() -> None:
    d = retry.decide(
        failure_reason=FailureReason.OOM,
        attempts_so_far=1,
        max_retries=2,
    )
    assert d.should_retry is False
    assert "not in the retryable set" in d.explanation


def test_decide_image_pull_stuck_at_boundary() -> None:
    """attempts_so_far=2 with max_retries=2 should still allow one final retry."""
    d = retry.decide(
        failure_reason=FailureReason.IMAGE_PULL_STUCK,
        attempts_so_far=2,
        max_retries=2,
    )
    assert d.should_retry is True
    assert d.attempt_after_this == 3


def test_chain_append_is_immutable() -> None:
    chain = retry.RetryChain(head_run_id="rid_1", attempts=["rid_1"])
    new_chain = retry.append_attempt(chain, "rid_2")
    assert chain.attempts == ["rid_1"]  # unchanged
    assert new_chain.attempts == ["rid_1", "rid_2"]
    assert new_chain.length == 2
