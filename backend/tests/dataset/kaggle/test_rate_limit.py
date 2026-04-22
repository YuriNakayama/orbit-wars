"""Unit tests for src/dataset/kaggle/rate_limit.py."""

from __future__ import annotations

import pytest

from dataset.kaggle.rate_limit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_bucket_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(capacity=0, window_sec=60.0)


def test_bucket_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_sec"):
        TokenBucket(capacity=5, window_sec=0.0)


def test_bucket_allows_up_to_capacity_without_sleep() -> None:
    fake = FakeClock()
    bucket = TokenBucket(
        capacity=3, window_sec=60.0, clock=fake.clock, sleeper=fake.sleep
    )
    for _ in range(3):
        with bucket.acquire():
            pass
    assert fake.sleeps == []


def test_bucket_waits_when_full() -> None:
    fake = FakeClock()
    bucket = TokenBucket(
        capacity=2, window_sec=10.0, clock=fake.clock, sleeper=fake.sleep
    )

    with bucket.acquire():
        pass  # t=0 の1件目
    fake.now = 1.0
    with bucket.acquire():
        pass  # t=1 の2件目
    fake.now = 2.0
    with bucket.acquire():
        pass  # t=2 で3件目 → 最初の t=0 を 10s 待つ必要がある

    assert fake.sleeps, "3件目は sleep が入るはず"
    assert fake.sleeps[0] == pytest.approx(8.0, abs=0.01)
