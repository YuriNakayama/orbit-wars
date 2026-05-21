"""Token bucket 形式のレートリミッタ。

Kaggle EpisodeService への呼び出しを 60 req/60s 以下に抑えるための
スレッドセーフな粗粒度制御。`with bucket.acquire(): ...` で利用する。

429 を観測したときに `report_429()` で capacity を一時的に半減させる
adaptive 機構を持つ。cooldown_sec 経過後の最初の `report_ok()` で
capacity が +1 ずつ復元する (initial_capacity が上限)。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TokenBucket:
    """固定ウィンドウ型トークンバケット（指定秒あたりのリクエスト数制限）。"""

    def __init__(
        self,
        capacity: int,
        window_sec: float,
        *,
        cooldown_sec: float = 60.0,
        min_capacity: int = 1,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {window_sec}")
        self._initial_capacity = capacity
        self._capacity = capacity
        self._min_capacity = max(1, min_capacity)
        self._window_sec = window_sec
        self._cooldown_sec = cooldown_sec
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._timestamps: list[float] = []
        self._cooldown_until: float = 0.0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def window_sec(self) -> float:
        return self._window_sec

    def report_429(self) -> None:
        """429 を観測した worker から呼ぶ。capacity を半減し cooldown を張る。

        同一 cooldown 中に複数 worker が呼んでも 1 度しか半減しない (再エントリ抑止)。
        """

        with self._lock:
            now = self._clock()
            if now < self._cooldown_until:
                return
            new_capacity = max(self._min_capacity, self._capacity // 2)
            if new_capacity == self._capacity:
                self._cooldown_until = now + self._cooldown_sec
                return
            self._capacity = new_capacity
            self._cooldown_until = now + self._cooldown_sec
            logger.warning(
                "rate_limit: 429 observed, capacity %d -> %d (cooldown %.0fs)",
                self._initial_capacity,
                self._capacity,
                self._cooldown_sec,
            )

    def report_ok(self) -> None:
        """成功 req 後に呼ぶ。cooldown 経過後なら capacity を +1 復元。"""

        with self._lock:
            if self._capacity >= self._initial_capacity:
                return
            now = self._clock()
            if now < self._cooldown_until:
                return
            self._capacity = min(self._initial_capacity, self._capacity + 1)
            logger.info(
                "rate_limit: capacity restored to %d / %d",
                self._capacity,
                self._initial_capacity,
            )

    @contextmanager
    def acquire(self) -> Iterator[None]:
        while True:
            with self._lock:
                now = self._clock()
                threshold = now - self._window_sec
                self._timestamps = [t for t in self._timestamps if t > threshold]
                if len(self._timestamps) < self._capacity:
                    self._timestamps.append(now)
                    break
                wait = self._window_sec - (now - self._timestamps[0])
            self._sleeper(max(wait, 0.01))
        yield
