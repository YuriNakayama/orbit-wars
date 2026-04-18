"""Token bucket 形式のレートリミッタ。

Kaggle EpisodeService への呼び出しを 60 req/60s 以下に抑えるための
スレッドセーフな粗粒度制御。`with bucket.acquire(): ...` で利用する。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class TokenBucket:
    """固定ウィンドウ型トークンバケット（指定秒あたりのリクエスト数制限）。"""

    def __init__(
        self,
        capacity: int,
        window_sec: float,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {window_sec}")
        self._capacity = capacity
        self._window_sec = window_sec
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def window_sec(self) -> float:
        return self._window_sec

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
