"""watcher.watch_pod のユニットテスト。

外部副作用 (時刻 / sleep / S3 / SDK / 通知) を全て差し替えてテストする。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from runpod_io.pod_state import PodState
from runpod_io.progress import ProgressMarker
from runpod_io.watcher import WatchResult, _classify_markers, watch_pod


def _pod(status: str, uptime: int = 60) -> PodState:
    return PodState(
        pod_id="pod-x",
        name="run-x",
        desired_status=status,
        cost_per_hr=0.46,
        uptime_seconds=uptime,
        gpu_display_name="RTX 3090",
        image_name="runpod/pytorch:1.0.3",
    )


def _marker(step: str, ts: str = "2026-05-02T07:00:00Z") -> ProgressMarker:
    return ProgressMarker(timestamp=ts, step=step)


def test_classify_markers_success() -> None:
    last, term = _classify_markers(["00_container_started", "99_done"])
    assert last == "99_done"
    assert term == "success"


def test_classify_markers_cleanup_exit_zero_is_success() -> None:
    last, term = _classify_markers(["00_container_started", "90_cleanup_exit_0"])
    assert term == "success"
    assert last == "90_cleanup_exit_0"


def test_classify_markers_cleanup_exit_nonzero_is_failure() -> None:
    last, term = _classify_markers(
        ["00_container_started", "30_before_uv_sync", "90_cleanup_exit_1"]
    )
    assert term == "failure"
    assert last == "90_cleanup_exit_1"


def test_classify_markers_in_progress() -> None:
    last, term = _classify_markers(["00_container_started", "30_before_uv_sync"])
    assert term is None
    assert last == "30_before_uv_sync"


def test_classify_markers_empty() -> None:
    assert _classify_markers([]) == (None, None)


class _StubProgress:
    def __init__(self, sequence: list[list[ProgressMarker]]) -> None:
        self._seq = sequence
        self._idx = 0

    def list_markers(
        self, run_id: str, *, profile: str | None = None
    ) -> list[ProgressMarker]:
        if self._idx < len(self._seq):
            res = self._seq[self._idx]
            self._idx += 1
        else:
            res = self._seq[-1]
        return res


class _StubPodState:
    def __init__(self, sequence: list[PodState | None]) -> None:
        self._seq = sequence
        self._idx = 0

    def get_pod(self, pod_id: str) -> PodState | None:
        if self._idx < len(self._seq):
            res = self._seq[self._idx]
            self._idx += 1
        else:
            res = self._seq[-1]
        return res


def _make_clock(values: list[float]) -> Any:
    """連続呼び出しで `values` を順に返すモック clock。"""
    iterator = iter(values)
    last = [values[-1]]

    def _clock() -> float:
        try:
            v = next(iterator)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    return _clock


def test_watch_pod_success_after_two_polls() -> None:
    notifier = MagicMock()
    sleep_calls: list[float] = []
    progress = _StubProgress(
        [
            [_marker("00_container_started")],
            [_marker("00_container_started"), _marker("99_done")],
        ]
    )
    pods = _StubPodState([_pod("RUNNING"), _pod("RUNNING")])
    # clock: [start, t1, t2] = elapsed 0s, 30s, 60s
    clock = _make_clock([0.0, 30.0, 60.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=sleep_calls.append,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
        poll_interval=30.0,
        max_wait_seconds=300.0,
    )
    assert isinstance(result, WatchResult)
    assert result.outcome == "success"
    assert result.last_step == "99_done"
    notifier.assert_called_once()
    title = notifier.call_args.args[0]
    assert "SUCCESS" in title


def test_watch_pod_failure_when_cleanup_nonzero() -> None:
    notifier = MagicMock()
    progress = _StubProgress(
        [
            [_marker("00_container_started"), _marker("90_cleanup_exit_1")],
        ]
    )
    pods = _StubPodState([_pod("RUNNING")])
    clock = _make_clock([0.0, 5.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
    )
    assert result.outcome == "failure"
    assert result.last_step == "90_cleanup_exit_1"
    title = notifier.call_args.args[0]
    assert "FAILED" in title


def test_watch_pod_failure_when_pod_disappears() -> None:
    """pod が None になった時点で 99_done 無ければ失敗扱い。"""
    notifier = MagicMock()
    progress = _StubProgress([[_marker("00_container_started")]])
    pods = _StubPodState([None])
    clock = _make_clock([0.0, 1.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
    )
    assert result.outcome == "failure"
    assert "gone" in result.detail.lower()


def test_watch_pod_failure_when_pod_terminated() -> None:
    notifier = MagicMock()
    progress = _StubProgress([[_marker("30_before_uv_sync")]])
    pods = _StubPodState([_pod("TERMINATED")])
    clock = _make_clock([0.0, 1.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
    )
    assert result.outcome == "failure"


def test_watch_pod_timeout() -> None:
    notifier = MagicMock()
    # progress も pod も健全だが elapsed が max_wait を超える
    progress = _StubProgress([[_marker("30_before_uv_sync")]])
    pods = _StubPodState([_pod("RUNNING")])
    # clock: 0 -> 100 (すぐ timeout)
    clock = _make_clock([0.0, 100.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
        max_wait_seconds=50.0,
    )
    assert result.outcome == "timeout"
    title = notifier.call_args.args[0]
    assert "TIMEOUT" in title


def test_watch_pod_stalled_when_no_marker_progress() -> None:
    """marker が更新されない時間が threshold を超えたら stalled で抜ける。"""
    notifier = MagicMock()
    # 同じ marker を返し続ける = step 進捗なし
    progress = _StubProgress([[_marker("30_before_uv_sync")]])
    pods = _StubPodState([_pod("RUNNING")])
    # clock: 0 (start), 0 (1回目: last_step 確定), 1000 (2回目: stall threshold 超え)
    clock = _make_clock([0.0, 0.0, 0.0, 1000.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
        stall_threshold_seconds=900.0,
    )
    assert result.outcome == "stalled"
    assert result.last_step == "30_before_uv_sync"
    title = notifier.call_args.args[0]
    assert "STALLED" in title


def test_watch_pod_no_stall_when_threshold_zero() -> None:
    """stall_threshold_seconds=0 で stall 検知 OFF (既存挙動互換)。"""
    notifier = MagicMock()
    progress = _StubProgress([[_marker("30_before_uv_sync")], [_marker("99_done")]])
    pods = _StubPodState([_pod("RUNNING"), _pod("RUNNING")])
    clock = _make_clock([0.0, 0.0, 1000.0, 1000.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=progress,
        pod_state_module=pods,
        stall_threshold_seconds=0,
    )
    # stall 無効なので 1000s 経過しても success marker が来れば success
    assert result.outcome == "success"


def test_watch_pod_progress_failure_does_not_kill_loop() -> None:
    """boto3 が一時 fail しても poll 継続できる (次回 success 取得)。"""
    notifier = MagicMock()

    class _FlakyProgress:
        def __init__(self) -> None:
            self.calls = 0

        def list_markers(
            self, run_id: str, *, profile: str | None = None
        ) -> list[ProgressMarker]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient s3 error")
            return [_marker("99_done")]

    pods = _StubPodState([_pod("RUNNING"), _pod("RUNNING")])
    clock = _make_clock([0.0, 30.0, 60.0])

    result = watch_pod(
        run_id="r1",
        pod_id="pod-x",
        notifier=notifier,
        sleep=lambda _s: None,
        clock=clock,
        progress_module=_FlakyProgress(),
        pod_state_module=pods,
    )
    assert result.outcome == "success"
