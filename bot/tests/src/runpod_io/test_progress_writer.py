"""Tests for runpod_io.progress.write_progress_marker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from runpod_io import progress


def test_write_progress_marker_writes_to_s3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "rid_abc")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_xyz")
    client = MagicMock()
    client.put_object.return_value = {}
    ok = progress.write_progress_marker(
        "30_train_step_0042",
        s3_client=client,
        timestamp="2026-05-05T10:00:00Z",
    )
    assert ok is True
    client.put_object.assert_called_once()
    kwargs = client.put_object.call_args.kwargs
    assert kwargs["Bucket"] == progress.PROGRESS_BUCKET
    assert kwargs["Key"] == (
        f"{progress.PROGRESS_PREFIX}/rid_abc/2026-05-05T10:00:00Z_30_train_step_0042"
    )
    assert b"pod=pod_xyz" in kwargs["Body"]


def test_write_progress_marker_returns_false_when_run_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORBIT_WARS_RUN_ID", raising=False)
    client = MagicMock()
    ok = progress.write_progress_marker("99_done", s3_client=client)
    assert ok is False
    client.put_object.assert_not_called()


def test_write_progress_marker_swallows_s3_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORBIT_WARS_RUN_ID", "rid_abc")
    client = MagicMock()
    client.put_object.side_effect = RuntimeError("network down")
    ok = progress.write_progress_marker("99_done", s3_client=client)
    assert ok is False
