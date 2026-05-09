"""Tests for runpod_io.runpod.cleanup pod-termination wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock

from runpod_io.runpod import cleanup


def test_terminate_pod_success() -> None:
    sdk = MagicMock()
    sdk.terminate_pod.return_value = None
    result = cleanup.terminate_pod(sdk, "pod_xyz")
    assert result.terminated is True
    assert result.error is None
    sdk.terminate_pod.assert_called_once_with("pod_xyz")


def test_terminate_pod_already_removed_is_idempotent() -> None:
    sdk = MagicMock()
    sdk.terminate_pod.side_effect = RuntimeError("Pod not found (404)")
    result = cleanup.terminate_pod(sdk, "pod_xyz")
    assert result.terminated is True
    assert result.error is None


def test_terminate_pod_unexpected_error_returns_failure() -> None:
    sdk = MagicMock()
    sdk.terminate_pod.side_effect = RuntimeError("upstream timeout")
    result = cleanup.terminate_pod(sdk, "pod_xyz")
    assert result.terminated is False
    assert result.error is not None
    assert "timeout" in result.error


def test_terminate_pod_empty_pod_id() -> None:
    sdk = MagicMock()
    result = cleanup.terminate_pod(sdk, "")
    assert result.terminated is False
    sdk.terminate_pod.assert_not_called()


def test_stop_pod_success() -> None:
    sdk = MagicMock()
    result = cleanup.stop_pod(sdk, "pod_xyz")
    assert result.terminated is True
    sdk.stop_pod.assert_called_once_with("pod_xyz")
