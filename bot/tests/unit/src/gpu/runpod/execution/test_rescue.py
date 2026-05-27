"""rescue_onstart_log のユニットテスト。

外部副作用 (SSH endpoint 解決 / SSH command / S3 upload) を全て差し替えて
テストする。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gpu.runpod.execution.rescue import RescueResult, rescue_onstart_log
from gpu.runpod.runpod.ssh import SshEndpoint, SshUnavailable


def _endpoint() -> SshEndpoint:
    return SshEndpoint(host="1.2.3.4", port=22222)


def test_happy_path_uploads_and_reports_bytes() -> None:
    sdk = MagicMock()
    resolver = MagicMock(return_value=_endpoint())
    grabber = MagicMock(return_value="hello onstart log\n")
    uploader = MagicMock(return_value="s3://bucket/key/onstart.rescue.log")

    out = rescue_onstart_log(
        sdk,
        "pod-x",
        "run-y",
        ssh_endpoint_resolver=resolver,
        log_grabber=grabber,
        uploader=uploader,
    )

    assert out == RescueResult(
        uploaded=True,
        s3_uri="s3://bucket/key/onstart.rescue.log",
        error=None,
        bytes_uploaded=len("hello onstart log\n".encode()),
    )
    resolver.assert_called_once_with(sdk, "pod-x")
    grabber.assert_called_once()
    uploader.assert_called_once_with("run-y", "hello onstart log\n", profile=None)


def test_ssh_unavailable_returns_skip_not_raise() -> None:
    sdk = MagicMock()
    resolver = MagicMock(side_effect=SshUnavailable("port 22 closed"))
    grabber = MagicMock()
    uploader = MagicMock()

    out = rescue_onstart_log(
        sdk,
        "pod-x",
        "run-y",
        ssh_endpoint_resolver=resolver,
        log_grabber=grabber,
        uploader=uploader,
    )

    assert out.uploaded is False
    assert out.s3_uri is None
    assert out.error is not None and "port 22 closed" in out.error
    grabber.assert_not_called()
    uploader.assert_not_called()


def test_log_grab_failure_records_error() -> None:
    sdk = MagicMock()
    resolver = MagicMock(return_value=_endpoint())
    grabber = MagicMock(side_effect=SshUnavailable("ssh timed out"))
    uploader = MagicMock()

    out = rescue_onstart_log(
        sdk,
        "pod-x",
        "run-y",
        ssh_endpoint_resolver=resolver,
        log_grabber=grabber,
        uploader=uploader,
    )

    assert out.uploaded is False
    assert out.error is not None and "ssh timed out" in out.error
    uploader.assert_not_called()


def test_s3_upload_failure_records_error() -> None:
    sdk = MagicMock()
    resolver = MagicMock(return_value=_endpoint())
    grabber = MagicMock(return_value="log body")
    uploader = MagicMock(side_effect=RuntimeError("403 AccessDenied"))

    out = rescue_onstart_log(
        sdk,
        "pod-x",
        "run-y",
        ssh_endpoint_resolver=resolver,
        log_grabber=grabber,
        uploader=uploader,
    )

    assert out.uploaded is False
    assert out.error is not None and "403 AccessDenied" in out.error
