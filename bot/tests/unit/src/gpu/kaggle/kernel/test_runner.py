"""gpu.kaggle.kernel.runner のユニットテスト (KaggleApi mock)。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpu.kaggle.kernel.runner import (
    KernelPollTimeout,
    KernelPushResult,
    poll_status,
    push_kernel,
)
from gpu.kaggle.kernel.state import KernelStatus


def _kernel_dir_with_metadata(tmp_path: Path, slug: str) -> Path:
    """kernel-metadata.json を含む dummy kernel dir。"""
    (tmp_path / "kernel-metadata.json").write_text(
        json.dumps({"id": slug, "code_file": "main.ipynb", "language": "python"}),
        encoding="utf-8",
    )
    (tmp_path / "main.ipynb").write_text("{}", encoding="utf-8")
    return tmp_path


def test_push_kernel_uses_metadata_when_response_empty(tmp_path: Path) -> None:
    api = MagicMock()
    api.kernels_push_cli.return_value = {}
    kernel_dir = _kernel_dir_with_metadata(tmp_path, "yuri/orbit-wars-run0")
    result = push_kernel(api, kernel_dir)
    api.kernels_push_cli.assert_called_once_with(str(kernel_dir), None, None)
    assert isinstance(result, KernelPushResult)
    assert result.slug == "yuri/orbit-wars-run0"
    assert result.version is None


def test_push_kernel_extracts_ref_from_response(tmp_path: Path) -> None:
    api = MagicMock()
    api.kernels_push_cli.return_value = {
        "ref": "yuri/from-response",
        "versionNumber": 7,
        "url": "https://www.kaggle.com/code/yuri/from-response",
    }
    kernel_dir = _kernel_dir_with_metadata(tmp_path, "yuri/orbit-wars-run0")
    result = push_kernel(api, kernel_dir)
    assert result.slug == "yuri/from-response"
    assert result.version == 7
    assert "kaggle.com" in (result.url or "")


def test_poll_status_completes_after_three_polls() -> None:
    api = MagicMock()
    api.kernels_status.side_effect = [
        {"status": "queued"},
        {"status": "running"},
        {"status": "complete"},
    ]
    sleep = MagicMock()
    status, msg = poll_status(api, "yuri/run0", interval=1.0, timeout=10.0, sleep=sleep)
    assert status == KernelStatus.COMPLETE
    assert msg is None
    assert api.kernels_status.call_count == 3
    assert sleep.call_count == 2


def test_poll_status_returns_error_message() -> None:
    api = MagicMock()
    api.kernels_status.side_effect = [
        {"status": "error", "failureMessage": "OOM"},
    ]
    sleep = MagicMock()
    status, msg = poll_status(api, "yuri/run0", interval=1.0, timeout=10.0, sleep=sleep)
    assert status == KernelStatus.ERROR
    assert msg == "OOM"


def test_poll_status_cancel_acknowledged() -> None:
    api = MagicMock()
    api.kernels_status.side_effect = [{"status": "cancel acknowledged"}]
    sleep = MagicMock()
    status, _ = poll_status(api, "yuri/run0", interval=1.0, timeout=10.0, sleep=sleep)
    assert status == KernelStatus.CANCEL_ACKNOWLEDGED


def test_poll_status_timeout_raises() -> None:
    api = MagicMock()
    api.kernels_status.return_value = {"status": "running"}
    sleep = MagicMock()
    with pytest.raises(KernelPollTimeout):
        poll_status(api, "yuri/run0", interval=1.0, timeout=2.0, sleep=sleep)


def test_poll_status_invalid_interval_raises() -> None:
    api = MagicMock()
    with pytest.raises(ValueError):
        poll_status(api, "yuri/run0", interval=0, timeout=10.0)
    with pytest.raises(ValueError):
        poll_status(api, "yuri/run0", interval=1.0, timeout=0)
