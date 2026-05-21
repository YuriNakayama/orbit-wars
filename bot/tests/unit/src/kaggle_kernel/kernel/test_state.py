"""kaggle_kernel.kernel.state のユニットテスト。"""

from __future__ import annotations

from kaggle_kernel.kernel.state import KernelStatus, parse_status


def test_is_terminal_flags() -> None:
    assert KernelStatus.COMPLETE.is_terminal
    assert KernelStatus.ERROR.is_terminal
    assert KernelStatus.CANCEL_ACKNOWLEDGED.is_terminal
    assert not KernelStatus.QUEUED.is_terminal
    assert not KernelStatus.RUNNING.is_terminal
    assert not KernelStatus.UNKNOWN.is_terminal


def test_parse_status_from_dict_status_key() -> None:
    assert parse_status({"status": "complete"}) == KernelStatus.COMPLETE
    assert parse_status({"status": "running"}) == KernelStatus.RUNNING


def test_parse_status_from_dict_state_key() -> None:
    assert parse_status({"state": "queued"}) == KernelStatus.QUEUED


def test_parse_status_from_string() -> None:
    assert parse_status("complete") == KernelStatus.COMPLETE
    assert parse_status("CANCEL ACKNOWLEDGED") == KernelStatus.CANCEL_ACKNOWLEDGED
    assert parse_status("Error") == KernelStatus.ERROR


def test_parse_status_unknown_default() -> None:
    assert parse_status({"status": "weird"}) == KernelStatus.UNKNOWN
    assert parse_status({}) == KernelStatus.UNKNOWN
    assert parse_status(None) == KernelStatus.UNKNOWN


def test_parse_status_from_object_attr() -> None:
    class Resp:
        status = "complete"

    assert parse_status(Resp()) == KernelStatus.COMPLETE


def test_parse_status_from_enum_with_name() -> None:
    """Kaggle SDK 1.7+ returns ``KernelWorkerStatus`` Enum; we read ``.name``."""

    class FakeEnum:
        name = "RUNNING"

    class Resp:
        status = FakeEnum()

    assert parse_status(Resp()) == KernelStatus.RUNNING


def test_parse_status_from_enum_repr_string() -> None:
    """`str(KernelWorkerStatus.COMPLETE)` returns ``KernelWorkerStatus.COMPLETE``."""
    assert parse_status("KernelWorkerStatus.COMPLETE") == KernelStatus.COMPLETE
    assert parse_status("KernelWorkerStatus.ERROR") == KernelStatus.ERROR
