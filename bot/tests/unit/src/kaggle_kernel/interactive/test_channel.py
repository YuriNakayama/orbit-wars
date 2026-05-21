"""kaggle_kernel.interactive.channel のユニットテスト (S3 mock)。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from kaggle_kernel.interactive.channel import (
    DEFAULT_BUCKET,
    ChannelTimeoutError,
    Command,
    CommandResult,
    await_result,
    cleanup_session,
    file_key,
    get_heartbeat,
    inbox_key,
    outbox_key,
    put_command,
    session_prefix,
    submit_command,
    try_fetch_result,
)


def _no_such_key_error() -> Exception:
    """botocore.exceptions.ClientError 互換の 404 例外を返す。"""
    err = Exception("NoSuchKey")
    err.response = {  # type: ignore[attr-defined]
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }
    return err


def _make_s3_get_response(body: bytes) -> dict[str, Any]:
    stream = MagicMock()
    stream.read.return_value = body
    return {"Body": stream}


# ─── Command / CommandResult round-trip ─────────────────────────────────────


def test_command_to_from_json_roundtrip() -> None:
    cmd = Command(
        argv=["python", "-c", "print(1)"],
        cwd="/tmp/foo",
        timeout=60.0,
        env={"FOO": "bar"},
        command_id="abc",
    )
    restored = Command.from_json(cmd.to_json())
    assert restored == cmd


def test_command_with_id_is_immutable() -> None:
    cmd = Command(argv=["ls"])
    assert cmd.command_id == ""
    cmd2 = cmd.with_id("new")
    assert cmd2.command_id == "new"
    assert cmd.command_id == ""  # frozen 元は不変


def test_command_result_caps_large_output() -> None:
    big = "x" * 2_000_000  # 2 MB
    result = CommandResult(command_id="r1", returncode=0, stdout=big, stderr=big)
    raw = result.to_json()
    import json as _json

    parsed = _json.loads(raw)
    # 1 MB cap each (channel.STDOUT_CAP / STDERR_CAP)
    assert len(parsed["stdout"]) == 1_000_000
    assert len(parsed["stderr"]) == 1_000_000


def test_command_result_roundtrip_with_error() -> None:
    result = CommandResult(
        command_id="r2",
        returncode=-9,
        stdout="hi",
        stderr="bye",
        elapsed_seconds=1.5,
        started_at="2026-05-21T00:00:00Z",
        finished_at="2026-05-21T00:00:02Z",
        error="timeout",
    )
    restored = CommandResult.from_json(result.to_json())
    assert restored == result


# ─── Key helpers ─────────────────────────────────────────────────────────────


def test_session_prefix() -> None:
    assert session_prefix("rid") == "kaggle_interactive/rid"


def test_inbox_outbox_keys() -> None:
    assert inbox_key("rid", "cmd1") == "kaggle_interactive/rid/inbox/cmd1.json"
    assert outbox_key("rid", "cmd1") == "kaggle_interactive/rid/outbox/cmd1.json"


def test_file_key_rejects_absolute() -> None:
    with pytest.raises(ValueError, match="unsafe relpath"):
        file_key("rid", "/etc/passwd")


def test_file_key_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe relpath"):
        file_key("rid", "../etc/passwd")


def test_file_key_valid() -> None:
    assert (
        file_key("rid", "bot/src/foo.py")
        == "kaggle_interactive/rid/state/files/bot/src/foo.py"
    )


# ─── put_command ─────────────────────────────────────────────────────────────


def test_put_command_assigns_uuid_when_empty() -> None:
    s3 = MagicMock()
    cmd = Command(argv=["echo", "hi"])
    submitted = put_command(s3, "rid", cmd, bucket="B")
    assert submitted.command_id
    s3.put_object.assert_called_once()
    kw = s3.put_object.call_args.kwargs
    assert kw["Bucket"] == "B"
    assert kw["Key"] == f"kaggle_interactive/rid/inbox/{submitted.command_id}.json"


def test_put_command_preserves_id_when_set() -> None:
    s3 = MagicMock()
    cmd = Command(argv=["echo"], command_id="fixed")
    submitted = put_command(s3, "rid", cmd)
    assert submitted.command_id == "fixed"
    kw = s3.put_object.call_args.kwargs
    assert kw["Key"].endswith("/fixed.json")


# ─── try_fetch_result ────────────────────────────────────────────────────────


def test_try_fetch_result_returns_none_on_no_such_key() -> None:
    s3 = MagicMock()
    s3.get_object.side_effect = _no_such_key_error()
    assert try_fetch_result(s3, "rid", "cmd1") is None


def test_try_fetch_result_returns_parsed_and_deletes() -> None:
    s3 = MagicMock()
    payload = CommandResult(command_id="cmd1", returncode=0, stdout="ok").to_json()
    s3.get_object.return_value = _make_s3_get_response(payload.encode("utf-8"))
    result = try_fetch_result(s3, "rid", "cmd1", bucket="B")
    assert result is not None
    assert result.command_id == "cmd1"
    assert result.stdout == "ok"
    s3.delete_object.assert_called_once_with(
        Bucket="B", Key="kaggle_interactive/rid/outbox/cmd1.json"
    )


def test_try_fetch_result_reraises_unknown_error() -> None:
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("network down")
    with pytest.raises(RuntimeError):
        try_fetch_result(s3, "rid", "cmd1")


# ─── await_result ────────────────────────────────────────────────────────────


def test_await_result_returns_when_present() -> None:
    s3 = MagicMock()
    payload = CommandResult(command_id="x", returncode=0).to_json()
    no_key = _no_such_key_error()
    # 最初の 2 回は no key、3 回目で found
    s3.get_object.side_effect = [
        no_key,
        no_key,
        _make_s3_get_response(payload.encode("utf-8")),
    ]
    sleep = MagicMock()
    result = await_result(s3, "rid", "x", poll_interval=1.0, timeout=10.0, sleep=sleep)
    assert result.command_id == "x"
    assert sleep.call_count == 2  # 2 回 sleep してから取得


def test_await_result_times_out() -> None:
    s3 = MagicMock()
    s3.get_object.side_effect = _no_such_key_error()
    sleep = MagicMock()
    with pytest.raises(ChannelTimeoutError):
        await_result(s3, "rid", "x", poll_interval=1.0, timeout=2.0, sleep=sleep)


def test_await_result_invalid_interval() -> None:
    s3 = MagicMock()
    with pytest.raises(ValueError):
        await_result(s3, "rid", "x", poll_interval=0, timeout=10.0)
    with pytest.raises(ValueError):
        await_result(s3, "rid", "x", poll_interval=1.0, timeout=0)


# ─── submit_command (put + await) ────────────────────────────────────────────


def test_submit_command_roundtrip() -> None:
    s3 = MagicMock()
    payload = CommandResult(
        command_id="rid-cmd", returncode=42, stdout="done"
    ).to_json()
    s3.get_object.return_value = _make_s3_get_response(payload.encode("utf-8"))
    cmd = Command(argv=["echo"], command_id="rid-cmd")
    sleep = MagicMock()
    result = submit_command(
        s3, "rid", cmd, poll_interval=1.0, await_timeout=10.0, sleep=sleep
    )
    assert result.returncode == 42
    s3.put_object.assert_called_once()
    s3.delete_object.assert_called_once()  # outbox cleanup


# ─── heartbeat / cleanup ─────────────────────────────────────────────────────


def test_get_heartbeat_returns_none_when_absent() -> None:
    s3 = MagicMock()
    s3.get_object.side_effect = _no_such_key_error()
    assert get_heartbeat(s3, "rid") is None


def test_get_heartbeat_parses_json() -> None:
    s3 = MagicMock()
    s3.get_object.return_value = _make_s3_get_response(
        b'{"state":"idle","iso":"2026-05-21T00:00:00Z"}'
    )
    hb = get_heartbeat(s3, "rid")
    assert hb is not None
    assert hb["state"] == "idle"


def test_cleanup_session_deletes_all_objects() -> None:
    s3 = MagicMock()
    s3.list_objects_v2.side_effect = [
        {
            "Contents": [
                {"Key": "kaggle_interactive/rid/inbox/a.json"},
                {"Key": "kaggle_interactive/rid/state/heartbeat.json"},
            ],
            "IsTruncated": True,
            "NextContinuationToken": "tok",
        },
        {
            "Contents": [{"Key": "kaggle_interactive/rid/outbox/b.json"}],
            "IsTruncated": False,
        },
    ]
    deleted = cleanup_session(s3, "rid", bucket="B")
    assert deleted == 3
    assert s3.delete_object.call_count == 3


def test_cleanup_session_handles_empty_prefix() -> None:
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    assert cleanup_session(s3, "rid") == 0


def test_default_bucket_is_dvc_bucket() -> None:
    assert DEFAULT_BUCKET == "orbit-wars-dvc-286854171013"
