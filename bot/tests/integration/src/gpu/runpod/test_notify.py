"""notify.py のユニットテスト。"""

from __future__ import annotations

import subprocess

import pytest

from gpu.runpod.notify import _osascript_command, notify


def test_osascript_command_escapes_quotes() -> None:
    """AppleScript 文字列リテラルの `"` と `\\` がエスケープされる。"""
    cmd = _osascript_command('Title with "quote"', "Message with \\backslash")
    assert cmd[0] == "osascript"
    assert cmd[1] == "-e"
    script = cmd[2]
    assert 'with title "Title with \\"quote\\""' in script
    assert 'display notification "Message with \\\\backslash"' in script


def test_notify_macos_invokes_osascript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gpu.runpod.notify.sys.platform", "darwin")
    monkeypatch.setattr(
        "gpu.runpod.notify.shutil.which", lambda _: "/usr/bin/osascript"
    )
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("gpu.runpod.notify.subprocess.run", fake_run)
    assert notify("hello", "world") is True
    assert captured
    assert captured[0][0] == "/usr/bin/osascript"
    assert captured[0][1] == "-e"
    assert "hello" in captured[0][2]
    assert "world" in captured[0][2]


def test_notify_linux_invokes_notify_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gpu.runpod.notify.sys.platform", "linux")
    monkeypatch.setattr(
        "gpu.runpod.notify.shutil.which", lambda _: "/usr/bin/notify-send"
    )
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("gpu.runpod.notify.subprocess.run", fake_run)
    assert notify("t", "m") is True
    assert captured == [["/usr/bin/notify-send", "t", "m"]]


def test_notify_falls_back_to_stdout_when_no_tool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("gpu.runpod.notify.sys.platform", "darwin")
    monkeypatch.setattr("gpu.runpod.notify.shutil.which", lambda _: None)
    assert notify("title", "msg") is False
    out = capsys.readouterr().out
    assert "[notify] title: msg" in out


def test_notify_subprocess_failure_returns_false(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("gpu.runpod.notify.sys.platform", "darwin")
    monkeypatch.setattr(
        "gpu.runpod.notify.shutil.which", lambda _: "/usr/bin/osascript"
    )

    def fake_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.SubprocessError("boom")

    monkeypatch.setattr("gpu.runpod.notify.subprocess.run", fake_run)
    # subprocess fails → falls back to stdout, returns False
    assert notify("t", "m") is False
    out = capsys.readouterr().out
    assert "[notify] t: m" in out
