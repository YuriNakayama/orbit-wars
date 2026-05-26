"""auth.py のユニットテスト。subprocess は monkeypatch で固定値を返す。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu.vast.auth import (
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_vast_api_key,
)


class _FakeCompleted:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_load_aws_creds_reads_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> _FakeCompleted:
        calls.append(cmd)
        if "aws_access_key_id" in cmd:
            return _FakeCompleted("AKIAFAKE\n")
        if "aws_secret_access_key" in cmd:
            return _FakeCompleted("secret\n")
        if "region" in cmd:
            return _FakeCompleted("ap-northeast-1\n")
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr("gpu.vast.auth.shutil.which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(subprocess, "run", fake_run)
    creds = load_aws_creds(profile="orbit-wars")
    assert creds == AwsCreds(
        access_key_id="AKIAFAKE",
        secret_access_key="secret",
        region="ap-northeast-1",
    )
    assert any("--profile" in c and "orbit-wars" in c for c in calls)


def test_load_aws_creds_aws_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gpu.vast.auth.shutil.which", lambda _x: None)
    with pytest.raises(CredentialsError, match="aws CLI not found"):
        load_aws_creds()


def test_load_aws_creds_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gpu.vast.auth.shutil.which", lambda _x: "/usr/bin/aws")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(""),
    )
    with pytest.raises(CredentialsError, match="no value"):
        load_aws_creds()


def test_load_vast_api_key_from_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("VAST_API_KEY=abcdef123\nOTHER=ignored\n", encoding="utf-8")
    assert load_vast_api_key(env_path=env_path) == "abcdef123"


def test_load_vast_api_key_from_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"  # not created
    monkeypatch.setenv("VAST_API_KEY", "env-fallback")
    assert load_vast_api_key(env_path=env_path) == "env-fallback"


def test_load_vast_api_key_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"  # not created
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(CredentialsError, match="VAST_API_KEY not found"):
        load_vast_api_key(env_path=env_path)


def test_load_vast_api_key_blank_value_in_file_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("VAST_API_KEY=\n", encoding="utf-8")
    monkeypatch.setenv("VAST_API_KEY", "fallback-from-env")
    assert load_vast_api_key(env_path=env_path) == "fallback-from-env"
