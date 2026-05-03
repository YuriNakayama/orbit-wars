"""runpod_io.auth のユニットテスト。

- load_runpod_api_key: .env 読み / env 変数 fallback / 未設定で actionable error
- load_aws_creds は vast.auth から re-export しているので smoke check のみ
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runpod_io.auth import (
    CredentialsError,
    load_aws_creds,
    load_runpod_api_key,
)


def test_load_runpod_api_key_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("RUNPOD_API_KEY=test-key-from-file\n", encoding="utf-8")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert load_runpod_api_key(env_path=env_path) == "test-key-from-file"


def test_load_runpod_api_key_from_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-from-env")
    assert load_runpod_api_key(env_path=env_path) == "test-key-from-env"


def test_load_runpod_api_key_env_file_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("RUNPOD_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-environ")
    assert load_runpod_api_key(env_path=env_path) == "from-file"


def test_load_runpod_api_key_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(CredentialsError, match="RUNPOD_API_KEY not found"):
        load_runpod_api_key(env_path=env_path)


def test_load_aws_creds_reexported() -> None:
    """vast.auth.load_aws_creds が runpod_io 経由で import 可能。"""
    assert callable(load_aws_creds)
