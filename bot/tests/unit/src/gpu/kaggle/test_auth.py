"""gpu.kaggle.auth のユニットテスト。

3 段 fallback: process env → bot/.env → ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu.kaggle.auth import (
    CredentialsError,
    KaggleCreds,
    load_kaggle_creds,
)


def test_load_kaggle_creds_from_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    kaggle_json = tmp_path / "missing-kaggle.json"
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds == KaggleCreds(username="env-user", key="env-key")


def test_load_kaggle_creds_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_USERNAME=file-user\nKAGGLE_KEY=file-key\n", encoding="utf-8"
    )
    kaggle_json = tmp_path / "missing-kaggle.json"
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds.username == "file-user"
    assert creds.key == "file-key"


def test_load_kaggle_creds_from_kaggle_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text(
        json.dumps({"username": "json-user", "key": "json-key"}), encoding="utf-8"
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds == KaggleCreds(username="json-user", key="json-key")


def test_load_kaggle_creds_process_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_USERNAME=file-user\nKAGGLE_KEY=file-key\n", encoding="utf-8"
    )
    kaggle_json = tmp_path / "missing-kaggle.json"
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.setenv("KAGGLE_KEY", "env-key")
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds.username == "env-user"


def test_load_kaggle_creds_env_file_wins_over_kaggle_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_USERNAME=file-user\nKAGGLE_KEY=file-key\n", encoding="utf-8"
    )
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text(
        json.dumps({"username": "json-user", "key": "json-key"}), encoding="utf-8"
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds.username == "file-user"


def test_load_kaggle_creds_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    kaggle_json = tmp_path / "missing-kaggle.json"
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(CredentialsError, match="Kaggle credentials not found"):
        load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)


def test_load_kaggle_creds_partial_env_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Username だけ env にあって key が無い場合は次の経路へ落ちる。"""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_USERNAME=file-user\nKAGGLE_KEY=file-key\n", encoding="utf-8"
    )
    kaggle_json = tmp_path / "missing-kaggle.json"
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    creds = load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
    assert creds.username == "file-user"


def test_load_kaggle_creds_malformed_kaggle_json_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "missing.env"
    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text("not-json{", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(CredentialsError, match="Failed to parse"):
        load_kaggle_creds(env_path=env_path, kaggle_json_path=kaggle_json)
