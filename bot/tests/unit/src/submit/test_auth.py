"""auth のユニットテスト。認証値そのものはテストに含めない。"""

from __future__ import annotations

from pathlib import Path

import pytest

from submit.auth import AuthError, ensure_credentials


def test_ensure_credentials_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert ensure_credentials() == "env"


def test_ensure_credentials_config_file(
    monkeypatch: pytest.MonkeyPatch,
    clean_env: None,
    tmp_path: Path,
) -> None:
    fake_home = tmp_path
    kaggle_dir = fake_home / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    assert ensure_credentials() == "config"


def test_ensure_credentials_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
    clean_env: None,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(AuthError):
        ensure_credentials()


def test_ensure_credentials_empty_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "")
    monkeypatch.setenv("KAGGLE_KEY", "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(AuthError):
        ensure_credentials()
