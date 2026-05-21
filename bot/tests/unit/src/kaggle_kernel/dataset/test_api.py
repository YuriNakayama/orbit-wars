"""kaggle_kernel.dataset.api のユニットテスト (KaggleApi mock)。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaggle_kernel.dataset.api import (
    create_new_dataset,
    dataset_status,
    latest_version_commit,
    push_dataset_version,
)
from kaggle_kernel.dataset.metadata import write_dataset_metadata


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    write_dataset_metadata(
        tmp_path,
        slug="yuri/orbit-wars-bot",
        title="Orbit Wars bot snapshot",
        commit_sha="abc1234deadbeef",
    )
    return tmp_path


def test_create_new_dataset_calls_api(dataset_dir: Path) -> None:
    api = MagicMock()
    api.dataset_create_new.return_value = {"status": "ok"}
    result = create_new_dataset(api, dataset_dir, commit_sha="abc1234deadbeef")
    api.dataset_create_new.assert_called_once_with(
        folder=str(dataset_dir),
        public=False,
        quiet=False,
        convert_to_csv=False,
        dir_mode="zip",
    )
    assert result.slug == "yuri/orbit-wars-bot"
    assert result.version_notes == "commit=abc1234"
    assert result.commit_sha == "abc1234"


def test_push_dataset_version_calls_api(dataset_dir: Path) -> None:
    api = MagicMock()
    api.dataset_create_version.return_value = {"status": "ok"}
    result = push_dataset_version(
        api, dataset_dir, commit_sha="abc1234deadbeef", label="tweak"
    )
    api.dataset_create_version.assert_called_once_with(
        folder=str(dataset_dir),
        version_notes="commit=abc1234 | tweak",
        quiet=False,
        convert_to_csv=False,
        dir_mode="zip",
    )
    assert result.slug == "yuri/orbit-wars-bot"
    assert result.version_notes == "commit=abc1234 | tweak"


def test_dataset_status_dict_passthrough() -> None:
    api = MagicMock()
    api.dataset_status.return_value = {
        "status": "ready",
        "versionNotes": "commit=def5678",
    }
    result = dataset_status(api, "yuri/orbit-wars-bot")
    assert result["status"] == "ready"
    assert result["versionNotes"] == "commit=def5678"


def test_latest_version_commit_extracts_sha() -> None:
    api = MagicMock()
    api.dataset_status.return_value = {"versionNotes": "commit=def5678 | tweak"}
    assert latest_version_commit(api, "yuri/orbit-wars-bot") == "def5678"


def test_latest_version_commit_none_when_missing() -> None:
    api = MagicMock()
    api.dataset_status.return_value = {"versionNotes": "free-form"}
    assert latest_version_commit(api, "yuri/orbit-wars-bot") is None


def test_create_new_dataset_missing_metadata_raises(tmp_path: Path) -> None:
    """dataset-metadata.json なしで呼ぶと slug 抽出で失敗。"""
    api = MagicMock()
    api.dataset_create_new.return_value = {"status": "ok"}
    with pytest.raises(FileNotFoundError, match="dataset-metadata.json"):
        create_new_dataset(api, tmp_path, commit_sha="abc1234deadbeef")


def test_create_new_dataset_invalid_metadata_id(tmp_path: Path) -> None:
    """id が存在しない / 非 str なら ValueError。"""
    api = MagicMock()
    api.dataset_create_new.return_value = {"status": "ok"}
    (tmp_path / "dataset-metadata.json").write_text(
        json.dumps({"title": "no id"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="id missing"):
        create_new_dataset(api, tmp_path, commit_sha="abc1234deadbeef")
