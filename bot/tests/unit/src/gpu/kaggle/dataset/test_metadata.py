"""gpu.kaggle.dataset.metadata のユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu.kaggle.dataset.metadata import (
    DATASET_METADATA_FILENAME,
    build_version_notes,
    parse_commit_from_version_notes,
    write_dataset_metadata,
)


def test_write_dataset_metadata_minimal(tmp_path: Path) -> None:
    out = write_dataset_metadata(
        tmp_path,
        slug="yuri/orbit-wars-bot",
        title="Orbit Wars bot snapshot",
        commit_sha="abc1234deadbeef",
    )
    assert out == tmp_path / DATASET_METADATA_FILENAME
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["id"] == "yuri/orbit-wars-bot"
    assert data["title"] == "Orbit Wars bot snapshot"
    assert data["isPrivate"] is True
    assert data["licenses"] == [{"name": "Apache-2.0"}]
    assert "commit=abc1234" in data["keywords"]


def test_write_dataset_metadata_invalid_slug(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid Kaggle dataset slug"):
        write_dataset_metadata(
            tmp_path,
            slug="no-slash-here",
            title="x",
            commit_sha="abc1234",
        )


def test_build_version_notes_basic() -> None:
    assert build_version_notes("abc1234deadbeef") == "commit=abc1234"
    assert (
        build_version_notes("abc1234deadbeef", label="dropout=0.3")
        == "commit=abc1234 | dropout=0.3"
    )


def test_build_version_notes_short_sha_raises() -> None:
    with pytest.raises(ValueError, match="commit_sha too short"):
        build_version_notes("abc")


def test_parse_commit_roundtrip() -> None:
    notes = build_version_notes("abc1234deadbeef", label="tweak")
    assert parse_commit_from_version_notes(notes) == "abc1234"


def test_parse_commit_none_when_missing() -> None:
    assert parse_commit_from_version_notes("free-form note") is None
    assert parse_commit_from_version_notes("") is None
