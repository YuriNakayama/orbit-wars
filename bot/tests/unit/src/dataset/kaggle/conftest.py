"""Fixtures for Kaggle dataset unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def kaggle_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "u", "key": "k"}), encoding="utf-8")
    return cfg
