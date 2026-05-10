"""Fixtures for rulebase case8 unit tests."""

from __future__ import annotations

import pytest

from pipeline.rulebase.case8.baseline.core.physics import reset_predict_cache


@pytest.fixture(autouse=True)
def clear_predict_cache_each_test() -> None:
    reset_predict_cache()
