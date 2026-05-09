"""Fixtures for pipeline e2e tests."""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def reset_random_state() -> Iterator[None]:
    """Restore global random state after each e2e test."""
    state = random.getstate()
    yield
    random.setstate(state)
