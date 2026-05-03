"""Pytest configuration for the simulator/rust Python-side test suite.

Registers the `slow` mark so `pytest -m slow` works without warnings.
"""

from __future__ import annotations


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.addinivalue_line(
        "markers",
        "slow: long-running benchmarks / fuzz tests (skipped by default)",
    )
