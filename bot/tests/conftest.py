"""Pytest marker synchronization for the classified test tree."""

from __future__ import annotations

from pathlib import Path

import pytest

_SCOPE_MARKS = {
    "integration": pytest.mark.integration,
    "e2e": pytest.mark.e2e,
}
_SCOPE_MARK_NAMES = frozenset(_SCOPE_MARKS)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply non-unit scope markers from tests/{integration,e2e}."""
    root = Path(config.rootpath)
    for item in items:
        try:
            parts = Path(item.path).relative_to(root).parts
        except ValueError:
            continue
        if not parts:
            continue
        scope = parts[1] if parts[0] == "tests" and len(parts) > 1 else parts[0]
        marker = _SCOPE_MARKS.get(scope)
        if marker is None:
            continue
        item.own_markers = [
            mark for mark in item.own_markers if mark.name not in _SCOPE_MARK_NAMES
        ]
        item.add_marker(marker)
