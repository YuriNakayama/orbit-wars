"""Fixtures for rulebase case1 unit tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture
def append_recorder() -> tuple[
    list[tuple[int, float, int]], Callable[[int, float, int], int]
]:
    moves: list[tuple[int, float, int]] = []

    def append_move(src_id: int, angle: float, ships: int) -> int:
        moves.append((src_id, angle, ships))
        return ships

    return moves, append_move
