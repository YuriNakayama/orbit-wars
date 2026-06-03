"""Fixtures for rulebase case1 unit tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import pytest


@pytest.fixture(autouse=True)
def _x64_parity_isolation(request: pytest.FixtureRequest) -> Iterator[None]:
    """Enable jax_enable_x64 ONLY for *_jax_parity tests, and restore after.

    The JAX→Python parity tests need float64 to isolate algorithm bugs from
    float32 drift. Setting jax_enable_x64 at MODULE level leaked the flag to
    every later test in the same xdist worker, breaking float32 agents/rollouts
    (int64↔int32 scatter TypeError). This autouse fixture scopes x64 to the
    parity modules and resets the prior value afterwards.
    """
    is_parity = request.module.__name__.endswith("_jax_parity")
    if not is_parity:
        yield
        return
    prev = bool(jax.config.jax_enable_x64)  # type: ignore[attr-defined]
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)  # type: ignore[no-untyped-call]


@pytest.fixture
def append_recorder() -> tuple[
    list[tuple[int, float, int]], Callable[[int, float, int], int]
]:
    moves: list[tuple[int, float, int]] = []

    def append_move(src_id: int, angle: float, ships: int) -> int:
        moves.append((src_id, angle, ships))
        return ships

    return moves, append_move
