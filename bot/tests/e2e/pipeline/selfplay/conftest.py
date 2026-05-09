"""Fixtures for self-play e2e tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dataset.schema import AgentSpec, MatchSpec
from dataset.selfplay.runner import RunSpec


@pytest.fixture
def run_spec_factory(tmp_path: Path) -> Callable[..., RunSpec]:
    """Build self-play RunSpec values rooted at the test tmp_path."""

    def build(
        *,
        mode: str = "1v1",
        agents: tuple[str, ...] = ("random", "random"),
        episodes: int = 2,
        parallel: int = 1,
        save_replay: bool = False,
    ) -> RunSpec:
        return RunSpec(
            agents=agents,
            mode=mode,
            episodes=episodes,
            seed=0,
            parallel=parallel,
            save_replay=save_replay,
            data_root=tmp_path,
        )

    return build


@pytest.fixture
def random_match_spec_factory(tmp_path: Path) -> Callable[..., MatchSpec]:
    """Build random-vs-random MatchSpec values rooted at the test tmp_path."""

    def build(
        *,
        match_id: str,
        seed: int = 0,
        save_replay: bool = False,
    ) -> MatchSpec:
        return MatchSpec(
            match_id=match_id,
            run_id="run_test",
            mode="1v1",
            seed=seed,
            agents=(
                AgentSpec(name="random", version="", callable_path="random"),
                AgentSpec(name="random", version="", callable_path="random"),
            ),
            save_replay=save_replay,
            data_root=str(tmp_path),
        )

    return build
