"""Unit + integration tests for src/dataset/runner.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset.selfplay import runner
from dataset.selfplay.runner import RunSpec


def _spec(
    tmp_path: Path,
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


def test_validate_rejects_unknown_mode(tmp_path: Path) -> None:
    spec = _spec(tmp_path, mode="2v2")
    with pytest.raises(ValueError, match="unknown mode"):
        runner._validate(spec)


def test_validate_rejects_mismatched_agent_count(tmp_path: Path) -> None:
    spec = _spec(tmp_path, mode="ffa4", agents=("random", "random"))
    with pytest.raises(ValueError, match="needs 4 agents"):
        runner._validate(spec)


def test_validate_rejects_non_positive_episodes(tmp_path: Path) -> None:
    spec = _spec(tmp_path, episodes=0)
    with pytest.raises(ValueError, match="episodes must be positive"):
        runner._validate(spec)


def test_validate_rejects_non_positive_parallel(tmp_path: Path) -> None:
    spec = _spec(tmp_path, parallel=0)
    with pytest.raises(ValueError, match="parallel must be positive"):
        runner._validate(spec)


def test_make_match_specs_uses_sequential_seeds(tmp_path: Path) -> None:
    spec = _spec(tmp_path, episodes=3)
    match_specs = runner._make_match_specs(spec, run_id="run_abc")
    assert len(match_specs) == 3
    assert [m.seed for m in match_specs] == [0, 1, 2]
    assert all(m.run_id == "run_abc" for m in match_specs)
    assert all(m.mode == "1v1" for m in match_specs)

