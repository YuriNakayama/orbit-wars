"""Unit + integration tests for src/env/runner.py."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from env import runner
from env.runner import RunSpec


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


@pytest.mark.integration
def test_run_episodes_serial_writes_parquet(tmp_path: Path) -> None:
    pytest.importorskip("kaggle_environments")
    spec = _spec(tmp_path, episodes=2, parallel=1)
    records = runner.run_episodes(spec)
    assert len(records) == 2
    df = pl.read_parquet(
        tmp_path / "matches/index.parquet/mode=1v1",
        hive_partitioning=True,
    )
    assert df.height == 2


@pytest.mark.integration
@pytest.mark.slow
def test_run_episodes_parallel_matches_serial_count(tmp_path: Path) -> None:
    pytest.importorskip("kaggle_environments")
    spec = _spec(tmp_path, episodes=2, parallel=2)
    records = runner.run_episodes(spec)
    assert len(records) == 2
