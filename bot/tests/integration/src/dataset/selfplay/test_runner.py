"""Self-play runner integration tests."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from dataset.selfplay import runner
from dataset.selfplay.runner import RunSpec


def test_run_episodes_serial_writes_parquet(
    run_spec_factory: Callable[..., RunSpec],
) -> None:
    spec = run_spec_factory(episodes=2, parallel=1)
    records = runner.run_episodes(spec)
    assert len(records) == 2
    df = pl.read_parquet(
        spec.data_root / "matches/index.parquet/mode=1v1",
        hive_partitioning=True,
    )
    assert df.height == 2


@pytest.mark.slow
def test_run_episodes_parallel_matches_serial_count(
    run_spec_factory: Callable[..., RunSpec],
) -> None:
    spec = run_spec_factory(episodes=2, parallel=2)
    records = runner.run_episodes(spec)
    assert len(records) == 2
