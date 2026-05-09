"""Fixtures for imitation case1 unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.imitation.case1.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case1.policy.templates import TEMPLATE_CTX_DIM


def _make_fixture_parquet(path: Path, n: int = 10) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        from_multihot = [bool(j == 0) for j in range(MAX_PLANETS)]
        target_per_src = [-1] * MAX_PLANETS
        ships_per_src = [-1] * MAX_PLANETS
        target_per_src[0] = int((i + 1) % MAX_PLANETS)
        ships_per_src[0] = int(i % 4)
        rows.append(
            {
                "planet_feats": rng.standard_normal(MAX_PLANETS * PLANET_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": [True] * MAX_PLANETS,
                "my_planet_mask": [bool(j == 0) for j in range(MAX_PLANETS)],
                "target_mask": [bool(j != 0) for j in range(MAX_PLANETS)],
                "template_ctx": rng.standard_normal(MAX_PLANETS * TEMPLATE_CTX_DIM)
                .astype(np.float32)
                .tolist(),
                "from_multihot": from_multihot,
                "target_per_src": target_per_src,
                "ships_per_src": ships_per_src,
                "is_noop": bool(i % 3 == 0 and not any(from_multihot)),
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _make_mini_parquet(path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        is_noop = bool(i % 4 == 0)
        from_multihot = [False] * MAX_PLANETS
        target_per_src = [-1] * MAX_PLANETS
        ships_per_src = [-1] * MAX_PLANETS
        if not is_noop:
            src = int(rng.integers(0, 4))
            from_multihot[src] = True
            target_per_src[src] = int(rng.integers(0, 8))
            ships_per_src[src] = int(rng.integers(0, 4))
        rows.append(
            {
                "planet_feats": rng.standard_normal(MAX_PLANETS * PLANET_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "global_feats": rng.standard_normal(GLOBAL_FEAT_DIM)
                .astype(np.float32)
                .tolist(),
                "planet_mask": [bool(j < 8) for j in range(MAX_PLANETS)],
                "my_planet_mask": [bool(j < 4) for j in range(MAX_PLANETS)],
                "target_mask": [bool(4 <= j < 8) for j in range(MAX_PLANETS)],
                "template_ctx": rng.standard_normal(MAX_PLANETS * TEMPLATE_CTX_DIM)
                .astype(np.float32)
                .tolist(),
                "from_multihot": from_multihot,
                "target_per_src": target_per_src,
                "ships_per_src": ships_per_src,
                "is_noop": is_noop,
            }
        )
    pl.DataFrame(rows).write_parquet(path)


@pytest.fixture
def fixture_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.parquet"
    _make_fixture_parquet(path, n=10)
    return path


@pytest.fixture
def mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    _make_mini_parquet(train_path, n=200, seed=0)
    _make_mini_parquet(val_path, n=80, seed=1)
    return train_path, val_path
