"""Fixtures for imitation case2 unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.imitation.case2.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case2.policy.templates import TEMPLATE_CTX_DIM


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


@pytest.fixture
def fixture_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.parquet"
    _make_fixture_parquet(path, n=10)
    return path
