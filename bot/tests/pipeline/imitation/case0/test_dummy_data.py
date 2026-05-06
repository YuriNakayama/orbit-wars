"""Determinism + content checks for the case0 dummy dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from pipeline.imitation.case0.training.dummy_data import (
    DummyDatasetSpec,
    generate,
    sha256_of_file,
    write,
)

EXPECTED_SHA12_DEFAULT = "1bfcc041ff16"


def _hash_df_values(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_numpy().tobytes()).hexdigest()


def test_generate_is_deterministic() -> None:
    spec = DummyDatasetSpec()
    df1 = generate(spec)
    df2 = generate(spec)
    assert df1.shape == (32, 9)  # 8 features + label
    assert _hash_df_values(df1) == _hash_df_values(df2)


def test_generate_changes_with_seed() -> None:
    df_seed_0 = generate(DummyDatasetSpec(seed=0))
    df_seed_1 = generate(DummyDatasetSpec(seed=1))
    assert _hash_df_values(df_seed_0) != _hash_df_values(df_seed_1)


def test_default_parquet_sha12(tmp_path: Path) -> None:
    """Default-spec parquet sha must match the value verified at scaffold time."""
    out = tmp_path / "case0_train.parquet"
    write(DummyDatasetSpec(), out)
    assert sha256_of_file(out)[:12] == EXPECTED_SHA12_DEFAULT


def test_dtypes_are_compact(tmp_path: Path) -> None:
    out = tmp_path / "case0_train.parquet"
    write(DummyDatasetSpec(num_samples=8), out)
    df = pd.read_parquet(out)
    for i in range(8):
        assert str(df[f"f{i}"].dtype) == "float32"
    assert str(df["label"].dtype) == "int64"
