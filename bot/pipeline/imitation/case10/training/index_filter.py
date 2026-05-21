"""Side-index filtering for case10 base mart.

The base mart (data/mart/imitation/case10/base/{train,val}.parquet) is built
ONCE with loser_swap=true and no top_K cap. Per-experiment filters are applied
in-memory by reading the row-aligned ``*_index.parquet`` and computing
``keep_row_indices`` to pass into ``CaseTenDataset(..., keep_row_indices=...)``.

Supported filters (composable; all default to "keep all"):

- ``top_submission_limit``: keep only matches whose either side's submission_id
  ranks within the top_K by max rating across the index. Mirrors
  ``preprocess._filter_index`` semantics, but applied at row (frame) level.
- ``use_loser_swap``: when ``False``, drop rows where ``is_loser_view=true``
  so the run sees only original-view frames (= R1 case10 既定再現).
- ``modes``: subset of {"1v1", "ffa4"}.
- ``min_turn_count`` / ``max_turn_count``: episode turn-count range.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl


def _expected_index_path(mart_path: Path | str) -> Path:
    p = Path(mart_path)
    return p.with_name(p.stem + "_index.parquet")


def load_index(mart_path: Path | str) -> pl.DataFrame:
    """Load the row-aligned side index for the given mart parquet."""
    return pl.read_parquet(_expected_index_path(mart_path))


def _top_submission_ids(index_df: pl.DataFrame, top_submission_limit: int) -> set[str]:
    """Pick top-K submission ids by max(p0_rating_mu, p1_rating_mu) across the
    index. Mirrors ``preprocess._filter_index`` ranking logic.
    """
    parts: list[pl.DataFrame] = []
    for sub_col, mu_col in (
        ("p0_submission_id", "p0_rating_mu"),
        ("p1_submission_id", "p1_rating_mu"),
    ):
        parts.append(
            index_df.select(
                pl.col(sub_col).alias("submission_id"),
                pl.col(mu_col).alias("mu"),
            )
        )
    ratings = pl.concat(parts).filter(
        (pl.col("mu") > 0) & (pl.col("submission_id") != "")
    )
    if ratings.is_empty():
        return set()
    ranked = (
        ratings.group_by("submission_id")
        .agg(pl.col("mu").max().alias("max_mu"))
        .sort("max_mu", descending=True)
    )
    return set(ranked.head(int(top_submission_limit))["submission_id"].to_list())


def compute_keep_row_indices(
    mart_path: Path | str,
    *,
    top_submission_limit: int | None = None,
    use_loser_swap: bool = True,
    modes: Iterable[str] | None = None,
    min_turn_count: int | None = None,
    max_turn_count: int | None = None,
) -> np.ndarray:
    """Resolve the side index into a numpy array of row indices to keep.

    The returned indices are in ascending order so the resulting Dataset
    preserves the original mart row order (important for deterministic split
    + reproducibility).
    """
    df = load_index(mart_path)
    mask = pl.lit(True)
    if not use_loser_swap:
        mask = mask & (~pl.col("is_loser_view"))
    if modes is not None:
        mask = mask & pl.col("mode").is_in(list(modes))
    if min_turn_count is not None:
        mask = mask & (pl.col("turn_count") >= int(min_turn_count))
    if max_turn_count is not None:
        mask = mask & (pl.col("turn_count") <= int(max_turn_count))
    if top_submission_limit is not None:
        keep_ids = _top_submission_ids(df, int(top_submission_limit))
        mask = mask & (
            pl.col("p0_submission_id").is_in(list(keep_ids))
            | pl.col("p1_submission_id").is_in(list(keep_ids))
        )
    filtered = df.filter(mask)
    return np.asarray(filtered["row_idx"].to_numpy(), dtype=np.int64)
