"""既取得 episode_id の抽出ヘルパ（Parquet index 経由）。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from env.types import SOURCE_KAGGLE

_INDEX_DIRNAME = "index.parquet"


def _index_root(data_root: Path) -> Path:
    return data_root / "matches" / _INDEX_DIRNAME


def existing_episode_ids(
    data_root: Path,
    modes: tuple[str, ...] | None = None,
) -> set[int]:
    """既に保存済みの episode_id を set で返す。

    Parquet index が未作成の場合は空 set。selfplay 行は除外し、
    `source == "kaggle"` の行だけを集計対象にする。
    """

    root = _index_root(data_root)
    if not root.exists():
        return set()
    files = list(root.glob("**/*.parquet"))
    if not files:
        return set()
    lf = pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
    lf = lf.filter(pl.col("source") == SOURCE_KAGGLE)
    if modes is not None and len(modes) > 0:
        lf = lf.filter(pl.col("mode").is_in(list(modes)))
    df = lf.select("episode_id").unique().collect()
    return {int(v) for v in df["episode_id"].to_list() if v is not None and int(v) >= 0}
