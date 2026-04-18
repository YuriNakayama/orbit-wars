"""Polars-based aggregations over the match index."""

from __future__ import annotations

from pathlib import Path

import polars as pl

INDEX_ROOT = "matches/index.parquet"


def scan_index(data_root: Path) -> pl.LazyFrame:
    """Lazy scan of the hive-partitioned match index."""
    root = data_root / INDEX_ROOT
    files = list(root.glob("**/*.parquet")) if root.exists() else []
    if not files:
        return pl.LazyFrame()
    return pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)


def agent_winrate(lf: pl.LazyFrame) -> pl.DataFrame:
    """Win rate per (agent_name, mode), summed across seat positions."""
    frames: list[pl.LazyFrame] = []
    for idx in range(4):
        name_col = f"agent_{idx}_name"
        frames.append(
            lf.filter(pl.col(name_col) != "").select(
                pl.col(name_col).alias("agent"),
                pl.col("mode"),
                (pl.col("winner") == idx).alias("won"),
                pl.col("draw"),
            )
        )
    unified = pl.concat(frames)
    return (
        unified.group_by(["agent", "mode"])
        .agg(
            pl.len().alias("games"),
            pl.col("won").sum().alias("wins"),
            pl.col("draw").sum().alias("draws"),
        )
        .with_columns((pl.col("wins") / pl.col("games")).alias("win_rate"))
        .sort(["mode", "win_rate"], descending=[False, True])
        .collect()
    )


def timing_distribution(lf: pl.LazyFrame) -> pl.DataFrame:
    """Per-agent p50/p95/max of per-turn time (max aggregated across matches)."""
    frames: list[pl.LazyFrame] = []
    for idx in range(4):
        frames.append(
            lf.filter(pl.col(f"agent_{idx}_name") != "").select(
                pl.col(f"agent_{idx}_name").alias("agent"),
                pl.col(f"agent_{idx}_turn_p50").alias("p50"),
                pl.col(f"agent_{idx}_turn_p95").alias("p95"),
                pl.col(f"agent_{idx}_turn_max").alias("turn_max"),
                pl.col(f"agent_{idx}_timeouts").alias("timeouts"),
            )
        )
    unified = pl.concat(frames)
    return (
        unified.group_by("agent")
        .agg(
            pl.col("p50").mean().alias("p50_mean"),
            pl.col("p95").mean().alias("p95_mean"),
            pl.col("turn_max").max().alias("turn_max"),
            pl.col("timeouts").sum().alias("timeouts"),
        )
        .sort("p95_mean", descending=True)
        .collect()
    )


def mode_summary(lf: pl.LazyFrame) -> pl.DataFrame:
    """Per-mode average turns / draw rate / match count."""
    return (
        lf.group_by("mode")
        .agg(
            pl.len().alias("games"),
            pl.col("turns").mean().alias("avg_turns"),
            pl.col("draw").mean().alias("draw_rate"),
        )
        .sort("mode")
        .collect()
    )
