"""Polars-based minority row duplication for imitation/case8.

Reads an existing train parquet, filters rows whose `target_per_src` fires
any minority template id, and concatenates duplicates. Zero-copy reference
sharing keeps memory footprint flat vs the list[dict] path that OOM'd at
~17 GB for (289k rows * 2x).

Usage:
  uv run python -m pipeline.imitation.case9.training.expand_minority \\
      --in-parquet data/mart/imitation/case8/train_q30.parquet \\
      --out-parquet data/mart/imitation/case8/train_q30_dup.parquet \\
      --minority 0,2,5,6,7 \\
      --multiplier 2
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _row_fires_minority(target_per_src: list[int], minority: set[int]) -> bool:
    for t in target_per_src:
        if t != -1 and t in minority:
            return True
    return False


@app.command()
def main(
    in_parquet: Path = typer.Option(..., "--in-parquet"),  # noqa: B008
    out_parquet: Path = typer.Option(..., "--out-parquet"),  # noqa: B008
    minority: str = typer.Option(
        "0,2,5,6,7", "--minority", help="Comma-separated template ids"
    ),
    multiplier: int = typer.Option(2, "--multiplier", min=1),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    minority_ids = {int(x) for x in minority.split(",")}
    logger.info(
        "expand_minority: in=%s minority=%s multiplier=%d",
        in_parquet,
        sorted(minority_ids),
        multiplier,
    )

    df = pl.read_parquet(in_parquet)
    logger.info("read %d rows (%.1f MB)", df.height, df.estimated_size("mb"))

    if multiplier <= 1:
        df.write_parquet(out_parquet)
        logger.info("multiplier<=1, pass-through to %s", out_parquet)
        return

    mask = df["target_per_src"].map_elements(
        lambda lst: _row_fires_minority(lst.to_list(), minority_ids),
        return_dtype=pl.Boolean,
    )
    minority_df = df.filter(mask)
    logger.info(
        "minority rows: %d / %d (%.1f%%)",
        minority_df.height,
        df.height,
        100.0 * minority_df.height / df.height,
    )

    extra_copies = multiplier - 1
    frames = [df]
    for _ in range(extra_copies):
        frames.append(minority_df)
    out_df = pl.concat(frames, how="vertical")
    logger.info(
        "expanded: %d -> %d rows (+%d duplicates, %.1f MB)",
        df.height,
        out_df.height,
        extra_copies * minority_df.height,
        out_df.estimated_size("mb"),
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_parquet)
    logger.info("wrote %s", out_parquet)


if __name__ == "__main__":
    app()
