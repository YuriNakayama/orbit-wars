"""One-time converter: case11 mart parquet -> per-column uncompressed .npy.

Why: ZSTD parquet + row-group lazy load makes 1 epoch ~2-3h on L4 (IO bound).
Converting once to plain `np.save` files lets Dataset use `mmap_mode="r"`,
which defers reads to the OS page cache. RSS stays at ~50MB while training,
and per-batch random reads become indistinguishable from in-memory access.

Output layout (per split):
  data/mart/imitation/case11/<split>_npy/
    planet_feats.npy           # (N, MAX_PLANETS, PLANET_FEAT_DIM) float32
    global_feats.npy           # (N, GLOBAL_FEAT_DIM) float32
    planet_mask.npy            # (N, MAX_PLANETS) bool
    target_mask.npy            # (N, MAX_PLANETS) bool
    template_ctx.npy           # (N, MAX_PLANETS, TEMPLATE_CTX_DIM) float32
    candidate_feats.npy        # (N, MAX_PLANETS, CAND_K, CAND_FEAT_DIM) f32
    candidate_mask.npy
    candidate_pid.npy
    effective_source_mask.npy  # (N, MAX_PLANETS) bool  (layer2 source mask)
    should_learn_ship.npy      # (N, MAX_PLANETS) bool  (layer3 ship-head mask)
    target_pid_per_src.npy     # (N, MAX_PLANETS) int64 (0..P-1=fire slot,
                               #   P=no-op sentinel)
    ship_pred_label.npy        # (N, MAX_PLANETS) float32 (log1p(ships) for
                               #   fired sources, -1.0 otherwise)
    is_noop.npy                # (N,) bool

Memory: streams row_group by row_group, writes one column at a time using
`np.memmap` so peak RSS stays under 8GB even for the 19GB train parquet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from pipeline.imitation.case11.policy.candidates import CAND_FEAT_DIM, CAND_K
from pipeline.imitation.case11.policy.featurizer import (
    GLOBAL_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
)
from pipeline.imitation.case11.policy.templates import TEMPLATE_CTX_DIM

logger = logging.getLogger(__name__)


# (parquet_column, dtype, shape_per_row)
_LIST_COLS: tuple[tuple[str, np.dtype, tuple[int, ...]], ...] = (
    ("planet_feats", np.dtype(np.float32), (MAX_PLANETS, PLANET_FEAT_DIM)),
    ("global_feats", np.dtype(np.float32), (GLOBAL_FEAT_DIM,)),
    ("planet_mask", np.dtype(np.bool_), (MAX_PLANETS,)),
    ("target_mask", np.dtype(np.bool_), (MAX_PLANETS,)),
    ("template_ctx", np.dtype(np.float32), (MAX_PLANETS, TEMPLATE_CTX_DIM)),
    ("candidate_feats", np.dtype(np.float32), (MAX_PLANETS, CAND_K, CAND_FEAT_DIM)),
    ("candidate_mask", np.dtype(np.bool_), (MAX_PLANETS, CAND_K)),
    ("candidate_pid", np.dtype(np.int64), (MAX_PLANETS, CAND_K)),
    ("effective_source_mask", np.dtype(np.bool_), (MAX_PLANETS,)),
    ("should_learn_ship", np.dtype(np.bool_), (MAX_PLANETS,)),
    ("target_pid_per_src", np.dtype(np.int64), (MAX_PLANETS,)),
    ("ship_pred_label", np.dtype(np.float32), (MAX_PLANETS,)),
)


def _flatten_arrow(arr: pa.Array) -> pa.Array:
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        arr = arr.flatten()
    return arr


def _chunk_to_numpy(arr: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    chunks = arr.chunks if arr.num_chunks > 0 else [arr.combine_chunks()]
    parts: list[np.ndarray] = []
    for ch in chunks:
        flat = _flatten_arrow(ch)
        parts.append(flat.to_numpy(zero_copy_only=False).astype(dtype, copy=True))
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _default_for_missing(
    col: str, n_rows: int, dtype: np.dtype, shape_per_row: tuple[int, ...]
) -> np.ndarray:
    """Return a default-filled buffer for a column missing from older parquet.

    case11 parquet from this point on always emits the full column set, so
    this only matters when re-using an older mart accidentally. Surfaces a
    clear KeyError when an unexpected column is requested.
    """
    out_shape = (n_rows,) + shape_per_row
    if col == "target_pid_per_src":
        return np.full(out_shape, MAX_PLANETS, dtype=dtype)
    if col == "ship_pred_label":
        return np.full(out_shape, -1.0, dtype=dtype)
    if col == "template_ctx":
        return np.zeros(out_shape, dtype=dtype)
    if col in ("effective_source_mask", "should_learn_ship"):
        return np.zeros(out_shape, dtype=dtype)
    raise KeyError(f"no default registered for missing column: {col}")


def convert_split(
    parquet_path: Path, out_dir: Path, max_rows: int | None = None
) -> None:
    """Stream a single split parquet -> per-column .npy under out_dir.

    If ``max_rows`` is given, stop after writing that many rows. Used for
    smoke runs so the full 14-min conversion isn't required to verify
    downstream logic.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(parquet_path))
    schema_names = set(pf.schema_arrow.names)
    full_rows = int(pf.metadata.num_rows)
    n_rows = full_rows if max_rows is None else min(full_rows, int(max_rows))
    n_groups = pf.num_row_groups
    logger.info(
        "convert split=%s rows=%d (full=%d cap=%s) groups=%d -> %s",
        parquet_path.name,
        n_rows,
        full_rows,
        str(max_rows),
        n_groups,
        out_dir,
    )

    # Pre-allocate memmaps for each column (sized to the capped row count).
    memmaps: dict[str, np.memmap] = {}
    for col, dtype, shape_per_row in _LIST_COLS:
        out_path = out_dir / f"{col}.npy"
        # We use np.lib.format.open_memmap which writes the .npy header.
        memmaps[col] = np.lib.format.open_memmap(
            str(out_path),
            mode="w+",
            dtype=dtype,
            shape=(n_rows,) + shape_per_row,
        )
    # is_noop is primitive (not list-typed).
    is_noop_mm = np.lib.format.open_memmap(
        str(out_dir / "is_noop.npy"),
        mode="w+",
        dtype=np.dtype(np.bool_),
        shape=(n_rows,),
    )

    cursor = 0
    for g in range(n_groups):
        if cursor >= n_rows:
            break
        table = pf.read_row_group(g)
        rg_rows = int(pf.metadata.row_group(g).num_rows)
        # Trim the last row_group if it would overshoot the cap.
        remaining = n_rows - cursor
        take = min(rg_rows, remaining)
        if g % 50 == 0 or g == n_groups - 1 or take < rg_rows:
            logger.info(
                "  group %d/%d rows %d-%d (take=%d/%d)",
                g,
                n_groups,
                cursor,
                cursor + take,
                take,
                rg_rows,
            )

        for col, dtype, shape_per_row in _LIST_COLS:
            if col not in schema_names:
                memmaps[col][cursor : cursor + take] = _default_for_missing(
                    col, take, dtype, shape_per_row
                )
                continue
            buf = _chunk_to_numpy(table[col], dtype).reshape((rg_rows,) + shape_per_row)
            memmaps[col][cursor : cursor + take] = buf[:take]

        is_noop_chunk = np.array(
            table["is_noop"].to_numpy(zero_copy_only=False),
            dtype=np.bool_,
            copy=True,
        )
        is_noop_mm[cursor : cursor + take] = is_noop_chunk[:take]

        cursor += take
        del table
    assert cursor == n_rows, f"row mismatch: cursor={cursor} expected={n_rows}"

    # Flush + close memmaps.
    for mm in memmaps.values():
        mm.flush()
        del mm
    is_noop_mm.flush()
    del is_noop_mm

    # Force fsync per-file. mm.flush() only msync()s the memory mapping
    # but on MFS the inode metadata sync is on a separate code path —
    # observed (commit 2cb655d) that planet_feats.npy / global_feats.npy
    # / target_pid_per_src.npy / ship_pred_label.npy / is_noop.npy were
    # absent in Path.exists() check ~1s after flush. Open each file by
    # fd and fsync to force the metadata flush.
    expected_files = [f"{col}.npy" for col, _dt, _sh in _LIST_COLS] + ["is_noop.npy"]
    for name in expected_files:
        p = out_dir / name
        try:
            with open(p, "rb+") as f:
                os.fsync(f.fileno())
        except FileNotFoundError:
            logger.warning("fsync skip: %s not present yet (will retry)", p)
    # And a global sync as belt-and-braces.
    os.sync()

    # Verify presence + size so downstream `[ -d "$out_dir" ]` checks won't
    # silently see an empty directory. If a file is missing or zero-byte,
    # retry the verify after a short sleep to let MFS metadata catch up
    # (commit 2cb655d observed 5 files missing right after flush despite
    # os.sync). Failure after retries raises RuntimeError.
    import time
    for attempt in range(1, 4):
        logger.info("verifying outputs in %s (attempt %d/3)", out_dir, attempt)
        missing: list[str] = []
        zero_byte: list[str] = []
        for name in expected_files:
            p = out_dir / name
            if not p.exists():
                missing.append(name)
                continue
            sz = p.stat().st_size
            if sz == 0:
                zero_byte.append(name)
                continue
        if not (missing or zero_byte):
            for name in expected_files:
                sz = (out_dir / name).stat().st_size
                logger.info("  %s: %.1f MB", name, sz / 1e6)
            break
        logger.warning(
            "verify attempt %d/3 found missing=%d zero=%d; sleeping 10s and retrying",
            attempt,
            len(missing),
            len(zero_byte),
        )
        os.sync()
        time.sleep(10)
    else:
        raise RuntimeError(
            f"verify failed after 3 attempts: missing={missing} "
            f"zero_byte={zero_byte} out_dir={out_dir}"
        )

    logger.info("done split=%s -> %s", parquet_path.name, out_dir)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bot").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else (_repo_root() / p).resolve()


app = typer.Typer(add_completion=False, help="case11 parquet -> per-column .npy")


@app.command()
def main(
    train_parquet: Path = typer.Option(  # noqa: B008
        Path("data/mart/imitation/case11/train.parquet"), "--train-parquet"
    ),
    val_parquet: Path = typer.Option(  # noqa: B008
        Path("data/mart/imitation/case11/val.parquet"), "--val-parquet"
    ),
    out_root: Path = typer.Option(  # noqa: B008
        Path("data/mart/imitation/case11"), "--out-root"
    ),
    max_train_rows: int = typer.Option(
        0, "--max-train-rows", help="cap train rows (0 = unlimited, smoke-only)"
    ),
    max_val_rows: int = typer.Option(
        0, "--max-val-rows", help="cap val rows (0 = unlimited, smoke-only)"
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Resolve relative to the repo root (one level above `bot/`) so the
    # script works whether invoked from worktree root or from inside `bot/`.
    train_parquet = _abs(train_parquet)
    val_parquet = _abs(val_parquet)
    out_root = _abs(out_root)
    convert_split(
        train_parquet,
        out_root / "train_npy",
        max_rows=max_train_rows if max_train_rows > 0 else None,
    )
    convert_split(
        val_parquet,
        out_root / "val_npy",
        max_rows=max_val_rows if max_val_rows > 0 else None,
    )


if __name__ == "__main__":
    app()
