"""One-shot migration: move local replays to S3 + rewrite index.parquet with replay_uri.

Migrates the current `data/lake/{kaggle_episodes,selfplay}/matches/` layout
from the legacy `replays/*.json.gz` directory tree into the new design where
replays live directly under `s3://{REPLAY_S3_BUCKET}/{REPLAY_S3_PREFIX}/{source}/`.

Steps (per source = kaggle/selfplay):

1. Ensure local matches/replays/ is fully populated (`dvc pull matches.dvc`
   should have been run beforehand).
2. Parallel-upload every `matches/replays/{match_id}.json.gz` to
   `s3://.../replays/{source}/{match_id}.json.gz`.
3. Rewrite every `matches/index.parquet/mode=*/run_*.parquet` to:
   - drop legacy `replay_path` column if present
   - add `replay_uri` column built from match_id + source
4. Stage the new index.parquet under DVC: `dvc add index.parquet`.

`--dry-run` skips S3 upload + Parquet rewrite (logs what would be done).
Resume-friendly: each S3 put is idempotent (skip if object exists with matching size).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import s3fs

from dataset.storage.paths import REPLAY_S3_BUCKET, REPLAY_S3_PREFIX, replay_uri

logger = logging.getLogger(__name__)


def _iter_local_replays(replays_dir: Path) -> list[Path]:
    return sorted(replays_dir.glob("*.json.gz"))


def _upload_one(
    fs: s3fs.S3FileSystem,
    local_path: Path,
    source: str,
    *,
    dry_run: bool,
) -> tuple[Path, bool, str | None]:
    match_id = local_path.name.removesuffix(".json.gz")
    uri = replay_uri(match_id, source)
    s3_key = uri.removeprefix("s3://")
    try:
        if fs.exists(s3_key):
            remote_size = fs.size(s3_key)
            local_size = local_path.stat().st_size
            if remote_size == local_size:
                return (local_path, True, "skip (already up-to-date)")
        if dry_run:
            return (local_path, True, "dry-run (would upload)")
        fs.put_file(str(local_path), s3_key)
        return (local_path, True, None)
    except Exception as exc:  # noqa: BLE001
        return (local_path, False, f"{type(exc).__name__}: {exc}")


def upload_replays(
    matches_dir: Path,
    source: str,
    *,
    workers: int,
    dry_run: bool,
) -> tuple[int, int]:
    replays_dir = matches_dir / "replays"
    if not replays_dir.exists():
        logger.warning("replays_dir missing: %s -- skipping uploads", replays_dir)
        return 0, 0
    local_paths = _iter_local_replays(replays_dir)
    logger.info(
        "upload_replays source=%s files=%d workers=%d dry_run=%s",
        source,
        len(local_paths),
        workers,
        dry_run,
    )
    fs = s3fs.S3FileSystem()
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_upload_one, fs, p, source, dry_run=dry_run)
            for p in local_paths
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            path, ok, msg = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                logger.warning("upload failed: %s -- %s", path, msg)
            if i % 1000 == 0 or i == len(local_paths):
                logger.info(
                    "upload_replays progress: %d/%d (success=%d failed=%d)",
                    i,
                    len(local_paths),
                    success,
                    failed,
                )
    return success, failed


def rewrite_index_parquet(matches_dir: Path, source: str, *, dry_run: bool) -> int:
    """Replace `replay_path` column with `replay_uri` in every index parquet file."""

    index_root = matches_dir / "index.parquet"
    if not index_root.exists():
        logger.warning("index_root missing: %s -- skipping rewrite", index_root)
        return 0
    files = sorted(index_root.glob("**/*.parquet"))
    logger.info(
        "rewrite_index_parquet source=%s files=%d dry_run=%s",
        source,
        len(files),
        dry_run,
    )
    rewritten = 0
    for fp in files:
        df = pl.read_parquet(fp)
        cols = df.columns
        if "replay_uri" in cols and "replay_path" not in cols:
            continue  # already migrated
        if "match_id" not in cols:
            logger.warning("skip (no match_id): %s", fp)
            continue

        def _to_uri(mid: object, src: str = source) -> str:
            return replay_uri(str(mid), src)

        new_df = df.with_columns(
            pl.col("match_id")
            .map_elements(_to_uri, return_dtype=pl.String)
            .alias("replay_uri")
        )
        if "replay_path" in cols:
            new_df = new_df.drop("replay_path")
        if dry_run:
            rewritten += 1
            continue
        # Atomic replace via tmp file
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        new_df.write_parquet(tmp)
        tmp.replace(fp)
        rewritten += 1
    logger.info("rewrite_index_parquet done: rewrote=%d", rewritten)
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("kaggle", "selfplay"),
        required=True,
        help="Which source layout to migrate.",
    )
    parser.add_argument(
        "--matches-dir",
        type=Path,
        help=(
            "matches/ directory (default depends on source: "
            "data/lake/kaggle_episodes/matches or data/lake/selfplay/matches)."
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-rewrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.matches_dir is None:
        args.matches_dir = (
            Path("data/lake/kaggle_episodes/matches")
            if args.source == "kaggle"
            else Path("data/lake/selfplay/matches")
        )
    if not args.matches_dir.exists():
        logger.error("matches_dir does not exist: %s", args.matches_dir)
        return 1

    logger.info(
        "migration start source=%s matches_dir=%s bucket=%s prefix=%s",
        args.source,
        args.matches_dir,
        REPLAY_S3_BUCKET,
        REPLAY_S3_PREFIX,
    )

    failed = 0
    if not args.skip_upload:
        _, failed = upload_replays(
            args.matches_dir,
            args.source,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    if not args.skip_rewrite:
        rewrite_index_parquet(args.matches_dir, args.source, dry_run=args.dry_run)

    logger.info("migration done (failed uploads=%d)", failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Make `bot/src` importable when run directly via `uv run python -m scripts.migrate_replays_to_s3`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
