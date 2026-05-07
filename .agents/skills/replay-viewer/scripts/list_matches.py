"""List recorded matches as JSON for the replay-viewer skill.

The skill agent runs this from the repository root via uv:

    uv run --directory backend python ../.agents/skills/replay-viewer/scripts/list_matches.py \
        --source selfplay --mode 1v1 --limit 20 --winner 0

Output: JSON array on stdout with one entry per match (newest first).
Schema is documented in the skill's SKILL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_LAKE = REPO_ROOT / "data" / "lake"
SOURCES: dict[str, Path] = {
    "selfplay": DATA_LAKE / "selfplay",
    "kaggle": DATA_LAKE / "kaggle_episodes",
}


def _ensure_backend_on_path() -> None:
    backend_src = REPO_ROOT / "backend" / "src"
    if backend_src.is_dir() and str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))


def _list_for_source(source: str, mode: str | None, limit: int) -> pl.DataFrame:
    _ensure_backend_on_path()
    from dataset.storage.loader import list_matches  # type: ignore[import-not-found]

    data_root = SOURCES[source]
    if not data_root.exists():
        return pl.DataFrame()
    df = list_matches(data_root=data_root, mode=mode, limit=None)
    if df.is_empty():
        return df
    return df.with_columns(pl.lit(source).alias("source"))


def _filter_by_agent(df: pl.DataFrame, agent: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    cols = [c for c in df.columns if c.endswith("_name") and c.startswith("agent_")]
    if not cols:
        return df
    expr = pl.lit(False)
    for c in cols:
        expr = expr | pl.col(c).cast(pl.Utf8).str.contains(agent, literal=True)
    return df.filter(expr)


def _filter_by_winner(df: pl.DataFrame, winner: int) -> pl.DataFrame:
    if df.is_empty() or "winner" not in df.columns:
        return df
    return df.filter(pl.col("winner") == winner)


def _filter_by_replay_existence(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    keep: list[bool] = []
    for row in df.iter_rows(named=True):
        source = row.get("source")
        match_id = row.get("match_id")
        if not source or not match_id or source not in SOURCES:
            keep.append(False)
            continue
        replay_path = SOURCES[source] / "matches" / "replays" / f"{match_id}.json.gz"
        keep.append(replay_path.exists())
    return df.filter(pl.Series("__has_replay", keep))


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _row_to_summary(row: dict[str, object]) -> dict[str, object]:
    agents: list[dict[str, object]] = []
    for idx in range(4):
        name = row.get(f"agent_{idx}_name") or ""
        if not name:
            continue
        agents.append(
            {
                "idx": idx,
                "name": name,
                "version": row.get(f"agent_{idx}_version") or "",
                "score": _coerce_int(row.get(f"agent_{idx}_score"), 0),
            }
        )
    return {
        "match_id": row.get("match_id"),
        "source": row.get("source"),
        "mode": row.get("mode"),
        "started_at": row.get("started_at"),
        "turns": _coerce_int(row.get("turns"), 0),
        "winner": _coerce_int(row.get("winner"), -1),
        "draw": bool(row.get("draw", False)),
        "elapsed_sec": _coerce_float(row.get("elapsed_sec"), 0.0),
        "episode_id": _coerce_int(row.get("episode_id"), -1),
        "agents": agents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="List Orbit Wars match records")
    parser.add_argument(
        "--source",
        choices=["selfplay", "kaggle", "both"],
        default="both",
        help="Which data lake to read from",
    )
    parser.add_argument("--mode", default=None, help="Filter by env mode (e.g. 1v1, ffa)")
    parser.add_argument("--agent", default=None, help="Substring match on any agent_*_name")
    parser.add_argument("--winner", type=int, default=None, help="Filter by winner index (0-3)")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to return")
    parser.add_argument(
        "--has-replay",
        action="store_true",
        help="Keep only matches whose replay payload exists on disk",
    )
    args = parser.parse_args()

    sources = ["selfplay", "kaggle"] if args.source == "both" else [args.source]
    frames = [_list_for_source(s, args.mode, args.limit) for s in sources]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        print("[]")
        return 0

    df = pl.concat(frames, how="diagonal_relaxed")
    if args.agent:
        df = _filter_by_agent(df, args.agent)
    if args.winner is not None:
        df = _filter_by_winner(df, args.winner)
    if args.has_replay:
        df = _filter_by_replay_existence(df)
    if "started_at" in df.columns:
        df = df.sort("started_at", descending=True)
    df = df.head(args.limit)

    rows = df.to_dicts()
    out = [_row_to_summary(r) for r in rows]
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
