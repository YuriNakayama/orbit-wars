"""Pick two replay paths for the experiment-analysis skill: a long match + a fastest defeat.

Stdout is intentionally minimal — two absolute paths labelled `long` and `fastest_loss`,
ready to feed into `replay_to_markdown.py`. The skill reads only this stdout, never the raw
index data, to keep context lean.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from statistics import quantiles


def list_replays(replays_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(replays_dir.glob("*.json.gz"))
    if limit is not None:
        paths = paths[-limit:]
    return paths


def summarize_replay(path: Path) -> dict | None:
    """Read just enough of the gzipped replay to know total_turns and rewards."""
    try:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    steps = data.get("steps") or []
    rewards = data.get("rewards") or []
    if not steps:
        return None
    return {
        "path": path,
        "total_turns": len(steps) - 1,
        "rewards": rewards,
        "winner": next((i for i, r in enumerate(rewards) if r is not None and r > 0), None),
    }


def pick(
    summaries: list[dict],
    self_player_id: int,
    long_quantile: float,
    fast_quantile: float,
) -> tuple[Path | None, Path | None]:
    if not summaries:
        return None, None
    turns = sorted(s["total_turns"] for s in summaries)
    if len(turns) >= 4:
        q = quantiles(turns, n=4)
        long_thresh = q[2]
        fast_thresh = q[0]
    else:
        long_thresh = max(turns)
        fast_thresh = min(turns)

    longs = [s for s in summaries if s["total_turns"] >= long_thresh]
    losses = [
        s
        for s in summaries
        if s["total_turns"] <= fast_thresh
        and s["winner"] is not None
        and s["winner"] != self_player_id
    ]
    if not losses:
        losses = [s for s in summaries if s["winner"] is not None and s["winner"] != self_player_id]
        losses.sort(key=lambda s: s["total_turns"])
        losses = losses[:1]

    longs.sort(key=lambda s: s["total_turns"], reverse=True)
    losses.sort(key=lambda s: s["total_turns"])

    long_pick = longs[0]["path"] if longs else None
    loss_pick = losses[0]["path"] if losses else None
    if long_pick is not None and loss_pick is not None and long_pick == loss_pick:
        if len(longs) > 1:
            long_pick = longs[1]["path"]
        elif len(losses) > 1:
            loss_pick = losses[1]["path"]
    return long_pick, loss_pick


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays-dir", required=True, type=Path, help="Directory of *.json.gz replays")
    parser.add_argument("--self-player-id", type=int, default=1, help="Player ID treated as 'self'")
    parser.add_argument("--limit", type=int, default=200, help="Look at the most recent N replays")
    parser.add_argument("--long-quantile", type=float, default=0.75)
    parser.add_argument("--fast-quantile", type=float, default=0.25)
    args = parser.parse_args()

    paths = list_replays(args.replays_dir, args.limit)
    summaries = [s for s in (summarize_replay(p) for p in paths) if s is not None]
    if not summaries:
        print("ERROR: no readable replays in directory", file=sys.stderr)
        return 1

    long_pick, loss_pick = pick(summaries, args.self_player_id, args.long_quantile, args.fast_quantile)
    if long_pick is None or loss_pick is None:
        print(
            f"ERROR: could not pick both labels (long={long_pick}, fastest_loss={loss_pick}); "
            f"母集団 {len(summaries)} 件",
            file=sys.stderr,
        )
        return 1

    print(f"long\t{long_pick.resolve()}")
    print(f"fastest_loss\t{loss_pick.resolve()}")
    print(
        f"# scanned {len(summaries)} replays; long_thresh~{sorted(s['total_turns'] for s in summaries)[max(0, len(summaries)*3//4-1)]}, "
        f"fast_thresh~{sorted(s['total_turns'] for s in summaries)[max(0, len(summaries)//4-1)]}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
