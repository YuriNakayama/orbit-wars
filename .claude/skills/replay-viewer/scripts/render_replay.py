"""Render a single match replay to standalone HTML and open it in the browser.

Run from the repository root via uv:

    uv run --directory backend python ../.claude/skills/replay-viewer/scripts/render_replay.py \
        --match-id selfplay_run_0_match_42 --source selfplay [--no-open]

Behaviour:
- Resolves the data root from --source (selfplay or kaggle)
- Reconstructs the kaggle_environments Environment via dataset.storage.loader.load_replay
- Calls env.render(mode='html') — the official Kaggle player
- Writes the HTML to /tmp/replay-viewer/<match_id>.html
- Opens the file in the default browser unless --no-open is passed
- Prints the resulting path on stdout (last line) so the caller can show it
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _banner import build_banner, inject  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCES: dict[str, Path] = {
    "selfplay": REPO_ROOT / "data" / "lake" / "selfplay",
    "kaggle": REPO_ROOT / "data" / "lake" / "kaggle_episodes",
}
OUTPUT_DIR = Path("/tmp/replay-viewer")
MAX_PLAYERS = 4


def _ensure_backend_on_path() -> None:
    backend_src = REPO_ROOT / "backend" / "src"
    if backend_src.is_dir() and str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))


def _open_in_browser(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["cmd", "/c", "start", "", str(path)], check=False, shell=False)


def _lookup_index_row(match_id: str, data_root: Path) -> dict[str, object] | None:
    index_root = data_root / "matches" / "index.parquet"
    if not index_root.exists():
        return None
    df = (
        pl.scan_parquet(str(index_root / "**/*.parquet"), hive_partitioning=True)
        .filter(pl.col("match_id") == match_id)
        .collect()
    )
    if df.is_empty():
        return None
    return df.row(0, named=True)


def _row_to_banner_inputs(row: dict[str, object]) -> dict[str, object]:
    agents: list[dict[str, object]] = []
    for idx in range(MAX_PLAYERS):
        name = row.get(f"agent_{idx}_name") or ""
        if not name:
            continue
        agents.append(
            {
                "idx": idx,
                "name": str(name),
                "version": str(row.get(f"agent_{idx}_version") or ""),
                "score": int(row.get(f"agent_{idx}_score") or 0),
            }
        )
    return {
        "match_id": str(row.get("match_id") or ""),
        "mode": str(row.get("mode") or ""),
        "seed": int(row.get("seed") or 0),
        "winner": int(row.get("winner") if row.get("winner") is not None else -1),
        "agents": agents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an Orbit Wars replay to HTML")
    parser.add_argument("--match-id", required=True, help="match_id stored in the index")
    parser.add_argument("--source", choices=["selfplay", "kaggle"], required=True)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the HTML but do not open it in the browser",
    )
    args = parser.parse_args()

    data_root = SOURCES[args.source]
    if not data_root.exists():
        sys.stderr.write(
            f"data root missing: {data_root}\n"
            "Hint: run `uv run --directory backend dvc pull` to fetch replay payloads.\n"
        )
        return 2

    replay_path = data_root / "matches" / "replays" / f"{args.match_id}.json.gz"
    if not replay_path.exists():
        sys.stderr.write(
            f"replay payload missing: {replay_path}\n"
            "This match was recorded without save_replay (or DVC payload not pulled).\n"
            "Hint: run `uv run --directory backend dvc pull` or pick a different match_id.\n"
        )
        return 4

    _ensure_backend_on_path()
    from dataset.storage.loader import load_replay  # type: ignore[import-not-found]

    env = load_replay(args.match_id, data_root=data_root)
    html = env.render(mode="html")
    if not isinstance(html, str):
        sys.stderr.write("env.render(mode='html') returned non-string\n")
        return 3

    row = _lookup_index_row(args.match_id, data_root)
    if row is not None:
        info = _row_to_banner_inputs(row)
        banner = build_banner(
            match_id=str(info["match_id"]),
            mode=str(info["mode"]),
            seed=int(info["seed"]),  # type: ignore[arg-type]
            source=args.source,
            winner_idx=int(info["winner"]),  # type: ignore[arg-type]
            agents=list(info["agents"]),  # type: ignore[arg-type]
        )
        html = inject(html, banner)
    else:
        sys.stderr.write(
            f"index row not found for {args.match_id}; rendering without banner\n"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.match_id}.html"
    out_path.write_text(html, encoding="utf-8")

    if not args.no_open:
        _open_in_browser(out_path)

    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
