"""Reproduce a self-play match by re-running it from index metadata, when the
saved replay payload is missing.

Use when `render_replay.py` exits with code 4 (replay payload missing).

The script reads the index parquet for the given match_id, looks up its
seed / mode / agents, re-runs the match via kaggle_environments, and writes
the HTML player to /tmp/replay-viewer/<match_id>.html — the same place
render_replay.py uses.

Caveat: the reproduction is only as deterministic as the agent code at the
current commit. If the agents have changed since recording, the outcome may
diverge. The script prints the observed winner so the caller can compare
with the recorded one.

When the user's request implies a specific outcome ("matches where X won"),
pass `--require-winner-name <agent_name>`. The script will then look up
*all* index rows where that agent won with the same matchup, try them in
order, and stop at the first reproduction that still produces the requested
winner. This costs more time but is the only reliable way to honour the
user's intent when agent code has drifted.

Usage:

    uv run --directory backend python ../.agents/skills/replay-viewer/scripts/rerun_match.py \
        --match-id <match_id> --source selfplay [--no-open] \
        [--require-winner-name <agent_name>] [--max-attempts 30]

Only `--source selfplay` is supported — Kaggle episodes are external runs and
cannot be reproduced locally.
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
SELFPLAY_ROOT = REPO_ROOT / "data" / "lake" / "selfplay"
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
        subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)


def _index_root() -> Path:
    return SELFPLAY_ROOT / "matches" / "index.parquet"


def _lookup_match(match_id: str) -> dict[str, object] | None:
    root = _index_root()
    if not root.exists():
        return None
    df = pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True).filter(
        pl.col("match_id") == match_id
    ).collect()
    if df.is_empty():
        return None
    return df.row(0, named=True)


def _lookup_matchup_wins(
    agent_names: list[str], winner_name: str, exclude_match_id: str
) -> list[dict[str, object]]:
    """Find rows where the same matchup happened and `winner_name` won."""
    root = _index_root()
    if not root.exists() or winner_name not in agent_names:
        return []
    target_idx = agent_names.index(winner_name)
    expr = pl.lit(True)
    for idx, name in enumerate(agent_names):
        expr = expr & (pl.col(f"agent_{idx}_name") == name)
    expr = expr & (pl.col("winner") == target_idx) & (~pl.col("draw"))
    expr = expr & (pl.col("match_id") != exclude_match_id)
    df = (
        pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
        .filter(expr)
        .sort("started_at", descending=True)
        .collect()
    )
    return df.to_dicts()


def _winner_from_state(state: list[dict[str, object]]) -> int:
    """Heuristic that mirrors selfplay/executor: last ACTIVE wins, else max reward."""
    active = [i for i, s in enumerate(state) if s.get("status") == "ACTIVE"]
    if len(active) == 1:
        return active[0]
    rewards = [float(s.get("reward") or 0.0) for s in state]
    top = max(rewards)
    leaders = [i for i, r in enumerate(rewards) if r == top]
    return leaders[0] if len(leaders) == 1 else -1


def _agent_names_from_row(row: dict[str, object]) -> list[str]:
    names: list[str] = []
    for idx in range(MAX_PLAYERS):
        name = row.get(f"agent_{idx}_name") or ""
        if name:
            names.append(str(name))
    return names


def _run_once(
    seed: int, agent_names: list[str], make: object, resolve: object
) -> tuple[object, int]:
    callables = [resolve(n) for n in agent_names]  # type: ignore[operator]
    env = make("orbit_wars", configuration={"agents": len(callables), "seed": seed})  # type: ignore[operator]
    env.run(callables)  # type: ignore[attr-defined]
    return env, _winner_from_state(env.state)  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce a self-play replay by re-running it")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--source", choices=["selfplay"], default="selfplay")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--require-winner-name",
        default=None,
        help="Retry across other matchup rows until this agent reproduces a win",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=20, help="Max seeds to try when --require-winner-name is set"
    )
    args = parser.parse_args()

    row = _lookup_match(args.match_id)
    if row is None:
        sys.stderr.write(f"match_id not found in index: {args.match_id}\n")
        return 2

    seed = int(row.get("seed", 0) or 0)
    mode = str(row.get("mode", "1v1") or "1v1")
    agent_names = _agent_names_from_row(row)
    if not agent_names:
        sys.stderr.write("no agents recorded in index\n")
        return 3
    recorded_winner = int(row.get("winner", -1) or -1)

    _ensure_backend_on_path()
    from dataset.selfplay.agents import resolve  # type: ignore[import-not-found]
    from kaggle_environments import make  # type: ignore[import-not-found]

    env, observed = _run_once(seed, agent_names, make, resolve)
    print(f"reran match_id={args.match_id} mode={mode} seed={seed}")
    print(f"agents={agent_names}")
    print(f"recorded_winner={recorded_winner} observed_winner={observed}")

    final_match_id = args.match_id
    final_seed = seed

    if args.require_winner_name and args.require_winner_name in agent_names:
        target_idx = agent_names.index(args.require_winner_name)
        if observed != target_idx:
            sys.stderr.write(
                f"first attempt did not produce {args.require_winner_name} as winner; searching alternatives...\n"
            )
            candidates = _lookup_matchup_wins(agent_names, args.require_winner_name, args.match_id)
            tried = 1
            for cand in candidates:
                if tried >= args.max_attempts:
                    break
                cand_seed = int(cand.get("seed", 0) or 0)
                env, observed = _run_once(cand_seed, agent_names, make, resolve)
                tried += 1
                cand_match_id = str(cand.get("match_id", ""))
                print(
                    f"attempt {tried}: match_id={cand_match_id} seed={cand_seed} observed_winner={observed}"
                )
                if observed == target_idx:
                    final_match_id = cand_match_id
                    final_seed = cand_seed
                    break
            else:
                sys.stderr.write(
                    f"exhausted {tried} attempts without reproducing {args.require_winner_name} as winner\n"
                )
                return 5
        if observed != target_idx:
            return 5
    elif observed != recorded_winner:
        sys.stderr.write(
            "WARNING: observed winner differs from recorded winner — agent code may have changed since recording.\n"
        )

    html = env.render(mode="html")  # type: ignore[attr-defined]
    if not isinstance(html, str):
        sys.stderr.write("env.render(mode='html') returned non-string\n")
        return 4

    # Score is unknown for re-runs (we don't extract it from env.state); use 0 as placeholder.
    agents_for_banner = [
        {"idx": i, "name": name, "version": "", "score": 0}
        for i, name in enumerate(agent_names)
    ]
    note = "rerun (replay payload missing): identifies are taken from index"
    if final_match_id != args.match_id:
        note += f"; original match {args.match_id} did not reproduce, showing seed={final_seed} instead"
    elif observed != recorded_winner:
        note += (
            f"; recorded winner was P{recorded_winner + 1} but current agent code reproduces P{observed + 1}"
        )
    banner = build_banner(
        match_id=final_match_id,
        mode=mode,
        seed=final_seed,
        source="selfplay (rerun)",
        winner_idx=observed,
        agents=agents_for_banner,
        note=note,
    )
    html = inject(html, banner)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{final_match_id}.html"
    out_path.write_text(html, encoding="utf-8")

    if not args.no_open:
        _open_in_browser(out_path)

    print(f"final: match_id={final_match_id} seed={final_seed} winner_index={observed}")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
