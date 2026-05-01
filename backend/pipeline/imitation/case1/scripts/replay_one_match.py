"""Replay a single il_v1 vs baseline_v1 match for a given seed with save_replay=True.

Usage (run from backend/):
    uv run python -m pipeline.imitation.case1.scripts.replay_one_match \
        --seed 19 --label win
    uv run python -m pipeline.imitation.case1.scripts.replay_one_match \
        --seed 0  --label loss

Uses the canonical selfplay runner (same code path as eval_vs_baseline) so the
outcome matches the original 100-game batch. Writes a short trace to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# backend/pipeline/imitation/case1/scripts/<this>.py
# parents[4] = backend/, backend/src を import path に追加.
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.selfplay.runner import RunSpec, run_episodes  # noqa: E402

OUT_DIR = Path("docs/experiment/imitation_case1_replays")


def run_one(seed: int, label: str, data_root: Path) -> None:
    spec = RunSpec(
        agents=("il_v1", "baseline_v1"),
        mode="1v1",
        episodes=1,
        seed=seed,
        parallel=1,
        save_replay=True,
        data_root=data_root,
    )
    records = run_episodes(spec)
    rec = records[0]
    print(f"=== {label.upper()} (seed={seed}) ===")
    print(f"match_id    : {rec.match_id}")
    print(f"winner idx  : {rec.winner}  (0=il_v1, 1=baseline_v1)  draw={rec.draw}")
    print(f"turns played: {rec.turns}")
    print(
        f"scores      : il_v1={rec.agent_scores[0]}  baseline_v1={rec.agent_scores[1]}"
    )
    print(
        f"timings p50 : il_v1={rec.agent_timings[0].p50:.3f}s  base={rec.agent_timings[1].p50:.3f}s"
    )

    replay_path = Path(rec.replay_path) if rec.replay_path else None
    if not replay_path or not replay_path.exists():
        print("(no replay file recorded)")
        return
    print(f"replay path : {replay_path}")

    data = json.loads(replay_path.read_text())
    steps = data.get("steps") or []
    if not steps:
        print("(empty replay)")
        return

    il_act = Counter[int]()
    base_act = Counter[int]()
    rows: list[
        tuple[int, int, int, int, int]
    ] = []  # turn, il_ships, base_ships, il_p, base_p
    for t, step in enumerate(steps):
        if t == 0:
            continue
        il_entry, base_entry = step[0], step[1]
        il_a = il_entry.get("action") or []
        base_a = base_entry.get("action") or []
        il_act[len(il_a)] += 1
        base_act[len(base_a)] += 1
        if t % 20 == 0 or t == len(steps) - 1:
            obs = il_entry.get("observation") or {}
            planets = obs.get("planets") or []
            il_ships = sum(int(p[5]) for p in planets if int(p[1]) == 0)
            base_ships = sum(int(p[5]) for p in planets if int(p[1]) == 1)
            il_p = sum(1 for p in planets if int(p[1]) == 0)
            base_p = sum(1 for p in planets if int(p[1]) == 1)
            rows.append((t, il_ships, base_ships, il_p, base_p))

    print(f"il_v1   actions/turn: {dict(sorted(il_act.items()))}")
    print(f"base    actions/turn: {dict(sorted(base_act.items()))}")
    print()
    print("turn | il_ships | base_ships | il_planets | base_planets")
    print("-" * 58)
    for turn, il_s, base_s, il_p, base_p in rows:
        print(f"{turn:4d} | {il_s:8d} | {base_s:10d} | {il_p:10d} | {base_p:12d}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"iter9_{label}_seed{seed}.json"
    out.write_text(
        json.dumps(
            {
                "seed": seed,
                "label": label,
                "match_id": rec.match_id,
                "winner_idx": rec.winner,
                "draw": rec.draw,
                "turns": rec.turns,
                "scores": list(rec.agent_scores),
                "replay_path": str(replay_path),
                "actions_per_turn_il": dict(il_act),
                "actions_per_turn_base": dict(base_act),
                "trace": [
                    {
                        "turn": t,
                        "il_ships": a,
                        "base_ships": b,
                        "il_planets": c,
                        "base_planets": d,
                    }
                    for t, a, b, c, d in rows
                ],
            },
            indent=2,
        )
    )
    print(f"\nsaved summary: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", type=str, required=True, choices=["win", "loss"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    run_one(args.seed, args.label, args.data_root)


if __name__ == "__main__":
    main()
