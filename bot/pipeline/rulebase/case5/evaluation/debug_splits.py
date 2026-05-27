"""Measure fleet-split frequency for case5 (and case4 for comparison).

Orbit Wars の fleet 速度は艦数に依存する (艦数多→速い) ため、同一
ターゲットへ分割送信すると速度が落ちて敵に先着を譲ってしまう。
このスクリプトは自己対戦を N エピソード回し、ターンごとに以下 2 指標の
発生件数を CSV で出力する:

- same-src-angle: 同一 src から同一方向へ複数アクション (= 同一 target 分割)
- cross-src-same-angle: 複数 src から同一方向へ (swarm 合流、許容される分割)

Phase 0 計測の目的は Phase 1/2 の効果を定量化すること。
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Fleet split frequency measurement.")


@dataclass
class TurnStats:
    turn: int
    seat: int
    same_src_angle: int
    cross_src_same_angle: int
    total_actions: int


def _count_splits(action: list[list[int]]) -> tuple[int, int]:
    """Return (same_src_angle_dups, cross_src_same_angle_dups) for a turn's actions."""
    by_src_angle: Counter[tuple[int, float]] = Counter()
    by_angle: Counter[float] = Counter()
    for move in action:
        if not move:
            continue
        src_id, angle, _ships = move
        key_angle = round(float(angle), 2)
        by_src_angle[(int(src_id), key_angle)] += 1
        by_angle[key_angle] += 1

    same_src_dups = sum(c - 1 for c in by_src_angle.values() if c > 1)
    cross_src_dups = (
        sum(c - 1 for angle, c in by_angle.items() if c > 1) - same_src_dups
    )
    return same_src_dups, cross_src_dups


def _load_agent(case: str):  # type: ignore[no-untyped-def]
    if case == "case5":
        from pipeline.rulebase.case5.baseline import agent as ag5

        return ag5
    if case == "case4":
        from pipeline.rulebase.case4.baseline import agent as ag4

        return ag4
    raise ValueError(f"unknown case: {case}")


def _run_episode(case: str, seed: int) -> list[TurnStats]:
    from orbit_wars_sim import make_orbit_wars_env

    ag = _load_agent(case)
    env = make_orbit_wars_env(agents=2, seed=seed)
    from orbit_wars_sim import run_orbit_wars_episode

    run_orbit_wars_episode(env, [ag, ag])

    rows: list[TurnStats] = []
    for turn_idx, step in enumerate(env.steps):
        for seat_idx, seat_state in enumerate(step):
            action = seat_state.get("action") or []
            if not action:
                continue
            same, cross = _count_splits(action)
            rows.append(
                TurnStats(
                    turn=turn_idx,
                    seat=seat_idx,
                    same_src_angle=same,
                    cross_src_same_angle=cross,
                    total_actions=len(action),
                )
            )
    return rows


@app.command()
def run(
    case: str = typer.Option("case5", "--case", help="case5 or case4"),
    episodes: int = typer.Option(5, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    out: Path = typer.Option(
        Path("data/debug_splits.csv"), "--out", help="CSV output path"
    ),
) -> None:
    """Run N episodes, write per-turn split counts to CSV, print summary."""
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[tuple[int, TurnStats]] = []
    for ep in range(episodes):
        rows = _run_episode(case, seed + ep)
        all_rows.extend((ep, r) for r in rows)
        typer.echo(
            f"  ep={ep} turns={len(rows)} "
            f"same_src_splits={sum(r.same_src_angle for r in rows)} "
            f"cross_src_swarm={sum(r.cross_src_same_angle for r in rows)}"
        )

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "episode",
                "turn",
                "seat",
                "same_src_angle",
                "cross_src_same_angle",
                "total_actions",
            ]
        )
        for ep, r in all_rows:
            writer.writerow(
                [
                    ep,
                    r.turn,
                    r.seat,
                    r.same_src_angle,
                    r.cross_src_same_angle,
                    r.total_actions,
                ]
            )

    total_same = sum(r.same_src_angle for _, r in all_rows)
    total_cross = sum(r.cross_src_same_angle for _, r in all_rows)
    typer.echo("\n=== Summary ===")
    typer.echo(f"  case            : {case}")
    typer.echo(f"  episodes        : {episodes}")
    typer.echo(
        f"  same-src splits : {total_same} (per ep: {total_same / episodes:.1f})"
    )
    typer.echo(
        f"  cross-src swarm : {total_cross} (per ep: {total_cross / episodes:.1f})"
    )
    typer.echo(f"  csv             : {out}")


if __name__ == "__main__":
    app()
