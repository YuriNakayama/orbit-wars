"""Diagnose planet ping-pong incidents in self-play matches.

A ping-pong incident is defined as: the same unordered planet pair {a, b} has
both (a -> b) and (b -> a) launches within the last PING_PONG_WINDOW turns.
The script also tracks per-pair launch counts (any direction).

Used to quantify the ping-pong rate of a candidate agent vs a baseline. Output:

    data/output/diagnostics/ping_pong/{run_id}/summary.json
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, invoke_without_command=True)

PING_PONG_WINDOW = 5


@dataclass
class EpisodeStats:
    """Plain-dict-friendly per-seat stats; lists/tuples picklable across processes."""

    incidents: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    total_turns: int = 0
    total_launches: int = 0


def _run_one(
    args: tuple[int, int, bool],
) -> tuple[int, EpisodeStats, EpisodeStats]:
    """Worker entry: (v9_seat, seed, no_anti) -> (v9_seat, v9_stats, v4_stats)."""
    from orbit_wars_sim import make_orbit_wars_env

    from pipeline.rulebase.case4.baseline import agent as agent_v4
    from pipeline.rulebase.case9.baseline import agent as agent_v9
    from pipeline.rulebase.case9.baseline.core import config as cfg
    from pipeline.rulebase.case9.baseline.core.types import Planet
    from pipeline.rulebase.case9.baseline.lookahead import _infer_action_target

    v9_seat, seed, no_anti = args
    if no_anti:
        cfg.ANTI_PING_PONG_ENABLED = False

    env = make_orbit_wars_env(agents=2, seed=seed)
    agents = (agent_v9, agent_v4) if v9_seat == 0 else (agent_v4, agent_v9)
    seat_history: list[list[tuple[int, int, int]]] = [[], []]
    seat_stats: list[EpisodeStats] = [EpisodeStats(), EpisodeStats()]

    turn = 0
    while not env.done:
        obs_by_seat = [step_info["observation"] for step_info in env.steps[-1]]
        actions_by_seat = [agents[i](obs_by_seat[i]) for i in (0, 1)]
        env.step(actions_by_seat)
        turn += 1

        for seat in (0, 1):
            planets = [Planet(*p) for p in obs_by_seat[seat].get("planets", [])]
            by_id = {p.id: p for p in planets}
            decoded: list[tuple[int, int]] = []
            for action in actions_by_seat[seat]:
                if not action:
                    continue
                src_id = int(action[0])
                angle = float(action[1])
                ships = int(action[2])
                src = by_id.get(src_id)
                if src is None or ships <= 0:
                    continue
                target = _infer_action_target(src, angle, planets, ships)
                if target is None:
                    continue
                decoded.append((src_id, int(target.id)))

            stats = seat_stats[seat]
            history = seat_history[seat]
            for src_id, dst_id in decoded:
                stats.total_launches += 1
                for past_turn, past_src, past_dst in reversed(history):
                    if turn - past_turn > PING_PONG_WINDOW:
                        break
                    if past_src == dst_id and past_dst == src_id:
                        a, b = min(src_id, dst_id), max(src_id, dst_id)
                        stats.incidents.append((turn, seat, a, b, past_turn))
                        break
            history.extend((turn, s, d) for (s, d) in decoded)
            stats.total_turns = turn

    if v9_seat == 0:
        return v9_seat, seat_stats[0], seat_stats[1]
    return v9_seat, seat_stats[1], seat_stats[0]


def _summarize(stats_list: list[EpisodeStats], label: str) -> dict[str, Any]:
    total_eps = len(stats_list)
    total_incidents = sum(len(s.incidents) for s in stats_list)
    total_launches = sum(s.total_launches for s in stats_list)
    total_turns = sum(s.total_turns for s in stats_list)
    pair_top: dict[tuple[int, int], int] = defaultdict(int)
    for stats in stats_list:
        for _turn, _seat, a, b, _past_turn in stats.incidents:
            pair_top[(a, b)] += 1
    top_pairs = sorted(pair_top.items(), key=lambda kv: -kv[1])[:10]
    return {
        "label": label,
        "episodes": total_eps,
        "total_turns": total_turns,
        "total_launches": total_launches,
        "ping_pong_incidents": total_incidents,
        "incidents_per_episode": (
            total_incidents / total_eps if total_eps > 0 else 0.0
        ),
        "incidents_per_100_launches": (
            100.0 * total_incidents / total_launches if total_launches else 0.0
        ),
        "top_pairs": [
            {"pair": list(pair), "count": count} for pair, count in top_pairs
        ],
    }


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    parallel: int = typer.Option(4, "-p", help="Worker processes."),
    no_anti: bool = typer.Option(
        False,
        "--no-anti",
        help="Disable ANTI_PING_PONG_ENABLED for the duration of the run.",
    ),
    out_dir: Path = typer.Option(
        Path("data/output/diagnostics/ping_pong"),
        "--out-dir",
        help="Base output directory for diagnostics summary JSON.",
    ),
) -> None:
    """Run baseline_v9 vs baseline_v4 episodes and report ping-pong incidents."""
    typer.echo(f"ANTI_PING_PONG_ENABLED={not no_anti}  parallel={parallel}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    target_dir = out_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, int, bool]] = []
    for v9_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            jobs.append((v9_seat, seed + ep_idx + v9_seat * 10_000, no_anti))

    v9_stats: list[EpisodeStats] = []
    v4_stats: list[EpisodeStats] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, parallel)) as pool:
        for idx, (_v9_seat, v9_s, v4_s) in enumerate(
            pool.imap_unordered(_run_one, jobs)
        ):
            v9_stats.append(v9_s)
            v4_stats.append(v4_s)
            if (idx + 1) % 20 == 0:
                typer.echo(
                    f"  {idx + 1}/{len(jobs)} "
                    f"v9_inc={sum(len(s.incidents) for s in v9_stats)} "
                    f"v4_inc={sum(len(s.incidents) for s in v4_stats)}"
                )

    v9_summary = _summarize(v9_stats, "baseline_v9")
    v4_summary = _summarize(v4_stats, "baseline_v4")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "anti_enabled": not no_anti,
        "ping_pong_window": PING_PONG_WINDOW,
        "v9": v9_summary,
        "v4": v4_summary,
    }
    out_path = target_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"\nwrote: {out_path}")
    typer.echo(
        f"v9 incidents/ep = {v9_summary['incidents_per_episode']:.2f}, "
        f"v4 incidents/ep = {v4_summary['incidents_per_episode']:.2f}"
    )


if __name__ == "__main__":
    app()
