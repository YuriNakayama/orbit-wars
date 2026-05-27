"""Lightweight head-to-head: baseline_v9 (case9) vs baseline_v4 (case4 / production).

Mirrors case6/evaluation/compare_v4.py. Runs N episodes in each seat order
and reports win rate. Use --no-anti to ablate ANTI_PING_PONG_ENABLED for
side-by-side comparison. -p enables multiprocessing.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    total_turns: int


def _run_one(args: tuple[int, int, bool]) -> tuple[int, int, int | None, int]:
    """Worker entry: (v9_seat, seed, no_anti) -> (v9_seat, seed, winner, turns)."""
    from orbit_wars_sim import make_orbit_wars_env

    from pipeline.rulebase.case4.baseline import agent as agent_v4
    from pipeline.rulebase.case9.baseline import agent as agent_v9
    from pipeline.rulebase.case9.baseline.core import config as cfg

    v9_seat, seed, no_anti = args
    if no_anti:
        cfg.ANTI_PING_PONG_ENABLED = False

    agents = [agent_v9, agent_v4] if v9_seat == 0 else [agent_v4, agent_v9]
    env = make_orbit_wars_env(agents=2, seed=seed)
    turn = 0
    while not env.done:
        obs_by_player = [step_info["observation"] for step_info in env.steps[-1]]
        actions = [agents[i](obs_by_player[i]) for i in (0, 1)]
        env.step(actions)
        turn += 1

    rewards = [s.get("reward", 0) or 0 for s in env.steps[-1]]
    winner: int | None = None
    if rewards[0] != rewards[1]:
        seat_winner = 0 if rewards[0] > rewards[1] else 1
        winner = 9 if seat_winner == v9_seat else 4
    return v9_seat, seed, winner, turn


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    parallel: int = typer.Option(4, "-p", help="Worker processes."),
    no_anti: bool = typer.Option(
        False,
        "--no-anti",
        help="Disable ANTI_PING_PONG_ENABLED for ablation.",
    ),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    typer.echo(f"ANTI config: ENABLED={not no_anti}  parallel={parallel}")

    jobs: list[tuple[int, int, bool]] = []
    for v9_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            jobs.append((v9_seat, seed + ep_idx + v9_seat * 10_000, no_anti))

    v9_wins = 0
    v4_wins = 0
    draws = 0
    seat_v9_wins: dict[int, int] = {0: 0, 1: 0}
    seat_v4_wins: dict[int, int] = {0: 0, 1: 0}
    seat_draws: dict[int, int] = {0: 0, 1: 0}
    total_turns_total = 0
    total_eps = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, parallel)) as pool:
        for idx, (v9_seat, _ep_seed, winner, turns) in enumerate(
            pool.imap_unordered(_run_one, jobs)
        ):
            if winner == 9:
                v9_wins += 1
                seat_v9_wins[v9_seat] += 1
            elif winner == 4:
                v4_wins += 1
                seat_v4_wins[v9_seat] += 1
            else:
                draws += 1
                seat_draws[v9_seat] += 1
            total_turns_total += turns
            total_eps += 1
            if (idx + 1) % 20 == 0:
                typer.echo(
                    f"  {idx + 1}/{len(jobs)} v9={v9_wins} v4={v4_wins} draws={draws}"
                )

    n = v9_wins + v4_wins + draws
    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v9 wins        : {v9_wins} ({100 * v9_wins / max(1, n):.1f}%)"
        f"  v4 wins: {v4_wins} ({100 * v4_wins / max(1, n):.1f}%)"
        f"  draws: {draws}"
    )
    for seat in (0, 1):
        seat_n = seat_v9_wins[seat] + seat_v4_wins[seat] + seat_draws[seat]
        if seat_n == 0:
            continue
        typer.echo(
            f"  seat={seat:>1} ({seat_n} eps): "
            f"v9={seat_v9_wins[seat]} ({100 * seat_v9_wins[seat] / seat_n:.1f}%) "
            f"v4={seat_v4_wins[seat]} draws={seat_draws[seat]}"
        )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
