"""Lightweight head-to-head: baseline_v10 (case10) vs baseline_v4 (case4 / production).

Mirrors case9/evaluation/compare_v4.py. Runs N episodes in each seat order
with mp Pool and reports win rate. -p enables multiprocessing.
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


def _run_one(args: tuple[int, int]) -> tuple[int, int, int | None, int]:
    """Worker entry: (v10_seat, seed) -> (v10_seat, seed, winner, turns)."""
    import kaggle_environments

    from pipeline.rulebase.case4.baseline import agent as agent_v4
    from pipeline.rulebase.case10.baseline import agent as agent_v10

    v10_seat, seed = args

    agents = [agent_v10, agent_v4] if v10_seat == 0 else [agent_v4, agent_v10]
    env = kaggle_environments.make(
        "orbit_wars", configuration={"agents": 2, "seed": seed}
    )
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
        winner = 10 if seat_winner == v10_seat else 4
    return v10_seat, seed, winner, turn


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    parallel: int = typer.Option(4, "-p", help="Worker processes."),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    typer.echo(f"case10 vs baseline_v4  parallel={parallel}")

    jobs: list[tuple[int, int]] = []
    for v10_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            jobs.append((v10_seat, seed + ep_idx + v10_seat * 10_000))

    v10_wins = 0
    v4_wins = 0
    draws = 0
    seat_v10_wins: dict[int, int] = {0: 0, 1: 0}
    seat_v4_wins: dict[int, int] = {0: 0, 1: 0}
    seat_draws: dict[int, int] = {0: 0, 1: 0}
    total_turns_total = 0
    total_eps = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, parallel)) as pool:
        for idx, (v10_seat, _ep_seed, winner, turns) in enumerate(
            pool.imap_unordered(_run_one, jobs)
        ):
            if winner == 10:
                v10_wins += 1
                seat_v10_wins[v10_seat] += 1
            elif winner == 4:
                v4_wins += 1
                seat_v4_wins[v10_seat] += 1
            else:
                draws += 1
                seat_draws[v10_seat] += 1
            total_turns_total += turns
            total_eps += 1
            if (idx + 1) % 20 == 0:
                typer.echo(
                    f"  {idx + 1}/{len(jobs)} v10={v10_wins} v4={v4_wins} "
                    f"draws={draws}"
                )

    n = v10_wins + v4_wins + draws
    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v10 wins        : {v10_wins} ({100 * v10_wins / max(1, n):.1f}%)"
        f"  v4 wins: {v4_wins} ({100 * v4_wins / max(1, n):.1f}%)"
        f"  draws: {draws}"
    )
    for seat in (0, 1):
        seat_n = seat_v10_wins[seat] + seat_v4_wins[seat] + seat_draws[seat]
        if seat_n == 0:
            continue
        typer.echo(
            f"  seat={seat:>1} ({seat_n} eps): "
            f"v10={seat_v10_wins[seat]} ({100 * seat_v10_wins[seat] / seat_n:.1f}%) "
            f"v4={seat_v4_wins[seat]} draws={seat_draws[seat]}"
        )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
