"""Head-to-head evaluation: baseline_v5 vs baseline_v1.

Mirrors compare_v4 but pits case5 against case1 (baseline confirmation).
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    total_turns: int


def _run_episode(v5_seat: int, seed: int) -> EpisodeStats:
    from env.orbit_wars import make_orbit_wars_env
    from pipeline.rulebase.case1.baseline import agent as agent_v1
    from pipeline.rulebase.case5.baseline import agent as agent_v5

    agents = [agent_v5, agent_v1] if v5_seat == 0 else [agent_v1, agent_v5]
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
        winner = 5 if seat_winner == v5_seat else 1
    return EpisodeStats(winner=winner, total_turns=turn)


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    v5_wins = 0
    v1_wins = 0
    draws = 0
    total_turns_total = 0
    total_eps = 0

    for v5_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            s = _run_episode(v5_seat=v5_seat, seed=seed + ep_idx + v5_seat * 10_000)
            if s.winner == 5:
                v5_wins += 1
            elif s.winner == 1:
                v1_wins += 1
            else:
                draws += 1
            total_turns_total += s.total_turns
            total_eps += 1
            if (ep_idx + 1) % 10 == 0:
                typer.echo(
                    f"  seat={v5_seat} {ep_idx + 1}/{episodes_per_side} "
                    f"v5={v5_wins} v1={v1_wins} draws={draws}"
                )

    n = v5_wins + v1_wins + draws
    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v5 wins        : {v5_wins} ({100 * v5_wins / n:.1f}%)"
        f"  v1 wins: {v1_wins} ({100 * v1_wins / n:.1f}%)"
        f"  draws: {draws}"
    )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
