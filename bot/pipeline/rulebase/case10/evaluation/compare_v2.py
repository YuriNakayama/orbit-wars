"""Lightweight head-to-head evaluation: baseline_v4 vs baseline_v2.

Runs N episodes in each seat order and reports win rate plus fleet-size stats.
Designed for quick feedback loops; does not persist replays or match indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    v4_launch_sizes: list[int]
    v2_launch_sizes: list[int]
    total_turns: int


def _run_episode(v4_seat: int, seed: int) -> EpisodeStats:
    import kaggle_environments

    from pipeline.rulebase.case2.baseline import agent as agent_v2
    from pipeline.rulebase.case10.baseline import agent as agent_v4

    agents = [agent_v4, agent_v2] if v4_seat == 0 else [agent_v2, agent_v4]
    env = kaggle_environments.make(
        "orbit_wars", configuration={"agents": 2, "seed": seed}
    )
    peaks_by_owner_fleet: dict[tuple[int, int], int] = {}
    turn = 0
    while not env.done:
        obs_by_player = [step_info["observation"] for step_info in env.steps[-1]]
        actions = [agents[i](obs_by_player[i]) for i in (0, 1)]
        env.step(actions)
        turn += 1
        fleets = env.steps[-1][0]["observation"].get("fleets", [])
        for fleet in fleets:
            fid, owner, _x, _y, _angle, _from_pid, ships = fleet
            key = (int(owner), int(fid))
            ships_int = int(ships)
            if peaks_by_owner_fleet.get(key, 0) < ships_int:
                peaks_by_owner_fleet[key] = ships_int
    rewards = [s.get("reward", 0) or 0 for s in env.steps[-1]]
    winner: int | None = None
    if rewards[0] != rewards[1]:
        seat_winner = 0 if rewards[0] > rewards[1] else 1
        winner = 4 if seat_winner == v4_seat else 2
    v4_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == v4_seat
    ]
    v2_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == 1 - v4_seat
    ]
    return EpisodeStats(
        winner=winner,
        v4_launch_sizes=v4_sizes,
        v2_launch_sizes=v2_sizes,
        total_turns=turn,
    )


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    v4_wins = 0
    v2_wins = 0
    draws = 0
    all_v4_sizes: list[int] = []
    all_v2_sizes: list[int] = []
    total_turns_total = 0
    total_eps = 0

    for v4_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            s = _run_episode(v4_seat=v4_seat, seed=seed + ep_idx + v4_seat * 10_000)
            if s.winner == 4:
                v4_wins += 1
            elif s.winner == 2:
                v2_wins += 1
            else:
                draws += 1
            all_v4_sizes.extend(s.v4_launch_sizes)
            all_v2_sizes.extend(s.v2_launch_sizes)
            total_turns_total += s.total_turns
            total_eps += 1
            if (ep_idx + 1) % 10 == 0:
                typer.echo(
                    f"  seat={v4_seat} {ep_idx + 1}/{episodes_per_side} "
                    f"v4={v4_wins} v2={v2_wins} draws={draws}"
                )

    n = v4_wins + v2_wins + draws
    v4_avg = sum(all_v4_sizes) / max(1, len(all_v4_sizes))
    v2_avg = sum(all_v2_sizes) / max(1, len(all_v2_sizes))
    v4_launches = len(all_v4_sizes) / max(1, total_eps)
    v2_launches = len(all_v2_sizes) / max(1, total_eps)

    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v4 wins        : {v4_wins} ({100 * v4_wins / n:.1f}%)"
        f"  v2 wins: {v2_wins} ({100 * v2_wins / n:.1f}%)"
        f"  draws: {draws}"
    )
    typer.echo(
        f"  avg fleet peak : v4={v4_avg:.1f}  v2={v2_avg:.1f}  "
        f"ratio={v4_avg / max(1.0, v2_avg):.2f}"
    )
    typer.echo(
        f"  launches/ep    : v4={v4_launches:.1f}  v2={v2_launches:.1f}  "
        f"ratio={v4_launches / max(0.01, v2_launches):.2f}"
    )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
