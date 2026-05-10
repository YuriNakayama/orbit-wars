"""Lightweight head-to-head evaluation: baseline_v6 (case6) vs baseline_v4.

Mirrors case4/evaluation/compare_v2.py. Runs N episodes in each seat order and
reports win rate plus fleet-size and episode-length stats so STAY-induced
stalemates show up directly in `avg episode len`.

STAY ablation flags ``--no-defense`` / ``--no-burst`` mutate
``pipeline.rulebase.case6.baseline.core.config`` for the duration of the run
so the same harness can produce the full / defense-only / burst-only
comparison numbers without editing config.py by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    v6_launch_sizes: list[int]
    v4_launch_sizes: list[int]
    total_turns: int


def _run_episode(v6_seat: int, seed: int) -> EpisodeStats:
    from env.orbit_wars import make_orbit_wars_env
    from pipeline.rulebase.case4.baseline import agent as agent_v4
    from pipeline.rulebase.case6.baseline import agent as agent_v6

    agents = [agent_v6, agent_v4] if v6_seat == 0 else [agent_v4, agent_v6]
    env = make_orbit_wars_env(agents=2, seed=seed)
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
        winner = 6 if seat_winner == v6_seat else 4
    v6_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == v6_seat
    ]
    v4_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == 1 - v6_seat
    ]
    return EpisodeStats(
        winner=winner,
        v6_launch_sizes=v6_sizes,
        v4_launch_sizes=v4_sizes,
        total_turns=turn,
    )


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    no_defense: bool = typer.Option(
        False, "--no-defense", help="Disable STAY_DEFENSE_ENABLED for ablation."
    ),
    no_burst: bool = typer.Option(
        False, "--no-burst", help="Disable STAY_BURST_ENABLED for ablation."
    ),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    from pipeline.rulebase.case6.baseline.core import config as cfg

    if no_defense:
        cfg.STAY_DEFENSE_ENABLED = False
    if no_burst:
        cfg.STAY_BURST_ENABLED = False
    typer.echo(
        f"STAY config: ENABLED={cfg.STAY_ENABLED} "
        f"DEFENSE={cfg.STAY_DEFENSE_ENABLED} BURST={cfg.STAY_BURST_ENABLED}"
    )

    v6_wins = 0
    v4_wins = 0
    draws = 0
    seat_v6_wins: dict[int, int] = {0: 0, 1: 0}
    seat_v4_wins: dict[int, int] = {0: 0, 1: 0}
    seat_draws: dict[int, int] = {0: 0, 1: 0}
    all_v6_sizes: list[int] = []
    all_v4_sizes: list[int] = []
    total_turns_total = 0
    total_eps = 0

    for v6_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            s = _run_episode(v6_seat=v6_seat, seed=seed + ep_idx + v6_seat * 10_000)
            if s.winner == 6:
                v6_wins += 1
                seat_v6_wins[v6_seat] += 1
            elif s.winner == 4:
                v4_wins += 1
                seat_v4_wins[v6_seat] += 1
            else:
                draws += 1
                seat_draws[v6_seat] += 1
            all_v6_sizes.extend(s.v6_launch_sizes)
            all_v4_sizes.extend(s.v4_launch_sizes)
            total_turns_total += s.total_turns
            total_eps += 1
            if (ep_idx + 1) % 10 == 0:
                typer.echo(
                    f"  seat={v6_seat} {ep_idx + 1}/{episodes_per_side} "
                    f"v6={v6_wins} v4={v4_wins} draws={draws}"
                )

    n = v6_wins + v4_wins + draws
    v6_avg = sum(all_v6_sizes) / max(1, len(all_v6_sizes))
    v4_avg = sum(all_v4_sizes) / max(1, len(all_v4_sizes))
    v6_launches = len(all_v6_sizes) / max(1, total_eps)
    v4_launches = len(all_v4_sizes) / max(1, total_eps)

    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v6 wins        : {v6_wins} ({100 * v6_wins / n:.1f}%)"
        f"  v4 wins: {v4_wins} ({100 * v4_wins / n:.1f}%)"
        f"  draws: {draws}"
    )
    for seat in (0, 1):
        seat_n = seat_v6_wins[seat] + seat_v4_wins[seat] + seat_draws[seat]
        if seat_n == 0:
            continue
        typer.echo(
            f"  seat={seat:>1} ({seat_n} eps): "
            f"v6={seat_v6_wins[seat]} ({100 * seat_v6_wins[seat] / seat_n:.1f}%) "
            f"v4={seat_v4_wins[seat]} draws={seat_draws[seat]}"
        )
    typer.echo(
        f"  avg fleet peak : v6={v6_avg:.1f}  v4={v4_avg:.1f}  "
        f"ratio={v6_avg / max(1.0, v4_avg):.2f}"
    )
    typer.echo(
        f"  launches/ep    : v6={v6_launches:.1f}  v4={v4_launches:.1f}  "
        f"ratio={v6_launches / max(0.01, v4_launches):.2f}"
    )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
