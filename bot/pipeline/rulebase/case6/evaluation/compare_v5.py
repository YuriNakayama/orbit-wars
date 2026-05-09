"""Lightweight head-to-head evaluation: baseline_v6 (case6) vs baseline_v5.

Mirrors compare_v4.py but targets baseline_v5 (case5, LB1224 port) — the
"pre-rewrite" agent that case6 directly derives from. Used for the STAY
ablation (full / defense-only / burst-only) against the immediate ancestor
so the delta is purely the STAY judge.

STAY ablation flags ``--no-defense`` / ``--no-burst`` mutate
``pipeline.rulebase.case6.baseline.core.config`` for the duration of the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    v6_launch_sizes: list[int]
    v5_launch_sizes: list[int]
    total_turns: int


def _run_episode(v6_seat: int, seed: int) -> EpisodeStats:
    from env.orbit_wars import make_orbit_wars_env
    from pipeline.rulebase.case5.baseline import agent as agent_v5
    from pipeline.rulebase.case6.baseline import agent as agent_v6

    agents = [agent_v6, agent_v5] if v6_seat == 0 else [agent_v5, agent_v6]
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
        winner = 6 if seat_winner == v6_seat else 5
    v6_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == v6_seat
    ]
    v5_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == 1 - v6_seat
    ]
    return EpisodeStats(
        winner=winner,
        v6_launch_sizes=v6_sizes,
        v5_launch_sizes=v5_sizes,
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
    max_hold_turns: int | None = typer.Option(
        None,
        "--max-hold-turns",
        help="Override STAY_BURST_MAX_HOLD_TURNS for ablation (default: config value).",
    ),
    max_target_turns: int | None = typer.Option(
        None,
        "--max-target-turns",
        help="Override STAY_BURST_MAX_TARGET_TURNS for ablation (default: config value).",
    ),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    from pipeline.rulebase.case6.baseline.core import config as cfg

    if no_defense:
        cfg.STAY_DEFENSE_ENABLED = False
    if no_burst:
        cfg.STAY_BURST_ENABLED = False
    if max_hold_turns is not None:
        cfg.STAY_BURST_MAX_HOLD_TURNS = max_hold_turns
    if max_target_turns is not None:
        cfg.STAY_BURST_MAX_TARGET_TURNS = max_target_turns
    typer.echo(
        f"STAY config: ENABLED={cfg.STAY_ENABLED} "
        f"DEFENSE={cfg.STAY_DEFENSE_ENABLED} BURST={cfg.STAY_BURST_ENABLED} "
        f"MAX_HOLD={cfg.STAY_BURST_MAX_HOLD_TURNS} "
        f"MAX_TGT={cfg.STAY_BURST_MAX_TARGET_TURNS}"
    )

    v6_wins = 0
    v5_wins = 0
    draws = 0
    seat_v6_wins: dict[int, int] = {0: 0, 1: 0}
    seat_v5_wins: dict[int, int] = {0: 0, 1: 0}
    seat_draws: dict[int, int] = {0: 0, 1: 0}
    all_v6_sizes: list[int] = []
    all_v5_sizes: list[int] = []
    total_turns_total = 0
    total_eps = 0

    for v6_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            s = _run_episode(v6_seat=v6_seat, seed=seed + ep_idx + v6_seat * 10_000)
            if s.winner == 6:
                v6_wins += 1
                seat_v6_wins[v6_seat] += 1
            elif s.winner == 5:
                v5_wins += 1
                seat_v5_wins[v6_seat] += 1
            else:
                draws += 1
                seat_draws[v6_seat] += 1
            all_v6_sizes.extend(s.v6_launch_sizes)
            all_v5_sizes.extend(s.v5_launch_sizes)
            total_turns_total += s.total_turns
            total_eps += 1
            if (ep_idx + 1) % 10 == 0:
                typer.echo(
                    f"  seat={v6_seat} {ep_idx + 1}/{episodes_per_side} "
                    f"v6={v6_wins} v5={v5_wins} draws={draws}"
                )

    n = v6_wins + v5_wins + draws
    v6_avg = sum(all_v6_sizes) / max(1, len(all_v6_sizes))
    v5_avg = sum(all_v5_sizes) / max(1, len(all_v5_sizes))
    v6_launches = len(all_v6_sizes) / max(1, total_eps)
    v5_launches = len(all_v5_sizes) / max(1, total_eps)

    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v6 wins        : {v6_wins} ({100 * v6_wins / n:.1f}%)"
        f"  v5 wins: {v5_wins} ({100 * v5_wins / n:.1f}%)"
        f"  draws: {draws}"
    )
    for seat in (0, 1):
        seat_n = seat_v6_wins[seat] + seat_v5_wins[seat] + seat_draws[seat]
        if seat_n == 0:
            continue
        typer.echo(
            f"  seat={seat:>1} ({seat_n} eps): "
            f"v6={seat_v6_wins[seat]} ({100 * seat_v6_wins[seat] / seat_n:.1f}%) "
            f"v5={seat_v5_wins[seat]} draws={seat_draws[seat]}"
        )
    typer.echo(
        f"  avg fleet peak : v6={v6_avg:.1f}  v5={v5_avg:.1f}  "
        f"ratio={v6_avg / max(1.0, v5_avg):.2f}"
    )
    typer.echo(
        f"  launches/ep    : v6={v6_launches:.1f}  v5={v5_launches:.1f}  "
        f"ratio={v6_launches / max(0.01, v5_launches):.2f}"
    )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
