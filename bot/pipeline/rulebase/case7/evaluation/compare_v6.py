"""Lightweight head-to-head evaluation: baseline_v7 (case7) vs baseline_v6.

case7 = case6 + multi-turn ACCUMULATE mission. compare_v6 measures the delta
purely from the new ACCUMULATE_* config knobs by running case7 against case6.

ACCUMULATE ablation flags:
  --no-accumulate     ACCUMULATE_ENABLED = False (case7 ≡ case6)
  --knee-ships        Override ACCUMULATE_KNEE_SHIPS
  --safety            Override ACCUMULATE_SAFETY_SHIPS
  --max-hold-turns    Override ACCUMULATE_MAX_HOLD_TURNS
  --min-target-turns  Override ACCUMULATE_MIN_TARGET_TURNS
  --max-target-turns  Override ACCUMULATE_MAX_TARGET_TURNS
  --threat-reserve-max  Override ACCUMULATE_THREAT_RESERVE_MAX

Mutates ``pipeline.rulebase.case7.baseline.core.config`` for the duration of the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

app = typer.Typer(add_completion=False, invoke_without_command=True)


@dataclass
class EpisodeStats:
    winner: int | None
    v7_launch_sizes: list[int]
    v6_launch_sizes: list[int]
    total_turns: int


def _run_episode(v7_seat: int, seed: int) -> EpisodeStats:
    from orbit_wars_sim import make_orbit_wars_env

    from pipeline.rulebase.case6.baseline import agent as agent_v6
    from pipeline.rulebase.case7.baseline import agent as agent_v7

    agents = [agent_v7, agent_v6] if v7_seat == 0 else [agent_v6, agent_v7]
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
        winner = 7 if seat_winner == v7_seat else 6
    v7_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == v7_seat
    ]
    v6_sizes = [
        ships
        for (owner, _fid), ships in peaks_by_owner_fleet.items()
        if owner == 1 - v7_seat
    ]
    return EpisodeStats(
        winner=winner,
        v7_launch_sizes=v7_sizes,
        v6_launch_sizes=v6_sizes,
        total_turns=turn,
    )


@app.command()
def run(
    episodes_per_side: int = typer.Option(50, "-n"),
    seed: int = typer.Option(1000, "--seed"),
    no_accumulate: bool = typer.Option(
        False, "--no-accumulate", help="Disable ACCUMULATE_ENABLED for ablation."
    ),
    knee_ships: int | None = typer.Option(
        None, "--knee-ships", help="Override ACCUMULATE_KNEE_SHIPS."
    ),
    safety: int | None = typer.Option(
        None, "--safety", help="Override ACCUMULATE_SAFETY_SHIPS."
    ),
    max_hold_turns: int | None = typer.Option(
        None, "--max-hold-turns", help="Override ACCUMULATE_MAX_HOLD_TURNS."
    ),
    min_target_turns: int | None = typer.Option(
        None, "--min-target-turns", help="Override ACCUMULATE_MIN_TARGET_TURNS."
    ),
    max_target_turns: int | None = typer.Option(
        None, "--max-target-turns", help="Override ACCUMULATE_MAX_TARGET_TURNS."
    ),
    threat_reserve_max: int | None = typer.Option(
        None, "--threat-reserve-max", help="Override ACCUMULATE_THREAT_RESERVE_MAX."
    ),
) -> None:
    """Run episodes_per_side games in each seat order and report stats."""
    from pipeline.rulebase.case7.baseline.core import config as cfg

    if no_accumulate:
        cfg.ACCUMULATE_ENABLED = False
    if knee_ships is not None:
        cfg.ACCUMULATE_KNEE_SHIPS = knee_ships
    if safety is not None:
        cfg.ACCUMULATE_SAFETY_SHIPS = safety
    if max_hold_turns is not None:
        cfg.ACCUMULATE_MAX_HOLD_TURNS = max_hold_turns
    if min_target_turns is not None:
        cfg.ACCUMULATE_MIN_TARGET_TURNS = min_target_turns
    if max_target_turns is not None:
        cfg.ACCUMULATE_MAX_TARGET_TURNS = max_target_turns
    if threat_reserve_max is not None:
        cfg.ACCUMULATE_THREAT_RESERVE_MAX = threat_reserve_max

    typer.echo(
        f"ACCUMULATE config: ENABLED={cfg.ACCUMULATE_ENABLED} "
        f"KNEE={cfg.ACCUMULATE_KNEE_SHIPS} SAFETY={cfg.ACCUMULATE_SAFETY_SHIPS} "
        f"MAX_HOLD={cfg.ACCUMULATE_MAX_HOLD_TURNS} "
        f"MIN_TGT={cfg.ACCUMULATE_MIN_TARGET_TURNS} "
        f"MAX_TGT={cfg.ACCUMULATE_MAX_TARGET_TURNS} "
        f"THREAT_RES={cfg.ACCUMULATE_THREAT_RESERVE_MAX}"
    )

    v7_wins = 0
    v6_wins = 0
    draws = 0
    seat_v7_wins: dict[int, int] = {0: 0, 1: 0}
    seat_v6_wins: dict[int, int] = {0: 0, 1: 0}
    seat_draws: dict[int, int] = {0: 0, 1: 0}
    all_v7_sizes: list[int] = []
    all_v6_sizes: list[int] = []
    total_turns_total = 0
    total_eps = 0

    for v7_seat in (0, 1):
        for ep_idx in range(episodes_per_side):
            s = _run_episode(v7_seat=v7_seat, seed=seed + ep_idx + v7_seat * 10_000)
            if s.winner == 7:
                v7_wins += 1
                seat_v7_wins[v7_seat] += 1
            elif s.winner == 6:
                v6_wins += 1
                seat_v6_wins[v7_seat] += 1
            else:
                draws += 1
                seat_draws[v7_seat] += 1
            all_v7_sizes.extend(s.v7_launch_sizes)
            all_v6_sizes.extend(s.v6_launch_sizes)
            total_turns_total += s.total_turns
            total_eps += 1
            if (ep_idx + 1) % 10 == 0:
                typer.echo(
                    f"  seat={v7_seat} {ep_idx + 1}/{episodes_per_side} "
                    f"v7={v7_wins} v6={v6_wins} draws={draws}"
                )

    n = v7_wins + v6_wins + draws
    v7_avg = sum(all_v7_sizes) / max(1, len(all_v7_sizes))
    v6_avg = sum(all_v6_sizes) / max(1, len(all_v6_sizes))
    v7_launches = len(all_v7_sizes) / max(1, total_eps)
    v6_launches = len(all_v6_sizes) / max(1, total_eps)

    typer.echo("\n=== Summary ===")
    typer.echo(f"  episodes       : {n}")
    typer.echo(
        f"  v7 wins        : {v7_wins} ({100 * v7_wins / n:.1f}%)"
        f"  v6 wins: {v6_wins} ({100 * v6_wins / n:.1f}%)"
        f"  draws: {draws}"
    )
    for seat in (0, 1):
        seat_n = seat_v7_wins[seat] + seat_v6_wins[seat] + seat_draws[seat]
        if seat_n == 0:
            continue
        typer.echo(
            f"  seat={seat:>1} ({seat_n} eps): "
            f"v7={seat_v7_wins[seat]} ({100 * seat_v7_wins[seat] / seat_n:.1f}%) "
            f"v6={seat_v6_wins[seat]} draws={seat_draws[seat]}"
        )
    typer.echo(
        f"  avg fleet peak : v7={v7_avg:.1f}  v6={v6_avg:.1f}  "
        f"ratio={v7_avg / max(1.0, v6_avg):.2f}"
    )
    typer.echo(
        f"  launches/ep    : v7={v7_launches:.1f}  v6={v6_launches:.1f}  "
        f"ratio={v7_launches / max(0.01, v6_launches):.2f}"
    )
    typer.echo(f"  avg episode len: {total_turns_total / max(1, total_eps):.1f} turns")


if __name__ == "__main__":
    app()
