"""Self-play CLI for the baseline agent.

1v1 と 4P FFA の自己対戦を実行し、リプレイ JSON / タイミング / サマリを保存する。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pipeline.case1.baseline.agent import agent as baseline_agent

app = typer.Typer(add_completion=False, help="Baseline agent self-play runner.")
console = Console()
logger = logging.getLogger("baseline.selfplay")


Mode = str  # "1v1" | "ffa4"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )


def _make_env(mode: Mode, seed: int) -> Any:
    from kaggle_environments import make

    config: dict[str, Any] = {"seed": seed}
    if mode == "1v1":
        config["agents"] = 2
    elif mode == "ffa4":
        config["agents"] = 4
    else:
        raise typer.BadParameter(f"unknown mode: {mode!r} (expected '1v1' or 'ffa4')")
    return make("orbit_wars", configuration=config)


def _agent_with_timing(timings: list[float]) -> Any:
    def _wrapped(obs: Any, *_: Any, **__: Any) -> list[list[int | float]]:
        start = time.perf_counter()
        try:
            return baseline_agent(obs)
        finally:
            timings.append(time.perf_counter() - start)

    return _wrapped


def _episode_winner(final_state: list[dict[str, Any]]) -> int:
    active = [i for i, s in enumerate(final_state) if s.get("status") == "ACTIVE"]
    if len(active) == 1:
        return active[0]
    rewards = [float(s.get("reward") or 0.0) for s in final_state]
    return rewards.index(max(rewards))


def _player_scores(env: Any) -> list[int]:
    steps = env.steps
    if not steps:
        return []
    last = steps[-1]
    scores: list[int] = []
    for agent_state in last:
        obs = agent_state.get("observation") or {}
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        player = obs.get("player", 0)
        ships = sum(int(p[5]) for p in planets if p[1] == player)
        ships += sum(int(f[6]) for f in fleets if f[1] == player)
        scores.append(ships)
    return scores


@app.command()
def run(
    episodes: int = typer.Option(1, "--episodes", "-n", help="Number of episodes."),
    mode: Mode = typer.Option("1v1", "--mode", help="'1v1' or 'ffa4'."),
    seed: int = typer.Option(0, "--seed", help="Base seed; episode i uses seed+i."),
    output_dir: Path = typer.Option(
        Path("data/replays/case1"),
        "--output-dir",
        help="Directory for replay output.",
    ),
    save_replay: bool = typer.Option(
        True, "--save-replay/--no-save-replay", help="Persist full env replay JSON."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run N self-play episodes of the baseline agent against itself."""
    _configure_logging(verbose)

    if mode not in ("1v1", "ffa4"):
        raise typer.BadParameter(f"mode must be '1v1' or 'ffa4', got {mode!r}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing outputs to %s", run_dir)

    num_agents = 2 if mode == "1v1" else 4
    summary_rows: list[dict[str, Any]] = []
    wins = [0] * num_agents
    total_timeouts = 0
    total_turns = 0

    for i in range(episodes):
        episode_seed = seed + i
        env = _make_env(mode, episode_seed)
        timings_per_agent: list[list[float]] = [[] for _ in range(num_agents)]
        agents = [
            _agent_with_timing(timings_per_agent[j]) for j in range(num_agents)
        ]

        t0 = time.perf_counter()
        env.run(agents)
        elapsed = time.perf_counter() - t0

        final_state = env.steps[-1] if env.steps else []
        winner = _episode_winner(final_state) if final_state else -1
        if 0 <= winner < num_agents:
            wins[winner] += 1

        turns = len(env.steps)
        total_turns += turns

        timeouts = 0
        for per_agent in timings_per_agent:
            timeouts += sum(1 for t in per_agent if t > 1.0)
        total_timeouts += timeouts

        scores = _player_scores(env)
        row: dict[str, Any] = {
            "episode": i,
            "winner": winner,
            "turns": turns,
            "timeouts": timeouts,
            "elapsed_sec": round(elapsed, 3),
        }
        for j in range(4):
            row[f"p{j}_score"] = scores[j] if j < len(scores) else ""
        summary_rows.append(row)

        if save_replay:
            replay_path = run_dir / f"episode_{i}.json"
            replay: Any
            if hasattr(env, "toJSON"):
                replay = env.toJSON()
            else:
                replay = env.steps
            if isinstance(replay, (dict, list)):
                replay_path.write_text(
                    json.dumps(replay, default=str), encoding="utf-8"
                )
            else:
                replay_path.write_text(str(replay), encoding="utf-8")

        timings_path = run_dir / f"episode_{i}_timings.json"
        timings_path.write_text(
            json.dumps(
                {
                    "seed": episode_seed,
                    "mode": mode,
                    "turns": turns,
                    "elapsed_sec": elapsed,
                    "per_agent_timings": timings_per_agent,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Episode %d: winner=%s turns=%d timeouts=%d elapsed=%.2fs",
            i,
            winner,
            turns,
            timeouts,
            elapsed,
        )

    summary_path = run_dir / "summary.csv"
    fieldnames = [
        "episode",
        "winner",
        "turns",
        "p0_score",
        "p1_score",
        "p2_score",
        "p3_score",
        "timeouts",
        "elapsed_sec",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    table = Table(title=f"Baseline self-play — mode={mode} episodes={episodes}")
    table.add_column("player", justify="right")
    table.add_column("wins", justify="right")
    table.add_column("win rate", justify="right")
    for j in range(num_agents):
        rate = wins[j] / episodes if episodes else 0.0
        table.add_row(f"p{j}", str(wins[j]), f"{rate:.1%}")
    table.add_row(
        "[bold]total[/bold]",
        str(sum(wins)),
        f"turns_avg={total_turns / max(1, episodes):.1f} timeouts={total_timeouts}",
    )
    console.print(table)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    app()
